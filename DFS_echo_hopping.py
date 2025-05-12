import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from qutip import (
    basis, mesolve, sigmax, sigmay, sigmaz, qeye, tensor, Options, expect, Qobj
)
from tqdm import tqdm # 用于显示进度条

# --- (字体配置与之前相同) ---
try:
    font_preferences = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'PingFang SC', 'Noto Sans CJK SC']
    found_font = False
    for font_name in font_preferences:
        try:
            matplotlib.font_manager.findfont(font_name)
            plt.rcParams['font.sans-serif'] = [font_name]
            print(f"中文配置：成功设置字体为 {font_name}")
            found_font = True
            break
        except Exception:
            continue
    if not found_font:
        print("中文配置：未能找到并设置优先选择的中文字体。中文可能无法正确显示。")
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置Matplotlib字体参数时发生错误: {e}")

# --- (Qobj 定义与之前相同) ---
g_state = basis(2, 0)
e_state = basis(2, 1)
ge = tensor(g_state, e_state)
eg = tensor(e_state, g_state)
sx1_op = tensor(sigmax(), qeye(2))
sy1_op = tensor(sigmay(), qeye(2))
sz1_op = tensor(sigmaz(), qeye(2))
sx2_op = tensor(qeye(2), sigmax())
sy2_op = tensor(qeye(2), sigmay())
sz2_op = tensor(qeye(2), sigmaz())
I4 = tensor(qeye(2), qeye(2))
sxsx_op = tensor(sigmax(), sigmax())
sysy_op = tensor(sigmay(), sigmay())
szsz_op = tensor(sigmaz(), sigmaz())
psi_p_single = (g_state + e_state).unit()
psi_p_initial = tensor(psi_p_single, psi_p_single) # 初始态 (|g>+|e>)(|g>+|e>)/2
rho_initial = psi_p_initial * psi_p_initial.dag()
psi_plus = (ge + eg).unit()
P_psi_plus_op = psi_plus * psi_plus.dag()
psi_minus = (ge - eg).unit()
P_psi_minus_op = psi_minus * psi_minus.dag()
P01_op = ge * ge.dag()
P10_op = eg * eg.dag()

# --- SWAP Operator ---
# SWAP = 0.5 * (I⊗I + σx⊗σx + σy⊗σy + σz⊗σz)
SWAP_op = 0.5 * (I4 + sxsx_op + sysy_op + szsz_op)
# Verify it's unitary and acts as SWAP:
# print("SWAP hermitian:", SWAP_op.isherm) # True
# print("SWAP^2 = I:", (SWAP_op * SWAP_op - I4).norm() < 1e-9) # True
# test_state = tensor(basis(2,0), basis(2,1)) # |01>
# swapped_state = SWAP_op * test_state
# print("SWAP |01> -> |10>?", (swapped_state - tensor(basis(2,1), basis(2,0))).norm() < 1e-9) # True

# --- 系统参数 ---
delta_drive_H = 1.0 * np.pi  # 主驱动哈密顿量的强度
H_drive_base = (delta_drive_H / 2.0) * (sz1_op + sz2_op) # 驱动的基础部分

delta_omega_DFS = 0.1 * np.pi # DFS子空间内|01>和|10>的能量劈裂
H_bias_DFS = (delta_omega_DFS / 2.0) * (P01_op - P10_op)

# 哈密顿量的固定部分
H_fixed_part = H_drive_base + H_bias_DFS

# --- 准静态集体噪声参数 ---
N_quasi_static_samples = 50 # 为了速度，可以先设小一点，如10-50 (原为50)
sigma_quasi_static_noise_strength = 0.1 * np.pi # 准静态噪声的标准差
H_quasi_static_operator = (sz1_op + sz2_op) / 2.0

# --- Hopping Parameters ---
N_max_hops = 0              # 每条轨迹最多发生几次hopping
time_precision_epsilon = 1e-9 # 用于避免事件时间完全重合

# --- 其他退相干参数 (动态噪声) ---
gamma_collective_rate = 0.02
gamma_DFS_dephasing_rate = 0.001

# --- 时间参数 ---
t_final = 45.0
t_pulse = t_final / 2.0
n_points_segment = 150
tlist_full = np.linspace(0, t_final, 2 * n_points_segment + 1)
pulse_idx = np.argmin(np.abs(tlist_full - t_pulse))
t_pulse_actual = tlist_full[pulse_idx]

# --- QuTiP 选项 ---
# We need store_states=True for segmented evolution
qutip_options_segmented_evo = Options(store_states=True, nsteps=5000, atol=1e-7, rtol=1e-5)

# --- 期望算符列表 (与之前相同) ---
e_ops_to_track = [
    sx1_op, sy1_op, sz1_op, sx2_op, sy2_op, sz2_op,
    P_psi_plus_op, P_psi_minus_op, sxsx_op, sysy_op, szsz_op,
    P01_op, P10_op
]
num_e_ops = len(e_ops_to_track)

# --- 塌缩算符 (动态噪声部分，与之前相同) ---
c_ops_dynamic = []
if gamma_collective_rate > 0:
    c_ops_dynamic.append(np.sqrt(gamma_collective_rate) * (sz1_op + sz2_op))
if gamma_DFS_dephasing_rate > 0:
    c_ops_dynamic.append(np.sqrt(gamma_DFS_dephasing_rate) * (sz1_op - sz2_op))

# --- Segmented Evolution Function ---
def run_path_simulation(H_const, rho_start_state, full_tlist_global, c_ops_list,
                        e_ops_list_global, events_this_path, qutip_options_cfg):
    """
    Simulates a quantum system's evolution path, applying unitary events at specified times.

    Args:
        H_const: The time-independent part of the Hamiltonian.
        rho_start_state: The initial density matrix for this path.
        full_tlist_global: The desired list of time points for final output.
        c_ops_list: List of collapse operators.
        e_ops_list_global: List of expectation value operators to track.
        events_this_path: A list of dictionaries, each {'time': t, 'type': 'unitary', 'op': U}.
                          MUST be sorted by time.
        qutip_options_cfg: QuTiP options object (needs store_states=True).

    Returns:
        A list of numpy arrays, where each array contains the expectation values
        for one operator in e_ops_list_global, evaluated at times in full_tlist_global.
    """
    current_rho = rho_start_state
    
    # Initialize storage for the expectation values for this specific path
    # These will align with full_tlist_global
    path_expects = [np.zeros_like(full_tlist_global, dtype=float) for _ in e_ops_list_global]

    # Define all segment boundaries: start, end, and all event times
    event_times_only = [evt['time'] for evt in events_this_path]
    all_segment_boundaries = np.unique(np.sort(np.concatenate(
        ([full_tlist_global[0]], event_times_only, [full_tlist_global[-1]])
    )))

    for i in range(len(all_segment_boundaries) - 1):
        t_seg_start = all_segment_boundaries[i]
        t_seg_end = all_segment_boundaries[i+1]

        if t_seg_end - t_seg_start < time_precision_epsilon / 10: # Skip zero-duration or tiny segments
            # Still apply events if any are exactly at t_seg_start (which is also t_seg_end here)
            for event in events_this_path:
                if np.isclose(event['time'], t_seg_start, atol=time_precision_epsilon):
                    if event['type'] == 'unitary':
                        current_rho = event['op'] * current_rho * event['op'].dag()
            continue

        # Determine mesolve tlist for this segment:
        # It should include t_seg_start, t_seg_end, and any points from full_tlist_global within this segment.
        
        # Points from full_tlist_global strictly between t_seg_start and t_seg_end
        strict_interior_indices = np.where(
            (full_tlist_global > t_seg_start + time_precision_epsilon) &
            (full_tlist_global < t_seg_end - time_precision_epsilon)
        )[0]
        
        times_for_mesolve_abs = [t_seg_start]
        if len(strict_interior_indices) > 0:
            times_for_mesolve_abs.extend(full_tlist_global[strict_interior_indices])
        times_for_mesolve_abs.append(t_seg_end)
        
        times_for_mesolve_abs = np.unique(np.sort(times_for_mesolve_abs))
        
        # Ensure bounds are respected after unique sort (due to potential float precision issues)
        times_for_mesolve_abs = times_for_mesolve_abs[
            (times_for_mesolve_abs >= t_seg_start - time_precision_epsilon) &
            (times_for_mesolve_abs <= t_seg_end + time_precision_epsilon)
        ]
        if not np.isclose(times_for_mesolve_abs[0], t_seg_start): times_for_mesolve_abs[0] = t_seg_start
        if not np.isclose(times_for_mesolve_abs[-1], t_seg_end): times_for_mesolve_abs[-1] = t_seg_end


        times_for_mesolve_rel = times_for_mesolve_abs - t_seg_start
        
        # Ensure relative times are non-negative and first is zero
        times_for_mesolve_rel[times_for_mesolve_rel < 0] = 0
        if not np.isclose(times_for_mesolve_rel[0],0.0): times_for_mesolve_rel = np.insert(times_for_mesolve_rel,0,0.0)
        times_for_mesolve_rel = np.unique(times_for_mesolve_rel)


        if len(times_for_mesolve_rel) == 0 : # Should not happen with current logic but as a safeguard
             result_segment_states = [current_rho] # No evolution, state remains
             result_segment_expect = [np.array([expect(op, current_rho)]) for op in e_ops_list_global]
        elif np.isclose(times_for_mesolve_rel[-1], 0.0) and len(times_for_mesolve_rel) == 1 : # Zero duration evolution
             result_segment_states = [current_rho]
             result_segment_expect = [np.array([expect(op, current_rho)]) for op in e_ops_list_global]
        else:
            result_segment = mesolve(H_const, current_rho, times_for_mesolve_rel,
                                     c_ops_list, e_ops_list_global, options=qutip_options_cfg)
            result_segment_states = result_segment.states
            result_segment_expect = result_segment.expect


        # Store expectation values into the path_expects array
        for op_idx_global in range(len(e_ops_list_global)):
            data_from_segment = result_segment_expect[op_idx_global]
            for t_idx_in_segment, t_abs_val_in_seg in enumerate(times_for_mesolve_abs):
                # Skip if this data point is the start of the current segment,
                # AND this segment is not the very first one (t_seg_start > t_global_start).
                # This avoids double-counting data at segment boundaries.
                if t_idx_in_segment == 0 and t_seg_start > full_tlist_global[0] + time_precision_epsilon:
                    continue

                # Find the corresponding index in full_tlist_global
                match_indices = np.where(np.isclose(full_tlist_global, t_abs_val_in_seg, atol=time_precision_epsilon/2))[0]
                if len(match_indices) > 0:
                    target_output_idx = match_indices[0]
                    path_expects[op_idx_global][target_output_idx] = data_from_segment[t_idx_in_segment]
        
        current_rho = result_segment_states[-1] # State at t_seg_end

        # Apply any unitary events scheduled exactly at t_seg_end
        for event in events_this_path:
            if np.isclose(event['time'], t_seg_end, atol=time_precision_epsilon):
                if event['type'] == 'unitary':
                    current_rho = event['op'] * current_rho * event['op'].dag()
    return path_expects


# --- 初始化用于存储平均期望值的数组 ---
avg_result_no_echo_expect = [np.zeros_like(tlist_full, dtype=float) for _ in range(num_e_ops)]
avg_results_echo_expect = [np.zeros_like(tlist_full, dtype=float) for _ in range(num_e_ops)]

U_pi_pulse = tensor(sigmay(), sigmay()) # Pi脉冲 operator

print(f"开始模拟，包含 {N_quasi_static_samples} 个准静态噪声样本和最多 {N_max_hops} 次hopping...")

for k_sample in tqdm(range(N_quasi_static_samples), desc="准静态噪声采样"):
    epsilon_k = np.random.normal(0, sigma_quasi_static_noise_strength)
    H_total_current_sample = H_fixed_part + epsilon_k * H_quasi_static_operator

    # Generate hop times for this specific k_sample
    num_actual_hops = np.random.randint(0, N_max_hops + 1)
    raw_hop_times = np.random.uniform(0 + time_precision_epsilon, t_final - time_precision_epsilon, num_actual_hops)
    # Ensure hops don't coincide with pi-pulse time for simplicity in event ordering
    hop_times_k = raw_hop_times[np.abs(raw_hop_times - t_pulse_actual) > time_precision_epsilon]
    hop_times_k = np.sort(hop_times_k)

    # 1. No Echo Path
    events_no_echo = []
    for t_hop in hop_times_k:
        events_no_echo.append({'time': t_hop, 'type': 'unitary', 'op': SWAP_op})
    events_no_echo.sort(key=lambda x: x['time']) # Sort by time

    expects_no_echo_k = run_path_simulation(
        H_total_current_sample, rho_initial, tlist_full, c_ops_dynamic,
        e_ops_to_track, events_no_echo, qutip_options_segmented_evo
    )
    for i in range(num_e_ops):
        avg_result_no_echo_expect[i] += expects_no_echo_k[i]

    # 2. Echo Path
    events_echo = []
    for t_hop in hop_times_k:
        events_echo.append({'time': t_hop, 'type': 'unitary', 'op': SWAP_op})
    events_echo.append({'time': t_pulse_actual, 'type': 'unitary', 'op': U_pi_pulse})
    events_echo.sort(key=lambda x: x['time']) # Sort by time

    expects_echo_k = run_path_simulation(
        H_total_current_sample, rho_initial, tlist_full, c_ops_dynamic,
        e_ops_to_track, events_echo, qutip_options_segmented_evo
    )
    for i in range(num_e_ops):
        avg_results_echo_expect[i] += expects_echo_k[i]

# --- 计算平均值 ---
for i in range(num_e_ops):
    avg_result_no_echo_expect[i] /= N_quasi_static_samples
    avg_results_echo_expect[i] /= N_quasi_static_samples

print("所有模拟和平均完成。")


# --- 绘图 (与之前几乎相同, 只是数据源改变) ---
fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True)

plot_linewidth = 1.5
plot_linestyle_no_echo = ':'
plot_linestyle_echo = '-'
pulse_line_color = 'gray'

# --- 预定义 LaTeX 字符串 (与之前相同) ---
tex_sx1 = r"$\langle \sigma_x^{(1)} \rangle$"
tex_sy1 = r"$\langle \sigma_y^{(1)} \rangle$"
tex_sz1 = r"$\langle \sigma_z^{(1)} \rangle$"
tex_sx2 = r"$\langle \sigma_x^{(2)} \rangle$"
tex_sy2 = r"$\langle \sigma_y^{(2)} \rangle$"
tex_sz2 = r"$\langle \sigma_z^{(2)} \rangle$"
tex_sxsx = r"$\langle \sigma_x^{(1)}\sigma_x^{(2)} \rangle$"
tex_sysy = r"$\langle \sigma_y^{(1)}\sigma_y^{(2)} \rangle$"
tex_szsz = r"$\langle \sigma_z^{(1)}\sigma_z^{(2)} \rangle$"
tex_P_psi_plus = r"$P_{|\Psi^+\rangle}$"
tex_P_psi_minus = r"$P_{|\Psi^-\rangle}$"
tex_P01 = r"$P_{|01\rangle}$"
tex_P10 = r"$P_{|10\rangle}$"


# --- 子图 0: 离子 1 演化 ---
ax = axes[0]
ax.plot(tlist_full, avg_result_no_echo_expect[0], label=f'无回波 {tex_sx1}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[1], label=f'无回波 {tex_sy1}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[2], label=f'无回波 {tex_sz1}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[0], label=f'回波 {tex_sx1}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[1], label=f'回波 {tex_sy1}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[2], label=f'回波 {tex_sz1}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0, label=f'$\pi$-pulse & Hops @ t={t_pulse_actual:.1f}')
ax.set_title('离子 1 的期望值演化 (含准静态噪声和随机Hopping)')
ax.set_ylabel('期望值')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-1.1, 1.1])

# --- 子图 1: 离子 2 演化 ---
ax = axes[1]
ax.plot(tlist_full, avg_result_no_echo_expect[3], label=f'无回波 {tex_sx2}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[4], label=f'无回波 {tex_sy2}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[5], label=f'无回波 {tex_sz2}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[3], label=f'回波 {tex_sx2}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[4], label=f'回波 {tex_sy2}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[5], label=f'回波 {tex_sz2}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title('离子 2 的期望值演化 (含准静态噪声和随机Hopping)')
ax.set_ylabel('期望值')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-1.1, 1.1])

# --- 子图 2: 双比特关联项 ---
ax = axes[2]
ax.plot(tlist_full, avg_result_no_echo_expect[8], label=f'无回波 {tex_sxsx}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[9], label=f'无回波 {tex_sysy}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[10], label=f'无回波 {tex_szsz}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[8], label=f'回波 {tex_sxsx}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[9], label=f'回波 {tex_sysy}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[10], label=f'回波 {tex_szsz}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title('双比特关联项演化 (含准静态噪声和随机Hopping)')
ax.set_ylabel('期望值')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-1.1, 1.1])

# --- 子图 3: DFS 态概率 (Psi+ 和 Psi-) ---
ax = axes[3]
ax.plot(tlist_full, avg_result_no_echo_expect[6], label=f'无回波 {tex_P_psi_plus}', linestyle=plot_linestyle_no_echo, color='purple', linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[7], label=f'无回波 {tex_P_psi_minus}', linestyle=plot_linestyle_no_echo, color='orange', linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[6], label=f'回波 {tex_P_psi_plus}', color='purple', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[7], label=f'回波 {tex_P_psi_minus}', color='orange', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title(r'DFS 子空间内贝尔态 ($|\Psi^+\rangle$, $|\Psi^-\rangle$) 的布居概率 (含准静态噪声和随机Hopping)')
ax.set_ylabel('概率')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-0.1, 1.1])

# --- 子图 4: DFS 基矢 |01> 和 |10> 的布居概率 ---
ax = axes[4]
ax.plot(tlist_full, avg_result_no_echo_expect[11], label=f'无回波 {tex_P01}', linestyle=plot_linestyle_no_echo, color='brown', linewidth=plot_linewidth)
ax.plot(tlist_full, avg_result_no_echo_expect[12], label=f'无回波 {tex_P10}', linestyle=plot_linestyle_no_echo, color='magenta', linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[11], label=f'回波 {tex_P01}', color='brown', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, avg_results_echo_expect[12], label=f'回波 {tex_P10}', color='magenta', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title(r'DFS 基矢 ($|01\rangle$, $|10\rangle$) 的布居概率 (含准静态噪声和随机Hopping)')
ax.set_xlabel(f'时间 (总时间={t_final:.1f}, $\gamma_{{coll}}$={gamma_collective_rate}, $\gamma_{{DFS}}$={gamma_DFS_dephasing_rate}, $\sigma_{{QS}}$={sigma_quasi_static_noise_strength/(np.pi):.2f}$\pi$, max hops={N_max_hops})')
ax.set_ylabel('概率')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-0.1, 0.6]) # Adjusted based on typical probabilities for these states

fig.suptitle(f'自旋回波下演化 (H_drive ~{delta_drive_H/(2*np.pi):.2f}*2pi, H_bias_DFS ~{delta_omega_DFS/(2*np.pi):.2f}*2pi, N_samples={N_quasi_static_samples})', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 0.85, 0.96]) # Adjust layout to make space for legend
plt.show()

# --- (期望值检查与之前相同, 但现在是基于平均后的初始值) ---
print("\n--- 期望值检查 (t=0, 基于平均结果) ---")
op_names = [
    "<sx1>", "<sy1>", "<sz1>", "<sx2>", "<sy2>", "<sz2>",
    "P(Psi+)", "P(Psi-)", "<sxsx>", "<sysy>", "<szsz>",
    "P01", "P10"
]
rho_initial_for_check = psi_p_initial * psi_p_initial.dag()
theory_initial_values_calc = [expect(op, rho_initial_for_check) for op in e_ops_to_track]

for i, name in enumerate(op_names):
    sim_val_t0 = avg_result_no_echo_expect[i][0] # 取平均后的 t=0 值
    theory_val = theory_initial_values_calc[i]
    print(f"初始 {name}: {sim_val_t0:.4f} (理论: {theory_val:.4f})")