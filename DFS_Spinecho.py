import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from qutip import (
    basis, mesolve, sigmax, sigmay, sigmaz, qeye, tensor, Options
)

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
psi_p_initial = tensor(psi_p_single, psi_p_single)
rho_initial = psi_p_initial * psi_p_initial.dag()
psi_plus = (ge + eg).unit()
P_psi_plus_op = psi_plus * psi_plus.dag() # 重命名以区分变量和字符串
psi_minus = (ge - eg).unit()
P_psi_minus_op = psi_minus * psi_minus.dag() # 重命名
P01_op = ge * ge.dag() # 重命名
P10_op = eg * eg.dag() # 重命名

# --- (系统参数与之前相同) ---
delta_drive_H = 1.0 * np.pi
H_drive = (delta_drive_H / 2.0) * (sz1_op + sz2_op)
delta_omega_DFS = 0.1 * np.pi
H_bias_DFS = (delta_omega_DFS / 2.0) * (P01_op - P10_op) # 使用重命名后的算符
H_total = H_drive + H_bias_DFS
gamma_collective_rate = 0.005
gamma_DFS_dephasing_rate = 0.001
t_final = 45.0
t_pulse = t_final / 2.0
n_points_segment = 150
tlist_full = np.linspace(0, t_final, 2 * n_points_segment + 1)
pulse_idx = np.argmin(np.abs(tlist_full - t_pulse))
t_pulse_actual = tlist_full[pulse_idx]
qutip_options_store_states = Options(store_states=True, nsteps=10000, atol=1e-8, rtol=1e-6)
qutip_options_no_store = Options(store_states=False, nsteps=10000, atol=1e-8, rtol=1e-6)

# --- 期望算符列表 ---
e_ops_to_track = [
    sx1_op, sy1_op, sz1_op,
    sx2_op, sy2_op, sz2_op,
    P_psi_plus_op, P_psi_minus_op, # 使用重命名后的算符
    sxsx_op, sysy_op, szsz_op,
    P01_op, P10_op                # 使用重命名后的算符
]

# --- (塌缩算符与之前相同) ---
c_op_main_collective = np.sqrt(gamma_collective_rate) * (sz1_op + sz2_op)
c_ops_total = [c_op_main_collective]
if gamma_DFS_dephasing_rate > 0:
    c_op_DFS_internal_dephasing = np.sqrt(gamma_DFS_dephasing_rate) * (sz1_op - sz2_op)
    c_ops_total.append(c_op_DFS_internal_dephasing)

# --- (模拟计算与之前相同) ---
print("开始模拟：无回波...")
result_no_echo = mesolve(H_total, rho_initial, tlist_full, c_ops_total, e_ops_to_track, options=qutip_options_no_store)
print("模拟完成：无回波。")
print(f"开始模拟：有回波 - 第一段演化至 t={t_pulse_actual:.2f}...")
tlist_part1 = tlist_full[:pulse_idx + 1]
result_part1 = mesolve(H_total, rho_initial, tlist_part1, c_ops_total, e_ops_to_track, options=qutip_options_store_states)
rho_at_pulse = result_part1.states[-1]
print("第一段演化完成。")
U_pi_pulse = tensor(sigmay(), sigmay())
print("施加 pi_y x pi_y 脉冲...")
rho_after_pulse = U_pi_pulse * rho_at_pulse * U_pi_pulse.dag()
print(f"开始模拟：有回波 - 第二段演化从 t={t_pulse_actual:.2f} 至 t={t_final:.2f}...")
tlist_part2_relative = tlist_full[pulse_idx:] - t_pulse_actual
result_part2 = mesolve(H_total, rho_after_pulse, tlist_part2_relative, c_ops_total, e_ops_to_track, options=qutip_options_no_store)
print("第二段演化完成。")
results_echo_expect = []
for i in range(len(e_ops_to_track)):
    combined_expect = np.concatenate((result_part1.expect[i][:-1], result_part2.expect[i]))
    results_echo_expect.append(combined_expect)
print("数据合并完成。")


# --- 绘图 ---
fig, axes = plt.subplots(5, 1, figsize=(14, 24), sharex=True)

plot_linewidth = 1.5
plot_linestyle_no_echo = ':'
plot_linestyle_echo = '-'
pulse_line_color = 'gray'

# --- 预定义 LaTeX 字符串 ---
tex_sx1 = r"$\langle \sigma_x^{(1)} \rangle$"
tex_sy1 = r"$\langle \sigma_y^{(1)} \rangle$"
tex_sz1 = r"$\langle \sigma_z^{(1)} \rangle$"
tex_sx2 = r"$\langle \sigma_x^{(2)} \rangle$"
tex_sy2 = r"$\langle \sigma_y^{(2)} \rangle$"
tex_sz2 = r"$\langle \sigma_z^{(2)} \rangle$"
tex_sxsx = r"$\langle \sigma_x^{(1)}\sigma_x^{(2)} \rangle$"
tex_sysy = r"$\langle \sigma_y^{(1)}\sigma_y^{(2)} \rangle$"
tex_szsz = r"$\langle \sigma_z^{(1)}\sigma_z^{(2)} \rangle$"
tex_P_psi_plus = r"$P_{|\Psi^+\rangle}$" # 移除了多余的 LaTeX 结构
tex_P_psi_minus = r"$P_{|\Psi^-\rangle}$"
tex_P01 = r"$P_{|01\rangle}$"
tex_P10 = r"$P_{|10\rangle}$"


# --- 子图 0: 离子 1 演化 ---
ax = axes[0]
ax.plot(tlist_full, result_no_echo.expect[0], label=f'无回波 {tex_sx1}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[1], label=f'无回波 {tex_sy1}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[2], label=f'无回波 {tex_sz1}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[0], label=f'回波 {tex_sx1}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[1], label=f'回波 {tex_sy1}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[2], label=f'回波 {tex_sz1}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0, label=f'$\pi$-pulse @ t={t_pulse_actual:.1f}')
ax.set_title('离子 1 的期望值演化')
ax.set_ylabel('期望值')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-1.1, 1.1])

# --- 子图 1: 离子 2 演化 ---
ax = axes[1]
ax.plot(tlist_full, result_no_echo.expect[3], label=f'无回波 {tex_sx2}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[4], label=f'无回波 {tex_sy2}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[5], label=f'无回波 {tex_sz2}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[3], label=f'回波 {tex_sx2}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[4], label=f'回波 {tex_sy2}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[5], label=f'回波 {tex_sz2}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title('离子 2 的期望值演化')
ax.set_ylabel('期望值')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-1.1, 1.1])

# --- 子图 2: 双比特关联项 ---
ax = axes[2]
ax.plot(tlist_full, result_no_echo.expect[8], label=f'无回波 {tex_sxsx}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[9], label=f'无回波 {tex_sysy}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[10], label=f'无回波 {tex_szsz}', linestyle=plot_linestyle_no_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[8], label=f'回波 {tex_sxsx}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[9], label=f'回波 {tex_sysy}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[10], label=f'回波 {tex_szsz}', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title('双比特关联项演化')
ax.set_ylabel('期望值')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-1.1, 1.1])

# --- 子图 3: DFS 态概率 (Psi+ 和 Psi-) ---
ax = axes[3]
# 对于 P_psi_plus 和 P_psi_minus，它们已经是概率，不需要额外的 LaTeX 结构如 <...>
ax.plot(tlist_full, result_no_echo.expect[6], label=f'无回波 {tex_P_psi_plus}', linestyle=plot_linestyle_no_echo, color='purple', linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[7], label=f'无回波 {tex_P_psi_minus}', linestyle=plot_linestyle_no_echo, color='orange', linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[6], label=f'回波 {tex_P_psi_plus}', color='purple', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[7], label=f'回波 {tex_P_psi_minus}', color='orange', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title('DFS 子空间内贝尔态 的布居概率') # 这里中文内的LaTeX直接写
ax.set_ylabel('概率')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-0.1, 1.1])

# --- 子图 4: DFS 基矢 |01> 和 |10> 的布居概率 ---
ax = axes[4]
ax.plot(tlist_full, result_no_echo.expect[11], label=f'无回波 {tex_P01}', linestyle=plot_linestyle_no_echo, color='brown', linewidth=plot_linewidth)
ax.plot(tlist_full, result_no_echo.expect[12], label=f'无回波 {tex_P10}', linestyle=plot_linestyle_no_echo, color='magenta', linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[11], label=f'回波 {tex_P01}', color='brown', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.plot(tlist_full, results_echo_expect[12], label=f'回波 {tex_P10}', color='magenta', linestyle=plot_linestyle_echo, linewidth=plot_linewidth)
ax.axvline(t_pulse_actual, color=pulse_line_color, linestyle='--', linewidth=1.0)
ax.set_title('DFS 基矢 |01, |10的布居概率') # 这里中文内的LaTeX直接写
ax.set_xlabel(f'时间 (总时间={t_final:.1f}, $\gamma_{{coll}}$={gamma_collective_rate}, $\gamma_{{DFS}}$={gamma_DFS_dephasing_rate})')
ax.set_ylabel('概率')
ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5))
ax.grid(True)
ax.set_ylim([-0.1, 0.6])

fig.suptitle(f'自旋回波下演化 (H_drive freq ~{delta_drive_H/(2*np.pi):.2f}*2pi, H_bias_DFS freq ~{delta_omega_DFS/(2*np.pi):.2f}*2pi)', fontsize=12)
plt.tight_layout(rect=[0, 0.03, 0.85, 0.96])
plt.show()

# --- (期望值检查与之前相同) ---
print("\n--- 期望值检查 (初始时刻) ---")
op_names = [
    "<sx1>", "<sy1>", "<sz1>",
    "<sx2>", "<sy2>", "<sz2>",
    "P(Psi+)", "P(Psi-)",
    "<sxsx>", "<sysy>", "<szsz>",
    "P01", "P10"
]
theory_initial_values = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.25, 0.25]
for i, name in enumerate(op_names):
    sim_val = result_no_echo.expect[i][0]
    theory_val = theory_initial_values[i]
    print(f"初始 {name}: {sim_val:.4f} (理论: {theory_val:.2f})")