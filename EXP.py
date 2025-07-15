import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from tqdm.auto import tqdm
from typing import List, Dict, Union

# --- 事件序列 ---
class EventSequence:
    def __init__(self, ops: dict, sequence: list):
        self.ops = ops
        self.sequence = sequence

    def generate_events(self, t_total: float) -> List[Dict]:
        events = []
        current_time = 0.0
        total_time_units = sum(item for item in self.sequence if isinstance(item, (int, float)))
        time_unit_duration = t_total / total_time_units if total_time_units > 0 else 0

        for item in self.sequence:
            if isinstance(item, str):
                op = self.ops[item]
                events.append({'time': current_time, 'op': op, 'type': 'unitary'})
            elif isinstance(item, (int, float)):
                current_time += item * time_unit_duration
        return events

    def add_event_sequence(self,event_sequence,num=1):
        self.ops.update(event_sequence.ops)
        self.sequence.extend(event_sequence.sequence*num)


# --- 事件管理器 ---
class QuantumEvents:
    def __init__(self, num_qubits: int = 2):
        self.events: List[Dict] = []
        self.num_qubits = num_qubits
        self.hopping_op = self._create_swapgate()

    def _create_swapgate(self) -> Qobj:
        SWAP_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        return tensor(sigmay(), sigmay())

    def add_event(self, event):
        self.events.append(event)

    def add_hopping_events(self, t_total: float, hop_mu: float = 0.03, hop_sigma: float = 1, max_hops: int = 5):
        num_hops = int(np.round(np.random.normal(hop_mu*t_total, hop_sigma*t_total)))
        num_hops = np.clip(num_hops, 0, max_hops)
        
        hop_times = sorted(np.random.uniform(0, t_total, num_hops))
        
        for t in hop_times:
            self.events.append({'time': t, 'op': self.hopping_op, 'type': 'unitary'})
        # self.events.append({'time': t_total/2, 'op': self.hopping_op, 'type': 'unitary'})    


    def extend(self, events: List[Dict]):
        self.events.extend(events)

    def get_sorted_events(self):
        return sorted(self.events, key=lambda e: e['time'])

# --- 密度矩阵演化 ---
def evolve_density_matrix(H, rho0, t_total, events, c_ops, options, eps=1e-9):
    current_rho = rho0
    boundaries = sorted(set([0.0] + [e['time'] for e in events] + [t_total]))
    unique_boundaries = []
    if boundaries:
        unique_boundaries.append(boundaries[0])
        for t in boundaries[1:]:
            if t - unique_boundaries[-1] > eps:
                unique_boundaries.append(t)

    for i in range(len(unique_boundaries) - 1):
        t0, t1 = unique_boundaries[i], unique_boundaries[i + 1]
        for e in events:
            if abs(e['time'] - t0) < eps and e['type'] == 'unitary':
                current_rho = e['op'] * current_rho * e['op'].dag()
        if abs(t1 - t0) > eps:
            result = mesolve(H, current_rho, [0, t1 - t0], c_ops, e_ops=[], options=options)
            current_rho = result.states[-1]
    for e in events:
        if abs(e['time'] - t_total) < eps and e['type'] == 'unitary':
            current_rho = e['op'] * current_rho * e['op'].dag()
    return current_rho

# --- 总模拟 ---
def time_scan(evolution_times, observables, event_sequence_obj, H_base, H_noise_op, rho0, c_ops, 
                            N_samples=100, sigma_noise=0.1 * np.pi, hop_mu=3, hop_sigma=1, max_hops=5,
                            options={'store_states': True, 'nsteps': 50000}):
    results = np.zeros((len(observables), len(evolution_times)))
    
    for i, t_total in enumerate(tqdm(evolution_times, desc="演化时间扫描")):
        avg_vals = np.zeros(len(observables))
        for _ in range(N_samples):
            event_manager = QuantumEvents(num_qubits=2)
            events = event_sequence_obj.generate_events(t_total)
            event_manager.extend(events)
            event_manager.add_hopping_events(t_total, hop_mu, hop_sigma, max_hops)
            all_events = event_manager.get_sorted_events()
            eps_k = np.random.normal(0, sigma_noise)
            H = H_base + eps_k * H_noise_op
            rho = evolve_density_matrix(H, rho0, t_total, all_events, c_ops, options)
            for j, obs in enumerate(observables):
                avg_vals[j] += expect(obs, rho)
        results[:, i] = avg_vals / N_samples
    return results

def Seq_num_scan(unit_t_total, Seq_num_scan, observables, sequence_struct,sequence_scan_label:list, H_base, H_noise_op, rho0, c_ops, 
                            N_samples=100, sigma_noise=0.1 * np.pi, hop_mu=3, hop_sigma=1, max_hops=5,
                            options={'store_states': True, 'nsteps': 50000}):
    results = np.zeros((len(observables), len(Seq_num_scan)))
    
    for i,  Seq_num in enumerate(tqdm( Seq_num_scan, desc="序列数量扫描")):
        avg_vals = np.zeros(len(observables))
        for _ in range(N_samples):
            t_total=Seq_num*unit_t_total
            event_sequence_obj=EventSequence(ops={}, sequence=[])
            for order in range(len(sequence_scan_label)):
                if sequence_scan_label[order]==1:
                    event_sequence_obj.add_event_sequence(sequence_struct[order],Seq_num)
                else:
                    event_sequence_obj.add_event_sequence(sequence_struct[order])
            
            event_manager = QuantumEvents(num_qubits=2)
            events = event_sequence_obj.generate_events(t_total)
            event_manager.extend(events)
            event_manager.add_hopping_events(t_total, hop_mu, hop_sigma, max_hops)
            all_events = event_manager.get_sorted_events()
            eps_k = np.random.normal(0, sigma_noise)
            H = H_base + eps_k * H_noise_op
            rho = evolve_density_matrix(H, rho0, t_total, all_events, c_ops, options)
            for j, obs in enumerate(observables):
                avg_vals[j] += expect(obs, rho)
        results[:, i] = avg_vals / N_samples
    return results



if __name__ == '__main__':
    g, e = basis(2, 0), basis(2, 1)
    ge, eg = tensor(g, e), tensor(e, g)
    gg = tensor(g, g)
    psi_plus = (ge + eg).unit()

    ops = {
        'sx1': tensor(sigmax(), qeye(2)),
        'sy1': tensor(sigmay(), qeye(2)),
        'sz1': tensor(sigmaz(), qeye(2)),
        'sx2': tensor(qeye(2), sigmax()),
        'sy2': tensor(qeye(2), sigmay()),
        'sz2': tensor(qeye(2), sigmaz()),
        'P01': ge.proj(),
        'P10': eg.proj(),
        'U_pi_YY': tensor(gates.ry(np.pi), gates.ry(np.pi)),
        'U_halfpi_YY': tensor(gates.ry(0.5*np.pi), gates.ry(0.5*np.pi)),
        'U_XX': tensor(gates.rx(0.5*np.pi), gates.rx(0.5*np.pi)), # 添加一个示例操作
        'P_psi_plus': psi_plus.proj()
    }
    # 示例 1: 传统的自旋回波 (N=5)
    # [脉冲, 等待, 脉冲, 等待, ...]
    spin_echo_sequence = ['U_halfpi_YY']
    for _ in range(10):
        spin_echo_sequence.append(1)
        spin_echo_sequence.append('U_pi_YY')
        spin_echo_sequence.append(1)
    spin_echo_sequence.append('U_halfpi_YY')
    print(f"自旋回波序列: {spin_echo_sequence}")
    S_spin_echo = EventSequence(ops, sequence=spin_echo_sequence)

    # 示例 2: Ramsey 实验序列
    # [等待 tau/2, 脉冲, 等待 tau/2]
    ramsey_sequence = ['U_halfpi_YY',1,'U_halfpi_YY']
    print(f"Ramsey 序列: {ramsey_sequence}")
    S_ramsey = EventSequence(ops, sequence=ramsey_sequence)
    # --- 选择一个序列来运行模拟 ---
    # 您可以在这里切换 S_spin_echo, S_ramsey来测试不同的序列
    chosen_sequence =S_spin_echo
    print(f"\n正在运行模拟...")

    halfpi=EventSequence(ops, sequence=['U_halfpi_YY'])
    spin_echo=EventSequence(ops, sequence=[1,'U_pi_YY',1])
    Spin_echo_struct=[halfpi,spin_echo,halfpi]
    scan_label=[0,1,0]
    obs_list = [ops['sx1'], ops['sy1'], ops['sz1'], tensor(sigmaz(), sigmaz())]
    evolution_times = np.linspace(0.01, 84.0, 101) # 减少时间范围以便更快看到结果
    H_base = (1 * np.pi / 2.0) * (ops['sz1'] + ops['sz2']) + \
                (0.2 * np.pi / 2.0) * (ops['P01'] - ops['P10'])
    H_noise_op = 0.1*(ops['sz1'] + ops['sz2']) / 2.0
    rho0 = gg.proj()

    c_ops = [np.sqrt(0.05) * (ops['sz1'] + ops['sz2'])]

    # results = time_scan(evolution_times,observables=obs_list,event_sequence_obj=chosen_sequence,
    #                                     H_base=H_base,H_noise_op=H_noise_op,rho0=rho0,c_ops=c_ops,N_samples=10, hop_mu=0, hop_sigma=0, max_hops=1000)
    results = Seq_num_scan(1,range(10),observables=obs_list,sequence_struct=Spin_echo_struct,sequence_scan_label=scan_label,
                                    H_base=H_base,H_noise_op=H_noise_op,rho0=rho0,c_ops=c_ops,N_samples=1, hop_mu=0, hop_sigma=0, max_hops=1000)


    # 绘图
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(["sx1", "sy1", "sz1", "szsz"]):
        plt.plot(range(10), results[i, :], '.-', label=label)

    plt.xlabel("Total Evolution Time (t_total)")
    plt.ylabel("Expectation Values")
    plt.title("Flexible Sequence Simulation with Noise")
    plt.legend()
    plt.grid(True)
    plt.show()