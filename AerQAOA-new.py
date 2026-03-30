import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None

# ===================== Shared QUBO setup =====================

N_QUBITS = 6

qubo_const = 5662.8
qubo_linear = {0:-1681.1,1:-1737.7,2:-2116.7,3:-828.3,4:-2173.3,5:-828.3}
qubo_quad = {(2,4):1306.8,(0,1):871.2,(2,3):871.2,(0,5):871.2,(4,5):871.2,(1,3):871.2}

ENERGY_SCALE = 435.6
INIT_STATES = ["000101", "100110", "011001", "111010"]
LAMBDA = 0.4

# ===================== Utilities =====================

def qubo_cost_from_bits(bits):
    val = qubo_const
    for i, a in qubo_linear.items():
        val += a * bits[i]
    for (i, j), a in qubo_quad.items():
        val += a * bits[i] * bits[j]
    return float(val)

def check_constraints_user_sec_geq1(bits):
    x01, x02, x10, x12, x20, x21 = bits
    return all([
        x10 + x20 == 2,
        x01 + x02 == 2,
        x10 + x12 == 1,
        x01 + x21 == 1,
        x20 + x21 == 1,
        x02 + x12 == 1,
        x10 + x20 >= 1,
    ])

def brute_force_feasible_optimum():
    best_e = float("inf")
    best_states = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        if not check_constraints_user_sec_geq1(bits):
            continue
        e = qubo_cost_from_bits(bits)
        if e < best_e - 1e-12:
            best_e = e
            best_states = [bits]
        elif abs(e - best_e) <= 1e-12:
            best_states.append(bits)
    return best_e, best_states

def bits_to_xstr(bits):
    return "".join(str(b) for b in bits)  # bits in variable order q0..q5

def qubo_to_ising_standard():
    J_zz = {}
    h_z = {i: 0.0 for i in range(N_QUBITS)}
    c0 = float(qubo_const)
    for (i, j), a in qubo_quad.items():
        a = float(a)
        J_zz[(i, j)] = J_zz.get((i, j), 0.0) + a / 4.0
        h_z[i] += -a / 4.0
        h_z[j] += -a / 4.0
        c0 += a / 4.0
    for i, a in qubo_linear.items():
        a = float(a)
        h_z[i] += -a / 2.0
        c0 += a / 2.0
    return J_zz, h_z, c0

def build_uniform_superposition_statevector(bitstrings, n):
    m = len(bitstrings)
    vec = np.zeros(2**n, dtype=complex)
    amp = 1.0 / np.sqrt(m)
    for s in bitstrings:
        bits = [int(ch) for ch in s]  # variable order q0..q_{n-1}
        idx = sum(bits[i] * (2**i) for i in range(n))  # little-endian basis index
        vec[idx] += amp
    vec /= np.linalg.norm(vec)
    return vec

def counts_to_xorder(counts):
    # Qiskit order q_{n-1}..q_0 -> variable order q_0..q_{n-1}
    return {k[::-1]: v for k, v in counts.items()}

def expected_energy_from_counts(counts_x):
    shots = sum(counts_x.values())
    e = 0.0
    for xstr, c in counts_x.items():
        bits = tuple(int(ch) for ch in xstr)
        e += (c / shots) * qubo_cost_from_bits(bits)
    return float(e)

# ===================== Ising coefficients and initial state =====================

J_zz, h_z, ising_const = qubo_to_ising_standard()
psi0_vec = build_uniform_superposition_statevector(INIT_STATES, N_QUBITS)

def apply_cost_layer(qc, gamma):
    for (i, j), J in J_zz.items():
        Js = J / ENERGY_SCALE
        if abs(Js) > 1e-12:
            qc.rzz(2.0 * gamma * Js, i, j)
    for i, h in h_z.items():
        hs = h / ENERGY_SCALE
        if abs(hs) > 1e-12:
            qc.rz(2.0 * gamma * hs, i)

def apply_custom_mixer(qc, beta, lam=LAMBDA):
    qc.rxx(2.0 * beta, 2, 3)
    qc.ryy(2.0 * beta, 2, 3)
    qc.rxx(2.0 * beta, 4, 5)
    qc.ryy(2.0 * beta, 4, 5)
    qc.rx(2.0 * beta * lam, 0)
    qc.rx(2.0 * beta * lam, 1)

def build_qaoa_circuit(gammas, betas, with_measurements=True):
    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(psi0_vec, list(range(N_QUBITS)))
    for l in range(len(gammas)):
        apply_cost_layer(qc, gammas[l])
        apply_custom_mixer(qc, betas[l], lam=LAMBDA)
    if with_measurements:
        qc.measure_all()
    return qc


def save_circuit_diagram(qc, filepath="qaoa_circuit.pdf", fold=-1, dpi=150):
    """Generate and save the circuit diagram as PDF (requires matplotlib). Behavior matches AerQAOA.py."""
    if not _HAS_MPL:
        print("[Circuit] 未安装 matplotlib，无法生成 PDF。请安装: pip install matplotlib")
        return False
    try:
        fig = qc.draw(output="mpl", fold=fold)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Circuit] 电路图已保存: {filepath}")
        return True
    except Exception as e:
        print(f"[Circuit] 保存 PDF 失败: {e}")
        return False


# ===================== Noise model =====================

def build_custom_noise_model():
    noise_model = NoiseModel()

    # Symmetric readout error: p01 = p10 = 0.001 (~99.9% readout fidelity).
    p01, p10 = 0.001, 0.001
    ro_err = ReadoutError([[1 - p01, p01],
                           [p10, 1 - p10]])
    for q in range(N_QUBITS):
        noise_model.add_readout_error(ro_err, [q])

    # Single-qubit depolarizing: mean gate infidelity r̄₁ = p₁/2 (Qiskit Aer λ).
    p1 = 0.00015  # r̄₁ = p₁/2 ≈ 7.5e-5 (single-qubit infidelity)
    err_1q = depolarizing_error(p1, 1)
    for gate in ["h", "rz", "rx"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)

    # Two-qubit depolarizing: mean gate infidelity r̄₂ = 3p₂/4 (Qiskit Aer λ).
    p2 = 0.00125  # r̄₂ = 3p₂/4 ≈ 9.375e-4 (two-qubit infidelity)
    err_2q = depolarizing_error(p2, 2)
    for gate in ["rzz", "rxx", "ryy"]:
        noise_model.add_all_qubit_quantum_error(err_2q, gate)

    return noise_model

noise_model = build_custom_noise_model()

# IMPORTANT: Do not fix seed_simulator here; pass it per run() for paired reproducibility.
NOISY_SIM = AerSimulator(noise_model=noise_model)

def transpile_for_noisy(qc, seed_transpiler=999):
    return transpile(
        qc,
        seed_transpiler=seed_transpiler,
        optimization_level=1,
        basis_gates=["h", "rz", "rx", "rzz", "rxx", "ryy"]
    )

# ===================== Noisy objective =====================

def qaoa_objective_noisy(theta, p, shots_obj, seed, batches, seed_transpiler=999):
    gammas = theta[:p]
    betas = theta[p:]
    vals = []
    for b in range(batches):
        qc = build_qaoa_circuit(gammas, betas, with_measurements=True)
        tqc = transpile_for_noisy(qc, seed_transpiler=seed_transpiler)
        result = NOISY_SIM.run(
            tqc,
            shots=shots_obj,
            seed_simulator=seed + b
        ).result()
        counts = result.get_counts()
        counts_x = counts_to_xorder(counts)
        vals.append(expected_energy_from_counts(counts_x) / ENERGY_SCALE)
    return float(np.mean(vals))

# ===================== Statistics helpers (same four metrics) =====================

def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return m, s

def mean_ci95_t(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    m, s = mean_std(arr)
    if n <= 1:
        return m, (m, m)
    t = 2.045 if n == 30 else 2.0
    half = t * s / sqrt(n)
    return m, (m - half, m + half)

def wilson_ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z / denom) * sqrt((phat*(1-phat)/n) + (z*z/(4*n*n)))
    return (max(0.0, center - half), min(1.0, center + half))

def tts_shots_for_success(p_star, target=0.99):
    if p_star <= 0.0:
        return float("inf")
    if p_star >= 1.0:
        return 1.0
    return float(ceil(log(1.0 - target) / log(1.0 - p_star)))

def sampling_rank(counts_dict, opt_xstr_set):
    """Rank of the optimal solution among sampled bitstrings ordered by frequency (1 = most frequent)."""
    sorted_xstrs = sorted(counts_dict.keys(), key=lambda x: -counts_dict[x])
    best_rank = None
    for xstr in opt_xstr_set:
        if xstr not in counts_dict:
            continue
        rank = 1 + sorted_xstrs.index(xstr)
        if best_rank is None or rank < best_rank:
            best_rank = rank
    return best_rank if best_rank is not None else len(counts_dict) + 1

# ===================== Ground truth: feasible optimum =====================

C_star_feas, opt_states_feas = brute_force_feasible_optimum()
opt_xstr_set = {bits_to_xstr(s) for s in opt_states_feas}

# Generate example circuit diagram (p=2, default angles; failure does not affect subsequent runs).
_example_g = [0.1, 0.2]
_example_b = [0.3, 0.4]
qc_example = build_qaoa_circuit(_example_g, _example_b, with_measurements=True)
save_circuit_diagram(qc_example, filepath="qaoa_circuit_new_example.pdf", fold=-1, dpi=150)

print("\n[Feasible optimum]")
print("C*_feas =", C_star_feas)
print("Optimal feasible state(s) =", opt_states_feas)
print("Optimal feasible xstr set =", opt_xstr_set)

# ===================== Run 30 experiments (paired) =====================

# Keep these consistent with the previous noisy baseline for fair comparison.
p = 2
num_restarts = 8
shots_obj = 2048
shots_final = 8192
batches_obj = 2
maxiter = 120
rhobeg = 0.5
tol = 2e-3
seed_transpiler = 999

N_RUNS = 30
BASE_SEED = 12345  # Use same value as baseline for paired comparison.

p_star_list = []
gap_exp_list = []
run_success_list = []
tts99_list = []
rank_list = []

for run_id in range(N_RUNS):
    seed_run = BASE_SEED + run_id
    rng = np.random.default_rng(seed_run)

    best_val = float("inf")
    best_res = None

    for r in range(num_restarts):
        x0 = np.concatenate([
            rng.uniform(-np.pi, np.pi, size=p),
            rng.uniform(0.0, np.pi/2.0, size=p),
        ])

        # Objective evaluation seed (paired with baseline).
        seed_obj = 7000 + 100 * r + 10_000 * run_id

        res = minimize(
            qaoa_objective_noisy, x0=x0,
            args=(p, shots_obj, seed_obj, batches_obj, seed_transpiler),
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol}
        )

        if res.fun < best_val:
            best_val = float(res.fun)
            best_res = res

    best_g = best_res.x[:p]
    best_b = best_res.x[p:]

    # Final sampling seed (paired with baseline).
    seed_final = 2026 + 10_000 * run_id

    qc_final = build_qaoa_circuit(best_g, best_b, with_measurements=True)
    tqc_final = transpile_for_noisy(qc_final, seed_transpiler=seed_transpiler)

    final = NOISY_SIM.run(
        tqc_final,
        shots=shots_final,
        seed_simulator=seed_final
    ).result()

    counts = final.get_counts()
    counts_x = counts_to_xorder(counts)

    # Compute the four evaluation metrics.
    k_star = sum(counts_x.get(xstr, 0) for xstr in opt_xstr_set)
    p_star = k_star / shots_final
    run_success = 1 if k_star >= 1 else 0

    E_hat = expected_energy_from_counts(counts_x)
    gap_exp = E_hat - C_star_feas

    tts99 = tts_shots_for_success(p_star, target=0.99)
    sampling_rank_run = sampling_rank(counts_x, opt_xstr_set)

    p_star_list.append(p_star)
    run_success_list.append(run_success)
    gap_exp_list.append(gap_exp)
    tts99_list.append(tts99)
    rank_list.append(sampling_rank_run)

    print(f"\n[Run {run_id+1:02d}/{N_RUNS} | seed={seed_run}]")
    print("  best scaled sampled objective =", best_val)
    print(f"  p* = {p_star:.6f} (k*={k_star}/{shots_final})  run-success={run_success}")
    print(f"  E_hat = {E_hat:.6f}  gap_exp = {gap_exp:.6f}")
    print(f"  TTS99 = {tts99}  sampling_rank = {sampling_rank_run}")

# Save the optimized circuit diagram from the last run.
save_circuit_diagram(qc_final, filepath="qaoa_circuit_new_optimized.pdf", fold=-1, dpi=1200)

# ===================== Summary =====================

p_mean, p_ci = mean_ci95_t(p_star_list)
p_std = float(np.std(p_star_list, ddof=1))

succ_k = int(sum(run_success_list))
succ_rate = succ_k / N_RUNS
succ_ci = wilson_ci95(succ_k, N_RUNS)

gap_mean, gap_ci = mean_ci95_t(gap_exp_list)
gap_std = float(np.std(gap_exp_list, ddof=1))

tts_arr = np.asarray(tts99_list, dtype=float)
finite_tts = tts_arr[np.isfinite(tts_arr)]
num_inf = int(np.sum(~np.isfinite(tts_arr)))

if len(finite_tts) > 0:
    tts_median = float(np.median(finite_tts))
    tts_q25 = float(np.percentile(finite_tts, 25))
    tts_q75 = float(np.percentile(finite_tts, 75))
else:
    tts_median = float("inf")
    tts_q25 = float("inf")
    tts_q75 = float("inf")

print("\n" + "="*70)
print(f"[Summary over {N_RUNS} runs | Noisy NEW init+mixer | shots_final={shots_final}]")
print("="*70)

print("\n(1) Mean optimal-state probability p* (final sampling)")
print(f"    mean = {p_mean:.6f}, std = {p_std:.6f}, 95% CI = [{p_ci[0]:.6f}, {p_ci[1]:.6f}]")

print("\n(2) Run-level success rate (>=1 optimal sample in final sampling)")
print(f"    success = {succ_k}/{N_RUNS} = {succ_rate:.6f}, 95% Wilson CI = [{succ_ci[0]:.6f}, {succ_ci[1]:.6f}]")

print("\n(3) Expected energy gap (E[C] - C*_feas) estimated from final counts")
print(f"    mean = {gap_mean:.6f}, std = {gap_std:.6f}, 95% CI = [{gap_ci[0]:.6f}, {gap_ci[1]:.6f}]")

print("\n(4) TTS shots for >=99% success (derived from p*), report finite median/IQR")
print(f"    median = {tts_median:.2f}, IQR = [{tts_q25:.2f}, {tts_q75:.2f}], #inf (p*=0) = {num_inf}/{N_RUNS}")

rank_mean, rank_ci = mean_ci95_t(rank_list)
rank_std = float(np.std(rank_list, ddof=1))
print("\n(5) Sampling rank (rank of optimal among bitstrings by frequency, 1=most frequent)")
print(f"    mean = {rank_mean:.2f}, std = {rank_std:.2f}, 95% CI = [{rank_ci[0]:.2f}, {rank_ci[1]:.2f}]")
print("="*70)

print("\n[Noise model summary]")
print(noise_model)