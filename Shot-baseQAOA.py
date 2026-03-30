import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ============================================================
# 0) Problem setup
# ============================================================

VAR_NAMES = ["x01", "x02", "x10", "x12", "x20", "x21"]
N_QUBITS = 6

qubo_const = 5662.8
qubo_linear = {
    0: -1681.1,
    1: -1737.7,
    2: -2116.7,
    3: -828.3,
    4: -2173.3,
    5: -828.3,
}
qubo_quad = {
    (2, 4): 1306.8,
    (0, 1): 871.2,
    (2, 3): 871.2,
    (0, 5): 871.2,
    (4, 5): 871.2,
    (1, 3): 871.2,
}

ENERGY_SCALE = 435.6  # Scaling factor for numerical stability only.

# ============================================================
# 1) Classical utilities
# ============================================================

def qubo_cost_from_bits(bits, q_const, q_lin, q_quad):
    x = list(bits)
    val = q_const
    for i, a in q_lin.items():
        val += a * x[i]
    for (i, j), a in q_quad.items():
        val += a * x[i] * x[j]
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

def brute_force_feasible_optimum(n, q_const, q_lin, q_quad):
    best_e = float("inf")
    best_states = []
    for bits in itertools.product([0, 1], repeat=n):
        if not check_constraints_user_sec_geq1(bits):
            continue
        e = qubo_cost_from_bits(bits, q_const, q_lin, q_quad)
        if e < best_e - 1e-12:
            best_e = e
            best_states = [bits]
        elif abs(e - best_e) <= 1e-12:
            best_states.append(bits)
    return best_e, best_states

def bits_to_xstr(bits):
    # bits in variable order (q0..q5) corresponding to (x01..x21).
    return "".join(str(b) for b in bits)

# ============================================================
# 2) QUBO to Ising and standard QAOA circuit (RZZ, RZ, RX)
# ============================================================

def qubo_to_ising_standard(q_const, q_lin, q_quad, n):
    J_zz = {}
    h_z = {i: 0.0 for i in range(n)}
    c0 = float(q_const)

    for (i, j), a in q_quad.items():
        a = float(a)
        J_zz[(i, j)] = J_zz.get((i, j), 0.0) + a / 4.0
        h_z[i] += -a / 4.0
        h_z[j] += -a / 4.0
        c0 += a / 4.0

    for i, a in q_lin.items():
        a = float(a)
        h_z[i] += -a / 2.0
        c0 += a / 2.0

    return J_zz, h_z, c0

J_zz, h_z, ising_const = qubo_to_ising_standard(qubo_const, qubo_linear, qubo_quad, N_QUBITS)
print("Ising constant term (ignored during optimization):", ising_const)

def build_qaoa_circuit(gammas, betas, J_zz, h_z, n, energy_scale=1.0, with_measurements=False):
    p = len(gammas)
    qc = QuantumCircuit(n)
    qc.h(range(n))  # Uniform superposition |+>^{\otimes n}

    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]

        for (i, j), J in J_zz.items():
            J_scaled = J / energy_scale
            if abs(J_scaled) > 1e-12:
                qc.rzz(2.0 * gamma * J_scaled, i, j)

        for i, h in h_z.items():
            h_scaled = h / energy_scale
            if abs(h_scaled) > 1e-12:
                qc.rz(2.0 * gamma * h_scaled, i)

        for q in range(n):
            qc.rx(2.0 * beta, q)

    if with_measurements:
        qc.measure_all()
    return qc

# ============================================================
# 3) Aer simulator (ideal, no noise)
# ============================================================

IDEAL_SIM = AerSimulator()  # Pass seed_simulator per run/eval for reproducibility.

def counts_to_xorder(counts):
    # Qiskit bit order q_{n-1}..q_0 -> variable order q_0..q_{n-1}.
    out = {}
    for label, c in counts.items():
        xstr = label[::-1]
        out[xstr] = out.get(xstr, 0) + c
    return out

def expected_energy_from_counts(counts_xorder):
    S = sum(counts_xorder.values())
    e = 0.0
    for xstr, c in counts_xorder.items():
        bits = tuple(int(ch) for ch in xstr)
        e += (c / S) * qubo_cost_from_bits(bits, qubo_const, qubo_linear, qubo_quad)
    return float(e)

def qaoa_objective_shot_based(theta, p, shots_obj, seed_sim, n_batches):
    gammas = theta[:p]
    betas = theta[p:]

    vals = []
    for b in range(n_batches):
        qc = build_qaoa_circuit(
            gammas, betas, J_zz, h_z, N_QUBITS,
            energy_scale=ENERGY_SCALE,
            with_measurements=True
        )
        tqc = transpile(qc, IDEAL_SIM, seed_transpiler=999, optimization_level=1)

        result = IDEAL_SIM.run(
            tqc,
            shots=shots_obj,
            seed_simulator=seed_sim + b
        ).result()

        counts = result.get_counts()
        counts_xorder = counts_to_xorder(counts)
        mean_e = expected_energy_from_counts(counts_xorder)  # Sample mean of QUBO cost.
        vals.append(mean_e / ENERGY_SCALE)

    return float(np.mean(vals))

# ============================================================
# 4) Statistics helpers for 30-run summary
# ============================================================

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
    t = 2.045 if n == 30 else 2.0  # t_{0.975,29} for 95% CI when n=30.
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

# ============================================================
# 5) Ground truth: feasible optimum (used by all four metrics)
# ============================================================

C_star_feas, opt_states_feas = brute_force_feasible_optimum(
    N_QUBITS, qubo_const, qubo_linear, qubo_quad
)
opt_xstr_set = {bits_to_xstr(s) for s in opt_states_feas}

print("\n[Feasible optimum by brute force]")
print("C*_feas =", C_star_feas)
print("Optimal feasible state(s) =", opt_states_feas)
print("Optimal feasible xstr set =", opt_xstr_set)

# ============================================================
# 6) Run 30 experiments
# ============================================================

# Keep these fixed for fair evaluation (aligned with AerQAOA: same shots and batch size).
p = 2
num_restarts = 8
shots_obj = 2048
shots_final = 8192
batches_obj = 2
maxiter = 120
rhobeg = 0.5
tol = 2e-3

N_RUNS = 30
BASE_SEED = 12345  # Use same value across methods for paired comparison.

p_star_list = []
gap_exp_list = []
run_success_list = []
tts99_list = []
rank_list = []

for run_id in range(N_RUNS):
    seed_run = BASE_SEED + run_id
    rng = np.random.default_rng(seed_run)

    best_result = None
    best_value = float("inf")

    # Multiple restarts with different initial points.
    for r in range(num_restarts):
        x0 = np.concatenate([
            rng.uniform(-np.pi, np.pi, size=p),
            rng.uniform(0.0, np.pi / 2.0, size=p),
        ])

        # Seed for objective evaluation (depends on run and restart for pairing).
        seed_obj = 5000 + 100 * r + 10_000 * run_id

        res = minimize(
            qaoa_objective_shot_based,
            x0=x0,
            args=(p, shots_obj, seed_obj, batches_obj),
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol}
        )

        if res.fun < best_value:
            best_value = float(res.fun)
            best_result = res

    best_gammas = best_result.x[:p]
    best_betas = best_result.x[p:]

    # Final sampling with optimized parameters.
    final_qc = build_qaoa_circuit(
        best_gammas, best_betas, J_zz, h_z, N_QUBITS,
        energy_scale=ENERGY_SCALE,
        with_measurements=True
    )
    tfinal_qc = transpile(final_qc, IDEAL_SIM, seed_transpiler=999, optimization_level=1)

    seed_final = 2026 + 10_000 * run_id  # Per-run final sampling seed.
    final_result = IDEAL_SIM.run(
        tfinal_qc,
        shots=shots_final,
        seed_simulator=seed_final
    ).result()

    counts = final_result.get_counts()
    counts_xorder = counts_to_xorder(counts)

    # Metric (1): optimal-state probability p*.
    k_star = sum(counts_xorder.get(xstr, 0) for xstr in opt_xstr_set)
    p_star = k_star / shots_final

    # Metric (2): run-level success (at least one optimal sample).
    run_success = 1 if k_star >= 1 else 0

    # Metric (3): expected energy gap from final counts.
    E_hat = expected_energy_from_counts(counts_xorder)
    gap_exp = E_hat - C_star_feas

    # Metric (4): TTS for 99% success probability.
    tts99 = tts_shots_for_success(p_star, target=0.99)
    sampling_rank_run = sampling_rank(counts_xorder, opt_xstr_set)

    p_star_list.append(p_star)
    run_success_list.append(run_success)
    gap_exp_list.append(gap_exp)
    tts99_list.append(tts99)
    rank_list.append(sampling_rank_run)

    print(f"\n[Run {run_id+1:02d}/{N_RUNS} | seed={seed_run}]")
    print("  best scaled sampled objective =", best_value)
    print(f"  p* = {p_star:.6f} (k*={k_star}/{shots_final})  run-success={run_success}")
    print(f"  E_hat = {E_hat:.6f}  gap_exp = {gap_exp:.6f}")
    print(f"  TTS99 = {tts99}  sampling_rank = {sampling_rank_run}")

# ============================================================
# 7) Summarize the four metrics over 30 runs
# ============================================================

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
print(f"[Summary over {N_RUNS} runs | Shot-based ideal Aer | shots_final={shots_final}]")
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