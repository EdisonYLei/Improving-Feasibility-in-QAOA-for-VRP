import numpy as np
import itertools
from math import pi, log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qaoa_new_4node_runtime import (
    DEFAULT_RESULTS_DIR,
    NOISY_NUM_RESTARTS,
    THREENODE_MAXITER_SWEEP,
    THREENODE_SHOT_BASE_SEED,
    THREENODE_SHOT_N_RUNS,
    build_results_csv_path,
    print_experiment_summary,
    print_maxiter_sweep_selection,
    save_experiment_csv,
    summarize_experiment,
)

# ============================================================
# 0) Problem setup (same QUBO and variable order as baseline)
#    Qubits q0..q5 correspond to [x01,x02,x10,x12,x20,x21].
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

ENERGY_SCALE = 435.6

# ============================================================
# 1) Utilities
# ============================================================

def qubo_cost_from_bits(bits, q_const, q_lin, q_quad):
    x = list(bits)
    val = q_const
    for i, a in q_lin.items():
        val += a * x[i]
    for (i, j), a in q_quad.items():
        val += a * x[i] * x[j]
    return float(val)

def qubo_to_pauli_z_ising_standard(q_const, q_lin, q_quad, n):
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

def build_cost_sparse_pauli_op(J_zz, h_z, n, scale=1.0):
    triples = []
    for (i, j), coeff in J_zz.items():
        c = coeff / scale
        if abs(c) > 1e-12:
            triples.append(("ZZ", [i, j], c))
    for i, coeff in h_z.items():
        c = coeff / scale
        if abs(c) > 1e-12:
            triples.append(("Z", [i], c))
    return SparsePauliOp.from_sparse_list(triples, num_qubits=n).simplify()

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
    """Return the feasible optimum energy and the list of feasible-optimal bitstrings."""
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

def bits_to_str_xorder(bits):
    # bits in variable order (x01..x21) = (q0..q5).
    return "".join(str(b) for b in bits)

def expected_energy_from_counts_xorder(counts_xorder):
    S = sum(counts_xorder.values())
    e = 0.0
    for xstr, c in counts_xorder.items():
        bits = tuple(int(ch) for ch in xstr)
        e += (c / S) * qubo_cost_from_bits(bits, qubo_const, qubo_linear, qubo_quad)
    return float(e)

# --- Statistics helpers ---
def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return m, s

def t_critical_975_df29():
    return 2.045  # t_{0.975,29} for 95% CI when n=30.

def mean_ci95_t(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    m, s = mean_std(arr)
    if n <= 1:
        return m, (m, m)
    t = t_critical_975_df29() if n == 30 else 2.0  # fallback
    half = t * s / sqrt(n)
    return m, (m - half, m + half)

def wilson_ci95(k, n):
    """Return the 95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z / denom) * sqrt((phat*(1-phat)/n) + (z*z/(4*n*n)))
    return (max(0.0, center - half), min(1.0, center + half))

def tts_shots_for_success(p_star, target=0.99):
    """Shots needed for at least one success with probability >= target (i.i.d. samples, per-shot success prob. p_star)."""
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
# 2) Build cost operator + QAOA circuit (same as your code)
# ============================================================

J_zz, h_z, ising_const = qubo_to_pauli_z_ising_standard(
    qubo_const, qubo_linear, qubo_quad, N_QUBITS
)
print("Ising constant term (ignored during optimization):", ising_const)

cost_op = build_cost_sparse_pauli_op(J_zz, h_z, N_QUBITS, scale=ENERGY_SCALE)

def build_qaoa_circuit(gammas, betas, J_zz, h_z, n, energy_scale=1.0, with_measurements=False):
    p = len(gammas)
    assert len(betas) == p
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
# 3) Estimator + Sampler (same as your code)
# ============================================================

estimator = StatevectorEstimator()
sampler = StatevectorSampler()

def qaoa_objective(theta, p, J_zz, h_z, cost_op, energy_scale):
    gammas = theta[:p]
    betas = theta[p:]
    qc = build_qaoa_circuit(
        gammas, betas, J_zz, h_z, N_QUBITS,
        energy_scale=energy_scale,
        with_measurements=False
    )
    result = estimator.run([(qc, cost_op)]).result()
    evs = result[0].data.evs
    return float(np.asarray(evs).reshape(-1)[0])

# ============================================================
# 4) Ground truth: feasible optimum
# ============================================================

C_star_feas, opt_states_feas = brute_force_feasible_optimum(
    N_QUBITS, qubo_const, qubo_linear, qubo_quad
)
opt_xstr_set = {bits_to_str_xorder(s) for s in opt_states_feas}

print("\n[Feasible optimum by brute force]")
print("C*_feas =", C_star_feas)
print("Optimal feasible state(s) =", opt_states_feas)
print("Optimal feasible xstr set =", opt_xstr_set)

# ============================================================
# 5) Run experiments with maxiter sweep
# ============================================================

CASE_NAME = "3node_standard_statevector"
SCALE_TAG = "3node"
METHOD_TAG = "standard_statevector"
SUMMARY_TITLE = "Standard QAOA | 3-node | statevector"

# Experiment settings (aligned with THREENODE_SHOT_N_RUNS / NOISY_NUM_RESTARTS).
p = 2
num_restarts = NOISY_NUM_RESTARTS
shots_final = 4096

N_RUNS = THREENODE_SHOT_N_RUNS
BASE_SEED = THREENODE_SHOT_BASE_SEED

all_sweep_rows = []
for maxiter in THREENODE_MAXITER_SWEEP:
    print(f"\n{'=' * 72}")
    print(f"[Maxiter sweep | case={CASE_NAME} | maxiter={maxiter} | N_RUNS={N_RUNS}]")
    print(f"{'=' * 72}")
    run_records = []
    for run_id in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + run_id)

        best_result = None
        best_value = float("inf")

        for r in range(num_restarts):
            x0 = np.concatenate([
                rng.uniform(-np.pi, np.pi, size=p),
                rng.uniform(0.0, np.pi / 2.0, size=p),
            ])
            res = minimize(
                qaoa_objective,
                x0=x0,
                args=(p, J_zz, h_z, cost_op, ENERGY_SCALE),
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-4}
            )
            if res.fun < best_value:
                best_value = float(res.fun)
                best_result = res

        best_gammas = best_result.x[:p]
        best_betas = best_result.x[p:]
        final_qc_meas = build_qaoa_circuit(
            best_gammas, best_betas, J_zz, h_z, N_QUBITS,
            energy_scale=ENERGY_SCALE,
            with_measurements=True
        )
        sample_result = sampler.run([final_qc_meas], shots=shots_final).result()
        counts_qiskit = sample_result[0].data.meas.get_counts()
        counts_xorder = {}
        for label, c in counts_qiskit.items():
            xstr = label[::-1]
            counts_xorder[xstr] = counts_xorder.get(xstr, 0) + c

        k_star = sum(counts_xorder.get(xstr, 0) for xstr in opt_xstr_set)
        p_star = k_star / shots_final
        gap_exp = expected_energy_from_counts_xorder(counts_xorder) - C_star_feas
        rank = sampling_rank(counts_xorder, opt_xstr_set)
        run_records.append(
            {
                "run_id": run_id + 1,
                "best_val": best_value,
                "p_star": p_star,
                "gap_exp": gap_exp,
                "success": 1 if k_star >= 1 else 0,
                "tts99": tts_shots_for_success(p_star),
                "rank": rank,
            }
        )
        print(
            f"[Run {run_id+1:02d}/{N_RUNS}] best={best_value:.6f} p*={p_star:.6f} "
            f"gap={gap_exp:.6f} rank={rank}"
        )

    stats = summarize_experiment(run_records, N_RUNS)
    all_sweep_rows.append({"lambda": None, "maxiter": maxiter, "stats": stats, "run_records": run_records})
    print_experiment_summary(SUMMARY_TITLE, stats, "COBYLA", N_RUNS, maxiter)
    csv_path = build_results_csv_path(
        CASE_NAME, None, maxiter, scale_tag=SCALE_TAG, method_tag=METHOD_TAG
    )
    save_experiment_csv(
        csv_path,
        CASE_NAME,
        None,
        maxiter,
        "COBYLA",
        p,
        num_restarts,
        N_RUNS,
        stats,
        run_records,
        scale_tag=SCALE_TAG,
        method_tag=METHOD_TAG,
    )
    print(f"[Saved] {csv_path}")

print_maxiter_sweep_selection(
    CASE_NAME,
    all_sweep_rows,
    summary_title=SUMMARY_TITLE,
    optimizer="COBYLA",
    n_runs=N_RUNS,
    output_dir=DEFAULT_RESULTS_DIR,
    scale_tag=SCALE_TAG,
    method_tag=METHOD_TAG,
    save_experiment_csv_fn=save_experiment_csv,
    experiment_meta={
        "p": p,
        "num_restarts": num_restarts,
        "n_runs": N_RUNS,
    },
)