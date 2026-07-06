import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qaoa_new_4node_runtime import (
    CachedStandardShotRunner,
    DEFAULT_RESULTS_DIR,
    THREENODE_MAXITER_SWEEP,
    THREENODE_SHOT_BASE_SEED,
    THREENODE_SHOT_N_RUNS,
    THREENODE_SHOT_SEED_TRANSPILER,
    THREENODE_SHOT_PROTOCOL,
    build_ideal_shot_simulator,
    build_results_csv_path,
    print_experiment_summary,
    print_maxiter_sweep_selection,
    save_experiment_csv,
    standard_noisy_transpile_kwargs,
    summarize_experiment,
    threenode_final_seed,
    threenode_objective_seed,
)

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
# 3) Ideal shot simulator (cached runner configured below)
# ============================================================

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
# 6) Run experiments with maxiter sweep
# ============================================================

CASE_NAME = "3node_standard_ideal_shot"
SCALE_TAG = "3node"
METHOD_TAG = "standard_ideal_shot"
SUMMARY_TITLE = "Standard QAOA | 3-node | ideal finite-shot"

# Paired with AerQAOA.py: same budget; only simulator differs (ideal vs noisy).
p = 2
num_restarts = THREENODE_SHOT_PROTOCOL["num_restarts"]
shots_obj = THREENODE_SHOT_PROTOCOL["shots_obj"]
shots_final = THREENODE_SHOT_PROTOCOL["shots_final"]
batches_obj = THREENODE_SHOT_PROTOCOL["batches_obj"]
rhobeg = THREENODE_SHOT_PROTOCOL["rhobeg"]
tol = THREENODE_SHOT_PROTOCOL["tol"]
seed_transpiler = THREENODE_SHOT_SEED_TRANSPILER

N_RUNS = THREENODE_SHOT_N_RUNS
BASE_SEED = THREENODE_SHOT_BASE_SEED

IDEAL_SIM = build_ideal_shot_simulator()
STANDARD_IDEAL_TRANSPILE_KWARGS = standard_noisy_transpile_kwargs(seed_transpiler)
SHOT_RUNNER = CachedStandardShotRunner(
    p_layers=p,
    n_qubits=N_QUBITS,
    J_zz=J_zz,
    h_z=h_z,
    energy_scale=ENERGY_SCALE,
    simulator=IDEAL_SIM,
    transpile_kwargs=STANDARD_IDEAL_TRANSPILE_KWARGS,
)

all_sweep_rows = []
for maxiter in THREENODE_MAXITER_SWEEP:
    print(f"\n{'=' * 72}")
    print(f"[Maxiter sweep | case={CASE_NAME} | maxiter={maxiter} | N_RUNS={N_RUNS}]")
    print(f"{'=' * 72}")
    run_records = []
    for run_id in range(N_RUNS):
        seed_run = BASE_SEED + run_id
        rng = np.random.default_rng(seed_run)

        best_result = None
        best_value = float("inf")

        for r in range(num_restarts):
            x0 = np.concatenate([
                rng.uniform(-np.pi, np.pi, size=p),
                rng.uniform(0.0, np.pi / 2.0, size=p),
            ])
            seed_obj = threenode_objective_seed(run_id, r)
            res = minimize(
                lambda th, _seed=seed_obj: SHOT_RUNNER.objective_energy(
                    th,
                    shots_obj,
                    _seed,
                    batches_obj,
                    expected_energy_from_counts,
                    counts_to_xorder,
                ),
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol}
            )
            if res.fun < best_value:
                best_value = float(res.fun)
                best_result = res

        counts_xorder = SHOT_RUNNER.run_final(
            best_result.x,
            shots_final,
            threenode_final_seed(run_id),
            counts_to_xorder,
        )
        k_star = sum(counts_xorder.get(xstr, 0) for xstr in opt_xstr_set)
        p_star = k_star / shots_final
        gap_exp = expected_energy_from_counts(counts_xorder) - C_star_feas
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