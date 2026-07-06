import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit

from qaoa_new_4node_runtime import (
    CachedInitializeProposedShotRunner,
    DEFAULT_RESULTS_DIR,
    THREENODE_LAMBDA_LIST,
    THREENODE_MAXITER_SWEEP,
    THREENODE_SHOT_PROTOCOL,
    THREENODE_SHOT_BASE_SEED,
    THREENODE_SHOT_N_RUNS,
    THREENODE_SHOT_SEED_TRANSPILER,
    build_ideal_shot_simulator,
    proposed_noisy_transpile_kwargs,
    print_experiment_summary,
    print_hyperparam_sweep_selection,
    save_experiment_csv,
    select_best_config_by_gap_mean,
    summarize_experiment,
    threenode_final_seed,
    threenode_objective_seed,
)

# ===================== Shared QUBO setup =====================

VAR_NAMES = ["x01", "x02", "x10", "x12", "x20", "x21"]
N_QUBITS = 6

qubo_const = 5662.8
qubo_linear = {0:-1681.1,1:-1737.7,2:-2116.7,3:-828.3,4:-2173.3,5:-828.3}
qubo_quad = {(2,4):1306.8,(0,1):871.2,(2,3):871.2,(0,5):871.2,(4,5):871.2,(1,3):871.2}

ENERGY_SCALE = 435.6
INIT_STATES = ["000101", "100110", "011001", "111010"]

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
        if len(s) != n or any(ch not in "01" for ch in s):
            raise ValueError(f"Bad bitstring: {s}")
        bits = [int(ch) for ch in s]  # variable order q0..q_{n-1}
        # Little-endian computational basis index.
        idx = sum(bits[i] * (2**i) for i in range(n))
        vec[idx] += amp
    vec /= np.linalg.norm(vec)
    return vec

def counts_to_xorder(counts):
    # Qiskit order q_{n-1}..q_0 -> variable order q_0..q_{n-1}.
    return {k[::-1]: v for k, v in counts.items()}

def expected_energy_from_counts(counts_x):
    shots = sum(counts_x.values())
    e = 0.0
    for xstr, c in counts_x.items():
        bits = tuple(int(ch) for ch in xstr)
        e += (c / shots) * qubo_cost_from_bits(bits)
    return float(e)

# ===================== Ising coefficients and initial state vector =====================

J_zz, h_z, ising_const = qubo_to_ising_standard()
psi0_vec = build_uniform_superposition_statevector(INIT_STATES, N_QUBITS)

# ===================== QAOA cost and mixer layers =====================

def apply_cost_layer(qc, gamma):
    for (i, j), J in J_zz.items():
        Js = J / ENERGY_SCALE
        if abs(Js) > 1e-12:
            qc.rzz(2.0 * gamma * Js, i, j)
    for i, h in h_z.items():
        hs = h / ENERGY_SCALE
        if abs(hs) > 1e-12:
            qc.rz(2.0 * gamma * hs, i)

def apply_custom_mixer(qc, beta, lam):
    # Two-qubit pairs (q2,q3) and (q4,q5): RXX, RYY.
    qc.rxx(2.0 * beta, 2, 3)
    qc.ryy(2.0 * beta, 2, 3)
    qc.rxx(2.0 * beta, 4, 5)
    qc.ryy(2.0 * beta, 4, 5)
    # Single-qubit mixers on q0, q1 with weight lambda.
    qc.rx(2.0 * beta * lam, 0)
    qc.rx(2.0 * beta * lam, 1)

def build_qaoa_circuit(gammas, betas, lam, with_measurements=True):
    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(psi0_vec, list(range(N_QUBITS)))
    for l in range(len(gammas)):
        apply_cost_layer(qc, gammas[l])
        apply_custom_mixer(qc, betas[l], lam=lam)
    if with_measurements:
        qc.measure_all()
    return qc

XY_PAIRS = [(2, 3), (4, 5)]
DEPOT_QUBITS = [0, 1]

# ===================== Statistics helpers (four metrics over 30 runs) =====================

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

# ===================== Ground truth: feasible optimum =====================

C_star_feas, opt_states_feas = brute_force_feasible_optimum()
opt_xstr_set = {bits_to_xstr(s) for s in opt_states_feas}

print("[Feasible optimum]")
print("C*_feas =", C_star_feas)
print("Optimal feasible state(s) =", opt_states_feas)
print("Optimal feasible xstr set =", opt_xstr_set)

CASE_NAME = "3node_proposed_ideal_shot"
SCALE_TAG = "3node"
METHOD_TAG = "proposed_ideal_shot"
SUMMARY_TITLE = "Proposed QAOA | 3-node | ideal finite-shot"
OPTIMIZER = "COBYLA"

# Paired with AerQAOA-new.py: same budget; only simulator differs (ideal vs noisy).
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
PROPOSED_IDEAL_TRANSPILE_KWARGS = proposed_noisy_transpile_kwargs(seed_transpiler)

all_sweep_rows = []
for lam in THREENODE_LAMBDA_LIST:
    SHOT_RUNNER = CachedInitializeProposedShotRunner(
        psi0_vec=psi0_vec,
        p_layers=p,
        n_qubits=N_QUBITS,
        lam=lam,
        xy_pairs=XY_PAIRS,
        depot_qubits=DEPOT_QUBITS,
        J_zz=J_zz,
        h_z=h_z,
        energy_scale=ENERGY_SCALE,
        simulator=IDEAL_SIM,
        transpile_kwargs=PROPOSED_IDEAL_TRANSPILE_KWARGS,
    )

    lam_sweep_rows = []

    for maxiter in THREENODE_MAXITER_SWEEP:
        print(f"\n{'=' * 72}")
        print(
            f"[Lambda sweep | case={CASE_NAME} | lambda={lam:g} | "
            f"maxiter={maxiter} | N_RUNS={N_RUNS}]"
        )
        print(f"{'=' * 72}")

        run_records = []
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
                if res.fun < best_val:
                    best_val = float(res.fun)
                    best_res = res

            seed_final = threenode_final_seed(run_id)
            counts_x = SHOT_RUNNER.run_final(
                best_res.x,
                shots_final,
                seed_final,
                counts_to_xorder,
            )
            k_star = sum(counts_x.get(xstr, 0) for xstr in opt_xstr_set)
            p_star = k_star / shots_final
            gap_exp = expected_energy_from_counts(counts_x) - C_star_feas
            rank = sampling_rank(counts_x, opt_xstr_set)
            run_records.append(
                {
                    "run_id": run_id + 1,
                    "best_val": best_val,
                    "p_star": p_star,
                    "gap_exp": gap_exp,
                    "success": 1 if k_star >= 1 else 0,
                    "tts99": tts_shots_for_success(p_star),
                    "rank": rank,
                }
            )
            print(
                f"[Run {run_id+1:02d}/{N_RUNS}] best={best_val:.6f} p*={p_star:.6f} "
                f"gap={gap_exp:.6f} rank={rank}"
            )

        stats = summarize_experiment(run_records, N_RUNS)
        row = {"lambda": lam, "maxiter": maxiter, "stats": stats, "run_records": run_records}
        lam_sweep_rows.append(row)
        all_sweep_rows.append(row)
        print_experiment_summary(SUMMARY_TITLE, stats, OPTIMIZER, N_RUNS, maxiter)

    best_for_lambda = select_best_config_by_gap_mean(lam_sweep_rows)
    print(
        f"[Selection | lambda={lam:g}] best maxiter={best_for_lambda['maxiter']} "
        f"gap_mean={best_for_lambda['stats']['gap_mean']:.6f}"
    )

print_hyperparam_sweep_selection(
    CASE_NAME,
    THREENODE_LAMBDA_LIST,
    all_sweep_rows,
    summary_title=SUMMARY_TITLE,
    optimizer=OPTIMIZER,
    n_runs=N_RUNS,
    scale_tag=SCALE_TAG,
    method_tag=METHOD_TAG,
    output_dir=DEFAULT_RESULTS_DIR,
    save_experiment_csv_fn=save_experiment_csv,
    experiment_meta={
        "p": p,
        "num_restarts": num_restarts,
        "n_runs": N_RUNS,
    },
)
