import numpy as np
import itertools
from math import log, ceil, sqrt

# ============================================================
# 0) 4-node, 2-vehicle VRP setup: case1_balanced_symmetric
#    Nodes: 0 is the depot; 1, 2, 3 are customers.
#    Qubits q0..q11 correspond to variables:
#    [x01,x02,x03,x10,x12,x13,x20,x21,x23,x30,x31,x32].
# ============================================================

CASE_NAME = "case1_balanced_symmetric"
SCALE_TAG = "overlap4_100state_4node"
METHOD_TAG = "proposed_noisy"
VAR_NAMES = ["x01", "x02", "x03", "x10", "x12", "x13", "x20", "x21", "x23", "x30", "x31", "x32"]
ARCS = [(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]
N_QUBITS = 12
N_VEHICLES = 2
DISTANCE_MATRIX = np.array([[0.0, 21.7, 34.2, 28.6], [21.7, 0.0, 17.4, 24.9], [34.2, 17.4, 0.0, 19.8], [28.6, 24.9, 19.8, 0.0]], dtype=float)

# Penalty P = 2 * sum_{i != j} |w_ij| for this instance.
PENALTY = 586.4
ENERGY_SCALE = PENALTY

# QUBO: F(x)=qubo_const + sum_i qubo_linear[i] x_i + sum_{i<j} qubo_quad[(i,j)] x_i x_j.
qubo_const = 8209.6
qubo_linear = {0: -2323.9, 1: -2311.4, 2: -2317, 3: -2323.9, 4: -1155.4, 5: -1147.9, 6: -2311.4, 7: -1155.4, 8: -1153, 9: -2317, 10: -1147.9, 11: -1153}
qubo_quad = {(0, 1): 1172.8, (0, 2): 1172.8, (0, 7): 1172.8, (0, 10): 1172.8, (1, 2): 1172.8, (1, 4): 1172.8, (1, 11): 1172.8, (2, 5): 1172.8, (2, 8): 1172.8, (3, 4): 1172.8, (3, 5): 1172.8, (3, 6): 1172.8, (3, 9): 1172.8, (4, 5): 1172.8, (4, 11): 1172.8, (5, 8): 1172.8, (6, 7): 1172.8, (6, 8): 1172.8, (6, 9): 1172.8, (7, 8): 1172.8, (7, 10): 1172.8, (9, 10): 1172.8, (9, 11): 1172.8, (10, 11): 1172.8}

# Overlap-4 100-state direct initialization: four selected overlapping
# customer-degree constraints (I1, I2, O1, O2) filter a 100-state support.
XY_PAIRS = [(0, 10), (1, 11), (3, 5), (6, 8)]
X_MIXER_QUBITS = [2, 9]
FROZEN_OVERLAP_QUBITS = [4, 7]

# Experiment settings; adjust these if the 12-qubit runs are too slow on your machine.
p = 2
num_restarts = NOISY_NUM_RESTARTS
shots_obj = 2048
shots_final = 8192
batches_obj = 2
N_RUNS = 30
BASE_SEED = 12345
rhobeg = 0.5
tol = 2e-3
seed_transpiler = 999
# Available optimizers: "cobyla", "spsa"; manuscript runs use "cobyla".
OPTIMIZER = "cobyla"

# ============================================================
# 1) Classical utilities
# ============================================================

def qubo_cost_from_bits(bits):
    val = qubo_const
    for i, a in qubo_linear.items():
        val += a * bits[i]
    for (i, j), a in qubo_quad.items():
        val += a * bits[i] * bits[j]
    return float(val)

def raw_vrp_cost_from_bits(bits):
    return float(sum(DISTANCE_MATRIX[i, j] * bits[k] for k, (i, j) in enumerate(ARCS)))

def check_constraints(bits):
    x01,x02,x03,x10,x12,x13,x20,x21,x23,x30,x31,x32 = bits
    return all([
        x01 + x21 + x31 == 1,  # incoming to customer 1
        x02 + x12 + x32 == 1,  # incoming to customer 2
        x03 + x13 + x23 == 1,  # incoming to customer 3
        x10 + x12 + x13 == 1,  # outgoing from customer 1
        x20 + x21 + x23 == 1,  # outgoing from customer 2
        x30 + x31 + x32 == 1,  # outgoing from customer 3
        x01 + x02 + x03 == N_VEHICLES,  # depot outgoing
        x10 + x20 + x30 == N_VEHICLES,  # depot incoming
    ])

def brute_force_feasible_optimum():
    best_e = float("inf")
    best_states = []
    all_feasible = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        if not check_constraints(bits):
            continue
        e = qubo_cost_from_bits(bits)
        all_feasible.append((bits, e))
        if e < best_e - 1e-12:
            best_e = e
            best_states = [bits]
        elif abs(e - best_e) <= 1e-12:
            best_states.append(bits)
    return best_e, best_states, all_feasible

def bits_to_xstr(bits):
    return "".join(str(b) for b in bits)

def routes_from_bits(bits):
    selected = [ARCS[k] for k, b in enumerate(bits) if b]
    starts = [j for (i, j) in selected if i == 0]
    routes = []
    for s in starts:
        route = [0, s]
        current = s
        guard = 0
        while current != 0 and guard < 10:
            nxt = [j for (i, j) in selected if i == current]
            if not nxt:
                break
            current = nxt[0]
            route.append(current)
            guard += 1
        routes.append(route)
    return routes

def qubo_to_ising_standard():
    # Binary-to-Ising convention: x_i = (I - Z_i)/2.
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

def expected_energy_from_counts_xorder(counts_xorder):
    S = sum(counts_xorder.values())
    e = 0.0
    for xstr, c in counts_xorder.items():
        bits = tuple(int(ch) for ch in xstr)
        e += (c / S) * qubo_cost_from_bits(bits)
    return float(e)

def sampling_rank(counts_dict, opt_xstr_set):
    sorted_xstrs = sorted(counts_dict.keys(), key=lambda x: -counts_dict[x])
    best_rank = None
    for xstr in opt_xstr_set:
        if xstr not in counts_dict:
            continue
        rank = 1 + sorted_xstrs.index(xstr)
        if best_rank is None or rank < best_rank:
            best_rank = rank
    return best_rank if best_rank is not None else len(counts_dict) + 1

def tts_shots_for_success(p_star, target=0.99):
    if p_star <= 0.0:
        return float("inf")
    if p_star >= 1.0:
        return 1.0
    return float(ceil(log(1.0 - target) / log(1.0 - p_star)))

def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0

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
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)

J_zz, h_z, ising_const = qubo_to_ising_standard()
C_star_feas, opt_states_feas, feasible_states = brute_force_feasible_optimum()
opt_xstr_set = {bits_to_xstr(s) for s in opt_states_feas}

print("\n[Problem]")
print("case =", CASE_NAME)
print("distance matrix =\n", DISTANCE_MATRIX)
print("penalty P =", PENALTY)
print("Ising constant term =", ising_const, "(ignored during optimization)")
print("number of feasible states =", len(feasible_states))
print("C*_feas =", C_star_feas)
print("optimal feasible bitstrings =", sorted(opt_xstr_set))
for s in opt_states_feas:
    print("routes", bits_to_xstr(s), "=", routes_from_bits(s))

import os

from qaoa_new_4node_runtime import (
    NOISY_NUM_RESTARTS,
    NOISY_LAMBDA_LIST,
    NOISY_MAXITER,
    build_proposed_noisy_simulator,
    run_multi_restart_optimization,
    select_best_config_by_gap_mean,
)
from qaoa_overlap4_100state_direct_4node_runtime import (
    CachedOverlap4DirectShotRunner,
    build_results_csv_path,
    p_feas_from_counts,
    print_experiment_summary_overlap4,
    print_overlap4_init_and_mixer_diagnostics,
    save_experiment_csv_overlap4,
    setup_overlap4_initialization,
    summarize_experiment_with_pfeas,
    verify_optimal_qubo_raw_match,
    print_hyperparam_sweep_selection_overlap4,
)

_SMOKE = os.environ.get("OVERLAP4_SMOKE", "0") == "1"
maxiter = 3 if _SMOKE else NOISY_MAXITER
if _SMOKE:
    N_RUNS = 1
    num_restarts = 1
    shots_final = 256
    if shots_obj > 0:
        shots_obj = 128
    if batches_obj > 0:
        batches_obj = 1

INIT_CTX = setup_overlap4_initialization(
    N_QUBITS,
    check_constraints,
    qubo_cost_from_bits,
    opt_states_feas,
)
OVERLAP4_INITIAL_STATEVECTOR = INIT_CTX.initial_statevector
verify_optimal_qubo_raw_match(opt_states_feas, qubo_cost_from_bits, raw_vrp_cost_from_bits)
print("Initialization support size =", len(INIT_CTX.init_bits))
print(
    "Fully feasible states in support =",
    len(INIT_CTX.fully_feasible_init_bits),
)
print(
    "Initial complete-feasibility probability =",
    INIT_CTX.initial_feasible_probability,
)

def counts_to_xorder(counts):
    return {k[::-1]: v for k, v in counts.items()}


from qaoa_overlap4_100state_direct_4node_runtime import STATE_PREPARATION_NOISE_MODELED

# ============================================================
# 2) Proposed QAOA | overlap-4 100-state direct init | noisy finite-shot objective
# ============================================================

NOISY_SIM = build_proposed_noisy_simulator(N_QUBITS)

SUMMARY_TITLE = "Proposed QAOA | overlap-4 100-state direct initialization | noisy finite-shot"

all_sweep_rows = []
for lam in NOISY_LAMBDA_LIST:
    print_overlap4_init_and_mixer_diagnostics(INIT_CTX, opt_states_feas, lam)
    FINAL_RUNNER = CachedOverlap4DirectShotRunner(
        p_layers=p,
        initial_statevector=OVERLAP4_INITIAL_STATEVECTOR,
        n_qubits=N_QUBITS,
        xy_pairs=XY_PAIRS,
        x_mixer_qubits=X_MIXER_QUBITS,
        lam=lam,
        J_zz=J_zz,
        h_z=h_z,
        energy_scale=ENERGY_SCALE,
        simulator=NOISY_SIM,
        seed_transpiler=seed_transpiler,
        optimization_level=1,
    )

    def qaoa_objective(theta, seed_sim):
        return FINAL_RUNNER.objective_energy(
            theta,
            shots_obj,
            seed_sim,
            batches_obj,
            expected_energy_from_counts_xorder,
            counts_to_xorder,
        )

    print(f"\n{'=' * 72}")
    print(f"[Sweep | case={CASE_NAME} | lambda={lam:g} | maxiter={maxiter} | N_RUNS={N_RUNS}]")
    print(f"{'=' * 72}")
    run_records = []
    for run_id in range(N_RUNS):
        rng = np.random.default_rng(BASE_SEED + run_id)
        seed_obj = 7000 + 10000 * run_id

        def objective(theta):
            return qaoa_objective(theta, seed_obj)

        best_val, best_x = run_multi_restart_optimization(
            objective,
            p,
            num_restarts,
            maxiter,
            OPTIMIZER,
            rng,
            rhobeg=rhobeg,
            tol=tol,
            allow_lbfgsb=False,
        )
        counts_x = FINAL_RUNNER.run_final(
            best_x,
            shots_final,
            2026 + 10000 * run_id,
            counts_to_xorder,
        )
        k_star = sum(counts_x.get(xstr, 0) for xstr in opt_xstr_set)
        p_star = k_star / shots_final
        p_feas = p_feas_from_counts(counts_x, shots_final, check_constraints)
        gap_exp = expected_energy_from_counts_xorder(counts_x) - C_star_feas
        rank = sampling_rank(counts_x, opt_xstr_set)
        run_records.append(
            {
                "run_id": run_id + 1,
                "best_val": best_val,
                "p_star": p_star,
                "p_feas": p_feas,
                "gap_exp": gap_exp,
                "success": 1 if k_star >= 1 else 0,
                "tts99": tts_shots_for_success(p_star),
                "rank": rank,
            }
        )
        print(
            f"[Run {run_id+1:02d}/{N_RUNS}] best={best_val:.6f} p*={p_star:.6f} "
            f"p_feas={p_feas:.6f} gap={gap_exp:.6f} rank={rank}"
        )

    stats = summarize_experiment_with_pfeas(run_records, N_RUNS)
    row = {
        "lambda": lam,
        "maxiter": maxiter,
        "stats": stats,
        "run_records": run_records,
    }
    all_sweep_rows.append(row)
    print_experiment_summary_overlap4(SUMMARY_TITLE, stats, OPTIMIZER, N_RUNS, maxiter)
    csv_path = build_results_csv_path(CASE_NAME, lam, maxiter, scale_tag=SCALE_TAG, method_tag=METHOD_TAG)
    save_experiment_csv_overlap4(
        csv_path,
        CASE_NAME,
        lam,
        maxiter,
        OPTIMIZER,
        p,
        num_restarts,
        N_RUNS,
        stats,
        run_records,
        scale_tag=SCALE_TAG,
        method_tag=METHOD_TAG,
    )
    print(f"[Saved] {csv_path}")

print_hyperparam_sweep_selection_overlap4(
    CASE_NAME,
    NOISY_LAMBDA_LIST,
    all_sweep_rows,
    summary_title=SUMMARY_TITLE,
    optimizer=OPTIMIZER,
    n_runs=N_RUNS,
    scale_tag=SCALE_TAG,
    method_tag=METHOD_TAG,
    experiment_meta={
        "p": p,
        "num_restarts": num_restarts,
        "n_runs": N_RUNS,
        "scale_tag": SCALE_TAG,
        "method_tag": METHOD_TAG,
    },
)
