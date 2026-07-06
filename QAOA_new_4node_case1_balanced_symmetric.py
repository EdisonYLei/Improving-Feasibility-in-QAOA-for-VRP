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
SCALE_TAG = "4node"
METHOD_TAG = "proposed_statevector"
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

# Non-overlap structured proposed variant: the initialization encodes exactly-one
# outgoing constraints for customer nodes 1, 2, and 3, producing a 216-state
# support. In the revised manuscript, the final four-node proposed experiments
# use the overlap-4 100-state direct initialization scripts.
INIT_STATES = ['000001001001', '000001001010', '000001001100', '000001010001', '000001010010', '000001010100', '000001100001', '000001100010', '000001100100', '000010001001', '000010001010', '000010001100', '000010010001', '000010010010', '000010010100', '000010100001', '000010100010', '000010100100', '000100001001', '000100001010', '000100001100', '000100010001', '000100010010', '000100010100', '000100100001', '000100100010', '000100100100', '001001001001', '001001001010', '001001001100', '001001010001', '001001010010', '001001010100', '001001100001', '001001100010', '001001100100', '001010001001', '001010001010', '001010001100', '001010010001', '001010010010', '001010010100', '001010100001', '001010100010', '001010100100', '001100001001', '001100001010', '001100001100', '001100010001', '001100010010', '001100010100', '001100100001', '001100100010', '001100100100', '010001001001', '010001001010', '010001001100', '010001010001', '010001010010', '010001010100', '010001100001', '010001100010', '010001100100', '010010001001', '010010001010', '010010001100', '010010010001', '010010010010', '010010010100', '010010100001', '010010100010', '010010100100', '010100001001', '010100001010', '010100001100', '010100010001', '010100010010', '010100010100', '010100100001', '010100100010', '010100100100', '011001001001', '011001001010', '011001001100', '011001010001', '011001010010', '011001010100', '011001100001', '011001100010', '011001100100', '011010001001', '011010001010', '011010001100', '011010010001', '011010010010', '011010010100', '011010100001', '011010100010', '011010100100', '011100001001', '011100001010', '011100001100', '011100010001', '011100010010', '011100010100', '011100100001', '011100100010', '011100100100', '100001001001', '100001001010', '100001001100', '100001010001', '100001010010', '100001010100', '100001100001', '100001100010', '100001100100', '100010001001', '100010001010', '100010001100', '100010010001', '100010010010', '100010010100', '100010100001', '100010100010', '100010100100', '100100001001', '100100001010', '100100001100', '100100010001', '100100010010', '100100010100', '100100100001', '100100100010', '100100100100', '101001001001', '101001001010', '101001001100', '101001010001', '101001010010', '101001010100', '101001100001', '101001100010', '101001100100', '101010001001', '101010001010', '101010001100', '101010010001', '101010010010', '101010010100', '101010100001', '101010100010', '101010100100', '101100001001', '101100001010', '101100001100', '101100010001', '101100010010', '101100010100', '101100100001', '101100100010', '101100100100', '110001001001', '110001001010', '110001001100', '110001010001', '110001010010', '110001010100', '110001100001', '110001100010', '110001100100', '110010001001', '110010001010', '110010001100', '110010010001', '110010010010', '110010010100', '110010100001', '110010100010', '110010100100', '110100001001', '110100001010', '110100001100', '110100010001', '110100010010', '110100010100', '110100100001', '110100100010', '110100100100', '111001001001', '111001001010', '111001001100', '111001010001', '111001010010', '111001010100', '111001100001', '111001100010', '111001100100', '111010001001', '111010001010', '111010001100', '111010010001', '111010010010', '111010010100', '111010100001', '111010100010', '111010100100', '111100001001', '111100001010', '111100001100', '111100010001', '111100010010', '111100010100', '111100100001', '111100100010', '111100100100']
XY_BLOCKS = [(3,4,5), (6,7,8), (9,10,11)]
X_MIXER_QUBITS = [0, 1, 2]

# Experiment settings; adjust these if the 12-qubit runs are too slow on your machine.
p = 2
num_restarts = 10
# maxiter sweep values come from qaoa_new_4node_runtime.MAXITER_SWEEP (200, 300, ..., 2000).
shots_obj = 0
shots_final = 4096
batches_obj = 0
N_RUNS = 30
BASE_SEED = 12345
rhobeg = 0.5
tol = 1e-4
seed_transpiler = 999
# Available optimizers: "cobyla", "spsa", "lbfgsb"; manuscript runs use "cobyla".
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

from qiskit.primitives import StatevectorSampler

from qaoa_new_4node_runtime import (
    LAMBDA_LIST,
    MAXITER_SWEEP,
    build_cost_sparse_pauli_op,
    build_qaoa_circuit_structured,
    build_results_csv_path,
    build_uniform_superposition_statevector,
    make_statevector_objective,
    print_experiment_summary,
    run_multi_restart_optimization,
    save_experiment_csv,
    summarize_experiment,
    select_best_config_by_gap_mean,
    print_hyperparam_sweep_selection,
)

# ============================================================
# 2) Proposed QAOA: constraint-aware init + hybrid XY-X mixer
# ============================================================

psi0_vec = build_uniform_superposition_statevector(INIT_STATES, N_QUBITS)
cost_op = build_cost_sparse_pauli_op(J_zz, h_z, scale=ENERGY_SCALE, n_qubits=N_QUBITS)
sampler = StatevectorSampler()

SUMMARY_TITLE = "Proposed QAOA | statevector objective"
all_sweep_rows = []

for lam in LAMBDA_LIST:
    def build_qaoa_circuit(gammas, betas, with_measurements=False, _lam=lam):
        return build_qaoa_circuit_structured(
            gammas,
            betas,
            N_QUBITS,
            XY_BLOCKS,
            X_MIXER_QUBITS,
            _lam,
            J_zz,
            h_z,
            ENERGY_SCALE,
            with_measurements=with_measurements,
        )

    qaoa_objective = make_statevector_objective(
        psi0_vec,
        cost_op,
        N_QUBITS,
        p,
        XY_BLOCKS,
        X_MIXER_QUBITS,
        lam,
        J_zz,
        h_z,
        ENERGY_SCALE,
    )

    lam_sweep_rows = []
    for maxiter in MAXITER_SWEEP:
        print(f"\n{'=' * 72}")
        print(
            f"[Sweep | case={CASE_NAME} | lambda={lam:g} | "
            f"maxiter={maxiter} | N_RUNS={N_RUNS}]"
        )
        print(f"{'=' * 72}")
        run_records = []
        for run_id in range(N_RUNS):
            seed_run = BASE_SEED + run_id
            rng = np.random.default_rng(seed_run)
            best_val, best_x = run_multi_restart_optimization(
                qaoa_objective,
                p,
                num_restarts,
                maxiter,
                OPTIMIZER,
                rng,
                rhobeg=rhobeg,
                tol=tol,
                allow_lbfgsb=True,
            )
            qc_meas = build_qaoa_circuit(best_x[:p], best_x[p:], with_measurements=True)
            sres = sampler.run([qc_meas], shots=shots_final).result()
            counts = sres[0].data.meas.get_counts()
            counts_x = {k[::-1]: v for k, v in counts.items()}
            k_star = sum(counts_x.get(xstr, 0) for xstr in opt_xstr_set)
            p_star = k_star / shots_final
            E_hat = expected_energy_from_counts_xorder(counts_x)
            gap_exp = E_hat - C_star_feas
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
        row = {
            "lambda": lam,
            "maxiter": maxiter,
            "stats": stats,
            "run_records": run_records,
        }
        lam_sweep_rows.append(row)
        all_sweep_rows.append(row)
        print_experiment_summary(SUMMARY_TITLE, stats, OPTIMIZER, N_RUNS, maxiter)
        csv_path = build_results_csv_path(CASE_NAME, lam, maxiter, scale_tag=SCALE_TAG, method_tag=METHOD_TAG)
        save_experiment_csv(
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

    best_for_lambda = select_best_config_by_gap_mean(lam_sweep_rows)
    print(
        f"[Selection | lambda={lam:g}] best maxiter={best_for_lambda['maxiter']} "
        f"gap_mean={best_for_lambda['stats']['gap_mean']:.6f}"
    )

print_hyperparam_sweep_selection(
    CASE_NAME,
    LAMBDA_LIST,
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
