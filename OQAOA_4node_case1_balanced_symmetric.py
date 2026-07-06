import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

from qaoa_new_4node_runtime import (
    DEFAULT_RESULTS_DIR,
    MAXITER_SWEEP,
    build_results_csv_path,
    save_experiment_csv,
    summarize_experiment,
    print_experiment_summary,
    print_maxiter_sweep_selection,
)

# ============================================================
# 0) 4-node, 2-vehicle VRP setup: case1_balanced_symmetric
#    Nodes: 0 is the depot; 1, 2, 3 are customers.
#    Qubits q0..q11 correspond to variables:
#    [x01,x02,x03,x10,x12,x13,x20,x21,x23,x30,x31,x32].
# ============================================================

CASE_NAME = "case1_balanced_symmetric"
SCALE_TAG = "4node"
METHOD_TAG = "oqaoa_statevector"
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

# Experiment settings; adjust these if the 12-qubit runs are too slow on your machine.
p = 2
num_restarts = 10
# maxiter sweep values come from qaoa_new_4node_runtime.MAXITER_SWEEP (200, 300, ..., 2000).
# legacy single maxiter removed; use MAXITER_SWEEP below
shots_obj = 0
shots_final = 4096
batches_obj = 0
N_RUNS = 30
BASE_SEED = 12345
rhobeg = 0.5
tol = 1e-4
seed_transpiler = 999

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


# ============================================================
# 2) Standard QAOA: statevector objective + final sampling
# ============================================================

def build_cost_sparse_pauli_op(J_zz, h_z, scale=1.0):
    triples = []
    for (i, j), coeff in J_zz.items():
        c = coeff / scale
        if abs(c) > 1e-12:
            triples.append(("ZZ", [i, j], c))
    for i, coeff in h_z.items():
        c = coeff / scale
        if abs(c) > 1e-12:
            triples.append(("Z", [i], c))
    return SparsePauliOp.from_sparse_list(triples, num_qubits=N_QUBITS).simplify()

cost_op = build_cost_sparse_pauli_op(J_zz, h_z, scale=ENERGY_SCALE)
estimator = StatevectorEstimator()
sampler = StatevectorSampler()

def apply_cost_layer(qc, gamma):
    for (i, j), J in J_zz.items():
        Js = J / ENERGY_SCALE
        if abs(Js) > 1e-12:
            qc.rzz(2.0 * gamma * Js, i, j)
    for i, h in h_z.items():
        hs = h / ENERGY_SCALE
        if abs(hs) > 1e-12:
            qc.rz(2.0 * gamma * hs, i)

def apply_standard_mixer(qc, beta):
    for q in range(N_QUBITS):
        qc.rx(2.0 * beta, q)

def build_qaoa_circuit(gammas, betas, with_measurements=False):
    qc = QuantumCircuit(N_QUBITS)
    qc.h(range(N_QUBITS))
    for layer in range(len(gammas)):
        apply_cost_layer(qc, gammas[layer])
        apply_standard_mixer(qc, betas[layer])
    if with_measurements:
        qc.measure_all()
    return qc

def qaoa_objective(theta):
    gammas = theta[:p]
    betas = theta[p:]
    qc = build_qaoa_circuit(gammas, betas, with_measurements=False)
    res = estimator.run([(qc, cost_op)]).result()
    return float(np.asarray(res[0].data.evs).reshape(-1)[0])

SUMMARY_TITLE = "Standard QAOA | statevector objective"
all_sweep_rows = []
for maxiter in MAXITER_SWEEP:
    print(f"\n{'=' * 72}")
    print(f"[Maxiter sweep | case={CASE_NAME} | maxiter={maxiter} | N_RUNS={N_RUNS}]")
    print(f"{'=' * 72}")
    run_records = []
    for run_id in range(N_RUNS):
        seed_run = BASE_SEED + run_id
        rng = np.random.default_rng(seed_run)
        best_val = float("inf")
        best_res = None
        for rr in range(num_restarts):
            x0 = np.concatenate([rng.uniform(-np.pi, np.pi, size=p), rng.uniform(0.0, np.pi/2.0, size=p)])
            res = minimize(qaoa_objective, x0=x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol})
            if res.fun < best_val:
                best_val, best_res = float(res.fun), res
        qc_meas = build_qaoa_circuit(best_res.x[:p], best_res.x[p:], with_measurements=True)
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
        print(f"[Run {run_id+1:02d}/{N_RUNS}] best={best_val:.6f} p*={p_star:.6f} gap={gap_exp:.6f} rank={rank}")

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
        "scale_tag": SCALE_TAG,
        "method_tag": METHOD_TAG,
    },
)
