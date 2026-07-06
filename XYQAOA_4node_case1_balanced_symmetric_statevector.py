"""Register-wise fixed-Hamming-weight + pure XY mixer QAOA (4-node case1 balanced symmetric, statevector).
Reference source: OQAOA.py / OQAOA_4node_case1_balanced_symmetric.py
External literature baseline; NOT proposed-method ablation (no overlapping init, no RX, no lambda)."""
import os
import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

_SMOKE = os.environ.get("XYQAOA_SMOKE", "0") == "1"

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp, Statevector

from qaoa_new_4node_runtime import (
    DEFAULT_XYQAOA_RESULTS_DIR,
    MAXITER_SWEEP,
    build_xyqaoa_csv_path,
    save_experiment_csv_with_pfeas,
    summarize_experiment_with_pfeas,
    print_experiment_summary_with_pfeas,
    print_maxiter_sweep_selection,
)


CASE_NAME = "case1_balanced_symmetric"
SCALE_TAG = "4node"
METHOD_TAG = "pure_xy_statevector"
VAR_NAMES = ["x01", "x02", "x03", "x10", "x12", "x13", "x20", "x21", "x23", "x30", "x31", "x32"]
ARCS = [(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]
N_QUBITS = 12
N_VEHICLES = 2
DISTANCE_MATRIX = np.array([[0.0, 21.7, 34.2, 28.6], [21.7, 0.0, 17.4, 24.9], [34.2, 17.4, 0.0, 19.8], [28.6, 24.9, 19.8, 0.0]], dtype=float)
PENALTY = 586.4
qubo_const = 8209.6
qubo_linear = {0: -2323.9, 1: -2311.4, 2: -2317, 3: -2323.9, 4: -1155.4, 5: -1147.9, 6: -2311.4, 7: -1155.4, 8: -1153, 9: -2317, 10: -1147.9, 11: -1153}
qubo_quad = {(0, 1): 1172.8, (0, 2): 1172.8, (0, 7): 1172.8, (0, 10): 1172.8, (1, 2): 1172.8, (1, 4): 1172.8, (1, 11): 1172.8, (2, 5): 1172.8, (2, 8): 1172.8, (3, 4): 1172.8, (3, 5): 1172.8, (3, 6): 1172.8, (3, 9): 1172.8, (4, 5): 1172.8, (4, 11): 1172.8, (5, 8): 1172.8, (6, 7): 1172.8, (6, 8): 1172.8, (6, 9): 1172.8, (7, 8): 1172.8, (7, 10): 1172.8, (9, 10): 1172.8, (9, 11): 1172.8, (10, 11): 1172.8}
ENERGY_SCALE = PENALTY

METHOD_TAG = "pure_xy_statevector"

p = 2
num_restarts = 10
# maxiter sweep values come from qaoa_new_4node_runtime.MAXITER_SWEEP (200, 300, ..., 2000).
shots_final = 4096
N_RUNS = 30
BASE_SEED = 12345
rhobeg = 0.5
tol = 1e-4


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
        x01 + x21 + x31 == 1,
        x02 + x12 + x32 == 1,
        x03 + x13 + x23 == 1,
        x10 + x12 + x13 == 1,
        x20 + x21 + x23 == 1,
        x30 + x31 + x32 == 1,
        x01 + x02 + x03 == N_VEHICLES,
        x10 + x20 + x30 == N_VEHICLES,
    ])

def brute_force_feasible_optimum():
    best_e = float("inf")
    best_states = []
    all_feasible = []
    for bits in itertools.product([0, 1], repeat=N_QUBITS):
        if not check_constraints(bits):
            continue
        e = qubo_cost_from_bits(bits)
        all_feasible.append(bits)
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
    total = sum(counts_xorder.values())
    if total <= 0:
        raise ValueError("Empty counts.")
    return float(
        sum(
            count * qubo_cost_from_bits(tuple(int(ch) for ch in xstr))
            for xstr, count in counts_xorder.items()
        ) / total
    )

def verify_optimal_costs(opt_states):
    for bits in opt_states:
        q = qubo_cost_from_bits(bits)
        r = raw_vrp_cost_from_bits(bits)
        if abs(q - r) > 1e-6:
            raise RuntimeError(f"Optimal cost mismatch: qubo={q}, raw={r}, bits={bits}")


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
        return m, s, (m, m)
    t = 2.045 if n == 30 else 2.0
    half = t * s / sqrt(n)
    return m, s, (m - half, m + half)

def wilson_ci95(k, n):
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)

def tts_shots_for_success(p_star, target=0.99):
    if p_star <= 0.0:
        return float("inf")
    if p_star >= 1.0:
        return 1.0
    return float(ceil(log(1.0 - target) / log(1.0 - p_star)))

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

def p_feas_from_counts(counts_x, shots):
    return sum(
        c for xstr, c in counts_x.items()
        if check_constraints(tuple(int(ch) for ch in xstr))
    ) / shots

def print_summary(regime_label, eff_n):
    p_mean, p_std, p_ci = mean_ci95_t(p_star_list)
    pf_mean, pf_std, pf_ci = mean_ci95_t(p_feas_list)
    gap_mean, gap_std, gap_ci = mean_ci95_t(gap_exp_list)
    rank_mean, rank_std, rank_ci = mean_ci95_t(rank_list)
    succ_k = int(sum(run_success_list))
    succ_ci = wilson_ci95(succ_k, eff_n)
    print(f"\n[Summary | {BASELINE_NAME} | {regime_label}]")
    print(f"p*: mean={p_mean:.6f}, std={p_std:.6f}, 95% CI=[{p_ci[0]:.6f}, {p_ci[1]:.6f}]")
    print(f"p_feas: mean={pf_mean:.6f}, std={pf_std:.6f}, 95% CI=[{pf_ci[0]:.6f}, {pf_ci[1]:.6f}]")
    print(f"success: {succ_k}/{eff_n}, Wilson 95% CI=[{succ_ci[0]:.6f}, {succ_ci[1]:.6f}]")
    print(
        "Expected energy gap (E[C] - C*_feas): "
        f"mean={gap_mean:.6f}, std={gap_std:.6f}, 95% CI=[{gap_ci[0]:.6f}, {gap_ci[1]:.6f}]"
    )
    print(
        f"TTS99 median={np.median(np.asarray(tts99_list, dtype=float)):.6f}; "
        f"sampling-rank mean={rank_mean:.6f}, std={rank_std:.6f}, "
        f"95% CI=[{rank_ci[0]:.6f}, {rank_ci[1]:.6f}]"
    )


BASELINE_NAME = "register-wise pure XY-QAOA"
EXPECTED_INIT_SUPPORT = 81
XY_REGISTERS = {
    "B0_depot_out": (0, 1, 2),
    "B1_cust1_out": (3, 4, 5),
    "B2_cust2_out": (6, 7, 8),
    "B3_cust3_out": (9, 10, 11),
}
XY_MIXER_BLOCKS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]

def local_amps_weight1_3q():
    amps = np.zeros(8, dtype=float)
    a = 1.0 / np.sqrt(3.0)
    for idx in (1, 2, 4):
        amps[idx] = a
    return amps

def local_amps_weight2_3q():
    amps = np.zeros(8, dtype=float)
    a = 1.0 / np.sqrt(3.0)
    for idx in (3, 5, 6):
        amps[idx] = a
    return amps

def apply_register_init(qc):
    qc.initialize(local_amps_weight2_3q(), [0, 1, 2])
    qc.initialize(local_amps_weight1_3q(), [3, 4, 5])
    qc.initialize(local_amps_weight1_3q(), [6, 7, 8])
    qc.initialize(local_amps_weight1_3q(), [9, 10, 11])

def apply_pure_xy_mixer(qc, beta):
    for block in XY_MIXER_BLOCKS:
        pairs = [(block[0], block[1]), (block[1], block[2]), (block[0], block[2])]
        for i, j in pairs:
            qc.rxx(2.0 * beta, i, j)
            qc.ryy(2.0 * beta, i, j)

def init_register_constraints(bits):
    return (
        bits[0] + bits[1] + bits[2] == 2
        and bits[3] + bits[4] + bits[5] == 1
        and bits[6] + bits[7] + bits[8] == 1
        and bits[9] + bits[10] + bits[11] == 1
    )

def verify_initial_state():
    qc = QuantumCircuit(N_QUBITS)
    apply_register_init(qc)
    sv = Statevector(qc)
    support = []
    for idx, amp in enumerate(sv.data):
        if abs(amp) > 1e-12:
            bits = tuple((idx >> q) & 1 for q in range(N_QUBITS))
            support.append(bits)
    observed = len(support)
    ok = observed == EXPECTED_INIT_SUPPORT and all(init_register_constraints(b) for b in support)
    norm = float(np.linalg.norm(sv.data))
    print("\n[Initialization]")
    print("baseline = register-wise fixed-Hamming-weight pure XY-QAOA")
    print("expected support =", EXPECTED_INIT_SUPPORT)
    print("observed support =", observed)
    print("state norm =", norm)
    print("register constraints verified =", ok)
    if not ok:
        raise RuntimeError("Initial-state verification failed.")
    return support

J_zz, h_z, ising_const = qubo_to_ising_standard()

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

def build_qaoa_circuit(gammas, betas, with_measurements=False):
    qc = QuantumCircuit(N_QUBITS)
    apply_register_init(qc)
    for layer in range(len(gammas)):
        apply_cost_layer(qc, gammas[layer])
        apply_pure_xy_mixer(qc, betas[layer])
    if with_measurements:
        qc.measure_all()
    return qc


def qaoa_objective(theta):
    gammas = theta[:p]
    betas = theta[p:]
    qc = build_qaoa_circuit(gammas, betas, with_measurements=False)
    res = estimator.run([(qc, cost_op)]).result()
    return float(np.asarray(res[0].data.evs).reshape(-1)[0])

C_star_feas, opt_states_feas, feasible_states = brute_force_feasible_optimum()
opt_xstr_set = {bits_to_xstr(s) for s in opt_states_feas}
verify_optimal_costs(opt_states_feas)
verify_initial_state()

print("\n[Problem]")
print("case name =", CASE_NAME)
print("number of qubits =", N_QUBITS)
print("distance matrix =\n", DISTANCE_MATRIX)
print("penalty =", PENALTY)
print("Ising constant term =", ising_const, "(ignored during optimization)")
print("number of feasible states =", len(feasible_states))
print("optimal feasible cost =", C_star_feas)
print("optimal feasible bitstrings =", sorted(opt_xstr_set))
for s in opt_states_feas:
    print("optimal routes", bits_to_xstr(s), "=", routes_from_bits(s))


print("\n[Baseline]")
print("baseline name =", BASELINE_NAME)
print("initialization support size =", EXPECTED_INIT_SUPPORT)
print("register definitions =", XY_REGISTERS)
print("mixer blocks (complete-ring XY order) =", XY_MIXER_BLOCKS)
print("QAOA depth p =", p)


eff_N_RUNS = 1 if _SMOKE else N_RUNS
eff_restarts = 1 if _SMOKE else num_restarts
eff_shots_final = 256 if _SMOKE else shots_final

MAXITER_LIST = [5] if _SMOKE else MAXITER_SWEEP

SUMMARY_TITLE = "Pure XY-QAOA | statevector objective"
all_sweep_rows = []
for maxiter in MAXITER_LIST:
    print(f"\n{'=' * 72}")
    print(f"[Maxiter sweep | case={CASE_NAME} | maxiter={maxiter} | N_RUNS={eff_N_RUNS}]")
    print(f"{'=' * 72}")
    run_records = []
    for run_id in range(eff_N_RUNS):
        rng = np.random.default_rng(BASE_SEED + run_id)
        best_val, best_res = float("inf"), None
        for rr in range(eff_restarts):
            x0 = np.concatenate([
                rng.uniform(-np.pi, np.pi, size=p),
                rng.uniform(0.0, np.pi / 2.0, size=p),
            ])
            res = minimize(
                qaoa_objective,
                x0=x0,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol},
            )
            if res.fun < best_val:
                best_val, best_res = float(res.fun), res
        qc_meas = build_qaoa_circuit(best_res.x[:p], best_res.x[p:], with_measurements=True)
        counts_q = sampler.run([qc_meas], shots=eff_shots_final).result()[0].data.meas.get_counts()
        counts_x = {k[::-1]: v for k, v in counts_q.items()}
        k_star = sum(counts_x.get(x, 0) for x in opt_xstr_set)
        p_star = k_star / eff_shots_final
        p_feas = p_feas_from_counts(counts_x, eff_shots_final)
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
            f"\n[Run {run_id + 1:02d}/{eff_N_RUNS}] best={best_val:.6f} "
            f"p*={p_star:.6f} p_feas={p_feas:.6f} gap={gap_exp:.6f} rank={rank}"
        )

    stats = summarize_experiment_with_pfeas(run_records, eff_N_RUNS)
    row = {"lambda": None, "maxiter": maxiter, "stats": stats, "run_records": run_records}
    all_sweep_rows.append(row)
    csv_path = build_xyqaoa_csv_path(
        CASE_NAME, maxiter, DEFAULT_XYQAOA_RESULTS_DIR,
        scale_tag=SCALE_TAG, method_tag=METHOD_TAG,
    )
    save_experiment_csv_with_pfeas(
        csv_path,
        CASE_NAME,
        None,
        maxiter,
        "COBYLA",
        p,
        num_restarts,
        eff_N_RUNS,
        stats,
        run_records,
        scale_tag=SCALE_TAG,
        method_tag=METHOD_TAG,
    )
    print(f"[Saved] {csv_path}")
    print_experiment_summary_with_pfeas(
        SUMMARY_TITLE, stats, "COBYLA", eff_N_RUNS, maxiter
    )

print_maxiter_sweep_selection(
    CASE_NAME,
    all_sweep_rows,
    summary_title=SUMMARY_TITLE,
    optimizer="COBYLA",
    n_runs=eff_N_RUNS,
    summary_printer=print_experiment_summary_with_pfeas,
    output_dir=DEFAULT_XYQAOA_RESULTS_DIR,
    scale_tag=SCALE_TAG,
    method_tag=METHOD_TAG,
    experiment_meta={
        "scale_tag": SCALE_TAG,
        "method_tag": METHOD_TAG,
        "p": p,
        "num_restarts": num_restarts,
        "n_runs": eff_N_RUNS,
    },
)
