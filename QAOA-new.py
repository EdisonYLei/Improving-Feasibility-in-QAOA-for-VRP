import numpy as np
import itertools
from math import log, ceil, sqrt
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

# ============================================================
# 0) Problem setup (same QUBO and variable order as baseline)
#    Qubits q0..q5 correspond to variables [x01,x02,x10,x12,x20,x21].
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

# New initial state: uniform superposition of four basis states (variable order q0..q5).
INIT_STATES = ["000101", "100110", "011001", "111010"]
LAMBDA = 1.0  # Default mixer weight; keep fixed across runs for fair comparison.

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

def bits_to_str_xorder(bits):
    return "".join(str(b) for b in bits)

def qubo_to_pauli_z_ising_standard(q_const, q_lin, q_quad, n):
    # QUBO variable x_i maps to (I - Z_i)/2 in Ising form.
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

def build_uniform_superposition_statevector(bitstrings, n):
    """
    Build |psi0> = (1/sqrt(m)) sum_k |bitstrings[k]>.
    Bitstrings are in variable order q0..q_{n-1} (left to right).
    Map to computational basis index via little-endian: idx = sum_i bit[i] * 2^i.
    """
    m = len(bitstrings)
    vec = np.zeros(2**n, dtype=complex)
    amp = 1.0 / np.sqrt(m)

    for s in bitstrings:
        if len(s) != n or any(ch not in "01" for ch in s):
            raise ValueError(f"Bad bitstring: {s}")
        bits = [int(ch) for ch in s]  # variable order q0..q_{n-1}
        idx = sum(bits[i] * (2**i) for i in range(n))
        vec[idx] += amp

    vec = vec / np.linalg.norm(vec)
    return vec

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
    t = t_critical_975_df29() if n == 30 else 2.0
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
# 2) Build Ising + cost operator
# ============================================================

J_zz, h_z, ising_const = qubo_to_pauli_z_ising_standard(
    qubo_const, qubo_linear, qubo_quad, N_QUBITS
)
cost_op = build_cost_sparse_pauli_op(J_zz, h_z, N_QUBITS, scale=ENERGY_SCALE)

psi0_vec = build_uniform_superposition_statevector(INIT_STATES, N_QUBITS)

# ============================================================
# 3) QAOA circuit: new initial state + custom mixer
# ============================================================

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
    # Pairs (q2,q3) and (q4,q5): exp(-i*beta*(XX+YY)) via RXX, RYY.
    qc.rxx(2.0 * beta, 2, 3)
    qc.ryy(2.0 * beta, 2, 3)
    qc.rxx(2.0 * beta, 4, 5)
    qc.ryy(2.0 * beta, 4, 5)
    # Single-qubit mixers on q0, q1 with weight lambda.
    qc.rx(2.0 * beta * lam, 0)
    qc.rx(2.0 * beta * lam, 1)

def build_qaoa_circuit(gammas, betas, with_measurements=False):
    p = len(gammas)
    qc = QuantumCircuit(N_QUBITS)

    # Custom initial state (uniform superposition of four basis states).
    qc.initialize(psi0_vec, list(range(N_QUBITS)))

    for l in range(p):
        apply_cost_layer(qc, gammas[l])
        apply_custom_mixer(qc, betas[l], lam=LAMBDA)

    if with_measurements:
        qc.measure_all()
    return qc

# ============================================================
# 4) Estimator objective + Sampler
# ============================================================

estimator = StatevectorEstimator()
sampler = StatevectorSampler()

def qaoa_objective(theta, p):
    gammas = theta[:p]
    betas = theta[p:]
    qc = build_qaoa_circuit(gammas, betas, with_measurements=False)
    res = estimator.run([(qc, cost_op)]).result()
    e_scaled = float(np.asarray(res[0].data.evs).reshape(-1)[0])
    return e_scaled

# ============================================================
# 5) Ground truth: feasible optimum (for your 4 metrics)
# ============================================================

C_star_feas, opt_states_feas = brute_force_feasible_optimum(
    N_QUBITS, qubo_const, qubo_linear, qubo_quad
)
opt_xstr_set = {bits_to_str_xorder(s) for s in opt_states_feas}

print("[Feasible optimum]")
print("C*_feas =", C_star_feas)
print("Optimal feasible state(s) =", opt_states_feas)
print("Optimal feasible xstr set =", opt_xstr_set)

# ============================================================
# 6) Run 30 experiments with controlled seeds + summarize 4 metrics
# ============================================================

# Keep these identical across methods for fair comparison.
p = 2
num_restarts = 10
maxiter = 250
shots_final = 4096

N_RUNS = 30
BASE_SEED = 12345  # Use the same value as baseline for paired comparison.

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

    # restarts (same distribution each run; paired across methods via seed_run)
    for rr in range(num_restarts):
        x0 = np.concatenate([
            rng.uniform(-np.pi, np.pi, size=p),
            rng.uniform(0.0, np.pi / 2.0, size=p),
        ])
        res = minimize(
            qaoa_objective,
            x0=x0,
            args=(p,),
            method="COBYLA",
            options={"maxiter": maxiter, "rhobeg": 0.5, "tol": 1e-4}
        )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_res = res

    best_gammas = best_res.x[:p]
    best_betas = best_res.x[p:]

    # Final sampling (set explicit seed if supported by your Qiskit version).
    qc_meas = build_qaoa_circuit(best_gammas, best_betas, with_measurements=True)
    sres = sampler.run([qc_meas], shots=shots_final).result()
    counts = sres[0].data.meas.get_counts()  # Keys in Qiskit order q_{n-1}..q_0.

    # Convert to variable order q_0..q_{n-1}.
    counts_x = {k[::-1]: v for k, v in counts.items()}

    # Metric (1): optimal-state probability p*.
    k_star = sum(counts_x.get(xstr, 0) for xstr in opt_xstr_set)
    p_star = k_star / shots_final

    # metric (2): run-level success
    run_success = 1 if k_star >= 1 else 0

    # Metric (3): expected energy gap from final counts.
    E_hat = expected_energy_from_counts_xorder(counts_x)
    gap_exp = E_hat - C_star_feas

    # metric (4): TTS99
    tts99 = tts_shots_for_success(p_star, target=0.99)
    sampling_rank_run = sampling_rank(counts_x, opt_xstr_set)

    p_star_list.append(p_star)
    run_success_list.append(run_success)
    gap_exp_list.append(gap_exp)
    tts99_list.append(tts99)
    rank_list.append(sampling_rank_run)

    print(f"\n[Run {run_id+1:02d}/{N_RUNS} | seed={seed_run}]")
    print("  best scaled objective =", best_val)
    print(f"  p* = {p_star:.6f} (k*={k_star}/{shots_final})  run-success={run_success}")
    print(f"  E_hat = {E_hat:.6f}  gap_exp = {gap_exp:.6f}")
    print(f"  TTS99 = {tts99}  sampling_rank = {sampling_rank_run}")

# Summarize the four metrics over 30 runs.
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
print(f"[Summary over {N_RUNS} runs | NEW init+mixer | shots_final={shots_final}]")
print("="*70)

print("\n(1) Mean optimal-state probability p* (final sampling)")
print(f"    mean = {p_mean:.6f}, std = {p_std:.6f}, 95% CI = [{p_ci[0]:.6f}, {p_ci[1]:.6f}]")

print("\n(2) Run-level success rate (>=1 optimal sample in final sampling)")
print(f"    success = {succ_k}/{N_RUNS} = {succ_rate:.6f}, 95% Wilson CI = [{succ_ci[0]:.6f}, {succ_ci[1]:.6f}]")

print("\n(3) Expected energy gap (E[C] - C*_feas) from final counts")
print(f"    mean = {gap_mean:.6f}, std = {gap_std:.6f}, 95% CI = [{gap_ci[0]:.6f}, {gap_ci[1]:.6f}]")

print("\n(4) TTS shots for >=99% success (derived from p*), report finite median/IQR")
print(f"    median = {tts_median:.2f}, IQR = [{tts_q25:.2f}, {tts_q75:.2f}], #inf (p*=0) = {num_inf}/{N_RUNS}")

rank_mean, rank_ci = mean_ci95_t(rank_list)
rank_std = float(np.std(rank_list, ddof=1))
print("\n(5) Sampling rank (rank of optimal among bitstrings by frequency, 1=most frequent)")
print(f"    mean = {rank_mean:.2f}, std = {rank_std:.2f}, 95% CI = [{rank_ci[0]:.2f}, {rank_ci[1]:.2f}]")
print("="*70)