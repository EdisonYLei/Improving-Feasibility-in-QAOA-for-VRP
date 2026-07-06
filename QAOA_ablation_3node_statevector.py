"""Proposed-method ablation for 3-node (statevector).

Reference: QAOA-new.py / Shot-baseQAOA-new.py / AerQAOA-new.py
Runs init-only and mixer-only at fixed (lambda, maxiter) from the full proposed sweep.
"""
import itertools
from math import log, ceil, sqrt

import numpy as np

from qaoa_ablation_runner import run_both_threenode_ablation_arms
from qaoa_new_4node_runtime import (
    DEFAULT_RESULTS_DIR,
    NOISY_NUM_RESTARTS,
    THREENODE_SHOT_BASE_SEED,
    THREENODE_SHOT_N_RUNS,
    THREENODE_SHOT_PROTOCOL,
    THREENODE_SHOT_SEED_TRANSPILER,
    build_cost_sparse_pauli_op,
    build_uniform_superposition_statevector,
)

N_QUBITS = 6
qubo_const = 5662.8
qubo_linear = {0: -1681.1, 1: -1737.7, 2: -2116.7, 3: -828.3, 4: -2173.3, 5: -828.3}
qubo_quad = {(2, 4): 1306.8, (0, 1): 871.2, (2, 3): 871.2, (0, 5): 871.2, (4, 5): 871.2, (1, 3): 871.2}
ENERGY_SCALE = 435.6
INIT_STATES = ["000101", "100110", "011001", "111010"]
XY_BLOCKS = [(2, 3), (4, 5)]
X_MIXER_QUBITS = [0, 1]
XY_PAIRS = [(2, 3), (4, 5)]
DEPOT_QUBITS = [0, 1]

p = 2
N_RUNS = THREENODE_SHOT_N_RUNS
BASE_SEED = THREENODE_SHOT_BASE_SEED
num_restarts = NOISY_NUM_RESTARTS
shots_obj = 0
shots_final = 4096
batches_obj = 0
rhobeg = 0.5
tol = 1e-4
seed_transpiler = THREENODE_SHOT_SEED_TRANSPILER
OPTIMIZER = "cobyla"


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
    return "".join(str(b) for b in bits)


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


def expected_energy_from_counts(counts_x):
    shots = sum(counts_x.values())
    energy = 0.0
    for xstr, count in counts_x.items():
        bits = tuple(int(ch) for ch in xstr)
        energy += (count / shots) * qubo_cost_from_bits(bits)
    return float(energy)


def counts_to_xorder(counts):
    return {k[::-1]: v for k, v in counts.items()}


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


J_zz, h_z, _ising_const = qubo_to_ising_standard()
C_star_feas, opt_states_feas = brute_force_feasible_optimum()
opt_xstr_set = {bits_to_xstr(s) for s in opt_states_feas}
psi0_vec = build_uniform_superposition_statevector(INIT_STATES, N_QUBITS)
cost_op = build_cost_sparse_pauli_op(J_zz, h_z, scale=ENERGY_SCALE, n_qubits=N_QUBITS)

print("[Feasible optimum]")
print("C*_feas =", C_star_feas)
print("optimal feasible xstr set =", opt_xstr_set)

lam, maxiter, source, _results = run_both_threenode_ablation_arms(
    regime="statevector",
    psi0_vec=psi0_vec,
    cost_op=cost_op,
    n_qubits=N_QUBITS,
    p_layers=p,
    xy_blocks=XY_BLOCKS,
    x_mixer_qubits=X_MIXER_QUBITS,
    xy_pairs=XY_PAIRS,
    depot_qubits=DEPOT_QUBITS,
    J_zz=J_zz,
    h_z=h_z,
    energy_scale=ENERGY_SCALE,
    c_star_feas=C_star_feas,
    opt_xstr_set=opt_xstr_set,
    expected_energy_from_counts=expected_energy_from_counts,
    counts_to_xorder=counts_to_xorder,
    sampling_rank=sampling_rank,
    tts_shots_for_success=tts_shots_for_success,
    n_runs=N_RUNS,
    base_seed=BASE_SEED,
    num_restarts=num_restarts,
    optimizer=OPTIMIZER,
    rhobeg=rhobeg,
    tol=tol,
    shots_obj=shots_obj,
    shots_final=shots_final,
    batches_obj=batches_obj,
    seed_transpiler=seed_transpiler,
    output_dir=DEFAULT_RESULTS_DIR,
)

print("\n[Ablation complete]")
print(f"3-node | regime=statevector | lambda={lam:g} | maxiter={maxiter}")
print(f"hyperparam source: {source}")
