"""Shared runtime helpers for the standard and structured QAOA experiment scripts."""

from __future__ import annotations

import csv
import os
import shutil
from math import sqrt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.library import SetStatevector
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from scipy.optimize import minimize

OPTIMIZER_SPSA = "spsa"
OPTIMIZER_COBYLA = "cobyla"
OPTIMIZER_LBFGSB = "lbfgsb"

MAXITER_SWEEP = list(range(200, 2001, 100))
THREENODE_MAXITER_SWEEP = MAXITER_SWEEP  # 200 .. 2000 step 100
THREENODE_LAMBDA_LIST = [round(0.5 + 0.1 * i, 10) for i in range(16)]  # 0.5 .. 2.0 step 0.1

# 3-node ideal/noisy shot pairs: identical protocol except simulator (ideal vs noisy).
THREENODE_SHOT_N_RUNS = 30
THREENODE_SHOT_BASE_SEED = 12345
THREENODE_SHOT_SEED_TRANSPILER = 999

NOISY_MAXITER = 200
NOISY_NUM_RESTARTS = 10

# Shared 3-node finite-shot protocol for AerQAOA / AerQAOA-new and paired ideal scripts.
THREENODE_SHOT_PROTOCOL = {
    "num_restarts": NOISY_NUM_RESTARTS,
    "shots_obj": 2048,
    "shots_final": 8192,
    "batches_obj": 2,
    "rhobeg": 0.5,
    "tol": 2e-3,
}

# Backward-compatible aliases (proposed vs standard differ only in circuit/simulator).
THREENODE_PROPOSED_SHOT_PROTOCOL = THREENODE_SHOT_PROTOCOL
THREENODE_STANDARD_SHOT_PROTOCOL = THREENODE_SHOT_PROTOCOL


def threenode_objective_seed(run_id, restart_idx):
    """Paired objective-evaluation seed (ideal/noisy scripts must match)."""
    return 7000 + 100 * restart_idx + 10_000 * run_id


def threenode_final_seed(run_id):
    """Paired final-sampling seed (ideal/noisy scripts must match)."""
    return 2026 + 10_000 * run_id

NOISY_LAMBDA_LIST = [0.5, 1.0, 1.5, 2.0, 2.5]
# For fixed-lambda debugging, set LAMBDA_LIST = [2.0].
LAMBDA_LIST = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5
]
DEFAULT_RESULTS_DIR = "results_4node_new"
DEFAULT_XYQAOA_RESULTS_DIR = "results_xyqaoa"

_ONEHOT_AMPS_3Q = None


def _onehot_amps_3q():
    global _ONEHOT_AMPS_3Q
    if _ONEHOT_AMPS_3Q is None:
        amps = np.zeros(8, dtype=float)
        for idx in (1, 2, 4):
            amps[idx] = 1.0 / np.sqrt(3.0)
        _ONEHOT_AMPS_3Q = amps
    return _ONEHOT_AMPS_3Q


def build_uniform_superposition_statevector(bitstrings, n):
    vec = np.zeros(2**n, dtype=complex)
    amp = 1.0 / np.sqrt(len(bitstrings))
    for s in bitstrings:
        bits = [int(ch) for ch in s]
        idx = sum(bits[i] * (2**i) for i in range(n))
        vec[idx] += amp
    vec /= np.linalg.norm(vec)
    return vec


def build_cost_sparse_pauli_op(J_zz, h_z, scale=1.0, n_qubits=None):
    triples = []
    for (i, j), coeff in J_zz.items():
        c = coeff / scale
        if abs(c) > 1e-12:
            triples.append(("ZZ", [i, j], c))
    for i, coeff in h_z.items():
        c = coeff / scale
        if abs(c) > 1e-12:
            triples.append(("Z", [i], c))
    if n_qubits is None:
        n_qubits = max(
            [max(pair) for pair in J_zz.keys()] + list(h_z.keys()) + [0]
        ) + 1
    return SparsePauliOp.from_sparse_list(triples, num_qubits=n_qubits).simplify()


def apply_structured_init(qc, xy_blocks, x_mixer_qubits):
    """Uniform superposition over INIT_STATES without full-state synthesis."""
    qc.h(x_mixer_qubits)
    amps = _onehot_amps_3q()
    for block in xy_blocks:
        qc.initialize(amps, block)


def apply_cost_layer(qc, gamma, J_zz, h_z, energy_scale):
    for (i, j), J in J_zz.items():
        Js = J / energy_scale
        if abs(Js) > 1e-12:
            qc.rzz(2.0 * gamma * Js, i, j)
    for i, h in h_z.items():
        hs = h / energy_scale
        if abs(hs) > 1e-12:
            qc.rz(2.0 * gamma * hs, i)


def apply_hybrid_mixer(qc, beta, xy_blocks, x_mixer_qubits, lam):
    for block in xy_blocks:
        for a in range(len(block)):
            for b in range(a + 1, len(block)):
                i, j = block[a], block[b]
                qc.rxx(2.0 * beta, i, j)
                qc.ryy(2.0 * beta, i, j)
    for q in x_mixer_qubits:
        qc.rx(2.0 * beta * lam, q)


def build_qaoa_circuit_structured(
    gammas,
    betas,
    n_qubits,
    xy_blocks,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    with_measurements=False,
):
    qc = QuantumCircuit(n_qubits)
    apply_structured_init(qc, xy_blocks, x_mixer_qubits)
    for layer in range(len(gammas)):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        apply_hybrid_mixer(qc, betas[layer], xy_blocks, x_mixer_qubits, lam)
    if with_measurements:
        qc.measure_all()
    return qc


def build_parametric_qaoa_circuit(
    p_layers,
    n_qubits,
    xy_blocks,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    with_measurements=True,
):
    gammas = [Parameter(f"γ_{idx}") for idx in range(p_layers)]
    betas = [Parameter(f"β_{idx}") for idx in range(p_layers)]
    qc = QuantumCircuit(n_qubits)
    apply_structured_init(qc, xy_blocks, x_mixer_qubits)
    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        apply_hybrid_mixer(qc, betas[layer], xy_blocks, x_mixer_qubits, lam)
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


def build_plus_statevector(n_qubits):
    """Uniform |+>^n statevector (standard QAOA initialization)."""
    amplitude = 1.0 / np.sqrt(2**n_qubits)
    return np.full(2**n_qubits, amplitude, dtype=np.complex128)


def apply_standard_rx_mixer(qc, beta, n_qubits):
    """Standard transverse-field mixer: exp(-i beta sum_j X_j)."""
    for q in range(n_qubits):
        qc.rx(2.0 * beta, q)


ABLATION_INIT_ONLY = "init_only"
ABLATION_MIXER_ONLY = "mixer_only"


def make_statevector_objective(
    psi0_vec,
    cost_op,
    n_qubits,
    p_layers,
    xy_blocks,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    *,
    ablation_arm=None,
):
    if ablation_arm == ABLATION_INIT_ONLY:
        psi0 = Statevector(psi0_vec)

        def apply_mixer(layer_qc, beta):
            apply_standard_rx_mixer(layer_qc, beta, n_qubits)

    elif ablation_arm == ABLATION_MIXER_ONLY:
        psi0 = Statevector(build_plus_statevector(n_qubits))

        def apply_mixer(layer_qc, beta):
            apply_hybrid_mixer(
                layer_qc, beta, xy_blocks, x_mixer_qubits, lam
            )

    else:
        psi0 = Statevector(psi0_vec)

        def apply_mixer(layer_qc, beta):
            apply_hybrid_mixer(
                layer_qc, beta, xy_blocks, x_mixer_qubits, lam
            )

    def qaoa_objective(theta):
        gammas = theta[:p_layers]
        betas = theta[p_layers:]
        state = psi0.copy()
        for layer in range(p_layers):
            layer_qc = QuantumCircuit(n_qubits)
            apply_cost_layer(layer_qc, gammas[layer], J_zz, h_z, energy_scale)
            apply_mixer(layer_qc, betas[layer])
            state = state.evolve(layer_qc)
        return float(state.expectation_value(cost_op).real)

    return qaoa_objective


def build_qaoa_circuit_ablation(
    gammas,
    betas,
    n_qubits,
    xy_blocks,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    proposed_psi0_vec,
    *,
    ablation_arm,
    init_mode="psi0_initialize",
    with_measurements=False,
):
    """Measurement circuit matching ``make_statevector_objective`` ablation arms."""
    qc = QuantumCircuit(n_qubits)
    if ablation_arm == ABLATION_INIT_ONLY:
        if init_mode == "structured":
            apply_structured_init(qc, xy_blocks, x_mixer_qubits)
        else:
            qc.initialize(proposed_psi0_vec, list(range(n_qubits)))
    elif ablation_arm == ABLATION_MIXER_ONLY:
        qc.h(range(n_qubits))
    else:
        raise ValueError(f"Unknown ablation arm: {ablation_arm!r}")

    for layer in range(len(gammas)):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        if ablation_arm == ABLATION_INIT_ONLY:
            apply_standard_rx_mixer(qc, betas[layer], n_qubits)
        else:
            apply_hybrid_mixer(
                qc, betas[layer], xy_blocks, x_mixer_qubits, lam
            )

    if with_measurements:
        qc.measure_all()
    return qc


class CachedShotRunner:
    """Build/transpile the parametric circuit once; rebind angles each evaluation."""

    def __init__(
        self,
        p_layers,
        n_qubits,
        xy_blocks,
        x_mixer_qubits,
        lam,
        J_zz,
        h_z,
        energy_scale,
        simulator,
        transpile_kwargs=None,
    ):
        transpile_kwargs = transpile_kwargs or {}
        template, self.params = build_parametric_qaoa_circuit(
            p_layers,
            n_qubits,
            xy_blocks,
            x_mixer_qubits,
            lam,
            J_zz,
            h_z,
            energy_scale,
            with_measurements=True,
        )
        self.simulator = simulator
        self.energy_scale = energy_scale
        self.tqc = transpile(template, backend=simulator, **transpile_kwargs)

    def _bind(self, theta):
        binding = {param: float(theta[idx]) for idx, param in enumerate(self.params)}
        return self.tqc.assign_parameters(binding)

    def objective_energy(self, theta, shots, seed_sim, batches, energy_from_counts, counts_to_xorder):
        vals = []
        bound = self._bind(theta)
        for batch_idx in range(batches):
            result = self.simulator.run(
                bound,
                shots=shots,
                seed_simulator=seed_sim + batch_idx,
            ).result()
            counts_x = counts_to_xorder(result.get_counts())
            vals.append(energy_from_counts(counts_x) / self.energy_scale)
        return float(np.mean(vals))

    def run_final(self, theta, shots, seed_sim, counts_to_xorder):
        bound = self._bind(theta)
        result = self.simulator.run(
            bound,
            shots=shots,
            seed_simulator=seed_sim,
        ).result()
        return counts_to_xorder(result.get_counts())


def build_standard_parametric_qaoa_circuit(
    p_layers,
    n_qubits,
    J_zz,
    h_z,
    energy_scale,
    *,
    with_measurements=True,
):
    """Standard QAOA: |+>^n init, Ising cost, RX mixer."""
    gammas = [Parameter(f"γ_{idx}") for idx in range(p_layers)]
    betas = [Parameter(f"β_{idx}") for idx in range(p_layers)]
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        for q in range(n_qubits):
            qc.rx(2.0 * betas[layer], q)
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


STATE_PREPARATION_NOISE_MODELED = False

PROPOSED_NOISY_BASIS = ["h", "rz", "rx", "rzz", "rxx", "ryy"]


def proposed_xy_pairs_from_blocks(xy_blocks):
    """All unordered pairs within each proposed XY block (matches apply_hybrid_mixer)."""
    pairs = []
    for block in xy_blocks:
        for a in range(len(block)):
            for b in range(a + 1, len(block)):
                pairs.append((block[a], block[b]))
    return pairs


def proposed_noisy_transpile_kwargs(seed_transpiler=999, *, direct_statevector_init=True):
    """Transpile kwargs for proposed noisy shot runners.

    When ``direct_statevector_init`` is True (SetStatevector prep), omit
    ``basis_gates`` so Aer keeps the native state-prep instruction.
    """
    kwargs = {
        "seed_transpiler": seed_transpiler,
        "optimization_level": 1,
    }
    if not direct_statevector_init:
        kwargs["basis_gates"] = PROPOSED_NOISY_BASIS
    return kwargs


STANDARD_NOISY_BASIS = ["h", "rz", "rx", "rzz"]


def standard_noisy_transpile_kwargs(seed_transpiler=999, *, use_backend_target=True):
    """Transpile kwargs for standard noisy QAOA shot runners.

    When transpiling with ``backend=simulator``, omit ``basis_gates`` so Qiskit
    does not warn about overriding the backend target / error rates.
    """
    kwargs = {
        "seed_transpiler": seed_transpiler,
        "optimization_level": 1,
    }
    if not use_backend_target:
        kwargs["basis_gates"] = STANDARD_NOISY_BASIS
    return kwargs


def apply_proposed_direct_initial_state(qc, initial_statevector):
    """Noiseless direct state preparation (SetStatevector; prep gates not noised)."""
    qc.append(SetStatevector(initial_statevector), list(range(qc.num_qubits)))


def build_parametric_initialize_proposed_circuit(
    psi0_vec,
    p_layers,
    n_qubits,
    lam,
    xy_pairs,
    depot_qubits,
    J_zz,
    h_z,
    energy_scale,
    *,
    with_measurements=True,
):
    """Proposed QAOA with direct statevector init and XY+RX(λ) mixer."""
    gammas = [Parameter(f"γ_{idx}") for idx in range(p_layers)]
    betas = [Parameter(f"β_{idx}") for idx in range(p_layers)]
    qc = QuantumCircuit(n_qubits)
    apply_proposed_direct_initial_state(qc, psi0_vec)
    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        for i, j in xy_pairs:
            qc.rxx(2.0 * betas[layer], i, j)
            qc.ryy(2.0 * betas[layer], i, j)
        for q in depot_qubits:
            qc.rx(2.0 * betas[layer] * lam, q)
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


def build_parametric_initialize_proposed_circuit_ablation(
    psi0_vec,
    p_layers,
    n_qubits,
    lam,
    xy_pairs,
    depot_qubits,
    J_zz,
    h_z,
    energy_scale,
    *,
    ablation_arm,
    with_measurements=True,
):
    """3-node proposed init via SetStatevector; ablation init-only or mixer-only."""
    gammas = [Parameter(f"γ_{idx}") for idx in range(p_layers)]
    betas = [Parameter(f"β_{idx}") for idx in range(p_layers)]
    qc = QuantumCircuit(n_qubits)
    if ablation_arm == ABLATION_INIT_ONLY:
        apply_proposed_direct_initial_state(qc, psi0_vec)
    elif ablation_arm == ABLATION_MIXER_ONLY:
        qc.h(range(n_qubits))
    else:
        raise ValueError(f"Unknown ablation arm: {ablation_arm!r}")

    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        if ablation_arm == ABLATION_INIT_ONLY:
            apply_standard_rx_mixer(qc, betas[layer], n_qubits)
        else:
            for i, j in xy_pairs:
                qc.rxx(2.0 * betas[layer], i, j)
                qc.ryy(2.0 * betas[layer], i, j)
            for q in depot_qubits:
                qc.rx(2.0 * betas[layer] * lam, q)
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


def build_standard_noisy_simulator(n_qubits):
    """Standard noisy Aer backend: H/RZ/RX + RZZ depolarizing and readout error."""
    noise_model = NoiseModel()
    p01 = p10 = 0.001
    ro_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])
    err_1q = depolarizing_error(0.00015, 1)
    for gate in ["h", "rz", "rx"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)
    err_2q = depolarizing_error(0.00125, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, "rzz")
    return _configure_aer_noisy_safe(AerSimulator(noise_model=noise_model))


def build_proposed_noisy_simulator(n_qubits):
    """Proposed noisy Aer backend: adds RXX/RYY two-qubit depolarizing channels."""
    noise_model = NoiseModel()
    p01 = p10 = 0.001
    ro_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])
    err_1q = depolarizing_error(0.00015, 1)
    for gate in ["h", "rz", "rx"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)
    err_2q = depolarizing_error(0.00125, 2)
    for gate in ["rzz", "rxx", "ryy"]:
        noise_model.add_all_qubit_quantum_error(err_2q, gate)
    return _configure_aer_noisy_safe(AerSimulator(noise_model=noise_model))


class CachedStandardShotRunner:
    """Cached parametric standard QAOA circuit for noisy/ideal shot simulation."""

    def __init__(
        self,
        p_layers,
        n_qubits,
        J_zz,
        h_z,
        energy_scale,
        simulator,
        transpile_kwargs=None,
    ):
        transpile_kwargs = transpile_kwargs or {}
        template, self.params = build_standard_parametric_qaoa_circuit(
            p_layers,
            n_qubits,
            J_zz,
            h_z,
            energy_scale,
            with_measurements=True,
        )
        self.simulator = simulator
        self.energy_scale = energy_scale
        self.tqc = transpile(template, backend=simulator, **transpile_kwargs)

    def _bind(self, theta):
        binding = {param: float(theta[idx]) for idx, param in enumerate(self.params)}
        return self.tqc.assign_parameters(binding)

    def objective_energy(self, theta, shots, seed_sim, batches, energy_from_counts, counts_to_xorder):
        vals = []
        bound = self._bind(theta)
        for batch_idx in range(batches):
            result = self.simulator.run(
                bound,
                shots=shots,
                seed_simulator=seed_sim + batch_idx,
            ).result()
            counts_x = counts_to_xorder(result.get_counts())
            vals.append(energy_from_counts(counts_x) / self.energy_scale)
        return float(np.mean(vals))

    def run_final(self, theta, shots, seed_sim, counts_to_xorder):
        bound = self._bind(theta)
        result = self.simulator.run(
            bound,
            shots=shots,
            seed_simulator=seed_sim,
        ).result()
        return counts_to_xorder(result.get_counts())


class CachedInitializeProposedShotRunner:
    """Cached parametric proposed circuit with statevector initialize."""

    def __init__(
        self,
        psi0_vec,
        p_layers,
        n_qubits,
        lam,
        xy_pairs,
        depot_qubits,
        J_zz,
        h_z,
        energy_scale,
        simulator,
        transpile_kwargs=None,
        *,
        ablation_arm=None,
    ):
        transpile_kwargs = transpile_kwargs or {}
        if ablation_arm is None:
            template, self.params = build_parametric_initialize_proposed_circuit(
                psi0_vec,
                p_layers,
                n_qubits,
                lam,
                xy_pairs,
                depot_qubits,
                J_zz,
                h_z,
                energy_scale,
                with_measurements=True,
            )
        else:
            template, self.params = build_parametric_initialize_proposed_circuit_ablation(
                psi0_vec,
                p_layers,
                n_qubits,
                lam,
                xy_pairs,
                depot_qubits,
                J_zz,
                h_z,
                energy_scale,
                ablation_arm=ablation_arm,
                with_measurements=True,
            )
        self.simulator = simulator
        self.energy_scale = energy_scale
        self.tqc = transpile(template, backend=simulator, **transpile_kwargs)

    def _bind(self, theta):
        binding = {param: float(theta[idx]) for idx, param in enumerate(self.params)}
        return self.tqc.assign_parameters(binding)

    def objective_energy(self, theta, shots, seed_sim, batches, energy_from_counts, counts_to_xorder):
        vals = []
        bound = self._bind(theta)
        for batch_idx in range(batches):
            result = self.simulator.run(
                bound,
                shots=shots,
                seed_simulator=seed_sim + batch_idx,
            ).result()
            counts_x = counts_to_xorder(result.get_counts())
            vals.append(energy_from_counts(counts_x) / self.energy_scale)
        return float(np.mean(vals))

    def run_final(self, theta, shots, seed_sim, counts_to_xorder):
        bound = self._bind(theta)
        result = self.simulator.run(
            bound,
            shots=shots,
            seed_simulator=seed_sim,
        ).result()
        return counts_to_xorder(result.get_counts())


def _configure_aer_threading(sim):
    """Use Aer defaults (multi-thread). Set QAOA_AER_SAFE=1 for single-thread fallback."""
    if os.environ.get("QAOA_AER_SAFE", "0") == "1":
        sim.set_options(
            max_parallel_threads=1,
            max_parallel_shots=1,
            max_parallel_experiments=1,
        )
    return sim


def _configure_aer_noisy_safe(sim):
    """Conservative Aer options for noisy shot simulation.

    qiskit-aer can SIGABRT inside ``Fusion::optimize_circuit`` when OpenMP
    parallel shots run on noisy statevector paths (macOS/Python 3.14 observed).
    Default: single-thread + fusion off. Set ``QAOA_AER_NOISY_FAST=1`` to use
    multi-thread Aer (same as ideal) at your own risk.
    """
    if os.environ.get("QAOA_AER_NOISY_FAST", "0") == "1":
        return _configure_aer_threading(sim)
    sim.set_options(
        max_parallel_threads=1,
        max_parallel_shots=1,
        max_parallel_experiments=1,
        fusion_enable=False,
    )
    return sim


def build_ideal_shot_simulator(**aer_kwargs):
    """Ideal AerSimulator for XY/overlap4-style shot QAOA (statevector sampling)."""
    aer_kwargs.setdefault("method", "statevector")
    return _configure_aer_threading(AerSimulator(**aer_kwargs))


PURE_XY_NOISY_BASIS = ["h", "rz", "rx", "rzz", "rxx", "ryy"]


def build_pure_xy_noisy_simulator(n_qubits):
    """Noisy Aer backend for pure-XY QAOA (RXX/RYY/RZZ + extended 1q basis)."""
    noise_model = NoiseModel()
    p01 = p10 = 0.001
    ro_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])
    err_1q = depolarizing_error(0.00015, 1)
    for gate in ["h", "rz", "rx", "u", "u1", "u2", "u3", "p", "sx", "x"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)
    err_2q = depolarizing_error(0.00125, 2)
    for gate in ["rzz", "rxx", "ryy", "cx", "cz", "swap"]:
        noise_model.add_all_qubit_quantum_error(err_2q, gate)
    sim = AerSimulator(noise_model=noise_model)
    return _configure_aer_noisy_safe(sim)


def pure_xy_noisy_transpile_kwargs(seed_transpiler=999, *, direct_statevector_init=True):
    """Transpile kwargs for pure-XY noisy shot runners.

    When ``direct_statevector_init`` is True (SetStatevector prep), omit
    ``basis_gates`` so Aer keeps the native state-prep instruction.
    """
    kwargs = {
        "seed_transpiler": seed_transpiler,
        "optimization_level": 1,
    }
    if not direct_statevector_init:
        kwargs["basis_gates"] = PURE_XY_NOISY_BASIS
    return kwargs


def apply_pure_xy_mixer_parametric(qc, beta, mixer_pairs):
    """Pure XY mixer: RXX+RYY on each (i, j) pair in the given order."""
    for i, j in mixer_pairs:
        qc.rxx(2.0 * beta, i, j)
        qc.ryy(2.0 * beta, i, j)


def pure_xy_mixer_pairs_from_blocks(mixer_blocks):
    """Expand register blocks to the legacy complete-ring pair order."""
    pairs = []
    for block in mixer_blocks:
        if len(block) == 2:
            pairs.append((block[0], block[1]))
        else:
            pairs.extend(
                [(block[0], block[1]), (block[1], block[2]), (block[0], block[2])]
            )
    return pairs


def statevector_from_register_init(apply_register_init, n_qubits):
    """Materialize register-wise init into one normalized statevector."""
    qc = QuantumCircuit(n_qubits)
    apply_register_init(qc)
    vector = np.asarray(Statevector(qc).data, dtype=complex)
    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0, atol=1e-12):
        raise RuntimeError(f"Register init statevector not normalized: norm={norm}")
    return vector


def _aer_simulator_is_noisy(simulator):
    noise_model = getattr(simulator, "options", {}).get("noise_model")
    if noise_model is not None:
        return True
    return getattr(simulator, "_noise_model", None) is not None


def apply_pure_xy_initial_state(qc, initial_statevector):
    qc.append(SetStatevector(initial_statevector), list(range(qc.num_qubits)))


def build_pure_xy_parametric_circuit(
    p_layers,
    n_qubits,
    J_zz,
    h_z,
    energy_scale,
    mixer_pairs,
    *,
    apply_register_init=None,
    initial_statevector=None,
    with_measurements=True,
):
    """Register-wise pure XY-QAOA with parametric cost/mixer layers."""
    if (apply_register_init is None) == (initial_statevector is None):
        raise ValueError("Provide exactly one of apply_register_init or initial_statevector.")
    gammas = [Parameter(f"γ_{idx}") for idx in range(p_layers)]
    betas = [Parameter(f"β_{idx}") for idx in range(p_layers)]
    qc = QuantumCircuit(n_qubits)
    if initial_statevector is not None:
        apply_pure_xy_initial_state(qc, initial_statevector)
    else:
        apply_register_init(qc)
    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        apply_pure_xy_mixer_parametric(qc, betas[layer], mixer_pairs)
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


class CachedPureXYShotRunner:
    """Cached parametric pure-XY QAOA circuit for ideal or noisy shot simulation."""

    def __init__(
        self,
        p_layers,
        n_qubits,
        J_zz,
        h_z,
        energy_scale,
        apply_register_init,
        mixer_pairs,
        simulator,
        transpile_kwargs=None,
    ):
        transpile_kwargs = transpile_kwargs or {}
        init_statevector = statevector_from_register_init(apply_register_init, n_qubits)
        template, self.params = build_pure_xy_parametric_circuit(
            p_layers,
            n_qubits,
            J_zz,
            h_z,
            energy_scale,
            mixer_pairs,
            initial_statevector=init_statevector,
            with_measurements=True,
        )
        self.simulator = simulator
        self.energy_scale = energy_scale
        self.tqc = transpile(template, backend=simulator, **transpile_kwargs)

    def _bind(self, theta):
        binding = {param: float(theta[idx]) for idx, param in enumerate(self.params)}
        return self.tqc.assign_parameters(binding)

    def objective_energy(self, theta, shots, seed_sim, batches, energy_from_counts, counts_to_xorder):
        vals = []
        bound = self._bind(theta)
        for batch_idx in range(batches):
            result = self.simulator.run(
                bound,
                shots=shots,
                seed_simulator=seed_sim + batch_idx,
            ).result()
            counts_x = counts_to_xorder(result.get_counts())
            vals.append(energy_from_counts(counts_x) / self.energy_scale)
        return float(np.mean(vals))

    def run_final(self, theta, shots, seed_sim, counts_to_xorder):
        bound = self._bind(theta)
        result = self.simulator.run(
            bound,
            shots=shots,
            seed_simulator=seed_sim,
        ).result()
        return counts_to_xorder(result.get_counts())


def sample_x0(rng, p_layers):
    return np.concatenate(
        [
            rng.uniform(-np.pi, np.pi, size=p_layers),
            rng.uniform(0.0, np.pi / 2.0, size=p_layers),
        ]
    )


def optimize_spsa(
    objective,
    x0,
    maxiter,
    rng,
    a=0.602,
    c=0.101,
    alpha=0.602,
    gamma=0.101,
    A=10,
):
    x = np.asarray(x0, dtype=float).copy()
    best_x = x.copy()
    best_val = float(objective(x))
    for step in range(1, maxiter + 1):
        ak = a / ((step + A) ** alpha)
        ck = c / (step**gamma)
        delta = rng.choice([-1.0, 1.0], size=len(x))
        f_plus = float(objective(x + ck * delta))
        f_minus = float(objective(x - ck * delta))
        ghat = (f_plus - f_minus) / (2.0 * ck) * delta
        x = x - ak * ghat
        val = float(objective(x))
        if val < best_val:
            best_val = val
            best_x = x.copy()
    return SimpleNamespace(x=best_x, fun=best_val, success=True, nit=maxiter)


def optimize_lbfgsb(objective, x0, maxiter, tol):
    def grad(theta):
        theta = np.asarray(theta, dtype=float)
        grad_vec = np.zeros_like(theta)
        base = float(objective(theta))
        eps = 1e-4
        for idx in range(len(theta)):
            perturbed = theta.copy()
            perturbed[idx] += eps
            grad_vec[idx] = (float(objective(perturbed)) - base) / eps
        return grad_vec

    return minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=grad,
        options={"maxiter": maxiter, "ftol": tol},
    )


def optimize_cobyla(objective, x0, maxiter, rhobeg, tol):
    return minimize(
        objective,
        x0=x0,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": rhobeg, "tol": tol},
    )


def run_multi_restart_optimization(
    objective,
    p_layers,
    num_restarts,
    maxiter,
    optimizer,
    rng,
    rhobeg=0.5,
    tol=1e-4,
    allow_lbfgsb=True,
):
    if optimizer == OPTIMIZER_LBFGSB and not allow_lbfgsb:
        raise ValueError(
            "L-BFGS-B is only supported for statevector objectives in these scripts."
        )

    best_val = float("inf")
    best_x = None
    for _ in range(num_restarts):
        x0 = sample_x0(rng, p_layers)
        if optimizer == OPTIMIZER_SPSA:
            res = optimize_spsa(objective, x0, maxiter, rng)
        elif optimizer == OPTIMIZER_COBYLA:
            res = optimize_cobyla(objective, x0, maxiter, rhobeg, tol)
        elif optimizer == OPTIMIZER_LBFGSB:
            res = optimize_lbfgsb(objective, x0, maxiter, tol)
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. "
                f"Choose from {OPTIMIZER_SPSA}, {OPTIMIZER_COBYLA}, {OPTIMIZER_LBFGSB}."
            )
        if res.fun < best_val:
            best_val = float(res.fun)
            best_x = np.asarray(res.x, dtype=float).copy()
    return best_val, best_x


def _mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _mean_ci95_t(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mean_val, std_val = _mean_std(arr)
    if n <= 1:
        return mean_val, std_val, (mean_val, mean_val)
    t_val = 2.045 if n == 30 else 2.0
    half = t_val * std_val / sqrt(n)
    return mean_val, std_val, (mean_val - half, mean_val + half)


def _wilson_ci95(k, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)


def summarize_experiment(run_records, n_runs):
    p_star_list = [row["p_star"] for row in run_records]
    gap_exp_list = [row["gap_exp"] for row in run_records]
    run_success_list = [row["success"] for row in run_records]
    tts99_list = [row["tts99"] for row in run_records]
    rank_list = [row["rank"] for row in run_records]

    p_mean, p_std, p_ci = _mean_ci95_t(p_star_list)
    gap_mean, gap_std, gap_ci = _mean_ci95_t(gap_exp_list)
    rank_mean, rank_std, rank_ci = _mean_ci95_t(rank_list)
    succ_k = int(sum(run_success_list))
    succ_ci = _wilson_ci95(succ_k, n_runs)
    tts99_median = float(np.median(np.asarray(tts99_list, dtype=float)))

    return {
        "p_star_mean": p_mean,
        "p_star_std": p_std,
        "p_star_ci_low": p_ci[0],
        "p_star_ci_high": p_ci[1],
        "success_count": succ_k,
        "success_rate": succ_k / n_runs if n_runs else 0.0,
        "success_wilson_ci_low": succ_ci[0],
        "success_wilson_ci_high": succ_ci[1],
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_ci_low": gap_ci[0],
        "gap_ci_high": gap_ci[1],
        "tts99_median": tts99_median,
        "rank_mean": rank_mean,
        "rank_std": rank_std,
        "rank_ci_low": rank_ci[0],
        "rank_ci_high": rank_ci[1],
    }


def print_experiment_summary(title, stats, optimizer, n_runs, maxiter):
    print(f"\n[Summary | {title} | optimizer={optimizer} | maxiter={maxiter}]")
    print(
        "p*: "
        f"mean={stats['p_star_mean']:.6f}, "
        f"std={stats['p_star_std']:.6f}, "
        f"95% CI=[{stats['p_star_ci_low']:.6f}, {stats['p_star_ci_high']:.6f}]"
    )
    print(
        "success: "
        f"{stats['success_count']}/{n_runs}, "
        f"Wilson 95% CI=[{stats['success_wilson_ci_low']:.6f}, "
        f"{stats['success_wilson_ci_high']:.6f}]"
    )
    print(
        "Expected energy gap (E[C] - C*_feas): "
        f"mean={stats['gap_mean']:.6f}, "
        f"std={stats['gap_std']:.6f}, "
        f"95% CI=[{stats['gap_ci_low']:.6f}, {stats['gap_ci_high']:.6f}]"
    )
    print(
        "TTS99 "
        f"median={stats['tts99_median']:.6f}; "
        "sampling-rank "
        f"mean={stats['rank_mean']:.6f}, "
        f"std={stats['rank_std']:.6f}, "
        f"95% CI=[{stats['rank_ci_low']:.6f}, {stats['rank_ci_high']:.6f}]"
    )


def select_best_config_by_gap_mean(config_rows):
    """Select the configuration with the smallest expected QUBO energy gap mean."""
    if not config_rows:
        raise ValueError("No sweep results available for selection.")
    return min(
        config_rows,
        key=lambda row: (
            float(row["stats"]["gap_mean"]),
            -float(row["stats"]["p_star_mean"]),
            int(row["maxiter"]),
        ),
    )


def build_experiment_stem(scale_tag, method_tag, case_name):
    """Unique CSV stem: scale + scenario + case."""
    return f"{scale_tag}__{method_tag}__{case_name}"


def _csv_stem(case_name, *, scale_tag="", method_tag=""):
    if scale_tag and method_tag:
        return build_experiment_stem(scale_tag, method_tag, case_name)
    return case_name


def _lambda_tag(lam):
    if lam is None:
        return None
    return f"{lam:g}".replace(".", "p")


def _wrap_csv_path_builder(fn, scale_tag, method_tag):
    if not (scale_tag and method_tag):
        return fn

    def wrapped(case_name, lam, maxiter, output_dir):
        return fn(
            case_name,
            lam,
            maxiter,
            output_dir,
            scale_tag=scale_tag,
            method_tag=method_tag,
        )

    return wrapped


def build_results_csv_path(
    case_name,
    lam,
    maxiter,
    output_dir=DEFAULT_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _csv_stem(case_name, scale_tag=scale_tag, method_tag=method_tag)
    lam_tag = _lambda_tag(lam)
    if lam_tag is None:
        return out_dir / f"{stem}_maxiter{maxiter}.csv"
    return out_dir / f"{stem}_lambda{lam_tag}_maxiter{maxiter}.csv"


def build_selected_per_lambda_csv_path(
    case_name,
    lam,
    maxiter,
    output_dir=DEFAULT_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _csv_stem(case_name, scale_tag=scale_tag, method_tag=method_tag)
    lam_tag = _lambda_tag(lam)
    return out_dir / f"{stem}_SELECTED_per_lambda{lam_tag}_maxiter{maxiter}.csv"


def build_selected_overall_csv_path(
    case_name,
    lam,
    maxiter,
    output_dir=DEFAULT_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _csv_stem(case_name, scale_tag=scale_tag, method_tag=method_tag)
    lam_tag = _lambda_tag(lam)
    return (
        out_dir
        / f"{stem}_SELECTED_overall_best_lambda{lam_tag}_maxiter{maxiter}.csv"
    )


def build_maxiter_selected_csv_path(
    case_name,
    maxiter,
    output_dir=DEFAULT_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _csv_stem(case_name, scale_tag=scale_tag, method_tag=method_tag)
    return out_dir / f"{stem}_SELECTED_maxiter{maxiter}.csv"


def build_selection_index_csv_path(
    case_name,
    output_dir=DEFAULT_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _csv_stem(case_name, scale_tag=scale_tag, method_tag=method_tag)
    return out_dir / f"{stem}_SELECTED_lambda_maxiter_index.csv"


def _wrap_index_path_builder(fn, scale_tag, method_tag):
    if not (scale_tag and method_tag):
        return fn

    def wrapped(case_name, output_dir):
        return fn(
            case_name,
            output_dir,
            scale_tag=scale_tag,
            method_tag=method_tag,
        )

    return wrapped


def _selection_index_fieldnames():
    return [
        "selection_kind",
        "scale_tag",
        "method_tag",
        "case_name",
        "lambda",
        "maxiter",
        "optimizer",
        "n_runs",
        "p_star_mean",
        "p_star_std",
        "p_star_ci_low",
        "p_star_ci_high",
        "success_count",
        "success_rate",
        "gap_mean",
        "gap_std",
        "gap_ci_low",
        "gap_ci_high",
        "tts99_median",
        "rank_mean",
        "p_feas_mean",
        "saved_csv_path",
        "source_sweep_csv_path",
    ]


def _selection_index_row(
    kind,
    case_name,
    row,
    *,
    optimizer,
    n_runs,
    saved_csv_path,
    source_sweep_csv_path="",
    scale_tag="",
    method_tag="",
):
    stats = row["stats"]
    record = {
        "selection_kind": kind,
        "scale_tag": scale_tag,
        "method_tag": method_tag,
        "case_name": case_name,
        "lambda": row["lambda"],
        "maxiter": row["maxiter"],
        "optimizer": optimizer,
        "n_runs": n_runs,
        "p_star_mean": stats["p_star_mean"],
        "p_star_std": stats["p_star_std"],
        "p_star_ci_low": stats["p_star_ci_low"],
        "p_star_ci_high": stats["p_star_ci_high"],
        "success_count": stats["success_count"],
        "success_rate": stats["success_rate"],
        "gap_mean": stats["gap_mean"],
        "gap_std": stats["gap_std"],
        "gap_ci_low": stats["gap_ci_low"],
        "gap_ci_high": stats["gap_ci_high"],
        "tts99_median": stats["tts99_median"],
        "rank_mean": stats["rank_mean"],
        "p_feas_mean": stats.get("p_feas_mean", ""),
        "saved_csv_path": str(saved_csv_path),
        "source_sweep_csv_path": str(source_sweep_csv_path) if source_sweep_csv_path else "",
    }
    return record


def _persist_selected_config_csv(
    row,
    dest_path,
    *,
    case_name,
    optimizer,
    output_dir,
    build_sweep_csv_path,
    save_experiment_csv_fn,
    experiment_meta,
):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    run_records = row.get("run_records")
    if run_records and save_experiment_csv_fn and experiment_meta:
        save_kwargs = {}
        for key in ("method_tag", "scale_tag"):
            if key in experiment_meta:
                save_kwargs[key] = experiment_meta[key]
        save_experiment_csv_fn(
            dest_path,
            case_name,
            row["lambda"],
            row["maxiter"],
            optimizer,
            experiment_meta["p"],
            experiment_meta["num_restarts"],
            experiment_meta["n_runs"],
            row["stats"],
            run_records,
            **save_kwargs,
        )
        return dest_path, ""

    source_path = row.get("source_csv_path")
    if not source_path and build_sweep_csv_path is not None:
        source_path = build_sweep_csv_path(
            case_name, row["lambda"], row["maxiter"], output_dir
        )
    source_path = Path(source_path) if source_path else None
    if source_path and source_path.exists():
        shutil.copy2(source_path, dest_path)
        return dest_path, source_path

    print(
        f"[Warning] No sweep CSV or run records available for lambda={row['lambda']}, "
        f"maxiter={row['maxiter']}; skipped {dest_path.name}."
    )
    return None, ""


def save_hyperparam_selection_csvs(
    case_name,
    lambda_list,
    all_rows,
    *,
    optimizer,
    n_runs,
    output_dir=DEFAULT_RESULTS_DIR,
    scale_tag="",
    method_tag="",
    build_sweep_csv_path=None,
    build_selected_per_lambda_csv_path_fn=None,
    build_selected_overall_csv_path_fn=None,
    build_selection_index_csv_path_fn=None,
    save_experiment_csv_fn=None,
    experiment_meta=None,
):
    """Save per-lambda best-maxiter CSVs, overall-best CSV, and an index summary."""
    if build_sweep_csv_path is None:
        build_sweep_csv_path = build_results_csv_path
    if build_selected_per_lambda_csv_path_fn is None:
        build_selected_per_lambda_csv_path_fn = build_selected_per_lambda_csv_path
    if build_selected_overall_csv_path_fn is None:
        build_selected_overall_csv_path_fn = build_selected_overall_csv_path
    if build_selection_index_csv_path_fn is None:
        build_selection_index_csv_path_fn = build_selection_index_csv_path

    build_sweep_csv_path = _wrap_csv_path_builder(
        build_sweep_csv_path, scale_tag, method_tag
    )
    build_selected_per_lambda_csv_path_fn = _wrap_csv_path_builder(
        build_selected_per_lambda_csv_path_fn, scale_tag, method_tag
    )
    build_selected_overall_csv_path_fn = _wrap_csv_path_builder(
        build_selected_overall_csv_path_fn, scale_tag, method_tag
    )
    build_selection_index_csv_path_fn = _wrap_index_path_builder(
        build_selection_index_csv_path_fn, scale_tag, method_tag
    )

    by_lambda = {}
    for row in all_rows:
        by_lambda.setdefault(row["lambda"], []).append(row)

    index_rows = []
    saved_paths = []

    for lam in lambda_list:
        lam_rows = by_lambda.get(lam, [])
        if not lam_rows:
            continue
        best = select_best_config_by_gap_mean(lam_rows)
        dest = build_selected_per_lambda_csv_path_fn(
            case_name, lam, best["maxiter"], output_dir
        )
        saved, source = _persist_selected_config_csv(
            best,
            dest,
            case_name=case_name,
            optimizer=optimizer,
            output_dir=output_dir,
            build_sweep_csv_path=build_sweep_csv_path,
            save_experiment_csv_fn=save_experiment_csv_fn,
            experiment_meta=experiment_meta,
        )
        if saved is None:
            continue
        saved_paths.append(saved)
        index_rows.append(
            _selection_index_row(
                "per_lambda_best",
                case_name,
                best,
                optimizer=optimizer,
                n_runs=n_runs,
                saved_csv_path=saved,
                source_sweep_csv_path=source,
                scale_tag=scale_tag,
                method_tag=method_tag,
            )
        )
        print(f"[Saved selection | per-lambda best] {saved}")

    overall = select_best_config_by_gap_mean(all_rows)
    overall_dest = build_selected_overall_csv_path_fn(
        case_name, overall["lambda"], overall["maxiter"], output_dir
    )
    overall_saved, overall_source = _persist_selected_config_csv(
        overall,
        overall_dest,
        case_name=case_name,
        optimizer=optimizer,
        output_dir=output_dir,
        build_sweep_csv_path=build_sweep_csv_path,
        save_experiment_csv_fn=save_experiment_csv_fn,
        experiment_meta=experiment_meta,
    )
    if overall_saved is not None:
        saved_paths.append(overall_saved)
        index_rows.append(
            _selection_index_row(
                "overall_best",
                case_name,
                overall,
                optimizer=optimizer,
                n_runs=n_runs,
                saved_csv_path=overall_saved,
                source_sweep_csv_path=overall_source,
                scale_tag=scale_tag,
                method_tag=method_tag,
            )
        )
        print(f"[Saved selection | overall best] {overall_saved}")
    else:
        print(
            f"[Warning] Skipped overall-best selection CSV for lambda={overall['lambda']}, "
            f"maxiter={overall['maxiter']}."
        )

    if not index_rows:
        print("[Warning] No selection CSVs were saved; skipping index file.")
        return {
            "per_lambda_best": [],
            "overall_best": overall,
            "index_path": None,
            "saved_paths": saved_paths,
        }

    index_path = build_selection_index_csv_path_fn(case_name, output_dir)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_selection_index_fieldnames())
        writer.writeheader()
        writer.writerows(index_rows)
    saved_paths.append(index_path)
    print(f"[Saved selection | index] {index_path}")

    return {
        "per_lambda_best": [r for r in index_rows if r["selection_kind"] == "per_lambda_best"],
        "overall_best": overall,
        "index_path": index_path,
        "saved_paths": saved_paths,
    }


def print_hyperparam_sweep_selection(
    case_name,
    lambda_list,
    all_rows,
    *,
    summary_title,
    optimizer,
    n_runs,
    summary_printer=None,
    save_selection=True,
    output_dir=DEFAULT_RESULTS_DIR,
    scale_tag="",
    method_tag="",
    build_sweep_csv_path=None,
    build_selected_per_lambda_csv_path_fn=None,
    build_selected_overall_csv_path_fn=None,
    build_selection_index_csv_path_fn=None,
    save_experiment_csv_fn=None,
    experiment_meta=None,
):
    if summary_printer is None:
        summary_printer = print_experiment_summary

    print(f"\n{'=' * 72}")
    print(
        f"[Hyperparameter selection | case={case_name} | "
        "primary metric=gap_mean (lower is better)]"
    )
    print(f"{'=' * 72}")
    by_lambda = {}
    for row in all_rows:
        by_lambda.setdefault(row["lambda"], []).append(row)

    selected_rows = []
    for lam in lambda_list:
        lam_rows = by_lambda.get(lam, [])
        if not lam_rows:
            print(f"lambda={lam:g}: no results")
            continue
        best = select_best_config_by_gap_mean(lam_rows)
        selected_rows.append(("per_lambda", lam, best))
        stats = best["stats"]
        print(
            f"lambda={lam:g}: best maxiter={best['maxiter']} "
            f"gap_mean={stats['gap_mean']:.6f} "
            f"p*_mean={stats['p_star_mean']:.6f} "
            f"success={stats['success_count']}"
        )

    overall = select_best_config_by_gap_mean(all_rows)
    selected_rows.append(("overall", overall["lambda"], overall))
    stats = overall["stats"]
    print(
        f"\nOverall best configuration: lambda={overall['lambda']:g}, "
        f"maxiter={overall['maxiter']} "
        f"gap_mean={stats['gap_mean']:.6f} "
        f"p*_mean={stats['p_star_mean']:.6f} "
        f"success={stats['success_count']}"
    )
    if "p_feas_mean" in stats:
        print(f"overall p_feas_mean={stats['p_feas_mean']:.6f}")

    print(f"\n{'=' * 72}")
    print(f"[Selected configurations | full statistics | case={case_name}]")
    print(f"{'=' * 72}")
    for kind, lam, row in selected_rows:
        if kind == "per_lambda":
            title = (
                f"{summary_title} | selected best | lambda={lam:g} | "
                f"maxiter={row['maxiter']}"
            )
        else:
            title = (
                f"{summary_title} | overall selected best | "
                f"lambda={row['lambda']:g} | maxiter={row['maxiter']}"
            )
        summary_printer(title, row["stats"], optimizer, n_runs, row["maxiter"])

    if save_selection:
        save_hyperparam_selection_csvs(
            case_name,
            lambda_list,
            all_rows,
            optimizer=optimizer,
            n_runs=n_runs,
            output_dir=output_dir,
            scale_tag=scale_tag,
            method_tag=method_tag,
            build_sweep_csv_path=build_sweep_csv_path,
            build_selected_per_lambda_csv_path_fn=build_selected_per_lambda_csv_path_fn,
            build_selected_overall_csv_path_fn=build_selected_overall_csv_path_fn,
            build_selection_index_csv_path_fn=build_selection_index_csv_path_fn,
            save_experiment_csv_fn=save_experiment_csv_fn,
            experiment_meta=experiment_meta,
        )


def summarize_experiment_with_pfeas(run_records, n_runs):
    stats = summarize_experiment(run_records, n_runs)
    pf_list = [row["p_feas"] for row in run_records]
    pf_mean, pf_std, pf_ci = _mean_ci95_t(pf_list)
    stats["p_feas_mean"] = pf_mean
    stats["p_feas_std"] = pf_std
    stats["p_feas_ci_low"] = pf_ci[0]
    stats["p_feas_ci_high"] = pf_ci[1]
    return stats


def print_experiment_summary_with_pfeas(title, stats, optimizer, n_runs, maxiter):
    print_experiment_summary(title, stats, optimizer, n_runs, maxiter)
    print(
        "p_feas: "
        f"mean={stats['p_feas_mean']:.6f}, "
        f"std={stats['p_feas_std']:.6f}, "
        f"95% CI=[{stats['p_feas_ci_low']:.6f}, {stats['p_feas_ci_high']:.6f}]"
    )


def print_maxiter_sweep_selection(
    case_name,
    all_rows,
    *,
    summary_title,
    optimizer,
    n_runs,
    summary_printer=None,
    lam=None,
    save_selection=True,
    output_dir=DEFAULT_XYQAOA_RESULTS_DIR,
    scale_tag="",
    method_tag="",
    experiment_meta=None,
    build_sweep_csv_path=None,
    save_experiment_csv_fn=None,
):
    """Select best maxiter by gap_mean and print full summary for the winner."""
    if summary_printer is None:
        summary_printer = print_experiment_summary

    print(f"\n{'=' * 72}")
    print(
        f"[Maxiter selection | case={case_name} | "
        "primary metric=gap_mean (lower is better)]"
    )
    print(f"{'=' * 72}")

    best = select_best_config_by_gap_mean(all_rows)
    stats = best["stats"]
    lam_prefix = f"lambda={lam:g} | " if lam is not None else ""
    print(
        f"Best maxiter={best['maxiter']} | {lam_prefix}"
        f"gap_mean={stats['gap_mean']:.6f} "
        f"p*_mean={stats['p_star_mean']:.6f} "
        f"success={stats['success_count']}"
    )
    if "p_feas_mean" in stats:
        print(f"p_feas_mean={stats['p_feas_mean']:.6f}")

    print(f"\n{'=' * 72}")
    print(f"[Selected configuration | full statistics | case={case_name}]")
    print(f"{'=' * 72}")
    title = (
        f"{summary_title} | selected best | {lam_prefix}maxiter={best['maxiter']}"
    )
    summary_printer(title, stats, optimizer, n_runs, best["maxiter"])

    if save_selection and scale_tag and method_tag and experiment_meta:
        if build_sweep_csv_path is None:
            build_sweep_csv_path = _wrap_csv_path_builder(
                build_results_csv_path, scale_tag, method_tag
            )
        if save_experiment_csv_fn is None:
            save_experiment_csv_fn = save_experiment_csv_with_pfeas
        dest = build_maxiter_selected_csv_path(
            case_name,
            best["maxiter"],
            output_dir,
            scale_tag=scale_tag,
            method_tag=method_tag,
        )
        saved, _source = _persist_selected_config_csv(
            best,
            dest,
            case_name=case_name,
            optimizer=optimizer,
            output_dir=output_dir,
            build_sweep_csv_path=build_sweep_csv_path,
            save_experiment_csv_fn=save_experiment_csv_fn,
            experiment_meta=experiment_meta,
        )
        if saved is not None:
            print(f"[Saved selection | best maxiter] {saved}")
        else:
            print(
                f"[Warning] Skipped maxiter selection CSV for maxiter={best['maxiter']}."
            )


def build_xyqaoa_csv_path(
    case_name,
    maxiter,
    output_dir=DEFAULT_XYQAOA_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    return build_results_csv_path(
        case_name,
        None,
        maxiter,
        output_dir,
        scale_tag=scale_tag,
        method_tag=method_tag,
    )


def build_xyqaoa_selected_csv_path(
    case_name,
    maxiter,
    output_dir=DEFAULT_XYQAOA_RESULTS_DIR,
    *,
    scale_tag="",
    method_tag="",
):
    return build_maxiter_selected_csv_path(
        case_name,
        maxiter,
        output_dir,
        scale_tag=scale_tag,
        method_tag=method_tag,
    )


def save_experiment_csv_with_pfeas(
    csv_path,
    case_name,
    lam,
    maxiter,
    optimizer,
    p_layers,
    num_restarts,
    n_runs,
    stats,
    run_records,
    *,
    scale_tag="",
    method_tag="",
):
    fieldnames = [
        "row_type",
        "run_id",
        "scale_tag",
        "method_tag",
        "case_name",
        "lambda",
        "maxiter",
        "optimizer",
        "p",
        "num_restarts",
        "n_runs",
        "best_val",
        "p_star",
        "p_feas",
        "gap_exp",
        "success",
        "tts99",
        "rank",
        "p_star_mean",
        "p_star_std",
        "p_star_ci_low",
        "p_star_ci_high",
        "p_feas_mean",
        "p_feas_std",
        "p_feas_ci_low",
        "p_feas_ci_high",
        "success_count",
        "success_rate",
        "success_wilson_ci_low",
        "success_wilson_ci_high",
        "gap_mean",
        "gap_std",
        "gap_ci_low",
        "gap_ci_high",
        "tts99_median",
        "rank_mean",
        "rank_std",
        "rank_ci_low",
        "rank_ci_high",
    ]
    lam_value = "" if lam is None else lam
    meta = {
        "scale_tag": scale_tag,
        "method_tag": method_tag,
        "case_name": case_name,
        "lambda": lam_value,
        "maxiter": maxiter,
        "optimizer": optimizer,
        "p": p_layers,
        "num_restarts": num_restarts,
        "n_runs": n_runs,
    }
    summary_fields = {
        "p_star_mean": stats["p_star_mean"],
        "p_star_std": stats["p_star_std"],
        "p_star_ci_low": stats["p_star_ci_low"],
        "p_star_ci_high": stats["p_star_ci_high"],
        "p_feas_mean": stats.get("p_feas_mean", ""),
        "p_feas_std": stats.get("p_feas_std", ""),
        "p_feas_ci_low": stats.get("p_feas_ci_low", ""),
        "p_feas_ci_high": stats.get("p_feas_ci_high", ""),
        "success_count": stats["success_count"],
        "success_rate": stats["success_rate"],
        "success_wilson_ci_low": stats["success_wilson_ci_low"],
        "success_wilson_ci_high": stats["success_wilson_ci_high"],
        "gap_mean": stats["gap_mean"],
        "gap_std": stats["gap_std"],
        "gap_ci_low": stats["gap_ci_low"],
        "gap_ci_high": stats["gap_ci_high"],
        "tts99_median": stats["tts99_median"],
        "rank_mean": stats["rank_mean"],
        "rank_std": stats["rank_std"],
        "rank_ci_low": stats["rank_ci_low"],
        "rank_ci_high": stats["rank_ci_high"],
    }
    empty_summary = {k: "" for k in summary_fields}

    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_records:
            writer.writerow(
                {
                    "row_type": "run",
                    "run_id": row["run_id"],
                    **meta,
                    "best_val": row["best_val"],
                    "p_star": row["p_star"],
                    "p_feas": row.get("p_feas", ""),
                    "gap_exp": row["gap_exp"],
                    "success": row["success"],
                    "tts99": row["tts99"],
                    "rank": row["rank"],
                    **empty_summary,
                }
            )
        writer.writerow(
            {
                "row_type": "summary",
                "run_id": "",
                **meta,
                **{
                    k: ""
                    for k in (
                        "best_val",
                        "p_star",
                        "p_feas",
                        "gap_exp",
                        "success",
                        "tts99",
                        "rank",
                    )
                },
                **summary_fields,
            }
        )


def save_experiment_csv(
    csv_path,
    case_name,
    lam,
    maxiter,
    optimizer,
    p_layers,
    num_restarts,
    n_runs,
    stats,
    run_records,
    *,
    scale_tag="",
    method_tag="",
):
    fieldnames = [
        "row_type",
        "run_id",
        "scale_tag",
        "method_tag",
        "case_name",
        "lambda",
        "maxiter",
        "optimizer",
        "p",
        "num_restarts",
        "n_runs",
        "best_val",
        "p_star",
        "gap_exp",
        "success",
        "tts99",
        "rank",
        "p_star_mean",
        "p_star_std",
        "p_star_ci_low",
        "p_star_ci_high",
        "success_count",
        "success_rate",
        "success_wilson_ci_low",
        "success_wilson_ci_high",
        "gap_mean",
        "gap_std",
        "gap_ci_low",
        "gap_ci_high",
        "tts99_median",
        "rank_mean",
        "rank_std",
        "rank_ci_low",
        "rank_ci_high",
    ]
    lam_value = "" if lam is None else lam
    meta = {
        "scale_tag": scale_tag,
        "method_tag": method_tag,
        "case_name": case_name,
        "lambda": lam_value,
        "maxiter": maxiter,
        "optimizer": optimizer,
        "p": p_layers,
        "num_restarts": num_restarts,
        "n_runs": n_runs,
    }
    summary_fields = {
        "p_star_mean": stats["p_star_mean"],
        "p_star_std": stats["p_star_std"],
        "p_star_ci_low": stats["p_star_ci_low"],
        "p_star_ci_high": stats["p_star_ci_high"],
        "success_count": stats["success_count"],
        "success_rate": stats["success_rate"],
        "success_wilson_ci_low": stats["success_wilson_ci_low"],
        "success_wilson_ci_high": stats["success_wilson_ci_high"],
        "gap_mean": stats["gap_mean"],
        "gap_std": stats["gap_std"],
        "gap_ci_low": stats["gap_ci_low"],
        "gap_ci_high": stats["gap_ci_high"],
        "tts99_median": stats["tts99_median"],
        "rank_mean": stats["rank_mean"],
        "rank_std": stats["rank_std"],
        "rank_ci_low": stats["rank_ci_low"],
        "rank_ci_high": stats["rank_ci_high"],
    }

    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_records:
            writer.writerow(
                {
                    "row_type": "run",
                    "run_id": row["run_id"],
                    **meta,
                    "best_val": row["best_val"],
                    "p_star": row["p_star"],
                    "gap_exp": row["gap_exp"],
                    "success": row["success"],
                    "tts99": row["tts99"],
                    "rank": row["rank"],
                }
            )
        writer.writerow({"row_type": "summary", **meta, **summary_fields})
