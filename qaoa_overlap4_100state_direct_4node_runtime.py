"""Runtime helpers for overlap-4 100-state direct initialization (4-node proposed QAOA)."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit_aer.library import SetStatevector

from qaoa_new_4node_runtime import apply_cost_layer, build_cost_sparse_pauli_op

EXPECTED_INIT_SUPPORT = 100
EXPECTED_SECTOR_HISTOGRAM = {(0, 0): 64, (1, 0): 16, (0, 1): 16, (1, 1): 4}
STATE_PREPARATION_NOISE_MODELED = False

XY_PAIRS = [(0, 10), (1, 11), (3, 5), (6, 8)]
X_MIXER_QUBITS = [2, 9]
FROZEN_OVERLAP_QUBITS = [4, 7]
SELECTED_CONSTRAINT_QUBITS = {0, 1, 3, 4, 5, 6, 7, 8, 10, 11}


def satisfies_overlap4_init_constraints(bits):
    if len(bits) != 12:
        return False
    (
        x01,
        x02,
        x03,
        x10,
        x12,
        x13,
        x20,
        x21,
        x23,
        x30,
        x31,
        x32,
    ) = bits
    return all(
        [
            x01 + x21 + x31 == 1,  # I1: exactly one incoming arc to customer 1
            x02 + x12 + x32 == 1,  # I2: exactly one incoming arc to customer 2
            x10 + x12 + x13 == 1,  # O1: exactly one outgoing arc from customer 1
            x20 + x21 + x23 == 1,  # O2: exactly one outgoing arc from customer 2
        ]
    )


def enumerate_overlap4_init_bits(n_qubits=12):
    init_bits = tuple(
        bits
        for bits in itertools.product([0, 1], repeat=n_qubits)
        if satisfies_overlap4_init_constraints(bits)
    )
    if len(init_bits) != EXPECTED_INIT_SUPPORT:
        raise RuntimeError(
            f"Unexpected initialization support: {len(init_bits)}; "
            f"expected {EXPECTED_INIT_SUPPORT}."
        )
    return init_bits


def validate_overlap4_init_states(init_bits):
    sector_histogram = {}
    for bits in init_bits:
        if not satisfies_overlap4_init_constraints(bits):
            raise RuntimeError(
                f"Initialization state violates selected constraints: {bits}"
            )
        sector = (int(bits[4]), int(bits[7]))
        sector_histogram[sector] = sector_histogram.get(sector, 0) + 1
    if sector_histogram != EXPECTED_SECTOR_HISTOGRAM:
        raise RuntimeError(
            f"Unexpected q4/q7 sector histogram: {sector_histogram}; "
            f"expected {EXPECTED_SECTOR_HISTOGRAM}."
        )
    return sector_histogram


def swap_pair_if_exchangeable(bits, i, j):
    bits = list(bits)
    if bits[i] == bits[j]:
        return None
    bits[i], bits[j] = bits[j], bits[i]
    return tuple(bits)


def validate_xy_pairs_preserve_constraints(init_bits, xy_pairs=XY_PAIRS):
    for bits in init_bits:
        for i, j in xy_pairs:
            swapped = swap_pair_if_exchangeable(bits, i, j)
            if swapped is None:
                continue
            if not satisfies_overlap4_init_constraints(swapped):
                raise RuntimeError(
                    f"XY pair {(i, j)} breaks selected constraints for state {bits}."
                )


def validate_mixer_configuration(xy_pairs=XY_PAIRS, x_mixer_qubits=X_MIXER_QUBITS):
    if any(q in SELECTED_CONSTRAINT_QUBITS for q in x_mixer_qubits):
        raise RuntimeError(
            "An X-mixer qubit appears in a selected initialization constraint."
        )
    if any(q in FROZEN_OVERLAP_QUBITS for q in x_mixer_qubits):
        raise RuntimeError("An X-mixer qubit is listed as a frozen overlap qubit.")
    frozen = set(FROZEN_OVERLAP_QUBITS)
    for i, j in xy_pairs:
        if i in frozen or j in frozen:
            raise RuntimeError(
                f"Frozen overlap qubit appears in XY pair {(i, j)}."
            )


def build_overlap4_initial_statevector(init_bits, n_qubits):
    vector = np.zeros(2**n_qubits, dtype=np.complex128)
    amplitude = 1.0 / np.sqrt(len(init_bits))
    for bits in init_bits:
        basis_index = sum(int(bits[q]) * (2**q) for q in range(n_qubits))
        vector[basis_index] = amplitude
    norm = np.linalg.norm(vector)
    if not np.isclose(norm, 1.0, atol=1e-12):
        raise RuntimeError(f"Initial statevector is not normalized: norm={norm}")
    observed_support = int(np.count_nonzero(np.abs(vector) > 1e-14))
    if observed_support != EXPECTED_INIT_SUPPORT:
        raise RuntimeError(
            f"Observed support={observed_support}; expected={EXPECTED_INIT_SUPPORT}."
        )
    return vector


def apply_overlap4_direct_initial_state(qc, initial_statevector):
    qc.append(SetStatevector(initial_statevector), list(range(qc.num_qubits)))


def apply_overlap4_hybrid_mixer(qc, beta, xy_pairs, x_mixer_qubits, lam):
    for i, j in xy_pairs:
        qc.rxx(2.0 * beta, i, j)
        qc.ryy(2.0 * beta, i, j)
    for q in x_mixer_qubits:
        qc.rx(2.0 * beta * lam, q)


def build_parametric_qaoa_circuit_overlap4_direct(
    p_layers,
    initial_statevector,
    n_qubits,
    xy_pairs,
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
    apply_overlap4_direct_initial_state(qc, initial_statevector)
    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        apply_overlap4_hybrid_mixer(
            qc, betas[layer], xy_pairs, x_mixer_qubits, lam
        )
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


def setup_overlap4_initialization(
    n_qubits,
    check_constraints_fn,
    qubo_cost_from_bits_fn,
    opt_states_feas,
):
    init_bits = enumerate_overlap4_init_bits(n_qubits)
    sector_hist = validate_overlap4_init_states(init_bits)
    validate_mixer_configuration()
    validate_xy_pairs_preserve_constraints(init_bits)

    fully_feasible = tuple(bits for bits in init_bits if check_constraints_fn(bits))
    if len(fully_feasible) != 6:
        raise RuntimeError(
            "Expected exactly 6 fully feasible states inside the "
            f"100-state initialization support, but found {len(fully_feasible)}."
        )

    init_bits_set = set(init_bits)
    for optimal_bits in opt_states_feas:
        if tuple(optimal_bits) not in init_bits_set:
            raise RuntimeError(
                "A theoretical optimal feasible state was excluded by the "
                f"four-constraint initialization: {optimal_bits}"
            )

    initial_statevector = build_overlap4_initial_statevector(init_bits, n_qubits)
    initial_expected_qubo = float(
        np.mean([qubo_cost_from_bits_fn(bits) for bits in init_bits])
    )
    return SimpleNamespace(
        init_bits=init_bits,
        fully_feasible_init_bits=fully_feasible,
        initial_statevector=initial_statevector,
        sector_histogram=sector_hist,
        initial_feasible_probability=len(fully_feasible) / EXPECTED_INIT_SUPPORT,
        initial_expected_qubo=initial_expected_qubo,
    )


def print_overlap4_init_and_mixer_diagnostics(init_ctx, opt_states_feas, lam):
    opt_in_support = sum(
        1 for bits in init_ctx.init_bits if bits in set(map(tuple, opt_states_feas))
    )
    initial_optimal_probability = opt_in_support / EXPECTED_INIT_SUPPORT
    print("\n[Overlap-4 initialization]")
    print("selected constraints = I1, I2, O1, O2")
    print("support size =", EXPECTED_INIT_SUPPORT)
    print("fully feasible states in support =", len(init_ctx.fully_feasible_init_bits))
    print(
        "initial feasible probability =",
        init_ctx.initial_feasible_probability,
    )
    print("q4/q7 sector histogram =", init_ctx.sector_histogram)
    print("initial expected QUBO cost =", init_ctx.initial_expected_qubo)
    print("initial optimal probability =", initial_optimal_probability)
    print("\n[Mixer]")
    print("XY pairs =", XY_PAIRS)
    print("X-mixer qubits =", X_MIXER_QUBITS)
    print("frozen overlap qubits =", FROZEN_OVERLAP_QUBITS)
    print("lambda =", lam)
    print("selected constraints preserved exactly = True")
    print("State-preparation noise modeled =", STATE_PREPARATION_NOISE_MODELED)
    print(
        "The simulator is initialized directly in the prescribed "
        "100-state logical superposition. Gate and readout noise are "
        "applied only to the subsequent QAOA evolution and measurement. "
        "Physical state-preparation errors are not modeled."
    )


def verify_optimal_qubo_raw_match(opt_states, qubo_cost_from_bits_fn, raw_vrp_cost_from_bits_fn):
    for bits in opt_states:
        q = qubo_cost_from_bits_fn(bits)
        r = raw_vrp_cost_from_bits_fn(bits)
        if not np.isclose(q, r, atol=1e-8):
            raise RuntimeError(f"Raw/QUBO optimum mismatch: {bits}, qubo={q}, raw={r}")


def overlap4_ablation_uses_set_statevector(ablation_arm) -> bool:
    """Return whether the overlap-4 circuit prep uses SetStatevector."""
    from qaoa_new_4node_runtime import ABLATION_MIXER_ONLY

    return ablation_arm != ABLATION_MIXER_ONLY


def audit_transpiled_ops(
    tqc,
    label="transpiled",
    *,
    template=None,
    require_set_statevector=None,
):
    ops = dict(tqc.count_ops())
    print(f"\n[Gate audit | {label}]")
    print("operation counts:", ops)
    if require_set_statevector is None:
        if template is not None:
            require_set_statevector = "set_statevector" in template.count_ops()
        else:
            require_set_statevector = True
    if require_set_statevector and "set_statevector" not in ops:
        raise RuntimeError(
            "Transpilation removed set_statevector instruction "
            f"(require_set_statevector={require_set_statevector}, "
            f"template had set_statevector="
            f"{bool(template and 'set_statevector' in template.count_ops())})."
        )
    return ops


class CachedOverlap4DirectShotRunner:
    """Build/transpile parametric overlap-4 direct circuit once per simulator."""

    def __init__(
        self,
        p_layers,
        initial_statevector,
        n_qubits,
        xy_pairs,
        x_mixer_qubits,
        lam,
        J_zz,
        h_z,
        energy_scale,
        simulator,
        seed_transpiler=999,
        optimization_level=1,
        basis_gates=None,
        *,
        ablation_arm=None,
    ):
        if ablation_arm is None:
            template, self.params = build_parametric_qaoa_circuit_overlap4_direct(
                p_layers,
                initial_statevector,
                n_qubits,
                xy_pairs,
                x_mixer_qubits,
                lam,
                J_zz,
                h_z,
                energy_scale,
                with_measurements=True,
            )
        else:
            template, self.params = build_parametric_qaoa_circuit_overlap4_ablation(
                p_layers,
                initial_statevector,
                n_qubits,
                xy_pairs,
                x_mixer_qubits,
                lam,
                J_zz,
                h_z,
                energy_scale,
                ablation_arm=ablation_arm,
                with_measurements=True,
            )
        self.simulator = simulator
        self.energy_scale = energy_scale
        transpile_kwargs = dict(
            backend=simulator,
            seed_transpiler=seed_transpiler,
            optimization_level=optimization_level,
        )
        if basis_gates is not None:
            transpile_kwargs["basis_gates"] = basis_gates
        require_set_statevector = overlap4_ablation_uses_set_statevector(ablation_arm)
        self.tqc = transpile(template, **transpile_kwargs)
        audit_transpiled_ops(
            self.tqc,
            "cached parametric template",
            require_set_statevector=require_set_statevector,
        )

    def _bind(self, theta):
        binding = {param: float(theta[idx]) for idx, param in enumerate(self.params)}
        return self.tqc.assign_parameters(binding)

    def objective_energy(
        self, theta, shots, seed_sim, batches, energy_from_counts, counts_to_xorder
    ):
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


def build_parametric_qaoa_circuit_overlap4_ablation(
    p_layers,
    initial_statevector,
    n_qubits,
    xy_pairs,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    *,
    ablation_arm,
    with_measurements=True,
):
    from qaoa_new_4node_runtime import (
        ABLATION_INIT_ONLY,
        ABLATION_MIXER_ONLY,
        apply_standard_rx_mixer,
    )

    gammas = [Parameter(f"γ_{idx}") for idx in range(p_layers)]
    betas = [Parameter(f"β_{idx}") for idx in range(p_layers)]
    qc = QuantumCircuit(n_qubits)
    if ablation_arm == ABLATION_INIT_ONLY:
        apply_overlap4_direct_initial_state(qc, initial_statevector)
    elif ablation_arm == ABLATION_MIXER_ONLY:
        qc.h(range(n_qubits))
    else:
        raise ValueError(f"Unknown ablation arm: {ablation_arm!r}")

    for layer in range(p_layers):
        apply_cost_layer(qc, gammas[layer], J_zz, h_z, energy_scale)
        if ablation_arm == ABLATION_INIT_ONLY:
            apply_standard_rx_mixer(qc, betas[layer], n_qubits)
        else:
            apply_overlap4_hybrid_mixer(
                qc, betas[layer], xy_pairs, x_mixer_qubits, lam
            )
    if with_measurements:
        qc.measure_all()
    return qc, gammas + betas


def make_aer_exact_objective(
    initial_statevector,
    cost_op,
    n_qubits,
    p_layers,
    xy_pairs,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    simulator,
    seed_transpiler=999,
    *,
    ablation_arm=None,
):
    if ablation_arm is None:
        template, params = build_parametric_qaoa_circuit_overlap4_direct(
            p_layers,
            initial_statevector,
            n_qubits,
            xy_pairs,
            x_mixer_qubits,
            lam,
            J_zz,
            h_z,
            energy_scale,
            with_measurements=False,
        )
    else:
        template, params = build_parametric_qaoa_circuit_overlap4_ablation(
            p_layers,
            initial_statevector,
            n_qubits,
            xy_pairs,
            x_mixer_qubits,
            lam,
            J_zz,
            h_z,
            energy_scale,
            ablation_arm=ablation_arm,
            with_measurements=False,
        )
    tqc = transpile(
        template,
        backend=simulator,
        seed_transpiler=seed_transpiler,
        optimization_level=1,
    )
    audit_transpiled_ops(
        tqc,
        "exact objective template",
        require_set_statevector=overlap4_ablation_uses_set_statevector(ablation_arm),
    )

    def objective(theta):
        bound = tqc.assign_parameters(
            {param: float(theta[idx]) for idx, param in enumerate(params)}
        )
        bound.save_statevector()
        result = simulator.run(bound).result()
        sv = result.get_statevector(bound)
        return float(Statevector(sv).expectation_value(cost_op).real)

    return objective


def verify_aer_objective_against_math(
    objective,
    initial_statevector,
    cost_op,
    n_qubits,
    p_layers,
    xy_pairs,
    x_mixer_qubits,
    lam,
    J_zz,
    h_z,
    energy_scale,
    rng,
):
    theta = np.concatenate(
        [
            rng.uniform(-np.pi, np.pi, size=p_layers),
            rng.uniform(0.0, np.pi / 2.0, size=p_layers),
        ]
    )
    aer_val = float(objective(theta))
    state = Statevector(initial_statevector)
    gammas = theta[:p_layers]
    betas = theta[p_layers:]
    for layer in range(p_layers):
        layer_qc = QuantumCircuit(n_qubits)
        apply_cost_layer(layer_qc, gammas[layer], J_zz, h_z, energy_scale)
        apply_overlap4_hybrid_mixer(
            layer_qc, betas[layer], xy_pairs, x_mixer_qubits, lam
        )
        state = state.evolve(layer_qc)
    math_val = float(state.expectation_value(cost_op).real)
    if not np.isclose(aer_val, math_val, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            f"Aer exact objective mismatch: aer={aer_val}, math={math_val}"
        )
    print(
        f"[Exact objective check] aer={aer_val:.8f}, math={math_val:.8f}, ok=True"
    )


OVERLAP4_SCALE_TAG = "overlap4_100state_4node"
DEFAULT_OVERLAP4_RESULTS_DIR = "results_4node_overlap4_100state_direct"


def build_results_csv_path(
    case_name,
    lam,
    maxiter,
    output_dir=DEFAULT_OVERLAP4_RESULTS_DIR,
    *,
    scale_tag=OVERLAP4_SCALE_TAG,
    method_tag="",
):
    from qaoa_new_4node_runtime import build_results_csv_path as _build

    return _build(
        case_name,
        lam,
        maxiter,
        output_dir,
        scale_tag=scale_tag or OVERLAP4_SCALE_TAG,
        method_tag=method_tag,
    )


def summarize_experiment_with_pfeas(run_records, n_runs):
    from qaoa_new_4node_runtime import summarize_experiment, _mean_ci95_t

    stats = summarize_experiment(run_records, n_runs)
    pf_list = [row["p_feas"] for row in run_records]
    pf_mean, pf_std, pf_ci = _mean_ci95_t(pf_list)
    stats["p_feas_mean"] = pf_mean
    stats["p_feas_std"] = pf_std
    stats["p_feas_ci_low"] = pf_ci[0]
    stats["p_feas_ci_high"] = pf_ci[1]
    return stats


def print_experiment_summary_overlap4(title, stats, optimizer, n_runs, maxiter):
    from qaoa_new_4node_runtime import print_experiment_summary

    print_experiment_summary(title, stats, optimizer, n_runs, maxiter)
    print(
        "p_feas: "
        f"mean={stats['p_feas_mean']:.6f}, "
        f"std={stats['p_feas_std']:.6f}, "
        f"95% CI=[{stats['p_feas_ci_low']:.6f}, {stats['p_feas_ci_high']:.6f}]"
    )


def p_feas_from_counts(counts_x, shots, check_constraints_fn):
    return sum(
        c
        for xstr, c in counts_x.items()
        if check_constraints_fn(tuple(int(ch) for ch in xstr))
    ) / shots


def save_experiment_csv_overlap4(
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
    scale_tag=OVERLAP4_SCALE_TAG,
    method_tag="",
):
    import csv
    from pathlib import Path

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
        "scale_tag": scale_tag or OVERLAP4_SCALE_TAG,
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
        "p_feas_mean": stats["p_feas_mean"],
        "p_feas_std": stats["p_feas_std"],
        "p_feas_ci_low": stats["p_feas_ci_low"],
        "p_feas_ci_high": stats["p_feas_ci_high"],
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
                    "p_feas": row["p_feas"],
                    "gap_exp": row["gap_exp"],
                    "success": row["success"],
                    "tts99": row["tts99"],
                    "rank": row["rank"],
                    **{k: "" for k in summary_fields},
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


def build_selected_per_lambda_csv_path(
    case_name,
    lam,
    maxiter,
    output_dir=DEFAULT_OVERLAP4_RESULTS_DIR,
    *,
    scale_tag=OVERLAP4_SCALE_TAG,
    method_tag="",
):
    from qaoa_new_4node_runtime import build_selected_per_lambda_csv_path as _build

    return _build(
        case_name,
        lam,
        maxiter,
        output_dir,
        scale_tag=scale_tag or OVERLAP4_SCALE_TAG,
        method_tag=method_tag,
    )


def build_selected_overall_csv_path(
    case_name,
    lam,
    maxiter,
    output_dir=DEFAULT_OVERLAP4_RESULTS_DIR,
    *,
    scale_tag=OVERLAP4_SCALE_TAG,
    method_tag="",
):
    from qaoa_new_4node_runtime import build_selected_overall_csv_path as _build

    return _build(
        case_name,
        lam,
        maxiter,
        output_dir,
        scale_tag=scale_tag or OVERLAP4_SCALE_TAG,
        method_tag=method_tag,
    )


def build_selection_index_csv_path(
    case_name,
    output_dir=DEFAULT_OVERLAP4_RESULTS_DIR,
    *,
    scale_tag=OVERLAP4_SCALE_TAG,
    method_tag="",
):
    from qaoa_new_4node_runtime import build_selection_index_csv_path as _build

    return _build(
        case_name,
        output_dir,
        scale_tag=scale_tag or OVERLAP4_SCALE_TAG,
        method_tag=method_tag,
    )


def print_hyperparam_sweep_selection_overlap4(
    case_name,
    lambda_list,
    all_rows,
    *,
    summary_title,
    optimizer,
    n_runs,
    output_dir="results_4node_overlap4_100state_direct",
    **kwargs,
):
    from qaoa_new_4node_runtime import print_hyperparam_sweep_selection

    return print_hyperparam_sweep_selection(
        case_name,
        lambda_list,
        all_rows,
        summary_title=summary_title,
        optimizer=optimizer,
        n_runs=n_runs,
        summary_printer=print_experiment_summary_overlap4,
        output_dir=output_dir,
        scale_tag=kwargs.pop("scale_tag", OVERLAP4_SCALE_TAG),
        build_sweep_csv_path=build_results_csv_path,
        build_selected_per_lambda_csv_path_fn=build_selected_per_lambda_csv_path,
        build_selected_overall_csv_path_fn=build_selected_overall_csv_path,
        build_selection_index_csv_path_fn=build_selection_index_csv_path,
        save_experiment_csv_fn=save_experiment_csv_overlap4,
        **kwargs,
    )
