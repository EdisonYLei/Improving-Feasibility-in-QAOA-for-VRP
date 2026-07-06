#!/usr/bin/env python3
"""
Proposed QAOA p-depth study — run this file directly.

PyCharm: right-click this file → Run
Terminal (default 4 workers per script):
    cd /Users/edisonray/PycharmProjects/QAOA
    source .venv/bin/activate
    python proposed_p_depth_study.py

Run all three p-depth studies in parallel (3×4=12 processes on 14-core Mac):
    python proposed_p_depth_study.py &
    python standard_p_depth_study.py &
    python xy_p_depth_study.py &
    wait

Outputs:
  - Circuit diagrams: figures/proposed_qaoa_circuits/
  - Summary tables: figures/proposed_qaoa_circuits/p_sweep_statistics/
  - Raw CSV: results_proposed_p_sweep/
"""

from __future__ import annotations

import json
import numbers
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qiskit

from qaoa_ablation_runner import resolve_ablation_hyperparams
from qaoa_p_sweep_common import resolve_n_workers, run_jobs_parallel
from qaoa_proposed_p_sweep_runner import (
    CASE_LABELS,
    DEFAULT_P_SWEEP_DIR,
    FIGURES_DIR,
    MODELS,
    P_LAYERS,
    REGIME_LABELS,
    SCRIPT_BY_MODEL,
    build_overlap4_circuit,
    build_threenode_circuit,
    export_p_sweep_statistics,
    import_all_p2,
    load_overlap4_ns,
    load_threenode_ns,
    run_p_sweep_config,
)

# =============================================================================
# Configuration (edit here)
# =============================================================================

# 1) Draw parametric circuits for 9 models × p=1,2,3,4 (a few minutes)
DRAW_CIRCUITS = True

# Circuit figures always include measure gates (even for statevector diagrams)
DRAW_WITH_MEASUREMENTS = True

# 2) Import p=2 statistics from existing SELECTED runs (seconds)
IMPORT_P2 = True

# 3) Export master CSV, LaTeX tables, and p*-gap-rank plots
EXPORT_STATS = True

# 4) Run full optimization for p=1,3,4 (30 runs × 9 models; very slow; off by default)
RUN_EXPERIMENTS = True

# p values to run (p=2 comes from IMPORT_P2; no need to re-run)
RUN_P_VALUES = [1, 3]

# True = one run per group, for smoke-testing the pipeline (not for papers)
SMOKE_TEST = False

# Config-level parallel workers (each runs one model×p job).
# Override at launch: QAOA_P_SWEEP_N_WORKERS=2 python proposed_p_depth_study.py
N_WORKERS = 4

# =============================================================================


CIRCUIT_NUM_DECIMALS = 4
CIRCUIT_FOLD = -1  # single horizontal row, no line breaks

_FLOAT_RE = re.compile(
    r"(?<![\w$])"
    r"(-?(?:\d+\.\d+|\d+\.(?!\d)|\.\d+|\d+)(?:[eE][+-]?\d+)?)"
    r"(?![\w])"
)


def _slug(*parts: str) -> str:
    return "__".join(p.replace(".", "p") for p in parts)


def _fmt_num(value: float, *, ndigits: int = CIRCUIT_NUM_DECIMALS) -> str:
    """Format a number with exactly ``ndigits`` digits after the decimal point."""
    return f"{float(value):.{ndigits}f}"


def _format_fixed_decimals(text: str, *, ndigits: int = CIRCUIT_NUM_DECIMALS) -> str:
    """Force every numeric literal in ``text`` to fixed ``ndigits`` decimals."""

    def _repl(match: re.Match[str]) -> str:
        return f"{float(match.group(1)):.{ndigits}f}"

    return _FLOAT_RE.sub(_repl, text)


def _format_mpl_math_label(text: str, *, ndigits: int = CIRCUIT_NUM_DECIMALS) -> str:
    """Convert mpl labels to LaTeX mathtext with fixed-decimal coefficients."""
    if not text:
        return text

    body = text.strip()
    if body.startswith("$") and body.endswith("$"):
        body = body[1:-1]

    body = body.replace("$-$", "-")
    body = _format_fixed_decimals(body, ndigits=ndigits)
    body = re.sub(r"\(\s*(-?\d+\.\d+)\s*\)", r"\1", body)
    body = (
        body.replace("γ", r"\gamma")
        .replace("β", r"\beta")
        .replace("λ", r"\lambda")
        .replace("π", r"\pi")
    )
    body = body.replace("*", r" \cdot ")
    body = re.sub(r"\$\\pi\$", r"\\pi", body)
    body = re.sub(r"(\\gamma|\\beta|\\lambda)_(\d+)", r"\1_{\2}", body)
    return f"${body}$"


@lru_cache(maxsize=1)
def _load_circuit_mpl_style() -> dict:
    """Textbook mpl style plus project-specific LaTeX gate labels."""
    style_path = (
        Path(qiskit.__file__).resolve().parent
        / "visualization"
        / "circuit"
        / "styles"
        / "textbook.json"
    )
    with style_path.open(encoding="utf-8") as handle:
        style = json.load(handle)
    displaytext = style.setdefault("displaytext", {})
    displaytext.setdefault("set_statevector", r"$|\psi_0\rangle$")
    displaytext["rzz"] = "R_{ZZ}"
    displaytext.setdefault("rz", "R_{Z}")
    displaytext.setdefault("rx", "R_{X}")
    displaytext.setdefault("rxx", "R_{XX}")
    displaytext.setdefault("ryy", "R_{YY}")
    return style


def _prepare_circuit_for_display(qc):
    """Copy circuit and attach compact LaTeX labels for figure export."""
    qc = qc.copy()
    updated = []
    for inst in qc.data:
        op = inst.operation
        if op.name == "set_statevector":
            labeled = op.copy()
            labeled.label = r"$|\psi_0\rangle$"
            inst = inst.replace(operation=labeled)
        updated.append(inst)
    qc.data = updated
    return qc


def _configure_circuit_latex_fonts() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def _mpl_gate_text_is_math_wrapped(gate_text: str) -> bool:
    return gate_text.startswith("$") and gate_text.endswith("$")


def _wrap_mpl_gate_text(gate_text: str) -> str:
    if not gate_text or _mpl_gate_text_is_math_wrapped(gate_text):
        return gate_text
    return f"$\\mathrm{{{gate_text}}}$"


@contextmanager
def _circuit_draw_settings():
    """LaTeX mpl drawer: textbook gates, fixed decimals, single-row layout."""
    import importlib

    import qiskit.visualization.circuit._utils as qutils
    import qiskit.visualization.circuit.matplotlib as mpl_drawer

    pi_check_py = importlib.import_module("qiskit.circuit.tools.pi_check")
    orig_get_param_str = qutils.get_param_str
    orig_get_gate_ctrl_text = qutils.get_gate_ctrl_text
    orig_pi_check = pi_check_py.pi_check
    orig_mpl_get_param_str = mpl_drawer.get_param_str
    orig_mpl_get_gate_ctrl_text = mpl_drawer.get_gate_ctrl_text
    orig_mpl_pi_check = mpl_drawer.pi_check
    _configure_circuit_latex_fonts()

    def pi_check(inpt, eps=1e-9, output="text", ndigits=None):
        nd = CIRCUIT_NUM_DECIMALS if output == "mpl" and ndigits is not None else ndigits

        if output == "mpl" and nd is not None:
            if isinstance(inpt, numbers.Real) and not isinstance(inpt, bool):
                symbolic = orig_pi_check(inpt, eps=eps, output=output, ndigits=None)
                if symbolic and any(
                    marker in symbolic
                    for marker in ("pi", "π", "\\pi", "/", "*", "^", "$")
                ):
                    return symbolic
                try:
                    float(symbolic.replace("$", ""))
                    return _fmt_num(float(inpt), ndigits=nd)
                except ValueError:
                    pass
            result = orig_pi_check(inpt, eps=eps, output=output, ndigits=nd)
            if isinstance(inpt, numbers.Real) and not isinstance(inpt, bool):
                return _fmt_num(float(inpt), ndigits=nd)
            return result

        return orig_pi_check(inpt, eps=eps, output=output, ndigits=ndigits)

    def get_param_str(op, drawer, ndigits=3):
        if drawer == "mpl":
            ndigits = CIRCUIT_NUM_DECIMALS
            if op.name == "set_statevector":
                return ""
        text = orig_get_param_str(op, drawer, ndigits=ndigits)
        if drawer == "mpl" and text:
            text = text.replace("$-$", "-")
            text = _format_mpl_math_label(text, ndigits=ndigits)
        return text

    def get_gate_ctrl_text(op, drawer, style=None):
        gate_text, ctrl_text, raw_gate_text = orig_get_gate_ctrl_text(
            op, drawer, style=style
        )
        if drawer == "mpl":
            gate_text = _wrap_mpl_gate_text(gate_text)
            if ctrl_text:
                ctrl_text = _wrap_mpl_gate_text(ctrl_text)
        return gate_text, ctrl_text, raw_gate_text

    pi_check_py.pi_check = pi_check
    qutils.pi_check = pi_check
    qutils.get_param_str = get_param_str
    qutils.get_gate_ctrl_text = get_gate_ctrl_text
    mpl_drawer.pi_check = pi_check
    mpl_drawer.get_param_str = get_param_str
    mpl_drawer.get_gate_ctrl_text = get_gate_ctrl_text
    try:
        yield
    finally:
        pi_check_py.pi_check = orig_pi_check
        qutils.pi_check = orig_pi_check
        qutils.get_param_str = orig_get_param_str
        qutils.get_gate_ctrl_text = orig_get_gate_ctrl_text
        mpl_drawer.pi_check = orig_mpl_pi_check
        mpl_drawer.get_param_str = orig_mpl_get_param_str
        mpl_drawer.get_gate_ctrl_text = orig_mpl_get_gate_ctrl_text


def _save_circuit_figure(qc, out_base, *, title: str) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    qc = _prepare_circuit_for_display(qc)
    with _circuit_draw_settings():
        drawer = qc.draw(
            output="mpl",
            fold=CIRCUIT_FOLD,
            idle_wires=False,
            plot_barriers=True,
            style=_load_circuit_mpl_style(),
        )
    fig = drawer.figure if hasattr(drawer, "figure") else drawer
    fig.suptitle(title, fontsize=10, y=1.02)
    for ext, dpi in (("png", 180), ("pdf", 300)):
        path = out_base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[Saved] {path}")
    plt.close(fig)


def draw_all_circuits() -> None:
    threenode_ns = load_threenode_ns()
    out_dir = FIGURES_DIR

    for scale_tag, case_name, regime in MODELS:
        lam, maxiter, source = resolve_ablation_hyperparams(
            scale_tag, case_name, regime
        )
        with_measurements = DRAW_WITH_MEASUREMENTS
        case_label = CASE_LABELS.get((scale_tag, case_name), case_name)
        regime_label = REGIME_LABELS[regime]

        if scale_tag == "3node":
            ns = threenode_ns
            builder = build_threenode_circuit
            out_subdir = out_dir / "3node" / regime
            model_slug = f"3node_{regime}"
        else:
            script = SCRIPT_BY_MODEL[(scale_tag, case_name, regime)]
            ns = load_overlap4_ns(script)
            builder = build_overlap4_circuit
            out_subdir = out_dir / case_name / regime
            model_slug = f"{case_name}_{regime}"

        lam_text = _fmt_num(lam)
        print(
            textwrap.dedent(
                f"""
                === {case_label} | {regime_label} ===
                lambda={lam_text}, maxiter={maxiter}
                source: {source}
                """
            ).strip()
        )

        for p_layers in P_LAYERS:
            qc = builder(ns, p_layers, lam, with_measurements)
            title = (
                f"Proposed QAOA | {case_label} | {regime_label}\n"
                f"$p={p_layers}$, $\\lambda={lam_text}$, maxiter={maxiter}"
                + (" | with measurements" if with_measurements else "")
            )
            out_base = out_subdir / f"{model_slug}_p{p_layers}_lambda{_slug(str(lam))}"
            _save_circuit_figure(qc, out_base, title=title)

    print(
        f"\n[Circuits done] {len(MODELS) * len(P_LAYERS)} PNG+PDF sets → {out_dir}"
    )


def _run_one_proposed_job(job: tuple[str, str, str, int, bool]) -> None:
    scale_tag, case_name, regime, p_layers, smoke = job
    run_p_sweep_config(
        scale_tag,
        case_name,
        regime,
        p_layers,
        smoke=smoke,
    )


def run_experiments(*, p_values: list[int], smoke: bool) -> None:
    n_workers = resolve_n_workers(N_WORKERS)
    print(
        f"\n[Starting experiments] p={p_values}, smoke={smoke}, "
        f"n_workers={n_workers}"
    )
    jobs: list[tuple[str, str, str, int, bool]] = []
    for scale_tag, case_name, regime in MODELS:
        for p_layers in p_values:
            if p_layers == 2 and not smoke:
                continue
            jobs.append((scale_tag, case_name, regime, p_layers, smoke))
    run_jobs_parallel(jobs, _run_one_proposed_job, n_workers=n_workers)
    print(f"[Experiments done] CSV → {DEFAULT_P_SWEEP_DIR}")


def main() -> None:
    print("=" * 72)
    print("Proposed QAOA p-depth study")
    print(
        f"  DRAW_CIRCUITS={DRAW_CIRCUITS}  IMPORT_P2={IMPORT_P2}  "
        f"EXPORT_STATS={EXPORT_STATS}  RUN_EXPERIMENTS={RUN_EXPERIMENTS}"
    )
    if RUN_EXPERIMENTS:
        print(
            f"  RUN_P_VALUES={RUN_P_VALUES}  SMOKE_TEST={SMOKE_TEST}  "
            f"N_WORKERS={resolve_n_workers(N_WORKERS)}"
        )
    print("=" * 72)

    if DRAW_CIRCUITS:
        draw_all_circuits()

    if IMPORT_P2:
        import_all_p2()

    if RUN_EXPERIMENTS:
        run_experiments(p_values=RUN_P_VALUES, smoke=SMOKE_TEST)

    if EXPORT_STATS:
        master, latex_paths, plot_paths = export_p_sweep_statistics()
        print("\n[Statistics export]")
        print(f"  Master CSV: {master}")
        print(f"  LaTeX tables: {len(latex_paths)} → .../p_sweep_statistics/latex/")
        print(f"  Plots:        {len(plot_paths)} → .../p_sweep_statistics/plots/")

    print("\n[All done]")


if __name__ == "__main__":
    main()
