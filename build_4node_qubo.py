"""
Build and verify 4-node, 2-vehicle VRP QUBO coefficients.

Formulation (matches README_derivations_and_optima.md and original 6-qubit code):
  - Objective: sum_{(i,j)} w_ij x_ij
  - Each equality sum_{k in A} x_k = b is penalized by P * (sum x_k - b)^2
  - P = 2 * sum_{i != j} |w_ij|
  - For binary x: P*(sum x - b)^2 = P*[b^2 + (1-2b)*sum x + 2*sum_{i<j} x_i x_j]
"""

from __future__ import annotations

import ast
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np

ARCS = [
    (0, 1), (0, 2), (0, 3),
    (1, 0), (1, 2), (1, 3),
    (2, 0), (2, 1), (2, 3),
    (3, 0), (3, 1), (3, 2),
]

# Variable indices for each equality constraint (see README_derivations_and_optima.md).
CONSTRAINTS: List[Tuple[str, List[int], int]] = [
    ("in_node_1", [0, 7, 10], 1),
    ("in_node_2", [1, 4, 11], 1),
    ("in_node_3", [2, 5, 8], 1),
    ("out_node_1", [3, 4, 5], 1),
    ("out_node_2", [6, 7, 8], 1),
    ("out_node_3", [9, 10, 11], 1),
    ("depot_out", [0, 1, 2], 2),
    ("depot_in", [3, 6, 9], 2),
]

CASES = {
    "case1_balanced_symmetric": np.array(
        [[0.0, 21.7, 34.2, 28.6], [21.7, 0.0, 17.4, 24.9], [34.2, 17.4, 0.0, 19.8], [28.6, 24.9, 19.8, 0.0]],
        dtype=float,
    ),
    "case2_customer_cluster": np.array(
        [[0.0, 42.3, 26.8, 38.5], [42.3, 0.0, 13.7, 29.4], [26.8, 13.7, 0.0, 16.2], [38.5, 29.4, 16.2, 0.0]],
        dtype=float,
    ),
    "case3_asymmetric_directed": np.array(
        [[0.0, 19.6, 31.4, 27.8], [22.4, 0.0, 18.9, 33.7], [29.1, 14.6, 0.0, 21.3], [35.5, 25.8, 16.7, 0.0]],
        dtype=float,
    ),
    "case4_far_pair_symmetric": np.array(
        [[0.0, 15.9, 48.7, 44.1], [15.9, 0.0, 31.5, 27.2], [48.7, 31.5, 0.0, 14.6], [44.1, 27.2, 14.6, 0.0]],
        dtype=float,
    ),
}


def compute_penalty(distance_matrix: np.ndarray) -> float:
    n = distance_matrix.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                total += abs(float(distance_matrix[i, j]))
    return 2.0 * total


def build_qubo(distance_matrix: np.ndarray) -> Tuple[float, Dict[int, float], Dict[Tuple[int, int], float], float]:
    """Return (qubo_const, qubo_linear, qubo_quad, penalty)."""
    penalty = compute_penalty(distance_matrix)
    qubo_const = 0.0
    qubo_linear: Dict[int, float] = {}
    qubo_quad: Dict[Tuple[int, int], float] = {}

    for k, (i, j) in enumerate(ARCS):
        qubo_linear[k] = qubo_linear.get(k, 0.0) + float(distance_matrix[i, j])

    for _name, indices, rhs in CONSTRAINTS:
        qubo_const += penalty * rhs * rhs
        lin_add = penalty * (1 - 2 * rhs)
        for idx in indices:
            qubo_linear[idx] = qubo_linear.get(idx, 0.0) + lin_add
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                key = (min(i, j), max(i, j))
                qubo_quad[key] = qubo_quad.get(key, 0.0) + 2.0 * penalty

    return qubo_const, qubo_linear, qubo_quad, penalty


def qubo_to_ising(qubo_const: float, qubo_linear: Dict[int, float], qubo_quad: Dict[Tuple[int, int], float], n: int):
    J_zz = {}
    h_z = {i: 0.0 for i in range(n)}
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


def _parse_embedded_qubo(source_path: str):
    with open(source_path, encoding="utf-8") as f:
        text = f.read()
    const = float(re.search(r"^qubo_const = (.+)$", text, re.M).group(1))
    linear = ast.literal_eval(re.search(r"^qubo_linear = (.+)$", text, re.M).group(1))
    quad_raw = ast.literal_eval(re.search(r"^qubo_quad = (.+)$", text, re.M).group(1))
    quad = {(min(i, j), max(i, j)): float(v) for (i, j), v in quad_raw.items()}
    penalty = float(re.search(r"^PENALTY = (.+)$", text, re.M).group(1))
    return const, linear, quad, penalty


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _dict_close(d1: Dict, d2: Dict, tol: float = 1e-6) -> bool:
    if set(d1.keys()) != set(d2.keys()):
        return False
    return all(_close(float(d1[k]), float(d2[k]), tol) for k in d1)


def verify_all_cases(root: str | None = None, tol: float = 1e-6) -> bool:
    root = root or os.path.dirname(os.path.abspath(__file__))
    ok = True
    for case_name, dist in CASES.items():
        ref_file = os.path.join(root, f"OQAOA_4node_{case_name}.py")
        c_calc, lin_calc, quad_calc, p_calc = build_qubo(dist)
        c_emb, lin_emb, quad_emb, p_emb = _parse_embedded_qubo(ref_file)

        case_ok = (
            _close(c_calc, c_emb, tol)
            and _close(p_calc, p_emb, tol)
            and _dict_close(lin_calc, lin_emb, tol)
            and _dict_close(quad_calc, quad_emb, tol)
        )
        status = "OK" if case_ok else "MISMATCH"
        print(f"[{status}] {case_name}: P={p_calc:.1f}, c={c_calc:.1f}")
        if not case_ok:
            ok = False
            if not _close(p_calc, p_emb, tol):
                print(f"  PENALTY embedded={p_emb} computed={p_calc}")
            if not _close(c_calc, c_emb, tol):
                print(f"  qubo_const embedded={c_emb} computed={c_calc}")
            for k in sorted(set(lin_calc) | set(lin_emb)):
                if k not in lin_calc or k not in lin_emb or not _close(lin_calc[k], lin_emb[k], tol):
                    print(f"  linear[{k}] embedded={lin_emb.get(k)} computed={lin_calc.get(k)}")
            for k in sorted(set(quad_calc) | set(quad_emb)):
                if k not in quad_calc or k not in quad_emb or not _close(quad_calc[k], quad_emb[k], tol):
                    print(f"  quad{k} embedded={quad_emb.get(k)} computed={quad_calc.get(k)}")
    return ok


def format_qubo_for_python(case_name: str) -> str:
    dist = CASES[case_name]
    c, lin, quad, p = build_qubo(dist)
    lin_items = ", ".join(f"{k}:{lin[k]}" for k in sorted(lin))
    quad_items = ", ".join(f"({i}, {j}):{quad[(i, j)]}" for (i, j) in sorted(quad))
    return (
        f"PENALTY = {p}\n"
        f"qubo_const = {c}\n"
        f"qubo_linear = {{{lin_items}}}\n"
        f"qubo_quad = {{{quad_items}}}\n"
    )


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    print("Verifying embedded QUBO coefficients against formulation...")
    all_ok = verify_all_cases(root)
    if all_ok:
        print("\nAll four cases match embedded coefficients.")
    else:
        print("\nSome cases do NOT match embedded coefficients.")
        raise SystemExit(1)
