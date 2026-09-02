import argparse

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_aer import Aer

import src.classical_funcs as cf
from src.qiskit_route_construction import (
    build_qiskit_tsp_construction_circuit,
    build_route_phase_table,
)


def build_normalized_cost_matrix(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, size=(n, n))
    np.fill_diagonal(raw, 0.0)
    walks = cf.generate_all_walks(n, start_node=n - 1)
    all_costs = cf.find_all_cost(raw, walks)
    return raw / float(np.max(all_costs))


def plot_table(df, out_path: str) -> None:
    fig_height = max(8, 0.18 * len(df))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qiskit Aer statevector example for route/validity/cost-phase construction."
    )
    parser.add_argument("--n", type=int, default=5, help="Number of cities.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--csv", default="qiskit_tsp_table.csv", help="Output CSV path.")
    parser.add_argument("--plot", default="qiskit_tsp_table.png", help="Output plot path.")
    args = parser.parse_args()

    cost_matrix = build_normalized_cost_matrix(args.n, args.seed)
    circuit, layout = build_qiskit_tsp_construction_circuit(
        cost_matrix=cost_matrix,
        start_node=args.n - 1,
    )

    backend = Aer.get_backend("statevector_simulator")
    compiled = transpile(circuit, backend=backend)
    result = backend.run(compiled).result()
    statevector = np.asarray(result.get_statevector(compiled))

    df = build_route_phase_table(
        statevector=statevector,
        layout=layout,
        cost_matrix=cost_matrix,
    )
    df["tour_key"] = df["tour"].apply(tuple)
    df = df.sort_values(by=["validity", "expected_phi", "tour_key"], ascending=[False, True, True]).reset_index(drop=True)
    df = df.drop(columns=["tour_key"])
    df.to_csv(args.csv, index=False)
    plot_table(df, args.plot)

    print(f"Saved table CSV: {args.csv}")
    print(f"Saved table plot: {args.plot}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
