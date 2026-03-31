from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from rich.console import Console
from rich.table import Table

from q_lab.types import BenchmarkResult


def render_results(console: Console, results: Sequence[BenchmarkResult]) -> None:
    table = Table(title="Q-Lab Benchmark Report", show_lines=False)
    table.add_column("Label", style="bold cyan")
    table.add_column("Backend", style="green")
    table.add_column("Target")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Sparsity %", justify="right")
    table.add_column("Accuracy Proxy %", justify="right")
    table.add_column("Cosine", justify="right")
    table.add_column("Notes", overflow="fold")

    for result in results:
        table.add_row(
            result.label,
            result.backend.value,
            result.execution_target,
            f"{result.stats.mean_latency_ms:.3f}",
            f"{result.stats.p95_latency_ms:.3f}",
            f"{result.size_mb:.3f}",
            f"{result.sparsity_pct:.2f}",
            (
                "-"
                if result.fidelity.accuracy_proxy_pct is None
                else f"{result.fidelity.accuracy_proxy_pct:.2f}"
            ),
            (
                "-"
                if result.fidelity.cosine_similarity is None
                else f"{result.fidelity.cosine_similarity:.4f}"
            ),
            result.notes,
        )


    console.print(table)


def save_results_csv(results: Sequence[BenchmarkResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([result.to_record() for result in results])
    dataframe.to_csv(path, index=False)
