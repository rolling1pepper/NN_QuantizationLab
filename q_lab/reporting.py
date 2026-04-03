from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pandas as pd
from rich.console import Console
from rich.table import Table

from q_lab.types import BenchmarkResult


def render_results(console: Console, results: Sequence[BenchmarkResult]) -> None:
    table = Table(title="Q-Lab Benchmark Report", show_lines=False)
    table.add_column("Label", style="bold cyan")
    table.add_column("Batch", justify="right")
    table.add_column("Backend", style="green")
    table.add_column("Target")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Throughput", justify="right")
    table.add_column("Peak Mem (MB)", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Sparsity %", justify="right")
    table.add_column("Accuracy Proxy %", justify="right")
    table.add_column("Eval Acc %", justify="right")
    table.add_column("Eval F1 %", justify="right")
    table.add_column("Cosine", justify="right")
    table.add_column("Notes", overflow="fold")

    for result in results:
        table.add_row(
            result.label,
            str(result.batch_size),
            result.backend.value,
            result.execution_target,
            f"{result.stats.mean_latency_ms:.3f}",
            f"{result.stats.p95_latency_ms:.3f}",
            (
                "-"
                if result.stats.throughput_items_per_sec is None
                else f"{result.stats.throughput_items_per_sec:.2f}"
            ),
            (
                "-"
                if result.stats.peak_memory_mb is None
                else f"{result.stats.peak_memory_mb:.3f}"
            ),
            f"{result.size_mb:.3f}",
            f"{result.sparsity_pct:.2f}",
            (
                "-"
                if result.fidelity.accuracy_proxy_pct is None
                else f"{result.fidelity.accuracy_proxy_pct:.2f}"
            ),
            (
                "-"
                if result.evaluation.top1_accuracy_pct is None
                else f"{result.evaluation.top1_accuracy_pct:.2f}"
            ),
            (
                "-"
                if result.evaluation.macro_f1_pct is None
                else f"{result.evaluation.macro_f1_pct:.2f}"
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


def save_results_html(
    results: Sequence[BenchmarkResult],
    path: Path,
    manifest: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([result.to_record() for result in results])
    summary_json = "" if manifest is None else json.dumps(manifest, indent=2, ensure_ascii=False)
    summary_block = (
        ""
        if not summary_json
        else (
            "<section><h2>Run Manifest</h2>"
            f"<pre>{summary_json}</pre>"
            "</section>"
        )
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Q-Lab Benchmark Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f1f1f;
      --accent: #0e5a8a;
      --muted: #5b5b5b;
      --border: #d7d1c5;
    }}
    body {{
      margin: 0;
      padding: 32px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #efe7da 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 28px;
      box-shadow: 0 18px 40px rgba(34, 34, 34, 0.08);
    }}
    h1, h2 {{
      margin: 0 0 16px 0;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    p {{
      color: var(--muted);
      margin: 0 0 20px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-bottom: 24px;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #ece4d8;
      color: var(--accent);
    }}
    tbody tr:nth-child(even) {{
      background: #faf6ee;
    }}
    pre {{
      padding: 16px;
      overflow-x: auto;
      border-radius: 12px;
      background: #151515;
      color: #f4f4f4;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Q-Lab Benchmark Report</h1>
    <p>Generated from a reproducible Q-Lab experiment run.</p>
    {dataframe.to_html(index=False, border=0, classes="results-table")}
    {summary_block}
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
