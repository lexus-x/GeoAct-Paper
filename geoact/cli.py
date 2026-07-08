"""CLI for GeoAct paper experiments."""

from __future__ import annotations

import json
import os
import time

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
def main():
    """GeoAct: Geometry-Aware Action Prediction for VLA Models."""
    pass


@main.command()
def theory():
    """Run theoretical verification experiments."""
    from geoact.theory import TheoryVerification

    console.print(Panel.fit(
        "[bold cyan]GeoAct — Theoretical Verification[/bold cyan]",
        title="📐 Theory",
    ))

    verifier = TheoryVerification(n_samples=5000)
    results = verifier.verify_all()

    table = Table(title="Theorem Verification", show_header=True, header_style="bold magenta")
    table.add_column("Theorem", style="cyan", min_width=35)
    table.add_column("Verified", justify="center")
    table.add_column("Key Evidence")

    for r in results:
        status = "[green]✓[/green]" if r.verified else "[red]✗[/red]"
        evidence_str = ", ".join(f"{k}={v}" for k, v in list(r.evidence.items())[:2])
        table.add_row(r.name, status, evidence_str)

    console.print(table)

    # Save
    output = {
        "theorems": [
            {
                "name": r.name,
                "statement": r.statement,
                "verified": bool(r.verified),
                "evidence": {k: float(v) if isinstance(v, (np.floating, float, np.integer, int)) else str(v)
                            for k, v in r.evidence.items()},
            }
            for r in results
        ]
    }
    os.makedirs("results/theory", exist_ok=True)
    with open("results/theory/verification.json", "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[dim]Saved to results/theory/verification.json[/dim]")


@main.command()
def benchmark():
    """Run full benchmark: all heads × all tasks."""
    from benchmark.evaluator import BenchmarkEvaluator, FlatL2Head, GeoActHead, QuaternionHead
    from benchmark.tasks import TASKS

    console.print(Panel.fit(
        "[bold cyan]ActionRep-Bench — Full Evaluation[/bold cyan]\n"
        f"Heads: Flat L2, Quaternion, GeoAct | Tasks: {len(TASKS)} | Episodes: 100",
        title="🏃 Benchmark",
    ))

    heads = [FlatL2Head(), QuaternionHead(), GeoActHead()]
    evaluator = BenchmarkEvaluator(n_episodes=100)
    all_results = {}

    for head in heads:
        console.print(f"\n[bold]Evaluating {head.name}...[/bold]")
        start = time.perf_counter()
        results = evaluator.evaluate_head(head)
        elapsed = time.perf_counter() - start

        all_results[head.name] = {
            task_key: task_result.to_dict()
            for task_key, task_result in results.items()
        }

        # Display
        table = Table(title=f"{head.name}", show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan")
        table.add_column("Success %", justify="right")
        table.add_column("Trans Err", justify="right")
        table.add_column("Rot Err", justify="right")

        for task_key, task_result in results.items():
            sr = task_result.success_rate * 100
            color = "green" if sr > 70 else "yellow" if sr > 40 else "red"
            table.add_row(
                task_result.task_name,
                f"[{color}]{sr:.0f}%[/{color}]",
                f"{task_result.mean_translation_error:.4f}",
                f"{task_result.mean_rotation_error:.4f}",
            )
        console.print(table)
        console.print(f"  Time: {elapsed:.1f}s")

    # Summary comparison
    console.print("\n[bold]Summary: GeoAct vs Baselines[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task", style="cyan")
    for head_name in all_results:
        table.add_column(head_name, justify="right")

    for task_key in TASKS:
        row = [TASKS[task_key].name]
        for head_name in all_results:
            sr = all_results[head_name].get(task_key, {}).get("success_rate", 0)
            row.append(f"{sr*100:.0f}%")
        table.add_row(*row)
    console.print(table)

    # Save
    os.makedirs("results/synthetic", exist_ok=True)
    with open("results/synthetic/benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\n[dim]Saved to results/synthetic/benchmark.json[/dim]")


@main.command()
def ablation():
    """Run ablation studies."""
    from benchmark.evaluator import BenchmarkEvaluator, GeoActHead
    from benchmark.tasks import TASKS

    console.print(Panel.fit(
        "[bold cyan]GeoAct — Ablation Studies[/bold cyan]",
        title="🔬 Ablation",
    ))

    evaluator = BenchmarkEvaluator(n_episodes=50)
    ablations = {}

    # Ablation 1: Number of mixture components
    console.print("\n[bold]Ablation 1: Mixture Components (K)[/bold]")
    for K in [1, 3, 5, 7, 10]:
        head = GeoActHead(n_components=K)
        results = evaluator.evaluate_head(head, ["precision_insert", "multimodal_reach"])
        avg_sr = np.mean([r.success_rate for r in results.values()])
        ablations[f"K={K}"] = {"success_rate": float(avg_sr), "n_components": K}
        console.print(f"  K={K}: success={avg_sr:.1%}")

    # Ablation 2: Refinement steps
    console.print("\n[bold]Ablation 2: Refinement Steps[/bold]")
    for steps in [0, 1, 2, 3, 5]:
        head = GeoActHead(n_refine=steps)
        results = evaluator.evaluate_head(head, ["precision_insert"])
        sr = list(results.values())[0].success_rate
        ablations[f"refine={steps}"] = {"success_rate": float(sr), "n_steps": steps}
        console.print(f"  steps={steps}: success={sr:.1%}")

    # Ablation 3: Noise scale
    console.print("\n[bold]Ablation 3: Noise Scale[/bold]")
    for noise in [0.01, 0.03, 0.05, 0.1, 0.2]:
        head = GeoActHead(noise_scale=noise)
        results = evaluator.evaluate_head(head, ["precision_insert"])
        sr = list(results.values())[0].success_rate
        ablations[f"noise={noise}"] = {"success_rate": float(sr), "noise": noise}
        console.print(f"  noise={noise}: success={sr:.1%}")

    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/ablation.json", "w") as f:
        json.dump(ablations, f, indent=2)
    console.print(f"\n[dim]Saved to results/ablation/ablation.json[/dim]")


if __name__ == "__main__":
    main()
