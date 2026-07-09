"""GeoAct CLI — benchmark geometry-aware vs flat action heads."""

from __future__ import annotations

import json
import os

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from geoact.so3 import exp_map_so3, geodesic_distance_so3, SE3
from geoact.action_head import GeoActHead, GeoActConfig, FlatActionHead
from geoact.benchmark import BenchmarkSuite

console = Console()


@click.group()
def main():
    """GeoAct — Geometry-aware action head benchmarks."""
    pass


@main.command()
@click.option("--n-trials", default=200, help="Number of trials per benchmark")
@click.option("--feature-dim", default=768, help="VLA feature dimension")
@click.option("-o", "--output", default="benchmarks/results/results.json")
def benchmark(n_trials: int, feature_dim: int, output: str):
    """Run full benchmark: GeoAct vs Flat action head."""
    console.print(Panel.fit(
        "[bold cyan]GeoAct — Geometry-Aware Action Head Benchmark[/bold cyan]\n"
        f"Trials: {n_trials} | Feature dim: {feature_dim}",
        title="🏃 Benchmarking GeoAct vs Flat Head",
    ))

    suite = BenchmarkSuite(feature_dim=feature_dim, n_trials=n_trials)
    report = suite.run_all()

    # Display results
    table = Table(title="📊 GeoAct vs Flat Action Head", show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", min_width=35)
    table.add_column("GeoAct", justify="right")
    table.add_column("Flat", justify="right")
    table.add_column("Winner")
    table.add_column("Improvement", justify="right")

    for r in report.results:
        winner = "[green]GeoAct[/green]" if r.geoact_wins else "[red]Flat[/red]"
        table.add_row(
            r.test_name,
            f"{r.geoact_score:.4f}",
            f"{r.flat_score:.4f}",
            winner,
            f"{r.improvement_pct:+.1f}%",
        )

    table.add_row(
        "[bold]Overall[/bold]",
        f"[bold]{report.geoact_wins_count}[/bold]",
        f"[bold]{report.total_tests - report.geoact_wins_count}[/bold]",
        "[bold]GeoAct[/bold]" if report.win_rate > 0.5 else "[bold]Flat[/bold]",
        f"[bold]{report.win_rate:.0%} win rate[/bold]",
        style="bold",
    )
    console.print(table)

    # Save results
    results = {
        "config": {"n_trials": n_trials, "feature_dim": feature_dim},
        "summary": {
            "geoact_wins": report.geoact_wins_count,
            "total_tests": report.total_tests,
            "win_rate": report.win_rate,
        },
        "tests": [
            {
                "name": r.test_name,
                "geoact_score": r.geoact_score,
                "flat_score": r.flat_score,
                "geoact_wins": r.geoact_wins,
                "improvement_pct": r.improvement_pct,
                "details": r.details,
            }
            for r in report.results
        ],
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[dim]Results saved to {output}[/dim]")


@main.command()
def demo():
    """Demo: GeoAct predicting actions from random features."""
    console.print("[bold]GeoAct Demo — Multi-modal SE(3) Action Prediction[/bold]\n")

    head = GeoActHead(GeoActConfig(feature_dim=128))
    features = np.random.randn(128).astype(np.float32)

    # Predict 5 action samples
    predictions = head.predict(features, n_samples=5)

    table = Table(title="Predicted Actions", show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Translation (x,y,z)")
    table.add_column("Rotation (axis-angle)")
    table.add_column("Confidence", justify="right")
    table.add_column("Mixture ID", justify="right")

    for i, pred in enumerate(predictions, 1):
        t = pred.translation
        r = pred.rotation_axis_angle
        table.add_row(
            str(i),
            f"({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})",
            f"({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})",
            f"{pred.confidence:.3f}",
            str(pred.mixture_id),
        )
    console.print(table)

    best = head.select_best(predictions)
    console.print(f"\n[green]Best action:[/green] {best.se3}")


@main.command()
def so3_demo():
    """Demo: SO(3) operations and geodesic distance."""
    from geoact.so3 import exp_map_so3, log_map_so3, geodesic_distance_so3

    console.print("[bold]SO(3) Lie Algebra Demo[/bold]\n")

    # Two rotations
    omega1 = np.array([0.0, 0.0, 0.5])
    omega2 = np.array([0.0, 0.0, 1.5])

    R1 = exp_map_so3(omega1)
    R2 = exp_map_so3(omega2)

    dist = geodesic_distance_so3(R1, R2)

    console.print(f"Rotation 1: axis-angle = {omega1}")
    console.print(f"Rotation 2: axis-angle = {omega2}")
    console.print(f"Geodesic distance: {dist:.4f} rad ({np.degrees(dist):.1f}°)")

    # Show that geodesic distance is smooth (no discontinuity)
    console.print("\n[bold]Geodesic distance near ±π (no discontinuity):[/bold]")
    for angle in [2.8, 3.0, 3.14, 3.14159, 3.2, 3.5]:
        R = exp_map_so3(np.array([0, 0, angle]))
        d = geodesic_distance_so3(np.eye(3), R)
        console.print(f"  angle={angle:.5f} → geodesic={d:.5f} rad")


if __name__ == "__main__":
    main()
