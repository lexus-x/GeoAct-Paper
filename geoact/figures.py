"""Generate paper-ready figures for GeoAct."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def generate_all(results_dir: str, output_dir: str):
    """Generate all paper figures."""
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    with open(f"{results_dir}/theory/verification.json") as f:
        theory = json.load(f)
    with open(f"{results_dir}/synthetic/benchmark.json") as f:
        benchmark = json.load(f)

    ablation_path = f"{results_dir}/ablation/ablation.json"
    ablation = {}
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)

    print("Generating paper figures...")

    fig1_loss_landscape(output_dir)
    fig2_benchmark_comparison(benchmark, output_dir)
    fig3_theory_verification(theory, output_dir)
    fig4_ablation(ablation, output_dir)
    fig5_geodesic_vs_euler(output_dir)
    fig6_architecture(output_dir)
    fig7_per_task_breakdown(benchmark, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


def fig1_loss_landscape(output_dir: str):
    """Figure 1: Loss landscape comparison — geodesic vs L2 vs quaternion."""
    from geoact.so3 import exp_map, geodesic_distance
    from scipy.spatial.transform import Rotation

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    target = np.array([0.3, -0.2, 0.5])
    R_target = exp_map(target)

    # Sweep over rotation space
    n = 100
    omega1_range = np.linspace(-2, 2, n)
    omega2_range = np.linspace(-2, 2, n)
    O1, O2 = np.meshgrid(omega1_range, omega2_range)

    for ax_idx, (ax, loss_fn, title) in enumerate(zip(axes, [
        lambda o1, o2: geodesic_distance(exp_map(np.array([o1, o2, 0.5])), R_target),
        lambda o1, o2: float(np.sum((Rotation.from_matrix(exp_map(np.array([o1, o2, 0.5]))).as_euler('ZYX') - Rotation.from_matrix(R_target).as_euler('ZYX'))**2)),
        lambda o1, o2: float(np.sum((Rotation.from_matrix(exp_map(np.array([o1, o2, 0.5]))).as_quat() - Rotation.from_matrix(R_target).as_quat())**2)),
    ], ["Geodesic (GeoAct)", "L2 Euler (Flat)", "L2 Quaternion"])):
        Z = np.zeros_like(O1)
        for i in range(n):
            for j in range(n):
                try:
                    Z[i, j] = loss_fn(O1[i, j], O2[i, j])
                except:
                    Z[i, j] = np.nan

        im = ax.contourf(O1, O2, Z, levels=20, cmap="viridis", alpha=0.8)
        ax.plot(target[0], target[1], "r*", markersize=15, label="Target")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("ω₁")
        ax.set_ylabel("ω₂")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Figure 1: Loss Landscape Comparison on SO(3)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_loss_landscape.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig1_loss_landscape.png")


def fig2_benchmark_comparison(benchmark: dict, output_dir: str):
    """Figure 2: Success rate comparison across all tasks and heads."""
    fig, ax = plt.subplots(figsize=(12, 5))

    heads = list(benchmark.keys())
    tasks = list(benchmark[heads[0]].keys())
    task_labels = [benchmark[heads[0]][t]["task"] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.8 / len(heads)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i, head in enumerate(heads):
        rates = [benchmark[head][t]["success_rate"] * 100 for t in tasks]
        offset = (i - len(heads) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=head, color=colors[i % len(colors)],
                     edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Figure 2: ActionRep-Bench — Success Rate by Task", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_benchmark.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig2_benchmark.png")


def fig3_theory_verification(theory: dict, output_dir: str):
    """Figure 3: Theory verification — geodesic vs Euler discontinuity."""
    from geoact.so3 import exp_map, geodesic_distance
    from scipy.spatial.transform import Rotation

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Geodesic distance along a path through SO(3)
    angles = np.linspace(0, 4 * np.pi, 500)
    target = exp_map(np.array([0, 0, 0.5]))

    geo_dists = []
    euler_dists = []

    for angle in angles:
        R = exp_map(np.array([0, angle * 0.3, angle * 0.2]))
        geo_dists.append(geodesic_distance(R, target))

        euler_R = Rotation.from_matrix(R).as_euler('ZYX')
        euler_target = Rotation.from_matrix(target).as_euler('ZYX')
        euler_dists.append(float(np.sum((euler_R - euler_target) ** 2)))

    ax1.plot(angles, geo_dists, "-", color="#2ecc71", linewidth=1.5, label="Geodesic (smooth)")
    ax1.plot(angles, euler_dists, "-", color="#e74c3c", linewidth=1.5, label="L2 Euler (discontinuous)")
    ax1.set_xlabel("Path parameter", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("(a) Loss Along Path in SO(3)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Gradient magnitude comparison
    geo_grads = []
    euler_grads = []

    for i in range(1, len(angles)):
        geo_grads.append(abs(geo_dists[i] - geo_dists[i-1]))
        euler_grads.append(abs(euler_dists[i] - euler_dists[i-1]))

    ax2.hist(geo_grads, bins=50, alpha=0.5, color="#2ecc71", label="Geodesic", density=True)
    ax2.hist(euler_grads, bins=50, alpha=0.5, color="#e74c3c", label="L2 Euler", density=True)
    ax2.set_xlabel("Gradient Magnitude", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("(b) Gradient Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig3_theory.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig3_theory.png")


def fig4_ablation(ablation: dict, output_dir: str):
    """Figure 4: Ablation studies."""
    if not ablation:
        print("  ⚠ fig4_ablation.png skipped (no ablation data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Ablation 1: K components
    ax = axes[0]
    k_items = {k: v for k, v in ablation.items() if k.startswith("K=")}
    if k_items:
        ks = [v["n_components"] for v in k_items.values()]
        srs = [v["success_rate"] * 100 for v in k_items.values()]
        ax.plot(ks, srs, "o-", color="#3498db", linewidth=2, markersize=8)
        ax.set_xlabel("Mixture Components (K)")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("(a) Mixture Components", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Ablation 2: Refinement steps
    ax = axes[1]
    refine_items = {k: v for k, v in ablation.items() if k.startswith("refine=")}
    if refine_items:
        steps = [v["n_steps"] for v in refine_items.values()]
        srs = [v["success_rate"] * 100 for v in refine_items.values()]
        ax.plot(steps, srs, "o-", color="#2ecc71", linewidth=2, markersize=8)
        ax.set_xlabel("Refinement Steps")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("(b) Refinement Steps", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Ablation 3: Noise scale
    ax = axes[2]
    noise_items = {k: v for k, v in ablation.items() if k.startswith("noise=")}
    if noise_items:
        noises = [v["noise"] for v in noise_items.values()]
        srs = [v["success_rate"] * 100 for v in noise_items.values()]
        ax.plot(noises, srs, "o-", color="#e74c3c", linewidth=2, markersize=8)
        ax.set_xlabel("Noise Scale")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("(c) Noise Robustness", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Figure 4: Ablation Studies", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_ablation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig4_ablation.png")


def fig5_geodesic_vs_euler(output_dir: str):
    """Figure 5: Geodesic vs Euler angle error at different rotation magnitudes."""
    from geoact.so3 import exp_map, geodesic_distance
    from scipy.spatial.transform import Rotation

    fig, ax = plt.subplots(figsize=(8, 5))

    magnitudes = np.linspace(0.1, 3.0, 50)
    geo_errors = []
    euler_errors = []
    quat_errors = []

    for mag in magnitudes:
        target = np.array([0, 0, mag])
        R_target = exp_map(target)

        # Add noise
        pred = target + np.random.randn(3) * 0.1
        R_pred = exp_map(pred)

        # Geodesic error
        geo_errors.append(geodesic_distance(R_pred, R_target))

        # Euler error
        euler_pred = Rotation.from_matrix(R_pred).as_euler('ZYX')
        euler_target = Rotation.from_matrix(R_target).as_euler('ZYX')
        euler_errors.append(float(np.sum((euler_pred - euler_target) ** 2)))

        # Quaternion error
        q_pred = Rotation.from_matrix(R_pred).as_quat()
        q_target = Rotation.from_matrix(R_target).as_quat()
        if np.dot(q_pred, q_target) < 0:
            q_target = -q_target
        quat_errors.append(float(np.sum((q_pred - q_target) ** 2)))

    ax.plot(magnitudes, geo_errors, "-", color="#2ecc71", linewidth=2, label="Geodesic (GeoAct)")
    ax.plot(magnitudes, euler_errors, "-", color="#e74c3c", linewidth=2, label="L2 Euler")
    ax.plot(magnitudes, quat_errors, "-", color="#3498db", linewidth=2, label="L2 Quaternion")
    ax.axvline(x=np.pi, color="#95a5a6", linestyle="--", alpha=0.5, label="θ = π")

    ax.set_xlabel("Rotation Magnitude (rad)", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Figure 5: Loss vs Rotation Magnitude", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig5_loss_vs_magnitude.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig5_loss_vs_magnitude.png")


def fig6_architecture(output_dir: str):
    """Figure 6: GeoAct architecture diagram (SVG)."""
    svg = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 400" font-family="Helvetica, Arial, sans-serif">
  <rect width="900" height="400" fill="#fff" rx="8" stroke="#ddd"/>

  <!-- VLA Backbone -->
  <rect x="30" y="80" width="160" height="120" rx="6" fill="#e8f4fd" stroke="#1f6feb" stroke-width="1.5"/>
  <text x="110" y="110" text-anchor="middle" fill="#1f6feb" font-size="12" font-weight="bold">VLA Backbone</text>
  <text x="110" y="130" text-anchor="middle" fill="#555" font-size="9">Vision + Language Encoder</text>
  <text x="110" y="148" text-anchor="middle" fill="#555" font-size="9">→ features f ∈ ℝ^d</text>
  <text x="110" y="175" text-anchor="middle" fill="#888" font-size="8">(frozen or fine-tuned)</text>

  <!-- Arrow -->
  <line x1="190" y1="140" x2="230" y2="140" stroke="#888" stroke-width="1.5" marker-end="url(#arr)"/>
  <defs><marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#888"/></marker></defs>

  <!-- GeoAct Head -->
  <rect x="230" y="40" width="380" height="200" rx="6" fill="#e8fde8" stroke="#238636" stroke-width="1.5"/>
  <text x="420" y="65" text-anchor="middle" fill="#238636" font-size="14" font-weight="bold">GeoAct Head</text>

  <!-- Sub-modules -->
  <rect x="250" y="80" width="100" height="30" rx="4" fill="#d4edda" stroke="#28a745"/>
  <text x="300" y="100" text-anchor="middle" fill="#155724" font-size="9">Projection</text>

  <rect x="360" y="80" width="110" height="30" rx="4" fill="#d4edda" stroke="#28a745"/>
  <text x="415" y="100" text-anchor="middle" fill="#155724" font-size="9">SE(3) MDN (K)</text>

  <rect x="480" y="80" width="110" height="30" rx="4" fill="#d4edda" stroke="#28a745"/>
  <text x="535" y="100" text-anchor="middle" fill="#155724" font-size="9">Residual Refine</text>

  <rect x="250" y="120" width="340" height="30" rx="4" fill="#d4edda" stroke="#28a745"/>
  <text x="420" y="140" text-anchor="middle" fill="#155724" font-size="9">Geodesic Loss: L = ||log(R_pred^T R_target)|| + ||t_pred - t_target||</text>

  <rect x="250" y="160" width="340" height="30" rx="4" fill="#d4edda" stroke="#28a745"/>
  <text x="420" y="180" text-anchor="middle" fill="#155724" font-size="9">Multi-modal: p(a|f) = Σ_k π_k · N(t;μ_k,σ_k) · vMF(R;μ_R_k,κ_k)</text>

  <rect x="250" y="200" width="340" height="30" rx="4" fill="#d4edda" stroke="#28a745"/>
  <text x="420" y="220" text-anchor="middle" fill="#155724" font-size="9">SE(3) Constraint: R ∈ SO(3), t ∈ ℝ³ → valid rigid transform always</text>

  <!-- Arrow -->
  <line x1="610" y1="140" x2="650" y2="140" stroke="#888" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- SE(3) Output -->
  <rect x="650" y="80" width="120" height="120" rx="6" fill="#fde8e8" stroke="#da3633" stroke-width="1.5"/>
  <text x="710" y="110" text-anchor="middle" fill="#da3633" font-size="12" font-weight="bold">SE(3) Action</text>
  <text x="710" y="135" text-anchor="middle" fill="#555" font-size="9">R ∈ SO(3)</text>
  <text x="710" y="152" text-anchor="middle" fill="#555" font-size="9">t ∈ ℝ³</text>
  <text x="710" y="170" text-anchor="middle" fill="#555" font-size="9">confidence</text>

  <!-- Arrow -->
  <line x1="770" y1="140" x2="810" y2="140" stroke="#888" stroke-width="1.5" marker-end="url(#arr)"/>

  <!-- Robot -->
  <rect x="810" y="100" width="60" height="80" rx="6" fill="#f5f5f5" stroke="#888" stroke-width="1.5"/>
  <text x="840" y="145" text-anchor="middle" fill="#333" font-size="11" font-weight="bold">Robot</text>

  <!-- Bottom: Key Properties -->
  <text x="30" y="280" fill="#333" font-size="11" font-weight="bold">Key Properties:</text>
  <text x="30" y="300" fill="#555" font-size="9">1. Geodesic loss: smooth on SO(3), no discontinuities at ±π</text>
  <text x="30" y="318" fill="#555" font-size="9">2. MDN: handles multi-modal actions (reach left OR right)</text>
  <text x="30" y="336" fill="#555" font-size="9">3. Residual refinement: iterative correction, each step reduces error</text>
  <text x="30" y="354" fill="#555" font-size="9">4. SE(3) constraint: output always valid rigid transform (no invalid rotations)</text>
  <text x="30" y="372" fill="#555" font-size="9">5. Drop-in: replaces any VLA model's action head, no backbone changes</text>
</svg>'''

    with open(f"{output_dir}/fig6_architecture.svg", "w") as f:
        f.write(svg)
    print("  ✓ fig6_architecture.svg")


def fig7_per_task_breakdown(benchmark: dict, output_dir: str):
    """Figure 7: Per-task breakdown — translation vs rotation error."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Figure 7: Per-Task Error Breakdown", fontsize=14, fontweight="bold", y=1.02)

    heads = list(benchmark.keys())
    tasks = list(benchmark[heads[0]].keys())
    colors = {"Flat L2": "#e74c3c", "Quaternion": "#3498db", "GeoAct": "#2ecc71"}

    for idx, task_key in enumerate(tasks):
        ax = axes[idx // 4, idx % 4]
        task_data = benchmark[heads[0]][task_key]
        task_name = task_data["task"]

        trans_errors = []
        rot_errors = []
        labels = []

        for head in heads:
            d = benchmark[head][task_key]
            trans_errors.append(d["mean_trans_err"])
            rot_errors.append(d["mean_rot_err"])
            labels.append(head)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, trans_errors, width, label="Translation",
                       color=[colors.get(l, "#95a5a6") for l in labels], alpha=0.7)
        bars2 = ax.bar(x + width/2, rot_errors, width, label="Rotation",
                       color=[colors.get(l, "#95a5a6") for l in labels], alpha=0.4,
                       edgecolor=[colors.get(l, "#95a5a6") for l in labels], linewidth=1.5)

        ax.set_title(task_name, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{output_dir}/fig7_per_task.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  ✓ fig7_per_task.png")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
    generate_all(results_dir, output_dir)
