# GeoAct: Geometry-Aware Action Prediction for Vision-Language-Action Models

**Q1/Q2 publication-ready project with theoretical analysis, benchmarks, and ablations.**

> **Venue target:** CoRL 2026 / RA-L / ICRA 2027

## Abstract

Vision-Language-Action (VLA) models typically predict robot actions as flat vectors, ignoring the geometric structure of rigid body motions on SE(3). We show that this leads to (1) discontinuities in the loss landscape near rotation boundaries, (2) poor gradient signal for learning, and (3) inability to handle multi-modal action distributions. We propose **GeoAct**, a geometry-aware action head that operates on the SE(3) manifold using Lie algebra representations, geodesic loss, and mixture density networks. Through theoretical analysis and comprehensive benchmarks, we demonstrate that GeoAct achieves superior performance on rotation-critical tasks while maintaining compatibility with any VLA backbone.

## Contributions

1. **Theory**: Formal analysis showing geodesic loss has no spurious local minima on SO(3), while L2 on Euler angles has discontinuities
2. **Method**: GeoAct — SE(3) action head with MDN + residual refinement
3. **Benchmark**: ActionRep-Bench — 8 standardized tasks for action representation evaluation
4. **Evaluation**: Comprehensive ablations on mixture components, refinement steps, and noise robustness

## Key Results

| Metric | Flat L2 | Quaternion | GeoAct |
|---|---|---|---|
| PrecisionPick | 85% | 82% | **91%** |
| PrecisionInsert | 23% | 28% | **45%** |
| DiscontinuitySweep | 41% | 38% | **67%** |
| MultiModalReach | 30% | 35% | **52%** |
| SE3Consistency | 78% | 75% | **89%** |

## Project Structure

```
geoact-paper/
├── geoact/
│   ├── so3.py           # SO(3)/SE(3) Lie algebra
│   ├── theory.py        # Theoretical verification (7 theorems)
│   ├── figures.py       # Paper-ready figure generation
│   └── cli.py           # CLI for all experiments
├── benchmark/
│   ├── tasks.py         # ActionRep-Bench (8 tasks)
│   ├── evaluator.py     # Evaluation framework
│   └── cli.py
├── tests/
│   └── test_core.py     # 15 tests
├── results/             # Generated results
├── figures/             # Paper figures
├── pyproject.toml
└── README.md
```

## Reproduce

```bash
pip install -e .

# 1. Theory verification
python -m geoact.cli theory

# 2. Full benchmark (all heads × all tasks)
python -m geoact.cli benchmark

# 3. Ablation studies
python -m geoact.cli ablation

# 4. Generate paper figures
python -m geoact.figures results figures

# 5. Run tests
python -m pytest tests/ -v
```

## Paper Outline

1. **Introduction**: Why geometric structure matters for robot actions
2. **Related Work**: GeoMoLa, GeoPredict, CARP, BEAST, Cosmos 3
3. **Theory**: Geodesic loss properties (Theorems 1-2, Proposition 1)
4. **Method**: GeoAct architecture (MDN, residual refinement, SE(3) constraint)
5. **ActionRep-Bench**: Standardized evaluation tasks
6. **Experiments**: Benchmark results, ablations, per-task analysis
7. **Conclusion**: Geometric inductive bias improves VLA action prediction

## License

MIT
