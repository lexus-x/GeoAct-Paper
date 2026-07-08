"""
ActionRep-Bench: Standardized benchmark tasks for action representation evaluation.

Each task tests a specific property of action representations:
1. PrecisionPick — tests translation precision
2. PrecisionInsert — tests rotation precision (tight tolerance)
3. DiscontinuitySweep — tests behavior near rotation discontinuities
4. MultiModalReach — tests handling of ambiguous actions
5. ContactRichPush — tests contact-rich manipulation (force-sensitive)
6. LongHorizonStack — tests error accumulation over long sequences
7. SE3Consistency — tests geometric consistency of predictions
8. OutOfDistribution — tests generalization to unseen rotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from geoact.so3 import exp_map, geodesic_distance, SE3


class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class TaskConfig:
    """Configuration for a benchmark task."""
    name: str
    difficulty: TaskDifficulty
    description: str
    n_episodes: int = 100
    max_steps: int = 50
    success_threshold_translation: float = 0.01  # meters
    success_threshold_rotation: float = 0.1  # radians


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    task_name: str
    success: bool
    final_translation_error: float
    final_rotation_error: float
    total_steps: int
    action_sequence: list[NDArray] = field(default_factory=list)
    error_timeline: list[float] = field(default_factory=list)


@dataclass
class TaskResult:
    """Aggregated results for a task."""
    task_name: str
    success_rate: float
    mean_translation_error: float
    mean_rotation_error: float
    std_translation_error: float
    std_rotation_error: float
    mean_steps: float
    episodes: list[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "success_rate": round(self.success_rate, 4),
            "mean_trans_err": round(self.mean_translation_error, 6),
            "mean_rot_err": round(self.mean_rotation_error, 6),
            "std_trans_err": round(self.std_translation_error, 6),
            "std_rot_err": round(self.std_rotation_error, 6),
            "mean_steps": round(self.mean_steps, 1),
        }


# --- Task Definitions ---

TASKS = {
    "precision_pick": TaskConfig(
        name="PrecisionPick",
        difficulty=TaskDifficulty.EASY,
        description="Pick and place with 1cm translation tolerance",
        success_threshold_translation=0.01,
        success_threshold_rotation=0.2,
    ),
    "precision_insert": TaskConfig(
        name="PrecisionInsert",
        difficulty=TaskDifficulty.HARD,
        description="Peg-in-hole with 1mm tolerance and 5° rotation tolerance",
        success_threshold_translation=0.001,
        success_threshold_rotation=0.087,  # 5 degrees
    ),
    "discontinuity_sweep": TaskConfig(
        name="DiscontinuitySweep",
        difficulty=TaskDifficulty.MEDIUM,
        description="Reach targets near rotation discontinuity (±π)",
        success_threshold_translation=0.02,
        success_threshold_rotation=0.15,
    ),
    "multimodal_reach": TaskConfig(
        name="MultiModalReach",
        difficulty=TaskDifficulty.MEDIUM,
        description="Reach target from ambiguous starting position (2 valid paths)",
        success_threshold_translation=0.015,
        success_threshold_rotation=0.15,
    ),
    "contact_push": TaskConfig(
        name="ContactPush",
        difficulty=TaskDifficulty.HARD,
        description="Push object along surface (contact-rich, force-sensitive)",
        success_threshold_translation=0.005,
        success_threshold_rotation=0.3,
    ),
    "long_horizon_stack": TaskConfig(
        name="LongHorizonStack",
        difficulty=TaskDifficulty.HARD,
        description="Stack 3 blocks (error accumulates over 150 steps)",
        success_threshold_translation=0.01,
        success_threshold_rotation=0.1,
    ),
    "se3_consistency": TaskConfig(
        name="SE3Consistency",
        difficulty=TaskDifficulty.EASY,
        description="Consistent predictions for equivalent SE(3) transforms",
        success_threshold_translation=0.005,
        success_threshold_rotation=0.05,
    ),
    "ood_rotation": TaskConfig(
        name="OODRotation",
        difficulty=TaskDifficulty.MEDIUM,
        description="Generalize to rotations outside training distribution",
        success_threshold_translation=0.02,
        success_threshold_rotation=0.2,
    ),
}


class TaskGenerator:
    """Generates benchmark episodes for a given task."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_episode(self, task_key: str) -> tuple[NDArray, NDArray]:
        """
        Generate a (start_pose, target_pose) pair for a task.

        Returns:
            start: (6,) [ω, t]
            target: (6,) [ω, t]
        """
        if task_key == "precision_pick":
            return self._precision_pick()
        elif task_key == "precision_insert":
            return self._precision_insert()
        elif task_key == "discontinuity_sweep":
            return self._discontinuity_sweep()
        elif task_key == "multimodal_reach":
            return self._multimodal_reach()
        elif task_key == "contact_push":
            return self._contact_push()
        elif task_key == "long_horizon_stack":
            return self._long_horizon_stack()
        elif task_key == "se3_consistency":
            return self._se3_consistency()
        elif task_key == "ood_rotation":
            return self._ood_rotation()
        else:
            raise ValueError(f"Unknown task: {task_key}")

    def _precision_pick(self) -> tuple[NDArray, NDArray]:
        start = np.concatenate([self.rng.randn(3) * 0.3, self.rng.randn(3) * 0.2])
        target = np.concatenate([self.rng.randn(3) * 0.1, np.array([0.3, 0.0, 0.2])])
        return start, target

    def _precision_insert(self) -> tuple[NDArray, NDArray]:
        start = np.concatenate([self.rng.randn(3) * 0.5, self.rng.randn(3) * 0.1])
        target = np.concatenate([np.array([0.0, 0.0, 0.5]), np.array([0.3, 0.0, 0.1])])
        return start, target

    def _discontinuity_sweep(self) -> tuple[NDArray, NDArray]:
        # Target near rotation discontinuity
        angle = np.pi + self.rng.randn() * 0.2
        target = np.concatenate([np.array([0, 0, angle]), self.rng.randn(3) * 0.1])
        start = np.concatenate([self.rng.randn(3) * 0.3, self.rng.randn(3) * 0.2])
        return start, target

    def _multimodal_reach(self) -> tuple[NDArray, NDArray]:
        # Two valid paths: reach left or reach right
        start = np.zeros(6)
        if self.rng.random() > 0.5:
            target = np.concatenate([np.array([0, 0, 0.3]), np.array([0.2, 0.2, 0])])
        else:
            target = np.concatenate([np.array([0, 0, -0.3]), np.array([0.2, -0.2, 0])])
        return start, target

    def _contact_push(self) -> tuple[NDArray, NDArray]:
        start = np.concatenate([self.rng.randn(3) * 0.1, np.array([0.1, 0.1, 0.05])])
        target = np.concatenate([np.zeros(3), np.array([0.4, 0.0, 0.05])])
        return start, target

    def _long_horizon_stack(self) -> tuple[NDArray, NDArray]:
        start = np.concatenate([self.rng.randn(3) * 0.2, self.rng.randn(3) * 0.3])
        target = np.concatenate([np.zeros(3), np.array([0.3, 0.0, 0.3])])
        return start, target

    def _se3_consistency(self) -> tuple[NDArray, NDArray]:
        # Same rotation, different representation (should give same result)
        omega = self.rng.randn(3) * 0.5
        start = np.concatenate([omega, self.rng.randn(3) * 0.1])
        target = np.concatenate([omega + self.rng.randn(3) * 0.01, self.rng.randn(3) * 0.1])
        return start, target

    def _ood_rotation(self) -> tuple[NDArray, NDArray]:
        # Rotations outside typical training range
        angle = self.rng.uniform(np.pi * 0.8, np.pi * 1.2)
        axis = self.rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        target = np.concatenate([axis * angle, self.rng.randn(3) * 0.2])
        start = np.concatenate([self.rng.randn(3) * 0.5, self.rng.randn(3) * 0.3])
        return start, target
