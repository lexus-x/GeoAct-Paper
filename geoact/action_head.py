"""
GeoAct — Geometry-Aware Action Head for VLA Models.

A drop-in replacement for any VLA model's action head that:
1. Represents actions on the SE(3) manifold (not flat vectors)
2. Uses geodesic loss for rotation learning (not L2 on Euler angles)
3. Predicts multi-modal action distributions (handles ambiguity)
4. Iteratively refines predictions (residual geometric correction)
5. Works with any VLA backbone (OpenVLA, Octo, pi0, etc.)

Architecture:
    VLA features → Projection → SE(3) Mixture Density Network → Action

    Where:
    - Projection: maps VLA feature dim → hidden dim
    - SE(3) MDN: predicts K mixture components, each with:
        - Translation: μ_t ∈ R³, σ_t ∈ R³ (Gaussian)
        - Rotation: μ_ω ∈ so(3) (axis-angle), κ ∈ R (von Mises-Fisher concentration)
        - Mixing weights: π ∈ R^K
    - Residual refinement: iteratively correct the prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .so3 import (
    exp_map_so3, log_map_so3, geodesic_distance_so3,
    SE3, se3_geodesic_loss,
)


@dataclass
class ActionPrediction:
    """A single predicted action from GeoAct."""
    translation: NDArray      # (3,) end-effector translation
    rotation_axis_angle: NDArray  # (3,) axis-angle rotation
    rotation_matrix: NDArray  # (3,3) rotation matrix
    confidence: float         # mixture component confidence
    mixture_id: int           # which mixture component
    se3: SE3                  # full SE(3) pose

    def to_vector(self) -> NDArray:
        """Convert to 6-vector [ω, t]."""
        return np.concatenate([self.rotation_axis_angle, self.translation])

    def to_euler(self) -> NDArray:
        """Convert to [roll, pitch, yaw, x, y, z]."""
        from .so3 import rotation_matrix_to_euler
        euler = rotation_matrix_to_euler(self.rotation_matrix)
        return np.concatenate([euler, self.translation])


@dataclass
class MixtureComponent:
    """A single component in the SE(3) mixture density network."""
    # Translation parameters (Gaussian)
    mu_translation: NDArray   # (3,) mean
    sigma_translation: NDArray  # (3,) std dev

    # Rotation parameters (von Mises-Fisher on SO(3))
    mu_rotation: NDArray      # (3,) axis-angle mean
    kappa: float              # concentration parameter (higher = more certain)

    # Mixing weight
    weight: float             # π_k


@dataclass
class GeoActConfig:
    """Configuration for the GeoAct action head."""
    # Input dimension from VLA backbone
    feature_dim: int = 768

    # Hidden dimension
    hidden_dim: int = 256

    # Number of mixture components (for multi-modal prediction)
    n_components: int = 5

    # Number of residual refinement iterations
    n_refine_steps: int = 3

    # Action dimension (6 for SE(3), 7 for SE(3)+gripper)
    action_dim: int = 7

    # Loss weights
    w_rotation: float = 1.0
    w_translation: float = 1.0
    w_gripper: float = 0.5

    # Multi-modal: weight for diversity loss (prevents mode collapse)
    w_diversity: float = 0.1


class GeoActHead:
    """
    Geometry-Aware Action Head — drop-in replacement for VLA action heads.

    Usage:
        head = GeoActHead(config)
        actions = head.predict(features)  # multi-modal predictions
        best_action = head.select_best(actions)  # highest confidence
        loss = head.compute_loss(features, target_action)  # for training
    """

    def __init__(self, config: GeoActConfig | None = None):
        self.config = config or GeoActConfig()

        # Projection weights (simulated — in practice these are learned)
        self.projection = self._init_projection()

        # Mixture component parameters (simulated)
        self.components = self._init_components()

        # Refinement weights
        self.refine_weights = self._init_refine()

    def _init_projection(self) -> NDArray:
        """Initialize projection matrix (feature_dim → hidden_dim)."""
        # Xavier initialization
        scale = np.sqrt(2.0 / (self.config.feature_dim + self.config.hidden_dim))
        return np.random.randn(self.config.feature_dim, self.config.hidden_dim).astype(np.float32) * scale

    def _init_components(self) -> list[MixtureComponent]:
        """Initialize mixture components with diverse starting points."""
        components = []
        for k in range(self.config.n_components):
            # Spread rotation means around the identity
            angle = 2 * np.pi * k / self.config.n_components
            mu_rot = np.array([0, 0, angle * 0.1], dtype=np.float64)

            components.append(MixtureComponent(
                mu_translation=np.random.randn(3).astype(np.float64) * 0.01,
                sigma_translation=np.ones(3, dtype=np.float64) * 0.1,
                mu_rotation=mu_rot,
                kappa=10.0,  # moderate concentration
                weight=1.0 / self.config.n_components,
            ))
        return components

    def _init_refine(self) -> NDArray:
        """Initialize refinement weights."""
        scale = np.sqrt(2.0 / self.config.hidden_dim)
        return np.random.randn(self.config.hidden_dim, 6).astype(np.float32) * scale

    def predict(self, features: NDArray, n_samples: int = 1) -> list[ActionPrediction]:
        """
        Predict actions from VLA features.

        Args:
            features: (feature_dim,) or (batch, feature_dim) VLA features
            n_samples: number of samples to draw from the mixture

        Returns:
            List of ActionPrediction objects (one per sample)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Project features to hidden space
        hidden = features @ self.projection  # (batch, hidden_dim)

        # Compute mixture weights from features (softmax)
        logits = hidden @ np.random.randn(self.config.hidden_dim, self.config.n_components).astype(np.float32) * 0.1
        weights = self._softmax(logits[0])

        # Update component weights
        for k, comp in enumerate(self.components):
            comp.weight = float(weights[k])

        # Sample from mixture
        predictions = []
        for _ in range(n_samples):
            # Select component
            comp_idx = np.random.choice(self.config.n_components, p=weights)
            comp = self.components[comp_idx]

            # Sample translation (Gaussian)
            translation = comp.mu_translation + np.random.randn(3) * comp.sigma_translation

            # Sample rotation (von Mises-Fisher approximation)
            # For high κ, sample from Gaussian around μ
            rotation_noise = np.random.randn(3) / np.sqrt(max(comp.kappa, 1.0))
            rotation = comp.mu_rotation + rotation_noise

            # Residual refinement
            for step in range(self.config.n_refine_steps):
                correction = hidden[0] @ self.refine_weights * 0.01
                translation += correction[:3] * 0.1
                rotation += correction[3:6] * 0.01

            # Build SE(3) pose
            R = exp_map_so3(rotation)
            se3 = SE3(R, translation)

            predictions.append(ActionPrediction(
                translation=translation,
                rotation_axis_angle=rotation,
                rotation_matrix=R,
                confidence=float(weights[comp_idx]),
                mixture_id=comp_idx,
                se3=se3,
            ))

        return predictions

    def select_best(self, predictions: list[ActionPrediction]) -> ActionPrediction:
        """Select the highest-confidence prediction."""
        return max(predictions, key=lambda p: p.confidence)

    def compute_loss(
        self,
        features: NDArray,
        target_action: NDArray,
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """
        Compute geodesic loss for training.

        Args:
            features: VLA features
            target_action: (6,) or (7,) target action [ω, t] or [ω, t, gripper]
            return_details: if True, return per-component losses

        Returns:
            Total loss (and optionally per-component details)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        hidden = features @ self.projection

        # Split target
        target_rot = target_action[:3]
        target_trans = target_action[3:6]
        target_gripper = target_action[6] if len(target_action) > 6 else 0.0

        # Compute loss for each mixture component
        component_losses = []
        for k, comp in enumerate(self.components):
            # Rotation loss: geodesic distance on SO(3)
            R_pred = exp_map_so3(comp.mu_rotation)
            R_target = exp_map_so3(target_rot)
            rot_loss = geodesic_distance_so3(R_pred, R_target)

            # Translation loss: L2
            trans_loss = float(np.linalg.norm(comp.mu_translation - target_trans))

            # Combined
            total_loss = (self.config.w_rotation * rot_loss +
                         self.config.w_translation * trans_loss)

            component_losses.append(total_loss)

        # Mixture loss: negative log-likelihood (select best component)
        weights = np.array([c.weight for c in self.components])
        nll_losses = np.array(component_losses) - np.log(weights + 1e-10)
        best_comp_loss = float(np.min(nll_losses))

        # Diversity loss: encourage components to cover different modes
        diversity_loss = 0.0
        for i in range(len(self.components)):
            for j in range(i + 1, len(self.components)):
                rot_dist = geodesic_distance_so3(
                    exp_map_so3(self.components[i].mu_rotation),
                    exp_map_so3(self.components[j].mu_rotation),
                )
                trans_dist = np.linalg.norm(
                    self.components[i].mu_translation - self.components[j].mu_translation
                )
                # Penalize components that are too close
                diversity_loss += max(0, 0.1 - rot_dist) + max(0, 0.01 - trans_dist)

        total = best_comp_loss + self.config.w_diversity * diversity_loss

        if return_details:
            return total, {
                "best_component_loss": best_comp_loss,
                "diversity_loss": diversity_loss,
                "component_losses": component_losses,
                "rotation_loss": float(np.mean([
                    geodesic_distance_so3(exp_map_so3(c.mu_rotation), exp_map_so3(target_rot))
                    for c in self.components
                ])),
                "translation_loss": float(np.mean([
                    np.linalg.norm(c.mu_translation - target_trans)
                    for c in self.components
                ])),
            }

        return total

    def _softmax(self, x: NDArray) -> NDArray:
        """Numerically stable softmax."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


class FlatActionHead:
    """
    Baseline: flat MLP action head (what most VLA models use).
    Used for comparison with GeoAct.
    """

    def __init__(self, feature_dim: int = 768, action_dim: int = 7):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        scale = np.sqrt(2.0 / (feature_dim + action_dim))
        self.weights = np.random.randn(feature_dim, action_dim).astype(np.float32) * scale
        self.bias = np.zeros(action_dim, dtype=np.float32)

    def predict(self, features: NDArray) -> NDArray:
        """Predict flat action vector."""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return (features @ self.weights + self.bias)[0]

    def compute_loss(self, features: NDArray, target: NDArray) -> float:
        """L2 loss (the standard loss used by most VLA models)."""
        pred = self.predict(features)
        return float(np.mean((pred - target) ** 2))
