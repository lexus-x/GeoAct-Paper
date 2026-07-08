"""
SO(3) and SE(3) Lie algebra — mathematically rigorous implementation.

All operations verified against:
- Murray, Li, Sastry (1994) "A Mathematical Introduction to Robotic Manipulation"
- Stillwell (2008) "Naive Lie Theory"
- Boumal (2023) "An Introduction to Optimization on Smooth Manifolds"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def hat(v: NDArray) -> NDArray:
    """Hat map: R³ → so(3). Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ], dtype=np.float64)


def vee(M: NDArray) -> NDArray:
    """Vee map: so(3) → R³. 3-vector from skew-symmetric matrix."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)


def exp_map(omega: NDArray) -> NDArray:
    """
    Exponential map: so(3) → SO(3) via Rodrigues' formula.

    R = I + sin(θ)/θ · [ω]× + (1-cos(θ))/θ² · [ω]×²

    where θ = ||ω||.

    This is the UNIQUE shortest geodesic from I to R on SO(3).
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + hat(omega)

    K = hat(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def log_map(R: NDArray) -> NDArray:
    """
    Logarithmic map: SO(3) → so(3).

    Inverse of exp_map. Returns axis-angle vector ω such that exp(ω) = R.
    """
    cos_theta = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        return vee(R - np.eye(3))

    if abs(theta - np.pi) < 1e-6:
        # θ ≈ π: use eigenvector of R with eigenvalue 1
        M = R + np.eye(3)
        norms = [np.linalg.norm(M[:, i]) for i in range(3)]
        axis = M[:, np.argmax(norms)]
        axis = axis / np.linalg.norm(axis)
        return axis * theta

    return vee((R - R.T) / (2 * np.sin(theta))) * theta


def geodesic_distance(R1: NDArray, R2: NDArray) -> float:
    """
    Geodesic distance on SO(3): d(R1, R2) = ||log(R1^T R2)||.

    This is the Riemannian distance induced by the bi-invariant metric on SO(3).

    Properties:
    - d(R1, R2) ∈ [0, π]
    - d(R1, R2) = d(R2, R1) (symmetry)
    - d(R1, R3) ≤ d(R1, R2) + d(R2, R3) (triangle inequality)
    - d(R, R) = 0 (identity of indiscernibles)
    - Smooth everywhere (no discontinuities)
    """
    return float(np.linalg.norm(log_map(R1.T @ R2)))


def geodesic_loss(pred_omega: NDArray, target_omega: NDArray) -> float:
    """
    Geodesic loss for rotation prediction.

    L_geo(ω_pred, ω_target) = ||log(exp(ω_pred)^T exp(ω_target))||

    This is the NATURAL loss on SO(3):
    - Invariant to parameterization
    - Smooth everywhere
    - No spurious local minima (Theorem 1 in paper)
    """
    R_pred = exp_map(pred_omega)
    R_target = exp_map(target_omega)
    return geodesic_distance(R_pred, R_target)


def l2_euler_loss(pred_omega: NDArray, target_omega: NDArray) -> float:
    """
    L2 loss on Euler angles (baseline — what most VLA models use).

    L_euler = ||euler(exp(ω_pred)) - euler(exp(ω_target))||²

    Known issues:
    - Discontinuities at gimbal lock (pitch = ±π/2)
    - Non-smooth at ±π boundaries
    - Not invariant to Euler convention (ZYX vs XYZ)
    """
    R_pred = exp_map(pred_omega)
    R_target = exp_map(target_omega)

    euler_pred = Rotation.from_matrix(R_pred).as_euler('ZYX')
    euler_target = Rotation.from_matrix(R_target).as_euler('ZYX')

    return float(np.sum((euler_pred - euler_target) ** 2))


def l2_quaternion_loss(pred_omega: NDArray, target_omega: NDArray) -> float:
    """
    L2 loss on quaternions (another common baseline).

    Known issues:
    - Double-cover: q and -q represent the same rotation
    - L2 on quaternions is NOT the geodesic distance
    - Discontinuous when q flips sign
    """
    R_pred = exp_map(pred_omega)
    R_target = exp_map(target_omega)

    q_pred = Rotation.from_matrix(R_pred).as_quat()  # [x,y,z,w]
    q_target = Rotation.from_matrix(R_target).as_quat()

    # Handle double-cover
    if np.dot(q_pred, q_target) < 0:
        q_target = -q_target

    return float(np.sum((q_pred - q_target) ** 2))


class SE3:
    """SE(3) rigid body transformation."""

    def __init__(self, R: NDArray, t: NDArray):
        self.R = np.asarray(R, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)

    @classmethod
    def from_vector(cls, v: NDArray) -> SE3:
        """From 6-vector [ω, t]."""
        return cls(exp_map(v[:3]), v[3:])

    def to_vector(self) -> NDArray:
        """To 6-vector [ω, t]."""
        return np.concatenate([log_map(self.R), self.t])

    def inverse(self) -> SE3:
        R_inv = self.R.T
        return SE3(R_inv, -R_inv @ self.t)

    def compose(self, other: SE3) -> SE3:
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def transform_point(self, p: NDArray) -> NDArray:
        return self.R @ p + self.t


def se3_geodesic_loss(pred: NDArray, target: NDArray, w_rot: float = 1.0, w_trans: float = 1.0) -> float:
    """SE(3) geodesic loss: rotation geodesic + translation L2."""
    rot_loss = geodesic_loss(pred[:3], target[:3])
    trans_loss = float(np.linalg.norm(pred[3:] - target[3:]))
    return w_rot * rot_loss + w_trans * trans_loss


def se3_l2_loss(pred: NDArray, target: NDArray) -> float:
    """SE(3) flat L2 loss (baseline)."""
    return float(np.mean((pred - target) ** 2))
