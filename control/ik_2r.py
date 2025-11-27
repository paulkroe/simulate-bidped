# control/ik_2r.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Planar2RLegConfig:
    L1: float
    L2: float
    knee_sign: float = 1.0

    hip_offset: float = 0.0
    knee_offset: float = 0.0
    ankle_offset: float = 0.0
 


class Planar2RLegIK:
    def __init__(self, cfg: Planar2RLegConfig):
        self.cfg = cfg

    def solve(
        self,
        target_x: float,
        target_z: float,
        enforce_reachability: bool = True,
        compute_ankle: bool = True,
        desired_foot_angle: float = 0.0,
    ) -> Tuple[float, float, float | None]:
        """
        Compute (hip_angle, knee_angle, ankle_angle) for a given ankle target (x,z).

        Args:
            target_x, target_z: desired ankle position in hip frame.
            enforce_reachability: if True, clamp target to reachable circle.
            compute_ankle: if True, compute ankle angle so foot is parallel to ground.
            desired_foot_angle: target foot pitch in world/hip frame.
                               0.0 -> parallel to ground.

        Returns:
            hip_angle, knee_angle, ankle_angle (or None if compute_ankle=False)
        """
        L1 = self.cfg.L1
        L2 = self.cfg.L2
        knee_sign = self.cfg.knee_sign

        x = float(target_x)
        z = float(target_z)

        r = np.sqrt(x * x + z * z)

        # Clamp to reachable workspace if requested
        max_r = L1 + L2
        min_r = abs(L1 - L2)

        if enforce_reachability:
            # If the target is too close, push it out a bit
            if r < min_r:
                if r > 1e-6:
                    scale = min_r / r
                    x *= scale
                    z *= scale
                    r = min_r
                else:
                    # target is basically at the hip; just point leg straight down
                    x = 0.0
                    z = -(L1 + L2) * 0.99
                    r = np.sqrt(x * x + z * z)

            # If the target is too far, clamp
            if r > max_r:
                scale = max_r / r
                x *= scale
                z *= scale
                r = max_r

        # Law of cosines for knee
        # cos(theta2) = (r^2 - L1^2 - L2^2) / (2 L1 L2)
        cos_knee = (r * r - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        sin_knee = knee_sign * np.sqrt(max(0.0, 1.0 - cos_knee * cos_knee))
        knee_angle = np.arctan2(sin_knee, cos_knee)

        # Hip angle:
        # theta1 = atan2(z, x) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))
        hip_to_target = np.arctan2(z, x)
        hip_correction = np.arctan2(L2 * sin_knee, L1 + L2 * cos_knee)
        hip_angle = hip_to_target - hip_correction

        if not compute_ankle:
            # convert to joint frame with offsets
            hip_joint = hip_angle + self.cfg.hip_offset
            knee_joint = knee_angle + self.cfg.knee_offset
            return float(hip_joint), float(knee_joint), None

        ankle_angle_math = (
            desired_foot_angle
            - (hip_angle + knee_angle)
        )

        ankle_joint = ankle_angle_math + self.cfg.ankle_offset

        hip_joint = hip_angle + self.cfg.hip_offset
        knee_joint = knee_angle + self.cfg.knee_offset

        return float(hip_joint), float(knee_joint), float(ankle_joint)