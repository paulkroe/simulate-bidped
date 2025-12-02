# tasks/biped/reward_forward_displacement.py
"""
Reward that REQUIRES forward displacement.

The robot can't get high reward by staying in place,
even if it's making walking motions.
"""
import mujoco
import numpy as np
from typing import Tuple, Dict


class ForwardDisplacementReward:
    """Track actual position over time to require movement."""
    
    def __init__(self):
        self.initial_x = None
        self.last_x = None
        self.episode_step = 0
        
    def reset(self):
        """Call this at episode start."""
        self.initial_x = None
        self.last_x = None
        self.episode_step = 0
        
    def __call__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        t: int,
        dt: float,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Reward actual forward displacement."""
        
        self.episode_step += 1
        
        # Get state
        torso_id = model.body("hips").id
        torso_pos = data.xpos[torso_id]
        torso_quat = data.xquat[torso_id]
        cvel = data.cvel[torso_id]
        
        torso_x = float(torso_pos[0])
        v_forward = float(cvel[3])
        v_lateral = float(cvel[4])
        height = float(torso_pos[2])
        
        quat = torso_quat
        pitch = 2.0 * quat[2]
        tilt = np.sqrt(quat[1]**2 + quat[2]**2)
        
        # Initialize tracking
        if self.initial_x is None:
            self.initial_x = torso_x
            self.last_x = torso_x
        
        # ================================================================
        # CRITICAL: FORWARD DISPLACEMENT REWARD
        # ================================================================
        
        # Total distance traveled from start
        total_distance = torso_x - self.initial_x
        
        # Distance since last step
        step_distance = torso_x - self.last_x
        self.last_x = torso_x
        
        # Reward based on TOTAL distance traveled
        # This prevents "move back and forth" exploits
        if total_distance > 0.5:
            displacement_reward = 5.0  # Huge reward for 0.5m
        elif total_distance > 0.3:
            displacement_reward = 3.0  # Good progress
        elif total_distance > 0.1:
            displacement_reward = 1.5  # Some progress
        else:
            displacement_reward = -1.0  # Not moving forward
        
        # ALSO reward instantaneous forward velocity
        # (encourages continued motion)
        if v_forward > 0.2:
            velocity_reward = 1.0
        elif v_forward > 0.1:
            velocity_reward = 0.5
        else:
            velocity_reward = -0.5
        
        # Penalty for moving backward
        if step_distance < -0.01:
            backward_penalty = 2.0 * abs(step_distance)
        else:
            backward_penalty = 0.0
        
        # ================================================================
        # STABILITY (secondary to forward motion)
        # ================================================================
        
        upright_reward = 1.0 * np.exp(-15.0 * tilt**2)
        
        target_height = 0.15
        height_reward = 0.8 * np.exp(-30.0 * (height - target_height)**2)
        
        # ================================================================
        # PENALTIES
        # ================================================================
        
        pitch_penalty = 1.0 * pitch**2
        lateral_penalty = 0.5 * v_lateral**2
        action_penalty = 0.005 * np.sum(action**2)
        
        alive_bonus = 0.1
        
        # ================================================================
        # TOTAL
        # ================================================================
        
        total = (
            displacement_reward      # MOST IMPORTANT
            + velocity_reward        # SECOND MOST IMPORTANT
            + upright_reward
            + height_reward
            + alive_bonus
            - backward_penalty
            - pitch_penalty
            - lateral_penalty
            - action_penalty
        )
        
        components = {
            "displacement": displacement_reward,
            "velocity": velocity_reward,
            "upright": upright_reward,
            "height": height_reward,
            "alive": alive_bonus,
            "backward_penalty": -backward_penalty,
            "pitch_penalty": -pitch_penalty,
            "lateral_cost": -lateral_penalty,
            "action_cost": -action_penalty,
            # Diagnostics
            "total_distance": total_distance,
            "step_distance": step_distance,
            "v_forward": v_forward,
            "height_actual": height,
            "episode_step": float(self.episode_step),
        }
        
        return float(total), components


# Create global instance
_displacement_reward = ForwardDisplacementReward()

def reward(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    t: int,
    dt: float,
    action: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Wrapper function."""
    # Reset tracking on episode start
    if t == 0:
        _displacement_reward.reset()
    return _displacement_reward(model, data, t, dt, action)