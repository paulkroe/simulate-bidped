# policies/reference_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from control.ik_2r import Planar2RLegConfig, Planar2RLegIK
from control.pd import PDConfig, PDController

@dataclass
class GaitParams:
    step_length: float       # like stepLength (front/back x)
    step_height: float       # like stepHeight (front leg z)
    step_clearance: float    # like stepClearance (difference in z)
    cycle_duration: float    # seconds per full step (R-front + L-front)


@dataclass
class FootTarget:
    pos: np.ndarray   # (x, z) in hip frame
    foot_angle: float # desired pitch (we can keep it 0 for now)

def gait_targets(t: float, params: GaitParams) -> Dict[str, FootTarget]:
    """
    Arduino-style gait:

      For 0 <= phase < 0.5:
        i goes from +step_length to -step_length
        right:  ( i, step_height)
        left:   (-i, step_height - step_clearance)

      For 0.5 <= phase < 1.0:
        i goes from +step_length to -step_length
        right:  (-i, step_height - step_clearance)
        left:   ( i, step_height)
    """
    phase = (t / params.cycle_duration) % 1.0

    L = params.step_length
    H = params.step_height
    C = params.step_clearance

    if phase < 0.5:
        # first "for loop" in Arduino
        s = phase / 0.5           # map [0,0.5) -> [0,1)
        i = L + ( -2 * L * s )    # i: +L -> -L

        right_pos = np.array([ i,  H], dtype=np.float32)
        left_pos  = np.array([-i,  H - C], dtype=np.float32)
    else:
        # second "for loop" in Arduino
        s = (phase - 0.5) / 0.5   # map [0.5,1) -> [0,1)
        i = L + ( -2 * L * s )    # i: +L -> -L

        right_pos = np.array([-i,  H - C], dtype=np.float32)
        left_pos  = np.array([ i,  H], dtype=np.float32)

    # For now keep feet roughly parallel to ground
    stance_angle = 0.0
    swing_angle = 0.0

    # Determine which is swing vs stance by z (just for future tweaks)
    if right_pos[1] > left_pos[1]:
        right_angle = swing_angle
        left_angle = stance_angle
    else:
        right_angle = stance_angle
        left_angle = swing_angle

    targets: Dict[str, FootTarget] = {
        "right": FootTarget(pos=right_pos, foot_angle=right_angle),
        "left":  FootTarget(pos=left_pos,  foot_angle=left_angle),
    }
    return targets

# ------------------------------------------------------------
# Reference walker policy (hand-crafted gait + IK + PD)
# ------------------------------------------------------------

@dataclass
class LegJointIndices:
    """
    Indices of the leg joints in the full q/qdot/ctrl/action vectors.
    These are robot-specific and MUST be adapted to your biped.xml.
    """
    hip: int
    knee: int
    ankle: int


@dataclass
class WalkerJointMap:
    left: LegJointIndices
    right: LegJointIndices


class ReferenceWalkerPolicy(nn.Module):
    """
    Hand-crafted baseline controller:

      gait (foot targets) -> 2R leg IK -> desired joint angles -> PD -> torques.

    The interface matches ActorCritic.act: it returns (action_tensor, logp_tensor),
    but logp is just zeros since this is not stochastic.
    """
class ReferenceWalkerPolicy(nn.Module):
    def __init__(
        self,
        env,  # MujocoEnv instance so we can get dt, act_dim, etc.
        gait_params: GaitParams,
        joint_map: WalkerJointMap,
        left_leg_geom: Planar2RLegConfig,
        right_leg_geom: Planar2RLegConfig,
        pd_config: PDConfig,
        desired_foot_angle: float = 0.0,
    ):
        super().__init__()

        self.env = env
        self.gait_params = gait_params
        self.joint_map = joint_map
        self.desired_foot_angle = desired_foot_angle

        # Simulation timestep (approx): dt = timestep * frame_skip
        dt = env.model.opt.timestep * env.cfg.frame_skip
        self.dt = float(dt)
        self._time = 0.0

        # IK for each leg
        self.ik_left = Planar2RLegIK(left_leg_geom)
        self.ik_right = Planar2RLegIK(right_leg_geom)

        # PD controller: one PD over all controlled joints
        self.pd = PDController(pd_config)

        # Action dimension (number of actuators)
        self.act_dim = env.spec.act.shape[0]

        # ------------------------------------------------------------------
        # Joint index bookkeeping
        # ------------------------------------------------------------------
        # Joint indices in q (qpos) space that we want to control
        jmL = self.joint_map.left
        jmR = self.joint_map.right

        # Vector of q indices for all controlled joints (left + right leg)
        self._q_indices = np.array(
            [
                jmL.hip, jmL.knee, jmL.ankle,
                jmR.hip, jmR.knee, jmR.ankle,
            ],
            dtype=int,
        )
        
        # For MuJoCo models with a floating base, nq != nv.
        # Typically: nq = 7 + n_hinge, nv = 6 + n_hinge → offset = nq - nv = 1.
        # For non-base joints: qd_index = q_index - offset
        model = self.env.model
        nq = model.nq
        nv = model.nv
        offset = nq - nv

        self._qd_indices = self._q_indices - offset

        # ------------------------------------------------------------------
        # Build mapping from q_idx -> actuator index
        # ------------------------------------------------------------------
        # joint_id -> q_idx (from MuJoCo)
        jnt_to_q: dict[int, int] = {}
        for j in range(model.njnt):
            q_idx = int(model.jnt_qposadr[j])
            if q_idx >= 0:
                jnt_to_q[j] = q_idx

        # q_idx -> actuator index: uses actuator_trnid[a, 0] = joint_id
        self._act_for_q: dict[int, int] = {}
        for a in range(model.nu):
            j_id = int(model.actuator_trnid[a, 0])
            if j_id in jnt_to_q:
                q_idx = jnt_to_q[j_id]
                self._act_for_q[q_idx] = a

    def reset(self):
        """Reset internal episode state (e.g. gait phase)."""
        self._time = 0.0

    def forward(self, *args, **kwargs):
        # We don't use forward; we use act() like ActorCritic.
        raise NotImplementedError

    def act(self, obs: torch.Tensor, deterministic: bool = True):
        obs_np = obs.detach().cpu().numpy().squeeze(0)
        nq = self.env.model.nq
        nv = self.env.model.nv

        q = obs_np[:nq]
        qd = obs_np[nq:nq + nv]

        t = self._time
        self._time += self.dt

        # 1) Arduino-style gait targets
        targets = gait_targets(t, self.gait_params)
        left_target = targets["left"]
        right_target = targets["right"]

        # 2) IK per leg
        hip_L_des, knee_L_des, ankle_L_des = self.ik_left.solve(
            target_x=float(left_target.pos[0]),
            target_z=float(left_target.pos[1]),
            compute_ankle=True,
            desired_foot_angle=float(left_target.foot_angle),
        )
        hip_R_des, knee_R_des, ankle_R_des = self.ik_right.solve(
            target_x=float(right_target.pos[0]),
            target_z=float(right_target.pos[1]),
            compute_ankle=True,
            desired_foot_angle=float(right_target.foot_angle),
        )

        # 3) Desired joint positions in full q vector
        q_des = q.copy()
        jmL = self.joint_map.left
        jmR = self.joint_map.right

        q_des[jmL.hip] = hip_L_des
        q_des[jmL.knee] = knee_L_des
        q_des[jmL.ankle] = ankle_L_des

        q_des[jmR.hip] = hip_R_des
        q_des[jmR.knee] = knee_R_des
        q_des[jmR.ankle] = ankle_R_des

        # 4) Extract controlled joints only
        q_ctrl = q[self._q_indices]           # shape (n_ctrl,)
        qd_ctrl = qd[self._qd_indices]        # shape (n_ctrl,)
        q_des_ctrl = q_des[self._q_indices]   # shape (n_ctrl,)

         # 5) PD control in joint space
        tau_ctrl = self.pd.compute(
            q=q_ctrl,
            qd=qd_ctrl,
            q_des=q_des_ctrl,
            qd_des=None,
        )  # shape (n_ctrl,)

        # 6) Scatter torques into full action vector
        action = np.zeros(self.act_dim, dtype=np.float32)

        for q_idx, tau in zip(self._q_indices, tau_ctrl):
            act_idx = self._act_for_q.get(int(q_idx), None)
            if act_idx is not None and 0 <= act_idx < self.act_dim:
                action[act_idx] = tau

        action_t = torch.as_tensor(
            action, dtype=obs.dtype, device=obs.device
        ).unsqueeze(0)
        logp_dummy = torch.zeros_like(action_t[..., 0:1])

        return action_t, logp_dummy