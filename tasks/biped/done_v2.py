# tasks/biped/done_v2_strict.py
"""
STRICTER termination conditions - robot should reset when it falls.
"""
import mujoco
import numpy as np


def done_v2(model: mujoco.MjModel, data: mujoco.MjData, t: int) -> bool:
    """
    Stricter episode termination - catch falls earlier.
    """
    
    # Get torso state
    torso_id = model.body("hips").id
    torso_pos = data.xpos[torso_id]
    torso_quat = data.xquat[torso_id]
    
    # 1. Height check (STRICTER)
    height = torso_pos[2]
    if height < 0.10:  # Increased from 0.08
        return True
    
    # 2. Orientation check (STRICTER)
    quat = torso_quat
    tilt = np.sqrt(quat[1]**2 + quat[2]**2)
    if tilt > 0.4:  # Reduced from 0.5 (more sensitive)
        return True
    
    # 3. Forward pitch check (NEW - catches knee-first falls)
    # If leaning forward too much, terminate
    pitch = 2.0 * quat[2]
    if abs(pitch) > 0.6:  # About 35 degrees
        return True
    
    # 4. Lateral bounds
    y_pos = torso_pos[1]
    if abs(y_pos) > 1.0:
        return True
    
    # 5. Check for NaN/Inf
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        return True
    
    # 6. Check if torso is too low relative to feet (NEW)
    # This catches cases where robot is on ground but height > threshold
    try:
        left_foot_id = model.site("left_foot_ik").id
        right_foot_id = model.site("right_foot_ik").id
        
        left_foot_h = data.site_xpos[left_foot_id][2]
        right_foot_h = data.site_xpos[right_foot_id][2]
        max_foot_h = max(left_foot_h, right_foot_h)
        
        # If torso is less than 8cm above the highest foot, robot has fallen
        if height - max_foot_h < 0.08:
            return True
            
    except:
        pass
    
    return False