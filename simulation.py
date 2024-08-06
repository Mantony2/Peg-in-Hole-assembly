import pybullet as p
import pybullet_data
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


p.loadURDF("plane.urdf")
peg_start_pos = [0.05, 0.05, 0.3]
hole_start_pos = [0, 0, 0]

initial_peg_orientation = p.getQuaternionFromEuler([0.1, 0.1, 0])

peg_id = p.loadURDF("peg.urdf", peg_start_pos, initial_peg_orientation, useFixedBase=False)
hole_id = p.loadURDF("hole_hollow.urdf", hole_start_pos, useFixedBase=True)

p.changeDynamics(peg_id, -1, lateralFriction=0.5, restitution=0.1)
p.changeDynamics(hole_id, -1, lateralFriction=0.5, restitution=0.1)

max_steps = 10000
step_size = 1 / 240.0
alignment_threshold_xy = 0.001
alignment_threshold_z = 0.005

position_gain = 0.01
orientation_gain = 0.01

def apply_tilt_correction(peg_pos, peg_orn, contact_info):
    contact_position = np.array(contact_info[0][6])
    contact_normal = np.array(contact_info[0][7])
    contact_force = contact_info[0][9]

    lever_arm = contact_position - np.array(peg_pos)
    torque = np.cross(lever_arm, contact_force * contact_normal)
    
    peg_orn_euler = np.array(p.getEulerFromQuaternion(peg_orn))
    corrective_rotation = orientation_gain * torque
    peg_orn_euler += corrective_rotation
    
    peg_orn = p.getQuaternionFromEuler(peg_orn_euler)
    
    return peg_orn, contact_force, torque

def is_peg_inserted(peg_pos, hole_pos):
    distance_z = peg_pos[2] - hole_pos[2]
    if abs(distance_z) < alignment_threshold_z and np.linalg.norm(np.array(peg_pos[:2]) - np.array(hole_pos[:2])) < alignment_threshold_xy:
        return True
    return False


for step in range(max_steps):
    p.stepSimulation()

    peg_pos, peg_orn = p.getBasePositionAndOrientation(peg_id)
    hole_pos, hole_orn = p.getBasePositionAndOrientation(hole_id)

    contact_force = np.zeros(3)
    torque = np.zeros(3)

    contact_info = p.getContactPoints(peg_id, hole_id)
    if contact_info:
        peg_orn, contact_force, torque = apply_tilt_correction(peg_pos, peg_orn, contact_info)

    if is_peg_inserted(peg_pos, hole_pos):
        print(f"Step: {step}, Peg inserted successfully!")
        break

    direction_xy = np.array(hole_pos[:2]) - np.array(peg_pos[:2])
    distance_xy = np.linalg.norm(direction_xy)
    if distance_xy > alignment_threshold_xy:
        direction_normalized_xy = direction_xy / distance_xy if distance_xy > 0 else [0, 0]
        new_peg_pos_xy = np.array(peg_pos[:2]) + direction_normalized_xy * step_size * position_gain
        new_peg_pos = [new_peg_pos_xy[0], new_peg_pos_xy[1], peg_pos[2]]
        p.resetBasePositionAndOrientation(peg_id, new_peg_pos, peg_orn)
    else:
        direction_z = hole_pos[2] - peg_pos[2]
        if abs(direction_z) > alignment_threshold_z:
            new_peg_pos = [peg_pos[0], peg_pos[1], peg_pos[2] + np.sign(direction_z) * step_size * position_gain]
            p.resetBasePositionAndOrientation(peg_id, new_peg_pos, peg_orn)
        else:
            p.applyExternalForce(peg_id, -1, [0, 0, -0.1], peg_pos, p.WORLD_FRAME)
    
    if step % 100 == 0:
        print(f"Step: {step}, Peg Pos: {peg_pos}, Peg Orn: {p.getEulerFromQuaternion(peg_orn)}, Contact Force: {contact_force}, Torque: {torque}")

    time.sleep(0.05)

p.disconnect()
