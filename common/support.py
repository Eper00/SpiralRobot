import mujoco
import numpy as np
import matplotlib.pyplot as plt
def _action_to_ctrl(action,actuator_low,actuator_high):
        return actuator_low + (action + 1.0) * 0.5 * (
            actuator_high - actuator_low
        )
def compute_spring_energy(model, data)->float:
    energy = 0.0

    qpos_addr = 0

    for j in range(model.njnt):

        joint_type = model.jnt_type[j]

        # ball joint
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:

            stiffness = model.jnt_stiffness[j]

            quat = data.qpos[qpos_addr:qpos_addr+4]

            theta = quat_angle(quat)

            energy += 0.5 * stiffness * theta**2

            qpos_addr += 4

    return energy
def quat_angle(quat)->float:
    quat = quat / np.linalg.norm(quat)

    w = np.clip(np.abs(quat[0]), -1.0, 1.0)

    return 2.0 * np.arccos(w)
def _get_tip_position(model, data) -> np.ndarray:
    tip_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "tip_center"
    )
    return data.site_xpos[tip_site_id].copy()


def _read_dataset(dataset_path):

    data = np.load(dataset_path)

    points = data["tips"]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x, y, z, s=2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Tentacle Workspace")

    plt.show()