import mujoco
import numpy as np
import matplotlib.pyplot as plt
import yaml
def _action_to_ctrl(action,actuator_low,actuator_high):
        return actuator_low + (action + 1.0) * 0.5 * (
            actuator_high - actuator_low
        )
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)
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
def _get_tip_position(model, data, site_name="tip_center") -> np.ndarray:

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name
    )

    return data.site_xpos[site_id].copy()

def _read_dataset(dataset_path):

    data = np.load(dataset_path)

    states = data["states"]
    actions = data["actions"]
    print("States shape:", states.shape)
    print("Actions shape:", actions.shape)
    return states,actions

def visualize_dataset(states, actions, max_trajectories=10):

    num_traj = min(states.shape[0], max_trajectories)

    fig = plt.figure(figsize=(16, 8))

    # ======================================================
    # 1. TIP + TARGET TRAJECTORIES
    # ======================================================
    ax1 = fig.add_subplot(121, projection="3d")

    for i in range(num_traj):

        traj = states[i]

        tips = traj[:, 0:3]
        targets = traj[:, 3:6]

        # tip trajectory
        ax1.plot(
            tips[:, 0],
            tips[:, 1],
            tips[:, 2],
            linewidth=1.5,
            alpha=0.7
        )

        # target trajectory (dashed)
        ax1.plot(
            targets[:, 0],
            targets[:, 1],
            targets[:, 2],
            linestyle="dashed",
            alpha=0.4
        )

        # start
        ax1.scatter(
            tips[0, 0],
            tips[0, 1],
            tips[0, 2],
            c="green",
            s=15
        )

        # end
        ax1.scatter(
            tips[-1, 0],
            tips[-1, 1],
            tips[-1, 2],
            c="red",
            s=15
        )

    ax1.set_title("Tip + Target Trajectories")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # ======================================================
    # 2. ACTION FIELD
    # ======================================================
    ax2 = fig.add_subplot(122, projection="3d")

    for i in range(num_traj):

        traj = states[i]
        acts = actions[i]

        tips = traj[:-1, 0:3]

        step = max(1, len(acts) // 40)
        idx = np.arange(0, len(acts), step)

        ax2.quiver(
            tips[idx, 0],
            tips[idx, 1],
            tips[idx, 2],
            acts[idx, 0],
            acts[idx, 1],
            acts[idx, 2],
            length=0.03,
            normalize=True,
            alpha=0.5
        )

    ax2.set_title("Action Field (Control Directions)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.tight_layout()
    plt.show()
def _normalize_position(pos,workspace_center,workspace_scale) -> np.ndarray:
    return (pos - workspace_center) / workspace_scale
def _normalize_actuator_lengths(lengths,actuator_low,actuator_high) -> np.ndarray:
        return (
            2.0
            * (lengths - actuator_low)
            / (actuator_high - actuator_low)
            - 1.0
        )

