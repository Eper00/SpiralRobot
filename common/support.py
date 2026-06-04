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


def _get_sites_positions(model, data, site_names) -> np.ndarray:
    """
    Returns:
        (N, 3) array of site positions
    """
    positions = []
    site_names=[site_names] if isinstance(site_names, str) else site_names
    for name in site_names:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            name
        )
        positions.append(data.site_xpos[site_id].copy())
    return np.array(positions)

def get_workspace_points(expert,
                         grid_n=200,
                         reach_threshold=0.01,
                         trajectories=300,
                         steps_per_traj=200):

    # 1) Uniform grid
    [y_min, z_min] = expert.target_bounds_min
    [y_max, z_max] = expert.target_bounds_max

    ys = np.linspace(y_min, y_max, grid_n)
    zs = np.linspace(z_min, z_max, grid_n)

    grid_points = np.array([(y, z) for y in ys for z in zs])
    feasible_mask = np.zeros(len(grid_points), dtype=bool)

    # 2) Futtatjuk az összes trajektóriát
    for _ in range(trajectories):

        # reset + coiling
        expert._base_reset()
        direction = np.random.choice([0,1])
        for _ in range(50):
            expert.coiling_policy(direction)
            expert._base_step(expert.action)

        # random exploration
        expert.random_policy()
        for _ in range(steps_per_traj):

           
            expert._base_step(expert.action)

            tip = _get_sites_positions(
                expert.model,
                expert.data,
                expert.marker_names[-1]
            )[0][1:]

            # 3) Minden grid pontra megnézzük, hogy közel van-e
            dists = np.linalg.norm(grid_points - tip, axis=1)

            # ahol közel van → megjelöljük fizibilisnek
            feasible_mask |= (dists < reach_threshold)

    # 4) Csak a fizibilis pontokat tartjuk meg
    feasible_points = grid_points[feasible_mask]

    # 5) Mentés + vizualizáció
    np.save("workspace_points.npy", feasible_points)

    plt.scatter(feasible_points[:,0], feasible_points[:,1], s=5)
    plt.title("Uniform Grid Workspace (Feasible Points Only)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.show()

    return feasible_points





def _normalize_position(positions,workspace_center,workspace_scale) -> np.ndarray:
    positions = np.asarray(positions)


    return (positions - workspace_center[None ,:]) / workspace_scale[None, :]

def _normalize_actuator_lengths(lengths,actuator_low,actuator_high) -> np.ndarray:
    lengths = np.asarray(lengths)
    actuator_low = np.asarray(actuator_low)
    actuator_high = np.asarray(actuator_high)
    return (
            2.0
            * (lengths - actuator_low)
            / (actuator_high - actuator_low)
            - 1.0
        )

