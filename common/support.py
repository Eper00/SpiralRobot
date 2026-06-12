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




def sample_target(workspace_center,workspace_inner_radius,workspace_outer_radius):
    r = np.sqrt(
        np.random.uniform(
            workspace_inner_radius**2,
            workspace_outer_radius**2
        )
    )
    theta = np.pi * np.random.rand()
    x = workspace_center[0] + r * np.cos(theta)
    y = workspace_center[1] + r * np.sin(theta)
    return np.array([x, y])


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

