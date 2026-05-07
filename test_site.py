import mujoco
from mujoco import viewer
import parameters
import numpy as np
def _get_tip_position(model, data) -> np.ndarray:
    tip_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "tip_center"
    )
    return data.site_xpos[tip_site_id].copy()

if __name__ == "__main__":
        model = mujoco.MjModel.from_xml_path(parameters.path_xml_tentacle)
        data = mujoco.MjData(model)

        with viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)

                print(_get_tip_position(model,data))




                viewer.sync()