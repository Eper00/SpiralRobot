from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "tentalce_2D-v2" / "robot.xml"

# IK parameters
SOLVER = "daqp"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 200


def converge_ik(
    configuration, tasks, dt, solver, pos_threshold, ori_threshold, max_iters
):
    """Runs up to 'max_iters' of IK steps. Returns True if position and orientation
    are below thresholds, otherwise False."""
    for _ in range(max_iters):
        vel = mink.solve_ik(configuration, tasks.values(), dt, solver, damping=1e-3)
        configuration.integrate_inplace(vel, dt)
        # Only checking the first FrameTask here (end_effector_task).
        # If you want to check multiple tasks, sum or combine their errors.
        err = tasks["eef"].compute_error(configuration)
        pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
        ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold

        if pos_achieved and ori_achieved:
            return True
    return False


def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="marker_25",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    tasks = {"eef": end_effector_task, "posture": posture_task}

    # Initialize viewer in passive mode
    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        mink.move_mocap_to_frame(model, data, "target", "marker_25", "site")
        initial_target_position = data.mocap_pos[0].copy()

       
        rate = RateLimiter(frequency=1000.0, warn=False)

        while viewer.is_running():
            dt = rate.dt

            # 1) IK konfiguráció frissítése a szimulációból
            configuration.update(data.qpos)

            # 2) Cél beállítása
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # 3) IK → q_des (csak configuration-ben)
            converge_ik(configuration, tasks, dt, SOLVER, POS_THRESHOLD, ORI_THRESHOLD, MAX_ITERS)
            q_des = configuration.q.copy()

            # 4) q_des → tendon_length_des a mink saját modelljén (NEM a szimulációs data-n!)
            configuration.data.qpos[:] = q_des
            mujoco.mj_forward(configuration.model, configuration.data)
            tendon_length_des = configuration.data.ten_length.copy()

            # 5) tendon_length_des → ctrl (mindkét tendon aktuátorra)
            ctrl = np.zeros(model.nu)
            for i in range(model.nu):
                ctrl_min, ctrl_max = configuration.model.actuator_ctrlrange[i]
                len_min, len_max = configuration.model.actuator_lengthrange[i]

                norm = (tendon_length_des[i] - len_min) / (len_max - len_min)
                ctrl[i] = ctrl_min + norm * (ctrl_max - ctrl_min)

            # 6) Kontroll input beadása a VALÓDI szimulációnak
            data.ctrl[:] = ctrl

            # 7) Fizika léptetése
            mujoco.mj_step(model, data)

            viewer.sync()






if __name__ == "__main__":
    main()