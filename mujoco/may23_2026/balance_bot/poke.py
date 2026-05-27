"""
Interactive MuJoCo viewer with mouse perturbation (click/drag to apply forces).

This uses `mujoco.viewer.launch_passive`, which includes MuJoCo's built-in
perturbation UI. It is the easiest way to "click the robot and push it".

Run:
  uv run python poke.py

Notes:
- The exact mouse modifier for "apply force" depends on the MuJoCo viewer build,
  but the viewer overlay (press `H`) shows the current controls.
- We call `mjv_applyPerturbForce` each step so any active mouse perturbation is
  applied as external forces on the selected body.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from robot_params import OUTPUT_PATH, render_xml


def main() -> None:
    render_xml()  # ensure assets/balance_bot.xml is up to date
    model = mujoco.MjModel.from_xml_path(str(OUTPUT_PATH))
    data = mujoco.MjData(model)

    # Launch viewer (user interacts with mouse/keyboard in this window)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
        # Run until the viewer window is closed.
        while viewer.is_running():
            # Clear any stale forces, then apply the current mouse perturb (if any).
            data.xfrc_applied[:] = 0.0
            mujoco.mjv_applyPerturbForce(model, data, viewer.perturb)

            mujoco.mj_step(model, data)
            viewer.sync()

            # Try to run near real-time.
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()

