"""Launch the headless MuJoCo G1 sim used for SONIC token replay.

The WBC deploy binary drives the sim over loopback CycloneDDS. Elastic band is
disabled at runtime -- do not set ENABLE_ELASTIC_BAND: False in
wbc_configs/g1_29dof_sonic_model12.yaml, since base_sim.py's else branch never
initializes self.elastic_band, which raises AttributeError in sim_step().

Must run from inside GR00T-WholeBodyControl/ with .venv_sim/bin/python (this
script never edits files there).

By default publishes no images -- only drives the g1_debug state stream
consumed by align_sim_replay.py. Pass --enable-image-publish to also render
offscreen via EGL (requires MUJOCO_GL=egl in the environment) for visual QA of
a replay; frames are then available on the ZMQ image server at --camera-port.

The offscreen camera tracks the "pelvis" body (same body base_sim.py's
onscreen viewer tracks at mjCAMERA_TRACKING, see base_sim.py:224) instead of
using a fixed free camera -- a free camera loses the robot once it walks more
than a few meters from the origin.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--enable-image-publish",
        action="store_true",
        help="offscreen EGL render + ZMQ PUB images (needs MUJOCO_GL=egl); off by default",
    )
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--domain-id", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.domain_id <= 232:
        parser.error("--domain-id must be between 0 and 232")

    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel

    config = SimLoopConfig()
    config.enable_offscreen = args.enable_image_publish
    config.enable_onscreen = False
    config.enable_image_publish = args.enable_image_publish
    config.camera_port = args.camera_port

    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    wbc_config["DOMAIN_ID"] = args.domain_id
    print(
        f"headless sim DDS domain={args.domain_id} "
        f"interface={wbc_config.get('INTERFACE', '<default>')}",
        flush=True,
    )

    init_channel(config=wbc_config)

    # scene_43dof.xml has no named camera (ncam=0); the repo default
    # {"mjcf_name": "head_camera"} raises "camera does not exist" -- use a free
    # camera instead when rendering.
    camera_configs = (
        {"ego_view": {"height": 480, "width": 640, "params": -1}} if args.enable_image_publish else {}
    )

    sim = BaseSimulator(
        config=wbc_config,
        env_name=config.env_name,
        onscreen=False,
        offscreen=args.enable_image_publish,
        enable_image_publish=args.enable_image_publish,
        camera_configs=camera_configs,
    )

    if sim.sim_env.elastic_band is not None:
        sim.sim_env.elastic_band.enable = False

    if args.enable_image_publish:
        import mujoco

        track_cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(track_cam)
        track_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        track_cam.trackbodyid = sim.sim_env.mj_model.body("pelvis").id
        track_cam.distance = 3.0
        track_cam.azimuth = 90.0
        track_cam.elevation = -15.0
        camera_configs["ego_view"]["params"] = track_cam

    SimulatorFactory.start_simulator(
        sim,
        as_thread=False,
        enable_image_publish=args.enable_image_publish,
        mp_start_method=config.mp_start_method,
        camera_port=config.camera_port,
    )


if __name__ == "__main__":
    main()
