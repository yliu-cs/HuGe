"""Huge-side MuJoCo adapter: offscreen viewing, assisted startup and fall reset."""

from __future__ import annotations

import queue
import time
from typing import Any

from run_real_camera_sim import VIDEO_FPS, command_reader, event, worker_main


def make_sim_config(camera_port: int) -> Any:
    """Construct upstream defaults without invoking its implicit Git query."""
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    class PreviewConfig(SimLoopConfig):
        def __post_init__(self) -> None:
            # Same sim defaults, without the upstream automatic Git query.
            self.interface = "lo"
            self.env_type = "sim"
            if self.gravity_compensation_joints is None:
                self.gravity_compensation_joints = ["arms"]

    return PreviewConfig(enable_onscreen=False, enable_offscreen=True,
                         enable_image_publish=True, camera_port=camera_port)


def run(config: dict[str, Any], viewer: bool = False) -> None:
    """Run the loopback simulator; no real-robot DDS interface is accepted."""
    import mujoco
    from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
    from gear_sonic.utils.mujoco_sim.simulator_factory import init_channel
    from unitree_sdk2py.idl.default import (
        unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__HandCmd_,
    )

    sim_config = make_sim_config(config["ports"]["video"])
    wbc_config = sim_config.load_wbc_yaml()
    wbc_config.update(DOMAIN_ID=0, INTERFACE="lo", ENABLE_ELASTIC_BAND=True,
                      USE_JOYSTICK=0, PRINT_SCENE_INFORMATION=False,
                      IMAGE_DT=1.0 / VIDEO_FPS)
    init_channel(wbc_config)
    camera = {"ego_view": {"height": 480, "width": 640, "params": -1}}
    sim = BaseSimulator(config=wbc_config, env_name="default", onscreen=False,
                        offscreen=True, enable_image_publish=True, camera_configs=camera)
    commands = command_reader()
    env = sim.sim_env
    original_step = env.sim_step
    original_reset = env.reset
    original_render = env.update_render_caches
    waiting_for_restart = False
    report_at = time.monotonic()
    physics_steps = rendered_frames = 0

    def render() -> Any:
        nonlocal rendered_frames
        result = original_render()
        rendered_frames += 1
        return result

    def reset_after_fall() -> None:
        nonlocal waiting_for_restart
        original_reset()
        mujoco.mj_forward(env.mj_model, env.mj_data)
        env.elastic_band.enable = True
        waiting_for_restart = True
        event("fell")

    def step() -> None:
        nonlocal waiting_for_restart, report_at, physics_steps, rendered_frames
        while True:
            try:
                command = commands.get_nowait()
            except queue.Empty:
                break
            if command == "stop":
                raise KeyboardInterrupt
            if command == "release":
                env.elastic_band.enable = False
                event("released")
            elif command == "reset":
                # The parent has now stopped the old control writers.
                original_reset()
                mujoco.mj_forward(env.mj_model, env.mj_data)
                bridge = sim.unitree_bridge
                with bridge.low_cmd_lock:
                    bridge.low_cmd = unitree_hg_msg_dds__LowCmd_()
                with bridge.left_hand_cmd_lock:
                    bridge.left_hand_cmd = unitree_hg_msg_dds__HandCmd_()
                with bridge.right_hand_cmd_lock:
                    bridge.right_hand_cmd = unitree_hg_msg_dds__HandCmd_()
                env.elastic_band.enable = True
                waiting_for_restart = False
                event("reset_ready")
        if waiting_for_restart:
            # Keep observations and the viewing stream alive while WBC restarts.
            env.unitree_bridge.PublishLowState(env.prepare_obs())
        else:
            original_step()
            physics_steps += 1
        publisher = env.image_publish_process
        if publisher is not None and publisher.process is not None and not publisher.process.is_alive():
            raise RuntimeError(f"Image publisher exited: {publisher.process.exitcode}")
        now = time.monotonic()
        elapsed = now - report_at
        if elapsed >= 5.0:
            event("timing", physics_hz=round(physics_steps / elapsed, 1),
                  render_fps=round(rendered_frames / elapsed, 1),
                  real_time_factor=round(physics_steps * env.sim_dt / elapsed, 3),
                  waiting_for_restart=waiting_for_restart)
            report_at, physics_steps, rendered_frames = now, 0, 0

    env.reset = reset_after_fall
    env.sim_step = step
    env.update_render_caches = render
    try:
        from OpenGL import GL

        renderer_name = GL.glGetString(GL.GL_RENDERER).decode()
        event("renderer", gl_renderer=renderer_name, target_fps=VIDEO_FPS)
        if any(name in renderer_name.lower() for name in ("llvmpipe", "softpipe", "swrast")):
            raise RuntimeError(
                f"Software renderer {renderer_name} blocks physics; NVIDIA EGL is required. "
                "Check /usr/share/glvnd/egl_vendor.d/10_nvidia.json and GPU access."
            )
        track = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(track)
        track.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        track.trackbodyid = env.mj_model.body("pelvis").id
        track.distance, track.azimuth, track.elevation = 3.0, 90.0, -15.0
        camera["ego_view"]["params"] = track
        sim.start_image_publish_subprocess(start_method="spawn", camera_port=config["ports"]["video"])
        event("ready", domain=0, interface="lo", elastic_band=True)
        sim.start()
    finally:
        sim.close()
        for renderer in env.renderers.values():
            renderer.close()


if __name__ == "__main__":
    worker_main(run)
