"""Minimal MuJoCo G1 environment for offline policy stability validation.

Wraps the 29-DoF G1 MJCF with a flat ground plane, a head-mounted ego camera
(used only when offscreen rendering is enabled — the validation pipeline
itself feeds policy obs from the source episode video, not from this camera),
and a position-PD-to-torque inner loop. The 31-D action layout is
``[jp(29), hand_bits(2)]`` in the same MuJoCo joint order the training data
uses (legs → waist → left arm → right arm); the two hand bits are recorded
but do not actuate the simplified 29-DoF body model.
"""
from __future__ import annotations

import os

# Pick a headless OpenGL backend before importing mujoco. Offscreen MuJoCo
# rendering requires either EGL (preferred on Linux servers) or OSMesa; the
# default GLFW backend fails on headless machines (no DISPLAY). Force EGL
# unconditionally — overriding instead of ``setdefault`` because some shells
# pre-populate ``MUJOCO_GL=glfw`` for desktop use, which would crash here.
if not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


# Joint order used by the training data: the same order as the actuator block
# in ``g1_29dof.xml`` and the ``DEFAULT_ANGLES_MUJOCO`` constant in
# ``src.tokenizer.sonic_encoder``.
JOINT_NAMES_29: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


# PD gains + armature aligned to the SONIC RL training-time IsaacLab config
# (``GR00T-WholeBodyControl/gear_sonic/envs/manager_env/
# robots/g1.py``  →  ``G1_CYLINDER_MODEL_12_DEX_CFG``). The base formula is the
# BeyondMimic / Unitree motor design point::
#
#     stiffness = armature * (10 * 2π)²        # natural freq 10 Hz
#     damping   = 4 * armature * (10 * 2π)     # damping ratio 2 (overdamped)
#
# Critically, ``feet`` (ankle pitch/roll) and ``waist`` (waist roll/pitch) use
# a 2× multiplier on stiffness/damping/armature in the training config — the
# 10 Hz natural frequency is preserved, but the actuator becomes 2× stiffer at
# its training point. The single-multiplier ``policy_parameters.hpp`` values
# from the deploy reference give *the same per-joint action_scale* (because
# the deploy halves both ``stiffness`` and ``effort_limit`` and ``0.25·E/S``
# cancels), but they apply *half the PD torque* on the joint for the same
# position error. The decoder was trained against the 2× stiff feet/waist —
# leaving those at 1× makes the foot/COP under-respond, which is the dominant
# residual contributor to base/yaw drift in our sim.
#
# Similarly, per-motor armature is intentionally non-uniform in training to
# keep all joints at the same 10 Hz natural frequency. Our previous uniform
# ``armature = 0.01`` injection collapsed that — hip_pitch/knee became 16 Hz
# (too fast) while ankles and wrists fell to 6 Hz (too slow). Now we apply
# motor-specific armature (with the 2× factor on feet/waist) so every joint
# matches the training-time ω_n = 10 Hz.
_NATURAL_FREQ = 10.0 * 2.0 * np.pi
_DAMPING_RATIO = 2.0
_ARMATURE = {
    "5020": 0.003609725,
    "7520_14": 0.010177520,
    "7520_22": 0.025101925,
    "4010": 0.00425,
}
_STIFFNESS = {k: a * _NATURAL_FREQ * _NATURAL_FREQ for k, a in _ARMATURE.items()}
_DAMPING = {k: 2.0 * _DAMPING_RATIO * a * _NATURAL_FREQ for k, a in _ARMATURE.items()}

# Motor type per MuJoCo joint (same ordering as ``JOINT_NAMES_29``):
# legs ×2 = [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
# waist = [yaw, roll, pitch]
# arms ×2 = [shoulder_p, shoulder_r, shoulder_y, elbow, wrist_r, wrist_p, wrist_y]
_MOTOR_BY_JOINT = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
)
# Joints whose ``ImplicitActuatorCfg`` group in training applies a 2× factor:
# the ``feet`` group (ankle pitch+roll) and the ``waist`` group (waist roll +
# pitch, NOT waist_yaw which is its own 7520_14 group at 1×).
_FEET_WAIST_DOUBLE_INDICES: tuple[int, ...] = (4, 5, 10, 11, 13, 14)


def _per_joint_multiplier() -> np.ndarray:
    mult = np.ones(29, dtype=np.float64)
    for i in _FEET_WAIST_DOUBLE_INDICES:
        mult[i] = 2.0
    return mult


_PD_MULTIPLIER = _per_joint_multiplier()
DEFAULT_KP = np.array(
    [_STIFFNESS[m] for m in _MOTOR_BY_JOINT], dtype=np.float64,
) * _PD_MULTIPLIER
DEFAULT_KD = np.array(
    [_DAMPING[m] for m in _MOTOR_BY_JOINT], dtype=np.float64,
) * _PD_MULTIPLIER

# Per-joint armature (used by ``_inject_joint_dynamics``). Indexed by joint
# name; the ×2 group keeps ω_n at 10 Hz despite the doubled stiffness.
_ARMATURE_BY_JOINT_NAME: dict[str, float] = {
    name: _ARMATURE[motor] * (2.0 if i in _FEET_WAIST_DOUBLE_INDICES else 1.0)
    for i, (name, motor) in enumerate(zip(JOINT_NAMES_29, _MOTOR_BY_JOINT))
}


# G1 default standing pose, in the same 29-D MuJoCo joint order as above.
# Matches ``DEFAULT_ANGLES_MUJOCO`` in ``src/tokenizer/sonic_encoder.py``.
DEFAULT_QPOS = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)


# Path to the G1 MJCF shipped with the top-level GR00T-WholeBodyControl
# submodule. We embed-include it from a generated scene that adds a ground,
# light, and ego camera.
def _default_g1_xml_path() -> Path:
    here = Path(__file__).resolve()
    repo = here.parent.parent.parent
    return (
        repo
        / "GR00T-WholeBodyControl"
        / "gear_sonic_deploy"
        / "g1"
        / "g1_29dof.xml"
    )


def _look_at_quat_wxyz(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) that orients a MuJoCo camera so it looks from
    ``cam_pos`` toward ``target`` with the world ``+Z`` axis as up.

    MuJoCo cameras look down their local ``-Z`` and use local ``+Y`` as up.
    The rotation matrix mapping camera-local to world is therefore
    ``[right | true_up | -forward]`` as columns.
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    R = np.column_stack([right, true_up, -forward])
    # Matrix → quaternion (Shepperd's method; chooses the largest diagonal
    # element to avoid numerical issues near 180°).
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


# Per-class joint damping + frictionloss from the older ``g1_29dof_old.xml``
# defaults. Damping here is the MJCF *static* joint damping (always-on, applied
# automatically by MuJoCo every substep, independent of the actuator PD). We
# keep it at the same constant 0.05 that both the deploy XML and the BeyondMimic
# URDF use; the joint-specific actuator damping comes from ``DEFAULT_KD`` and
# is applied separately in ``_apply_pd_torque``.
_DEFAULT_DAMPING = 0.05
_DEFAULT_FRICTIONLOSS = 0.2
_WRIST_FRICTIONLOSS = 0.1
# Fallback armature for any hinge joint not in ``_ARMATURE_BY_JOINT_NAME``
# (e.g. hand fingers in the with-hand model). Matches the legacy uniform
# value the file shipped with for unmapped joints.
_FALLBACK_ARMATURE = 0.01


def _inject_joint_dynamics(spec: mujoco.MjSpec) -> None:
    """Apply per-joint armature, MJCF damping, and frictionloss.

    Armature is keyed per joint to match the IsaacLab training-time motor
    config (``_ARMATURE_BY_JOINT_NAME``). Skips the free root joint.
    Unmapped hinge joints fall back to ``_FALLBACK_ARMATURE = 0.01`` so
    auxiliary models (with-hand etc.) still compile.
    """
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        joint.armature = _ARMATURE_BY_JOINT_NAME.get(joint.name, _FALLBACK_ARMATURE)
        # ``damping`` is exposed as a 3-vector for SDF compatibility; hinge
        # only consumes element 0.
        joint.damping[0] = _DEFAULT_DAMPING
        joint.frictionloss = (
            _WRIST_FRICTIONLOSS if "wrist" in joint.name else _DEFAULT_FRICTIONLOSS
        )


def _add_tracking_camera(spec: mujoco.MjSpec) -> None:
    """Add a third-person camera placed slightly above and in front of the
    robot, automatically rotating to keep the pelvis centered.

    Uses ``TARGETBODYCOM`` mode pointed at the pelvis: the camera position
    stays fixed in world frame while its orientation always tracks the
    pelvis COM. Keeps the robot in frame even when free-base physics sends
    it tumbling. The earlier ``TRACKCOM`` setup with a hand-tuned quat
    produced a tilted view because ``TRACKCOM`` does not steer the camera —
    it only translates with the COM.
    """
    cam = spec.worldbody.add_camera()
    cam.name = "track"
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM
    cam.targetbody = "pelvis"
    # Slightly elevated front view; far enough that the angular sweep needed
    # to follow the COM stays small during a typical fall.
    cam_pos = np.array([3.0, 0.0, 1.6], dtype=np.float64)
    cam.pos = cam_pos
    # The initial quat is overwritten by MuJoCo each render under
    # TARGETBODYCOM but we set a sensible look-at default for static viewers.
    cam.quat = _look_at_quat_wxyz(cam_pos, np.array([0.0, 0.0, 0.85]))
    cam.fovy = 55.0


def _try_attach_head_camera(spec: mujoco.MjSpec) -> bool:
    """Attach a forward-looking head-mounted camera. The 29-DoF G1 model
    represents the head as a mesh on ``torso_link``; we add the camera as a
    child of that body with a vertical offset to roughly match the head
    position. Returns True on success."""
    host = spec.body("torso_link")
    if host is None:
        return False
    cam = host.add_camera()
    cam.name = "ego_view"
    # Head visual geom is at z ≈ -0.054 relative to torso_link with a
    # mesh extending upward; place the camera slightly above and forward of
    # the torso link origin (≈ 0.5m up from torso_link, 0.1m forward).
    cam.pos = np.array([0.10, 0.0, 0.50])
    # MuJoCo cameras look down their local -z; we want forward = +x of torso.
    # Rotation that maps cam-local -z onto +x and cam-local +y onto +z (so the
    # rendered image is upright) is the quaternion (w,x,y,z)=(0.5,0.5,0.5,0.5).
    cam.quat = np.array([0.5, 0.5, 0.5, 0.5])
    cam.fovy = 75.0
    return True


@dataclass
class StepStats:
    """Per-control-step quantities returned by ``G1MujocoEnv.step``."""
    qpos: np.ndarray  # (29,) joint positions
    qvel: np.ndarray  # (29,) joint velocities
    root_z: float
    root_quat_xyzw: np.ndarray  # (4,)
    root_rpy: np.ndarray  # (3,) roll, pitch, yaw (radians)
    fell: bool
    joint_limit_violation: int  # number of joints currently outside [low, high]
    contact_force_total: float


class G1MujocoEnv:
    """G1 simulator with PD position control on 29 body joints.

    Two operating modes:

    * ``fixed_base=True`` (default for offline action validation) — the pelvis
      is pinned in space, so the rollout isolates *joint-level* action
      generation: the metrics reflect whether policy outputs are smooth,
      respect joint limits, and produce reasonable joint dynamics under
      gravity acting on the limbs. Whole-body balance is out of scope.
    * ``fixed_base=False`` — full floating-base physics. The robot will fall
      without a balance controller.

    Parameters
    ----------
    xml_path : optional override for the G1 MJCF.
    sim_dt : MuJoCo physics timestep (seconds).
    ctrl_dt : outer control timestep (seconds); ``step()`` advances physics
        ``round(ctrl_dt / sim_dt)`` substeps.
    fixed_base : pin the pelvis (recommended for offline validation).
    fall_z_threshold / fall_tilt_threshold : ignored when ``fixed_base``.
    enable_offscreen / render_width / render_height : offscreen renderer.
    kp / kd : optional override gains, shape (29,).
    """

    def __init__(
        self,
        xml_path: Optional[Path] = None,
        sim_dt: float = 0.002,
        ctrl_dt: float = 1.0 / 30.0,
        fixed_base: bool = True,
        fall_z_threshold: float = 0.55,
        fall_tilt_threshold: float = 1.0,
        enable_offscreen: bool = True,
        render_width: int = 320,
        render_height: int = 224,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ):
        self.xml_path = Path(xml_path) if xml_path else _default_g1_xml_path()
        if not self.xml_path.exists():
            raise FileNotFoundError(f"G1 MJCF not found: {self.xml_path}")

        self.sim_dt = float(sim_dt)
        self.ctrl_dt = float(ctrl_dt)
        if self.ctrl_dt < self.sim_dt - 1e-9:
            raise ValueError(f"ctrl_dt {ctrl_dt} < sim_dt {sim_dt}")
        self.substeps = max(1, int(round(self.ctrl_dt / self.sim_dt)))

        self.fixed_base = bool(fixed_base)
        self.fall_z_threshold = float(fall_z_threshold)
        self.fall_tilt_threshold = float(fall_tilt_threshold)

        self.kp = np.asarray(kp if kp is not None else DEFAULT_KP, dtype=np.float64).copy()
        self.kd = np.asarray(kd if kd is not None else DEFAULT_KD, dtype=np.float64).copy()
        if self.kp.shape != (29,) or self.kd.shape != (29,):
            raise ValueError(f"kp/kd must be (29,), got {self.kp.shape}/{self.kd.shape}")

        # Load the G1 MJCF directly so its ``meshdir`` (relative to the file)
        # resolves correctly. Add tracking + head-mounted cameras to the spec
        # before compiling.
        spec = mujoco.MjSpec.from_file(str(self.xml_path))
        # The shipped ``g1_29dof.xml`` strips the per-joint ``<default class="...">``
        # block that the older ``g1_29dof_old.xml`` carried, so all 29 hinge joints
        # land with armature=0/damping=0/frictionloss=0. The deploy PD gains were
        # designed for the armatured model (STIFFNESS = armature × (2π·10)²), so
        # missing armature collapses the effective joint inertia to just the
        # rigid-body chain — orders of magnitude smaller than the design point.
        # PD then drives the under-damped distal joints (wrists, ankles) into
        # 200+ rad/s oscillations from step 0 and the sim numerically diverges
        # within ~15 ticks. Inject the old defaults onto every hinge so the
        # joint dynamics match what SONIC and the action stats were trained for.
        _inject_joint_dynamics(spec)
        _add_tracking_camera(spec)
        self._has_ego_camera = _try_attach_head_camera(spec)
        self.model = spec.compile()
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        self._actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in JOINT_NAMES_29
        ], dtype=np.int64)
        if (self._actuator_ids < 0).any():
            missing = [n for n, i in zip(JOINT_NAMES_29, self._actuator_ids) if i < 0]
            raise RuntimeError(f"Missing actuators in G1 MJCF: {missing}")

        self._joint_qposadr = np.zeros(29, dtype=np.int64)
        self._joint_dofadr = np.zeros(29, dtype=np.int64)
        self._joint_range = np.zeros((29, 2), dtype=np.float64)
        self._joint_range_valid = np.zeros(29, dtype=bool)
        for i, name in enumerate(JOINT_NAMES_29):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"Joint not in MJCF: {name}")
            self._joint_qposadr[i] = self.model.jnt_qposadr[jid]
            self._joint_dofadr[i] = self.model.jnt_dofadr[jid]
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                self._joint_range[i] = (lo, hi)
                self._joint_range_valid[i] = True

        # Floating-base joint (free joint at root); qpos[0:7] = (xyz + wxyz).
        self._free_qposadr = 0
        self._free_dofadr = 0

        self._renderer: mujoco.Renderer | None = None
        if enable_offscreen:
            self._renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self._render_size = (render_width, render_height)

        # Cached commanded targets (held between policy queries).
        self._target_qpos = DEFAULT_QPOS.copy()
        self._last_hand_bits = np.zeros(2, dtype=np.float32)

        # Baseline for the policy-facing finite-diff jv (matches training's
        # 50 Hz ``(jp[t]-jp[t-1])*fps`` recipe). The closed-loop sim proprio
        # assembler calls ``latch_proprio_baseline`` once per outer tick so
        # ``proprio_64_with_jv`` reports rad/s in the same units as training,
        # independent of the inner MuJoCo sim_dt or any data.qvel artifacts.
        self._jp_baseline: Optional[np.ndarray] = None
        self._jp_baseline_t: Optional[float] = None

        # WBC head state. Off by default; switched on via ``enable_wbc_mode``,
        # which retunes sim_dt to the WBC training resolution (0.005 s) and
        # exposes ``step_with_wbc``. Other heads (sonic, gt_wbc) are
        # untouched.
        self._wbc = None  # set by enable_wbc_mode
        self._wbc_decimation = 4  # WBC fires every N sim substeps (20 ms at 5 ms)
        self._wbc_sub_counter = 0

        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        qpos_init: Optional[np.ndarray] = None,
        init_hand_bits: Optional[np.ndarray] = None,
    ) -> None:
        """Reset the simulator to a default standing pose. Auto-computes the
        spawn height so the lowest foot geom rests on the floor (the bent-knee
        default pose puts the feet ~5cm above the MJCF nominal pelvis height,
        causing free-fall otherwise), then settles for ~0.5s with PD active.

        ``init_hand_bits`` seeds ``_last_hand_bits`` (the 2-D hand_bits mirror
        consumed by ``proprio_64_with_jv``) so the first policy-facing proprio
        vector lands in-distribution. Pass the dataset's frame-0 GT bits when
        running closed-loop sim proprio; leave None for the legacy zeros init.
        """
        mujoco.mj_resetData(self.model, self.data)
        q = qpos_init if qpos_init is not None else DEFAULT_QPOS

        # First, place the robot high above the ground in the desired pose
        # and run forward kinematics to read off geom positions.
        self.data.qpos[self._free_qposadr : self._free_qposadr + 7] = np.array([
            0.0, 0.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
        ])
        for i, addr in enumerate(self._joint_qposadr):
            self.data.qpos[addr] = q[i]
            self.data.qvel[self._joint_dofadr[i]] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Lowest geom z (collidable geoms only; group 0 are the contact meshes
        # in this MJCF, group 1 is visual-only).
        zs = []
        for gi in range(self.model.ngeom):
            if int(self.model.geom_contype[gi]) == 0 and int(self.model.geom_conaffinity[gi]) == 0:
                continue
            if self.model.geom_bodyid[gi] == 0:  # skip the world floor itself
                continue
            zs.append(float(self.data.geom_xpos[gi, 2]) - float(self.model.geom_size[gi, 0]))
        # In fixed-base mode, hover the robot well above the floor so the feet
        # never touch — otherwise pinning the base while the legs are commanded
        # to bend (e.g. GT replay where the knee target sweeps from 0.7 → 1.0
        # rad) makes the feet drive into the floor, and friction creates large
        # reaction torques the PD loop cannot overcome → 1+ rad tracking error
        # and joint-limit violations on every step. fixed_base is a joint-
        # isolation mode by design, so removing floor contact is correct.
        floor_clearance = 0.5 if self.fixed_base else 1e-3
        z_offset = (2.0 - min(zs)) + floor_clearance if zs else 0.793
        self.data.qpos[self._free_qposadr + 2] = z_offset

        mujoco.mj_forward(self.model, self.data)

        # Cache the spawn pose so we can re-pin the floating base each substep
        # in fixed_base mode.
        self._base_qpos_init = self.data.qpos[
            self._free_qposadr : self._free_qposadr + 7
        ].copy()

        self._target_qpos = q.copy().astype(np.float64)
        if init_hand_bits is not None:
            self._last_hand_bits = (
                np.asarray(init_hand_bits, dtype=np.float32)[:2].copy()
            )
        else:
            self._last_hand_bits = np.zeros(2, dtype=np.float32)

        # Settle: hold the default pose for ~0.5s so passive contacts converge
        # before the rollout begins.
        n_settle = int(round(0.5 / self.sim_dt))
        for _ in range(n_settle):
            self._apply_pd_torque(self._target_qpos)
            self._step_physics_once()

        # Latch the post-settle jp as the baseline for the next finite-diff
        # jv computation. Without this, the first ``proprio_64_with_jv`` after
        # reset would have to invent a dt and would report a degenerate zero
        # jv (or worse, a junk value if t_prev is None).
        self._jp_baseline = self.joint_pos().copy()
        self._jp_baseline_t = float(self.data.time)

    def step(self, action_31: np.ndarray) -> StepStats:
        """Advance physics one control-step under PD position tracking.

        The first 29 dims of ``action_31`` are absolute joint angle targets
        (already de-normalized by the caller); the last 2 are gripper bits,
        recorded for logging only since the simplified 29-DoF body has no
        finger DoFs.
        """
        if action_31.shape != (31,):
            raise ValueError(f"action shape must be (31,), got {action_31.shape}")
        self._target_qpos = action_31[:29].astype(np.float64)
        self._last_hand_bits = action_31[29:31].astype(np.float32)

        for _ in range(self.substeps):
            self._apply_pd_torque(self._target_qpos)
            self._step_physics_once()
        return self._collect_stats()

    def kinematic_step(
        self,
        action_31: np.ndarray,
        base_qpos: Optional[np.ndarray] = None,
    ) -> StepStats:
        """Visualize one control step by writing joint angles directly.

        No physics integration, no PD loop — used for replaying a recorded
        joint trajectory exactly as captured. The simplified offline rig has
        no whole-body controller, so PD-tracking GT joint targets with the
        legs hanging from a pinned base produces large oscillations on the
        under-damped distal joints (wrists swing past their limits and bounce
        off the constraint). Direct kinematic replay sidesteps that — what
        gets rendered is exactly the dataset's joint trajectory, which is the
        useful visualization for "what does the GT look like in sim?".

        ``base_qpos`` is the 7-D MuJoCo free-joint qpos for the pelvis
        ``[x, y, z, qw, qx, qy, qz]``. When provided, the floating base is
        moved to this pose so a walking episode actually translates across
        the scene; when None, the base is pinned to ``_base_qpos_init``
        (legacy behavior, useful for joint-only visualization).
        """
        if action_31.shape != (31,):
            raise ValueError(f"action shape must be (31,), got {action_31.shape}")
        target = action_31[:29].astype(np.float64)
        self._target_qpos = target
        self._last_hand_bits = action_31[29:31].astype(np.float32)
        for i, addr in enumerate(self._joint_qposadr):
            self.data.qpos[addr] = target[i]
            self.data.qvel[self._joint_dofadr[i]] = 0.0
        if base_qpos is not None:
            base = np.asarray(base_qpos, dtype=np.float64)
            if base.shape != (7,):
                raise ValueError(f"base_qpos must be (7,), got {base.shape}")
            self.data.qpos[self._free_qposadr : self._free_qposadr + 7] = base
            self.data.qvel[self._free_dofadr : self._free_dofadr + 6] = 0.0
        elif self.fixed_base:
            self.data.qpos[self._free_qposadr : self._free_qposadr + 7] = (
                self._base_qpos_init
            )
            self.data.qvel[self._free_dofadr : self._free_dofadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.time += self.ctrl_dt
        return self._collect_stats()

    def enable_wbc_mode(self, wbc, sim_dt: float = 0.005) -> None:
        """Switch this env into WBC head mode.

        Retunes ``sim_dt`` to the WBC training resolution (default 0.005 s, the
        IsaacLab → MuJoCo deploy timestep) and attaches the locomotion network
        so ``step_with_wbc`` can drive legs+waist. Other env heads are
        unaffected — they continue to call ``step`` / ``kinematic_step`` and
        will be running at whatever sim_dt was set at construction.

        The ``substeps`` count is recomputed so the outer ctrl_dt stays close
        to its original value (slight drift is acceptable; the WBC training
        period of 20 ms is preserved exactly via ``wbc_decimation = 4``).
        """
        self._wbc = wbc
        self.sim_dt = float(sim_dt)
        self.substeps = max(1, int(round(self.ctrl_dt / self.sim_dt)))
        self.model.opt.timestep = self.sim_dt
        self._wbc_sub_counter = 0

    def step_with_wbc(
        self,
        arm_target_14: np.ndarray,
        hand_bits_2: np.ndarray,
        loco_cmd_3: np.ndarray,
        height_cmd: float,
        rpy_cmd_3: np.ndarray,
    ) -> StepStats:
        """Advance physics one control step with split PD.

        Legs + waist (joints 0..14) tracked by the WBC's 15-D target output;
        arms (joints 15..28) tracked by ``arm_target_14``. Both share the env's
        ``self.kp`` / ``self.kd`` (same gains as ``step``).

        The WBC fires every ``_wbc_decimation`` substeps (20 ms at sim_dt=5 ms,
        matching the deployment script). Between firings the legs+waist target
        is held at the last value the WBC produced.
        """
        if self._wbc is None:
            raise RuntimeError("WBC not attached; call enable_wbc_mode() first.")
        if arm_target_14.shape != (14,):
            raise ValueError(f"arm_target must be (14,); got {arm_target_14.shape}")
        if hand_bits_2.shape != (2,):
            raise ValueError(f"hand_bits must be (2,); got {hand_bits_2.shape}")
        self._last_hand_bits = hand_bits_2.astype(np.float32)

        loco = np.asarray(loco_cmd_3, dtype=np.float64)
        rpy = np.asarray(rpy_cmd_3, dtype=np.float64)
        arm_target = arm_target_14.astype(np.float64)

        for _ in range(self.substeps):
            if self._wbc_sub_counter % self._wbc_decimation == 0:
                target_15 = self._wbc.step(
                    qpos_29=self.joint_pos(),
                    qvel_29=self.joint_vel(),
                    base_quat_wxyz=self.root_quat_wxyz(),
                    base_omega_body=self.base_angular_velocity_body(),
                    loco_cmd_3=loco,
                    height_cmd=float(height_cmd),
                    rpy_cmd_3=rpy,
                )
                self._target_qpos = np.concatenate(
                    [target_15.astype(np.float64), arm_target]
                )
            else:
                # Refresh arms only; legs+waist hold last WBC target.
                self._target_qpos[15:29] = arm_target
            self._wbc_sub_counter += 1
            self._apply_pd_torque(self._target_qpos)
            self._step_physics_once()
        return self._collect_stats()

    def _step_physics_once(self) -> None:
        """One ``mj_step`` substep with optional fixed-base pinning."""
        mujoco.mj_step(self.model, self.data)
        if self.fixed_base:
            # Re-clamp the floating-base joint after each substep. Resetting
            # both qpos and qvel each step removes any drift that integration
            # would have introduced and leaves the body joints (legs/torso/
            # arms) free to respond to PD + gravity normally.
            self.data.qpos[self._free_qposadr : self._free_qposadr + 7] = (
                self._base_qpos_init
            )
            self.data.qvel[self._free_dofadr : self._free_dofadr + 6] = 0.0

    def _apply_pd_torque(self, target_q: np.ndarray) -> None:
        q = np.array([self.data.qpos[a] for a in self._joint_qposadr])
        qd = np.array([self.data.qvel[a] for a in self._joint_dofadr])
        tau = self.kp * (target_q - q) - self.kd * qd
        # Saturate to torque-control limits given by the MJCF actuator
        # (forcerange is per-actuator). MuJoCo will clamp internally too.
        for i, aid in enumerate(self._actuator_ids):
            self.data.ctrl[aid] = float(tau[i])

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self._joint_qposadr], dtype=np.float64)

    def joint_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self._joint_dofadr], dtype=np.float64)

    def root_z(self) -> float:
        return float(self.data.qpos[self._free_qposadr + 2])

    def root_quat_wxyz(self) -> np.ndarray:
        return np.array(self.data.qpos[self._free_qposadr + 3 : self._free_qposadr + 7], dtype=np.float64)

    def base_angular_velocity_body(self) -> np.ndarray:
        """Pelvis angular velocity (rad/s) in body frame.

        Free-joint qvel layout for MuJoCo is ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]``
        with linear vel in world frame and angular vel in body frame, so we
        can return it directly without a rotation.
        """
        return np.array(
            self.data.qvel[self._free_dofadr + 3 : self._free_dofadr + 6],
            dtype=np.float64,
        )

    def root_quat_xyzw(self) -> np.ndarray:
        w, x, y, z = self.root_quat_wxyz()
        return np.array([x, y, z, w], dtype=np.float64)

    def root_rpy(self) -> np.ndarray:
        # ZYX Tait-Bryan from a quaternion (w, x, y, z) with intrinsic order
        # roll-pitch-yaw.
        w, x, y, z = self.root_quat_wxyz()
        # Roll (X)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (Y)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        # Yaw (Z)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def proprio_64(self, hand_bits: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the 64-D raw proprio vector (jp[29] + hand_bits[2] +
        jv[29] + odom_quat_xyzw[4]) used by the training data pipeline.

        ``hand_bits`` defaults to the last commanded bits.

        Note: jv here is ``data.qvel`` (continuous-time sim velocity). For
        closed-loop policy feed prefer ``proprio_64_with_jv`` so the jv units
        match the training cache's ``(jp[t]-jp[t-1])*fps`` finite-diff.
        """
        bits = self._last_hand_bits if hand_bits is None else np.asarray(hand_bits, np.float32)
        return np.concatenate([
            self.joint_pos().astype(np.float32),
            bits.astype(np.float32),
            self.joint_vel().astype(np.float32),
            self.root_quat_xyzw().astype(np.float32),
        ])

    def latch_proprio_baseline(self) -> None:
        """Snapshot the current jp + sim time for the next finite-diff jv.

        Called by ``SimProprioAssembler`` after every outer control tick (and
        during WBC warmup, so the first post-warmup inference's jv has the
        right ``t_prev``). Keeping the baseline on the env means callers don't
        have to track jp between rollout fns.
        """
        self._jp_baseline = self.joint_pos().copy()
        self._jp_baseline_t = float(self.data.time)

    def proprio_64_with_jv(
        self,
        jv_29: np.ndarray,
        quat_xyzw: np.ndarray,
        hand_bits: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Same 64-D layout as ``proprio_64`` but with explicit ``jv`` and
        ``quat_xyzw`` injection.

        Used by the closed-loop sim proprio path so the assembler can supply
        finite-diff jv (matching the training cache's ``(jp[t]-jp[t-1])*fps``
        units) and a frame-aligned quat (sim spawn → dataset frame-0 system),
        without this method having to know about the dataset.
        """
        jv = np.asarray(jv_29, dtype=np.float32)
        q = np.asarray(quat_xyzw, dtype=np.float32)
        if jv.shape != (29,):
            raise ValueError(f"jv_29 must be (29,), got {jv.shape}")
        if q.shape != (4,):
            raise ValueError(f"quat_xyzw must be (4,), got {q.shape}")
        bits = (
            self._last_hand_bits
            if hand_bits is None
            else np.asarray(hand_bits, np.float32)
        )
        return np.concatenate([
            self.joint_pos().astype(np.float32),
            bits.astype(np.float32),
            jv,
            q,
        ])

    def render(self, camera: str = "track") -> np.ndarray:
        """Return an RGB uint8 image (H, W, 3). ``camera`` defaults to the
        external tracking shot; pass ``ego_view`` for the head-mounted view."""
        if self._renderer is None:
            raise RuntimeError("Offscreen rendering disabled at construction time")
        if camera == "ego_view" and not self._has_ego_camera:
            camera = "track"
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _collect_stats(self) -> StepStats:
        qpos = self.joint_pos()
        qvel = self.joint_vel()
        rpy = self.root_rpy()
        z = self.root_z()
        # Joint limit checks.
        viol = 0
        if self._joint_range_valid.any():
            for i in range(29):
                if not self._joint_range_valid[i]:
                    continue
                lo, hi = self._joint_range[i]
                if qpos[i] < lo - 1e-3 or qpos[i] > hi + 1e-3:
                    viol += 1
        # Contact magnitude (sum of normal + friction force magnitudes).
        contact_total = 0.0
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, c, f)
            contact_total += float(np.linalg.norm(f[:3]))
        if self.fixed_base:
            fell = False
        else:
            fell = (
                z < self.fall_z_threshold
                or abs(rpy[0]) > self.fall_tilt_threshold
                or abs(rpy[1]) > self.fall_tilt_threshold
            )
        return StepStats(
            qpos=qpos.astype(np.float32),
            qvel=qvel.astype(np.float32),
            root_z=z,
            root_quat_xyzw=self.root_quat_xyzw().astype(np.float32),
            root_rpy=rpy.astype(np.float32),
            fell=bool(fell),
            joint_limit_violation=int(viol),
            contact_force_total=float(contact_total),
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
