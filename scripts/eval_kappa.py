"""Kappa model inference pipeline.

Loads a trained Kappa checkpoint, runs it on an episode from humanoid-everyday,
decodes the predicted SONIC tokens via the ONNX SONIC decoder, and renders
the reconstructed trajectory in MuJoCo side-by-side with the ground truth.

Structure mirrors `humanoid everyday/main.py`, but the SONIC ONNX encoder is
replaced by the Kappa VLM, which predicts SONIC tokens from egocentric image
+ task text via flow-matching denoising.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
# Headless MuJoCo rendering: force EGL (fallback to osmesa via env override).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch

# torch must be imported before onnxruntime so cuDNN is discoverable
import onnxruntime as ort
ort.set_default_logger_severity(3)

import cv2
import mujoco
from PIL import Image
from safetensors.torch import load_file
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENE_XML = os.path.join(
    BASE_DIR, "humanoid everyday", "GR00T-WholeBodyControl",
    "gear_sonic_deploy", "g1", "scene_29dof.xml",
)
SONIC_DECODER_PATH = os.path.join(BASE_DIR, "resource", "model", "sonic", "model_decoder.onnx")
QWEN_MODEL_PATH = os.path.join(BASE_DIR, "resource", "model", "Qwen3-VL-2B-Instruct")
DATA_ROOT = os.path.join(BASE_DIR, "resource", "dataset", "humanoid-everyday")
DEFAULT_CKPT = os.path.join(
    BASE_DIR, ".runs", "kappa",
    "exp.kappa.b16.gpus1.2604230954", "checkpoints", "ckpt_10000",
)

EPISODE_INDEX = 7013
FPS = 30
RENDER_W, RENDER_H = 640, 480

ACTION_CHUNK_SIZE = 10
SONIC_LATENT_DIM = 64  # original SONIC ONNX decoder input
HAND_BITS = 2          # left/right binary open/close
SONIC_DIM = SONIC_LATENT_DIM + HAND_BITS  # 66
NUM_INFERENCE_STEPS = 10

# IsaacLab <-> MuJoCo joint ordering (from main.py)
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
MUJOCO_TO_ISAACLAB = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

NATURAL_FREQ = 10 * 2.0 * np.pi
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
STIFFNESS = {m: ARMATURE[m] * NATURAL_FREQ**2 for m in ARMATURE}
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
MOTOR_TYPE_PER_JOINT = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]
ACTION_SCALE_MUJOCO = np.array([0.25 * EFFORT[m] / STIFFNESS[m] for m in MOTOR_TYPE_PER_JOINT])
DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])


# ---------------------------------------------------------------------------
# Episode loading (matches main.py; LeRobot gives joint / odom / imu / images)
# ---------------------------------------------------------------------------
def load_episode(episode_idx: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        "USC-GVL/humanoid-everyday",
        root=DATA_ROOT,
        episodes=[episode_idx],
        tolerance_s=1e-2,
    )
    n = len(ds)
    arm = np.zeros((n, 14), dtype=np.float32)
    leg = np.zeros((n, 15), dtype=np.float32)
    hand = np.zeros((n, 14), dtype=np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)
    pos = np.zeros((n, 3), dtype=np.float32)
    gyro = np.zeros((n, 3), dtype=np.float32)
    pil_frames: list[Image.Image] = []
    rgb_frames: list[np.ndarray] = []

    task_idx: int | None = None
    for i in range(n):
        s = ds[i]
        arm[i] = s["observation.arm_joints"].numpy()
        leg[i] = s["observation.leg_joints"].numpy()
        hand[i] = s["observation.hand_joints"].numpy()
        quat[i] = s["observation.odometry.quat"].numpy()
        pos[i] = s["observation.odometry.position"].numpy()
        gyro[i] = s["observation.imu.gyroscope"].numpy()
        img = s["observation.images.egocentric"].numpy()  # (3, H, W) float in [0,1]
        img_u8 = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(img_u8))
        rgb_frames.append(cv2.resize(img_u8, (RENDER_W, RENDER_H)))
        if task_idx is None and "task_index" in s:
            try:
                task_idx = int(s["task_index"])
            except Exception:
                pass

    joint_pos_mujoco = np.zeros((n, 29), dtype=np.float64)
    joint_pos_mujoco[:, :15] = leg
    joint_pos_mujoco[:, 15:] = arm

    task_text = "perform the task"
    tasks_path = os.path.join(DATA_ROOT, "meta", "tasks.jsonl")
    if task_idx is not None and os.path.exists(tasks_path):
        with open(tasks_path) as f:
            for line in f:
                entry = json.loads(line)
                if int(entry["task_index"]) == task_idx:
                    task_text = entry.get("description") or entry.get("task") or task_text
                    break

    return joint_pos_mujoco, quat, pos, gyro, hand, pil_frames, rgb_frames, task_text


# ---------------------------------------------------------------------------
# Kappa model load + processor input construction
# (matches src/trainers/kappa.py::KappaCollator so the saved checkpoint sees
# input in the exact format it was trained on.)
# ---------------------------------------------------------------------------
def load_kappa_model(ckpt_dir: str, device: torch.device, dtype: torch.dtype):
    from src.models.kappa_model import KappaModelWrapper

    print(f"Loading Kappa backbone from {QWEN_MODEL_PATH} ...")
    wrapper = KappaModelWrapper(
        model_name_or_path=QWEN_MODEL_PATH,
        tune_vision=False, tune_llm=True, tune_mlp=True,
        device=device,
    )
    print("Converting backbone to MoT architecture (sonic_dim=64) ...")
    wrapper.convert_to_mot(sonic_dim=SONIC_DIM)
    wrapper.model.to(dtype=dtype, device=device)

    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    print(f"Loading checkpoint weights from {sf_path} ...")
    sd = load_file(sf_path)
    missing, unexpected = wrapper.model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys. First: {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys. First: {unexpected[:3]}")
    wrapper.model.eval()
    return wrapper


def build_processor_inputs(processor, pil_image: Image.Image, text: str, device: torch.device):
    """Construct (input_ids, attention_mask, pixel_values, image_grid_thw,
    mm_token_type_ids) identical to KappaCollator for a single sample."""
    from qwen_vl_utils import process_vision_info

    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": text},
        ],
    }]]
    texts = [text]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    image_token_id = 151655
    spatial_merge_size = 2
    image_grid_thw = inputs.get("image_grid_thw", None)
    input_ids = inputs["input_ids"][0]

    if image_grid_thw is not None:
        t = image_grid_thw[0]
        num_patches = int((t[0] * t[1] * t[2]) // (spatial_merge_size ** 2))
    else:
        num_patches = 0

    if num_patches > 0:
        image_tokens = torch.full((num_patches,), image_token_id, dtype=input_ids.dtype)
        new_input_ids = torch.cat([image_tokens, input_ids])
        mm_types = torch.cat([
            torch.ones((num_patches,), dtype=torch.long),
            torch.zeros((input_ids.shape[0],), dtype=torch.long),
        ])
    else:
        new_input_ids = input_ids
        mm_types = torch.zeros((input_ids.shape[0],), dtype=torch.long)

    pad_id = processor.tokenizer.pad_token_id or 0
    input_ids = new_input_ids.unsqueeze(0)
    attention_mask = input_ids.ne(pad_id)

    result = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "mm_token_type_ids": mm_types.unsqueeze(0).to(device),
    }
    if "pixel_values" in inputs:
        result["pixel_values"] = inputs["pixel_values"].to(device)
    if "image_grid_thw" in inputs:
        result["image_grid_thw"] = inputs["image_grid_thw"].to(device)
    return result


@torch.no_grad()
def predict_chunk_tokens(
    model, batch: dict, device: torch.device, dtype: torch.dtype,
    num_steps: int = NUM_INFERENCE_STEPS, chunk_size: int = ACTION_CHUNK_SIZE,
) -> np.ndarray:
    """Flow-matching Euler integration from noise (t=1) to clean (t=0).

    Training defines noisy = t*noise + (1-t)*sonic with velocity_target =
    noise - sonic, so dx/dt = v. Integrating backward: x_{t-dt} = x_t - v*dt.
    """
    B = 1
    dt = 1.0 / num_steps
    x = torch.randn(B, chunk_size, SONIC_DIM, device=device, dtype=torch.float32)

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        timesteps = torch.full((B,), t_val, device=device, dtype=torch.float32)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
            noisy_sonic_tokens=x.to(dtype),
            timesteps=timesteps,
            velocity_target=None,
        )
        velocity = out.logits.to(torch.float32)
        x = x - velocity * dt
    return x[0].detach().cpu().numpy().astype(np.float32)


def run_kappa_inference(
    wrapper, pil_frames, task_text: str,
    device: torch.device, dtype: torch.dtype,
    stride: int, num_steps: int,
) -> np.ndarray:
    """Predict SONIC tokens for the full episode (N, 64) via non-overlapping chunks."""
    n = len(pil_frames)
    tokens = np.zeros((n, SONIC_DIM), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)

    chunk_starts = list(range(0, n, stride))
    print(f"Running Kappa on {len(chunk_starts)} chunks "
          f"(stride={stride}, chunk_size={ACTION_CHUNK_SIZE}, steps/chunk={num_steps}) ...")

    model = wrapper.model
    processor = wrapper.processor

    for ci, start in enumerate(chunk_starts):
        batch = build_processor_inputs(processor, pil_frames[start], task_text, device)
        chunk_tokens = predict_chunk_tokens(
            model, batch, device, dtype,
            num_steps=num_steps, chunk_size=ACTION_CHUNK_SIZE,
        )
        for t in range(ACTION_CHUNK_SIZE):
            idx = start + t
            if idx >= n or filled[idx]:
                continue
            tokens[idx] = chunk_tokens[t]
            filled[idx] = True
        if (ci + 1) % 10 == 0 or ci == len(chunk_starts) - 1:
            print(f"  chunk {ci + 1}/{len(chunk_starts)} (frame {start})")

    if not filled.all():
        last = 0
        for i in range(n):
            if filled[i]:
                last = i
            else:
                tokens[i] = tokens[last]
    return tokens


# ---------------------------------------------------------------------------
# SONIC decoder -> PD targets (parallel to main.py)
# ---------------------------------------------------------------------------
def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def compute_gravity_dir(quat_wxyz):
    rot = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz))
    return rot.inv().apply(np.array([0.0, 0.0, -1.0]))


def build_decoder_input(token_state, hist_pos, hist_vel, hist_act, hist_angvel, hist_grav):
    obs = np.zeros(994, dtype=np.float32)
    offset = 0
    # Decoder consumes the original 64-D SONIC latent only; the trailing 2 hand
    # bits are stripped here and consumed separately for hand-state prediction.
    obs[offset:offset + 64] = token_state[:SONIC_LATENT_DIM]; offset += 64
    obs[offset:offset + 30] = hist_angvel.flatten(); offset += 30
    obs[offset:offset + 290] = hist_pos.flatten(); offset += 290
    obs[offset:offset + 290] = hist_vel.flatten(); offset += 290
    obs[offset:offset + 290] = hist_act.flatten(); offset += 290
    obs[offset:offset + 30] = hist_grav.flatten(); offset += 30
    return obs.reshape(1, -1)


def decode_tokens_to_pd_targets(tokens, joint_pos_mujoco, odom_quat, imu_gyro):
    dec_sess = ort.InferenceSession(SONIC_DECODER_PATH)
    n = len(tokens)

    jv = np.zeros_like(joint_pos_mujoco)
    jv[1:] = (joint_pos_mujoco[1:] - joint_pos_mujoco[:-1]) / (1.0 / FPS)

    body_q_isaac = (joint_pos_mujoco - DEFAULT_ANGLES_MUJOCO)[:, MUJOCO_TO_ISAACLAB]
    gt_action_isaac = ((joint_pos_mujoco - DEFAULT_ANGLES_MUJOCO) / ACTION_SCALE_MUJOCO)[:, MUJOCO_TO_ISAACLAB]
    jv_isaac = jv[:, MUJOCO_TO_ISAACLAB]

    pd_targets = np.zeros((n, 29), dtype=np.float64)
    for i in range(n):
        hp = np.zeros((10, 29), dtype=np.float32)
        hv = np.zeros((10, 29), dtype=np.float32)
        ha = np.zeros((10, 29), dtype=np.float32)
        hang = np.zeros((10, 3), dtype=np.float32)
        hgrav = np.zeros((10, 3), dtype=np.float32)
        for h in range(10):
            idx = max(0, i - (9 - h))
            hp[h] = body_q_isaac[idx].astype(np.float32)
            hv[h] = jv_isaac[idx].astype(np.float32)
            if idx > 0:
                ha[h] = gt_action_isaac[idx - 1].astype(np.float32)
            hang[h] = imu_gyro[idx]
            hgrav[h] = compute_gravity_dir(odom_quat[idx]).astype(np.float32)
        dec_input = build_decoder_input(tokens[i], hp, hv, ha, hang, hgrav)
        action_isaac = dec_sess.run(None, {"obs_dict": dec_input})[0][0]
        action_mujoco = action_isaac[ISAACLAB_TO_MUJOCO]
        pd_targets[i] = action_mujoco * ACTION_SCALE_MUJOCO + DEFAULT_ANGLES_MUJOCO
    return pd_targets


def smooth_and_correct(pd_targets, joint_pos_mujoco):
    recon = np.zeros_like(pd_targets)
    for j in range(29):
        pd_std = np.std(pd_targets[:, j])
        gt_std = np.std(joint_pos_mujoco[:, j])
        if pd_std > 1e-6:
            scale = gt_std / pd_std
            recon[:, j] = (pd_targets[:, j] - np.mean(pd_targets[:, j])) * scale + np.mean(joint_pos_mujoco[:, j])
        else:
            recon[:, j] = pd_targets[:, j] + (np.mean(joint_pos_mujoco[:, j]) - np.mean(pd_targets[:, j]))
    for j in range(29):
        recon[:, j] = gaussian_filter1d(recon[:, j], sigma=2.0)
    return recon


# ---------------------------------------------------------------------------
# MuJoCo rendering (parallel to main.py)
# ---------------------------------------------------------------------------
def setup_mujoco():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
    return model, data, renderer


def compute_min_geom_z(model, data):
    min_z = np.inf
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if model.geom_bodyid[g] == 0:
            continue
        min_z = min(min_z, data.geom_xpos[g, 2] - model.geom_rbound[g])
    return min_z


def render_trajectory(model, data, renderer, joint_pos_mujoco, odom_quat, odom_pos):
    n = len(joint_pos_mujoco)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("pelvis").id
    cam.distance = 3.0; cam.azimuth = -130; cam.elevation = -20
    cam.lookat[:] = [0, 0, 0.8]

    raw_z = np.zeros(n)
    for i in range(n):
        data.qpos[:3] = odom_pos[i]
        data.qpos[3:7] = odom_quat[i]
        data.qpos[7:36] = joint_pos_mujoco[i]
        mujoco.mj_forward(model, data)
        raw_z[i] = max(0.0, -compute_min_geom_z(model, data))
    z_off = np.maximum(gaussian_filter1d(raw_z, sigma=3.0), raw_z)

    frames = []
    for i in range(n):
        data.qpos[:3] = odom_pos[i]
        data.qpos[2] += z_off[i]
        data.qpos[3:7] = odom_quat[i]
        data.qpos[7:36] = joint_pos_mujoco[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, cam)
        frames.append(renderer.render().copy())
    return frames


def save_video(frames, path, label):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (RENDER_W, RENDER_H))
    for fr in frames:
        fr = cv2.resize(fr, (RENDER_W, RENDER_H))
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(bgr)
    writer.release()
    print(f"Saved video: {path}  ({len(frames)} frames, label: {label})")


def compute_error(gt, recon, label: str):
    diff = gt - recon
    mse = np.mean(diff ** 2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)
    print(f"--- {label} ---")
    print(f"  Overall MSE: {np.mean(mse):.6f} rad^2   MAE: {np.mean(mae):.6f} rad")
    print(f"  Max per-joint MSE: {np.max(mse):.6f} (joint {int(np.argmax(mse))})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Kappa -> SONIC decoder -> MuJoCo eval pipeline")
    ap.add_argument("--episode", type=int, default=EPISODE_INDEX)
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--video-dir", type=str, default="kappa_eval_video")
    ap.add_argument("--token-out", type=str, default="kappa_eval_tokens.npy")
    ap.add_argument("--load-tokens", type=str, default=None,
                    help="If set, load predicted SONIC tokens from this .npy and skip Kappa inference.")
    ap.add_argument("--stride", type=int, default=ACTION_CHUNK_SIZE,
                    help="Chunk stride (frames). stride==chunk_size => non-overlapping (fast).")
    ap.add_argument("--num-steps", type=int, default=NUM_INFERENCE_STEPS)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"\n=== Loading episode {args.episode} ===")
    joint_pos_mujoco, quat, pos, gyro, hand_joints, pil_frames, rgb_frames, task_text = load_episode(args.episode)
    n = len(pil_frames)
    print(f"  {n} frames. Task text: {task_text!r}")

    # Resolve cache_root (where sonic_tokens & hand_threshold live).
    cache_root = DATA_ROOT if os.access(DATA_ROOT, os.W_OK) else f"{DATA_ROOT}-cache"
    threshold_path = os.path.join(cache_root, "meta", "hand_threshold.json")
    if os.path.exists(threshold_path):
        from src.tokenizer.sonic_encoder import load_hand_thresholds, binarize_hands
        left_thr, right_thr = load_hand_thresholds(threshold_path)
        gt_hand_bits = binarize_hands(hand_joints, left_thr, right_thr)  # (n, 2) {0,1}
        print(f"  Hand thresholds: left={left_thr:.4f}, right={right_thr:.4f}")
    else:
        gt_hand_bits = None
        print(f"  [warn] no hand threshold at {threshold_path}; skipping hand eval")

    if args.load_tokens is not None:
        print(f"\n=== Loading precomputed SONIC tokens from {args.load_tokens} ===")
        tokens = np.load(args.load_tokens)
        assert tokens.shape == (n, SONIC_DIM), (
            f"Loaded tokens shape {tokens.shape} does not match expected ({n}, {SONIC_DIM})"
        )
    else:
        print(f"\n=== Loading Kappa checkpoint ===\n  {args.ckpt}")
        wrapper = load_kappa_model(args.ckpt, device=device, dtype=dtype)

        print("\n=== Running Kappa inference (flow-matching denoise) ===")
        tokens = run_kappa_inference(
            wrapper, pil_frames, task_text,
            device=device, dtype=dtype,
            stride=args.stride, num_steps=args.num_steps,
        )
        out_dir = os.path.dirname(args.token_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(args.token_out, tokens)
        print(f"Saved predicted SONIC tokens {tokens.shape} -> {args.token_out}")

        del wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gt_path = os.path.join(cache_root, "sonic_tokens", f"episode_{args.episode:06d}.npy")
    if os.path.exists(gt_path):
        gt_tokens = np.load(gt_path)
        m = min(gt_tokens.shape[0], tokens.shape[0])
        latent_mse = float(np.mean((tokens[:m, :SONIC_LATENT_DIM] - gt_tokens[:m, :SONIC_LATENT_DIM]) ** 2))
        print(f"Latent MSE  (pred vs GT, first {SONIC_LATENT_DIM} dims, {m} frames): {latent_mse:.6f}")
        # Hand-bit eval: predicted hand prob is the last 2 dims of `tokens`. Tokens are
        # produced by flow-matching denoising and are not strictly in {0,1}; threshold
        # at 0.5 to recover the binary state.
        pred_hand_bits = (tokens[:m, SONIC_LATENT_DIM:] > 0.5).astype(np.float32)
        gt_hand_bits_npy = (gt_tokens[:m, SONIC_LATENT_DIM:] > 0.5).astype(np.float32)
        left_acc = float(np.mean(pred_hand_bits[:, 0] == gt_hand_bits_npy[:, 0]))
        right_acc = float(np.mean(pred_hand_bits[:, 1] == gt_hand_bits_npy[:, 1]))
        print(f"Hand-bit accuracy (pred vs GT, {m} frames): left={left_acc:.3f}, right={right_acc:.3f}")
        if gt_hand_bits is not None:
            # Sanity: GT bits derived directly from raw hand_joints should agree with
            # the bits stored in the precomputed token npy.
            consistency = float(np.mean(gt_hand_bits[:m] == gt_hand_bits_npy))
            print(f"  GT-bit consistency (raw vs npy): {consistency:.4f} (expected 1.0)")

    print("\n=== Decoding tokens via SONIC decoder (ONNX) ===")
    pd_targets = decode_tokens_to_pd_targets(tokens, joint_pos_mujoco, quat, gyro)
    recon = smooth_and_correct(pd_targets, joint_pos_mujoco)

    print()
    compute_error(joint_pos_mujoco, recon, "Kappa reconstruction (after smoothing+affine correction)")
    compute_error(joint_pos_mujoco, pd_targets, "Raw PD targets from SONIC decoder")

    print("\n=== Rendering trajectories in MuJoCo ===")
    mj_model, mj_data, renderer = setup_mujoco()
    orig_frames = render_trajectory(mj_model, mj_data, renderer, joint_pos_mujoco, quat, pos)
    recon_frames = render_trajectory(mj_model, mj_data, renderer, recon, quat, pos)

    video_dir = os.path.join(BASE_DIR, args.video_dir)
    os.makedirs(video_dir, exist_ok=True)
    save_video(rgb_frames, os.path.join(video_dir, "kappa_ego_video.mp4"),
               f"Ego view (ep {args.episode})")
    save_video(orig_frames, os.path.join(video_dir, "kappa_orig_trajectory.mp4"),
               "Original Trajectory (GT)")
    save_video(recon_frames, os.path.join(video_dir, "kappa_recon_trajectory.mp4"),
               "Kappa Reconstructed Trajectory")
    print(f"\nAll outputs written under: {video_dir}")


if __name__ == "__main__":
    main()
