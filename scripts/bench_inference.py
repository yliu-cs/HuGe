"""Inference latency benchmark for the Kappa MoT model.

Loads the same model the trainer uses (``KappaModelWrapper`` +
``convert_to_mot``) and drives the flow-matching denoising loop on synthetic
PIL inputs, so the bench does not depend on the dataset or any checkpoint.

Usage::

    .venv/bin/python scripts/bench_inference.py
    .venv/bin/python scripts/bench_inference.py --batch-size 4 --num-iter 50
    .venv/bin/python scripts/bench_inference.py --num-inference-steps 4
    .venv/bin/python scripts/bench_inference.py --no-cache    # uncached baseline
    .venv/bin/python scripts/bench_inference.py \\
        --checkpoint .runs/<run>/ckpt_xxx/model.safetensors

The reported numbers correspond to one full action prediction
(``action_chunk_size`` SONIC tokens) for a single observation = image + text
instruction. The flow-matching loop integrates from t=1 to t=0 in
``num_inference_steps`` integration steps, using either the 1st-order Euler
solver (1 NFE per step) or the 2nd-order Heun predictor-corrector (2 NFE per
step except the final t=0 step which falls back to Euler since the model is
trained on t∈(0, 1) only). The training-time noise schedule is
``x_t = t*noise + (1-t)*clean`` so ``dx/dt = noise - clean = v``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from PIL import Image

# Make ``src.*`` resolvable when launched from anywhere in the repo.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _build_model(cfg, device: torch.device):
    from src.models import KappaModelWrapper

    wrapper = KappaModelWrapper(
        model_name_or_path=cfg.model_name_or_path,
        tune_vision=False,
        tune_llm=False,
        tune_mlp=False,
        device=device,
    )
    if cfg.use_mot:
        wrapper.convert_to_mot(
            sonic_dim=cfg.sonic_dim,
            action_loss_weight=cfg.action_loss_weight,
        )
    return wrapper


def _build_inputs(processor, batch_size: int, image_width: int, image_height: int, text: str, device: torch.device):
    """Build a batch identical to what ``KappaCollator`` produces, minus the
    flow-matching fields (those are constructed in the inference loop)."""
    from qwen_vl_utils import process_vision_info

    img = Image.new("RGB", (image_width, image_height), (128, 128, 128))
    messages = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }]
        for _ in range(batch_size)
    ]
    texts = [
        processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        for conv in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    out: dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
    return out


def _solver_nfe(num_steps: int, solver: str) -> int:
    """Total model forwards (NFE) consumed by ``num_steps`` integration steps."""
    if solver == "euler":
        return num_steps
    if solver == "heun":
        # 2 NFE per step; the final step falls back to Euler (1 NFE) because
        # the corrector point lands at t=0 where the model is not trained.
        return max(2 * num_steps - 1, 1)
    raise ValueError(f"Unknown solver: {solver!r}")


@torch.no_grad()
def _flow_match_infer(
    model,
    batch: dict[str, torch.Tensor],
    sonic_dim: int,
    chunk: int,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    solver: str = "heun",
    use_cache: bool = True,
) -> torch.Tensor:
    """Integrate the flow-matching ODE from t=1 (pure noise) to t=0.

    ``solver="euler"`` is the 1st-order scheme (1 NFE per step). ``solver="heun"``
    is a 2nd-order predictor-corrector that needs 2 NFE per step except the
    final step to t=0, which degrades to Euler — the model is trained on
    t∈[0.001, ~1) only and the corrector evaluation at t=0 would be
    out-of-distribution.

    With ``use_cache=True`` the language/vision prefix is encoded once via
    ``model.encode_latent`` and the per-step velocity evaluation only runs the
    action expert through cached K/V — order of magnitude faster on the inner
    NFEs (the first NFE pays the encode cost). With ``use_cache=False`` every
    NFE re-runs the full forward, which is the right baseline to compare
    against and the right path while validating numerical alignment.
    """
    B = batch["input_ids"].shape[0]
    x = torch.randn(B, chunk, sonic_dim, device=device, dtype=dtype)
    dt = 1.0 / num_steps

    if use_cache:
        cache = model.encode_latent(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            T_action=chunk,
        )

        def _v(x_in: torch.Tensor, t_val: float) -> torch.Tensor:
            timesteps = torch.full((B,), t_val, device=device, dtype=torch.float32)
            return model.denoise_step(cache, x_in, timesteps).to(x_in.dtype)
    else:
        def _v(x_in: torch.Tensor, t_val: float) -> torch.Tensor:
            timesteps = torch.full((B,), t_val, device=device, dtype=torch.float32)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                noisy_sonic_tokens=x_in,
                timesteps=timesteps,
            )
            # MoT path stashes velocity_pred in `logits` at inference.
            return out.logits.to(x_in.dtype)

    if solver == "euler":
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            v = _v(x, t_val)
            x = x - dt * v
    elif solver == "heun":
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            t_next = t_val - dt
            k1 = _v(x, t_val)
            x_pred = x - dt * k1
            if i == num_steps - 1:
                # Final step: corrector at t=0 is OOD, so degrade to Euler.
                x = x_pred
            else:
                k2 = _v(x_pred, t_next)
                x = x - 0.5 * dt * (k1 + k2)
    else:
        raise ValueError(f"Unknown solver: {solver!r}")
    return x


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument(
        "--num-inference-steps", type=int, default=None,
        help="Override KappaModelConfig.num_inference_steps (default: 4).",
    )
    ap.add_argument(
        "--solver", type=str, default=None, choices=["euler", "heun"],
        help="Override KappaModelConfig.solver. 'heun' is 2nd-order "
             "(2 NFE/step) and ~2x more accurate per step than 'euler'.",
    )
    ap.add_argument(
        "--image-width", type=int, default=320,
        help="Input width in pixels. Matches KappaModelTransform default (320W×224H). "
             "Both edges should be multiples of 32 (patch_size×merge_size) and the "
             "area should exceed processor shortest_edge=65536 to skip smart_resize.",
    )
    ap.add_argument(
        "--image-height", type=int, default=224,
        help="Input height in pixels. Matches KappaModelTransform default (320W×224H).",
    )
    ap.add_argument(
        "--text", type=str, default="perform the task",
        help="Instruction string. Length affects sequence length and per-step latency.",
    )
    ap.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Compute dtype on CUDA. CPU/MPS always use float32.",
    )
    ap.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional safetensors path to load trained weights "
             "(missing/unexpected keys are tolerated; default: random init for action expert).",
    )
    ap.add_argument(
        "--no-mot", action="store_true",
        help="Skip MoT conversion (benchmarks the bare Qwen3-VL forward, not the action expert).",
    )
    ap.add_argument(
        "--no-cache", action="store_true",
        help="Disable latent KV cross-step caching (every NFE re-runs the full forward). "
             "Use this to measure the uncached baseline or while validating numerical alignment.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(args.dtype, device)
    print(f"device={device}  dtype={dtype}")

    from src.config.model_kappa import KappaModelConfig

    cfg = KappaModelConfig()
    if args.no_mot:
        cfg.use_mot = False
    if args.num_inference_steps is not None:
        cfg.num_inference_steps = args.num_inference_steps
    if args.solver is not None:
        cfg.solver = args.solver

    print(f"Loading backbone from {cfg.model_name_or_path}")
    wrapper = _build_model(cfg, device)
    model = wrapper.model
    model.eval()

    if args.checkpoint:
        from safetensors.torch import load_file

        print(f"Loading checkpoint weights from {args.checkpoint}")
        state = load_file(args.checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  state_dict: missing={len(missing)}  unexpected={len(unexpected)}")

    batch = _build_inputs(
        wrapper.processor,
        batch_size=args.batch_size,
        image_width=args.image_width,
        image_height=args.image_height,
        text=args.text,
        device=device,
    )
    print(f"input_ids:    {tuple(batch['input_ids'].shape)}")
    if "pixel_values" in batch:
        print(f"pixel_values: {tuple(batch['pixel_values'].shape)}")
    if "image_grid_thw" in batch:
        print(f"image_grid_thw: {batch['image_grid_thw'].tolist()}")

    if not cfg.use_mot:
        # Without MoT the model is the bare HF VLM; just run a single forward
        # per "iteration" so the timing call still works.
        @torch.no_grad()
        def _step():
            return model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )
        run = _step
        denom_label = "forward"
    else:
        use_cache = not args.no_cache
        def run():
            return _flow_match_infer(
                model, batch,
                sonic_dim=cfg.sonic_dim,
                chunk=cfg.action_chunk_size,
                num_steps=cfg.num_inference_steps,
                device=device,
                dtype=dtype,
                solver=cfg.solver,
                use_cache=use_cache,
            )
        nfe = _solver_nfe(cfg.num_inference_steps, cfg.solver)
        cache_label = "cached" if use_cache else "uncached"
        denom_label = (
            f"{cfg.num_inference_steps}-step {cfg.solver} denoise "
            f"(NFE={nfe}, {cache_label})"
        )

    # Warmup: amortize cuDNN autotune, kernel JIT, and any lazy allocations.
    for _ in range(args.num_warmup):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(args.num_iter):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times) * 1000.0  # ms
    per_sample = arr / args.batch_size

    print()
    print("=== Inference latency ===")
    print(f"  mode:                {denom_label}")
    print(f"  batch_size:          {args.batch_size}")
    if cfg.use_mot:
        print(f"  num_inference_steps: {cfg.num_inference_steps}")
        print(f"  solver:              {cfg.solver}  (NFE={nfe})")
        print(f"  latent_kv_cache:     {'on' if not args.no_cache else 'off'}")
        print(f"  sonic_dim:           {cfg.sonic_dim}")
        print(f"  action_chunk_size:   {cfg.action_chunk_size}")
    print(f"  warmup / iter:       {args.num_warmup} / {args.num_iter}")
    print(
        f"  per-batch  ms:  mean={arr.mean():7.2f}  "
        f"p50={np.percentile(arr,50):7.2f}  "
        f"p90={np.percentile(arr,90):7.2f}  "
        f"min={arr.min():7.2f}"
    )
    print(
        f"  per-sample ms:  mean={per_sample.mean():7.2f}  "
        f"p50={np.percentile(per_sample,50):7.2f}  "
        f"p90={np.percentile(per_sample,90):7.2f}  "
        f"min={per_sample.min():7.2f}"
    )
    if cfg.use_mot:
        # Divide by NFE (actual model forwards), not num_inference_steps —
        # under Heun each integration step except the last consumes 2 forwards,
        # so num_inference_steps would mis-state per-forward latency.
        per_step = per_sample / nfe
        print(
            f"  per-NFE    ms:  mean={per_step.mean():7.2f}  "
            f"p50={np.percentile(per_step,50):7.2f}  "
            f"p90={np.percentile(per_step,90):7.2f}"
        )
        print(f"  throughput:          {1000.0 / per_sample.mean():.2f} obs/s "
              f"(end-to-end action prediction)")


if __name__ == "__main__":
    main()
