"""Kappa model configuration."""
import os
from pathlib import Path
from .config import ModelConfig


class KappaModelConfig(ModelConfig):
    """Configuration for Kappa model."""

    model_name_or_path: str = os.path.join(
        Path(__file__).parent.parent.parent,
        "resource/model/Qwen3-VL-2B-Instruct"
    )
    tune_mm_llm: bool = True
    tune_mm_vision: bool = False
    tune_mm_mlp: bool = True

    weight_decay: float = 0.01
    gradient_checkpointing: bool = True

    mm_projector_lr: float = 5e-5

    model_max_length: int = 8192

    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora: bool = False

    use_mot: bool = True
    # 64 SONIC latent + 2 hand-binary bits (left/right open/close)
    sonic_dim: int = 66
    action_chunk_size: int = 50
    action_loss_weight: float = 1.0

    num_inference_steps: int = 4
    # ODE solver for the flow-matching denoise loop. "heun" is a 2nd-order
    # predictor-corrector (2 NFE per step except the final t=0 step which falls
    # back to Euler since the model wasn't trained at t=0); "euler" is the
    # original 1st-order scheme (1 NFE per step).
    solver: str = "heun"
