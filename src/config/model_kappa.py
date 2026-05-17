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

    # Post-train extras (3rd MoT expert + RTC + proprio). Defaults preserve
    # the pretrain stage 1-2 layout exactly so existing configs keep working.
    use_g1_expert: bool = False
    # 23-D = arms(14) + hand_bits(2) + loco_cmd(7). Legs+waist are not
    # predicted (WBC handles them at deployment); see
    # ``src.data.preprocess.build_g1_action_23`` for the channel layout.
    g1_dim: int = 23
    g1_loss_weight: float = 0.3
    use_proprio: bool = False
    proprio_dim: int = 64
    use_rtc: bool = False
    rtc_on_sonic: bool = True
    g1_max_delay: int = 10
    sonic_max_delay: int = 8
    proprio_noise_std: float = 0.02
    # Proprio now lives in the latent prefix (visible to BOTH SONIC and G1
    # via attention), so a non-zero default drop avoids the experts collapsing
    # onto proprio and ignoring the visual conditioning.
    proprio_drop_prob: float = 0.1
    g1_action_stats_path: str | None = None
    g1_proprio_stats_path: str | None = None

    # Post-train freeze toggles.
    freeze_for_post_train: bool = False
    init_g1_from_sonic: bool = False
