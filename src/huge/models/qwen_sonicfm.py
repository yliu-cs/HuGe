"""QwenSonicFM model facade."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from huge.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM
from huge.models.action_heads.starvla_gr00t_head import StarVLAGR00TActionHead
from huge.models.action_heads.starvla_oft_head import StarVLAOFTActionHead
from huge.models.action_heads.starvla_pi_head import StarVLAPIActionHead


PORTABLE_SAFETENSORS_SCHEMA = "huge_portable_safetensors_v1"
_SAFETENSORS_SCHEMA_KEY = "huge_schema"
_SAFETENSORS_ALIASES_KEY = "huge_shared_tensor_aliases"


def _portable_dtype_compatible(saved: torch.dtype, runtime: torch.dtype) -> bool:
    if saved == runtime:
        return True
    return saved in {torch.bfloat16, torch.float16} and runtime == torch.float32


def _shared_tensor_aliases(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Return deterministic aliases for exact shared model-state tensor views."""
    names = set(state_dict)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(group: list[str]) -> None:
        if len(group) < 2:
            return
        root = find(group[0])
        for name in group[1:]:
            other = find(name)
            if other != root:
                parent[other] = root

    object_groups: dict[int, list[str]] = {}
    for name, tensor in (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ):
        if name in names:
            object_groups.setdefault(id(tensor), []).append(name)
    for group in object_groups.values():
        union(group)

    view_groups: dict[tuple[Any, ...], list[str]] = {}
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"model state contains a non-tensor value: {name}")
        if tensor.device.type == "meta" or tensor.numel() == 0:
            continue
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        size = storage.nbytes()
        if pointer == 0 or size == 0:
            continue
        signature = (
            str(tensor.device),
            pointer,
            size,
            tensor.storage_offset(),
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        view_groups.setdefault(signature, []).append(name)
    for group in view_groups.values():
        union(group)

    components: dict[str, list[str]] = {}
    for name in sorted(names):
        components.setdefault(find(name), []).append(name)
    aliases: dict[str, str] = {}
    for group in components.values():
        if len(group) < 2:
            continue
        canonical = min(group)
        for name in sorted(group):
            if name != canonical:
                aliases[name] = canonical
    return aliases


def _save_portable_safetensors(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    destination: Path,
) -> None:
    from safetensors.torch import save_file

    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    actual_keys = set(state_dict)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "portable checkpoint state key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"unexpected={sorted(actual_keys - expected_keys)}"
        )
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"portable state contains a non-tensor value: {name}")
        if (
            (torch.is_floating_point(tensor) or torch.is_complex(tensor))
            and not bool(torch.isfinite(tensor).all().item())
        ):
            raise ValueError(f"portable checkpoint contains non-finite tensor: {name}")
    aliases = _shared_tensor_aliases(model, expected_state)
    for alias, canonical in aliases.items():
        alias_tensor = state_dict[alias]
        canonical_tensor = state_dict[canonical]
        if not torch.is_tensor(alias_tensor) or not torch.is_tensor(canonical_tensor):
            raise TypeError(f"portable shared state contains a non-tensor: {alias} -> {canonical}")
        if alias_tensor.shape != canonical_tensor.shape or alias_tensor.dtype != canonical_tensor.dtype:
            raise RuntimeError(f"portable shared alias shape/dtype mismatch: {alias} -> {canonical}")
        same_view = (
            alias_tensor.device == canonical_tensor.device
            and alias_tensor.untyped_storage().data_ptr()
            == canonical_tensor.untyped_storage().data_ptr()
            and alias_tensor.storage_offset() == canonical_tensor.storage_offset()
            and alias_tensor.stride() == canonical_tensor.stride()
        )
        if not same_view and not torch.equal(alias_tensor, canonical_tensor):
            raise RuntimeError(f"portable shared alias values differ: {alias} -> {canonical}")

    tensors: dict[str, torch.Tensor] = {}
    for name in sorted(expected_keys - set(aliases)):
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"portable state contains a non-tensor value: {name}")
        expected = expected_state[name]
        if tensor.shape != expected.shape or not _portable_dtype_compatible(
            tensor.dtype,
            expected.dtype,
        ):
            raise RuntimeError(
                f"portable tensor shape/dtype mismatch: {name}: "
                f"saved={tensor.dtype}, runtime={expected.dtype}"
            )
        tensors[name] = tensor.detach().cpu().contiguous()
    metadata = {
        _SAFETENSORS_SCHEMA_KEY: PORTABLE_SAFETENSORS_SCHEMA,
        _SAFETENSORS_ALIASES_KEY: json.dumps(
            aliases,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ),
    }
    save_file(tensors, str(destination), metadata=metadata)


def load_portable_safetensors_strict(
    model: nn.Module,
    path: str | Path,
    *,
    device: str | torch.device,
) -> None:
    """Strictly load historical or v1 alias-aware Huge weights."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"model.safetensors not found: {source}")
    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    expected_aliases = _shared_tensor_aliases(model, expected_state)
    with safe_open(source, framework="pt", device="cpu") as stream:
        checkpoint_keys = set(stream.keys())
        metadata = stream.metadata()

    canonical_keys = expected_keys - set(expected_aliases)
    no_metadata = metadata is None or metadata == {}
    legacy_full_key_format = no_metadata and checkpoint_keys == expected_keys
    legacy_deduplicated_format = no_metadata and checkpoint_keys == canonical_keys
    historical_save_model_format = metadata == expected_aliases and bool(expected_aliases)
    if no_metadata:
        if not legacy_full_key_format and not legacy_deduplicated_format:
            raise RuntimeError(
                "checkpoint key mismatch for metadata-free safetensors: "
                f"actual={sorted(checkpoint_keys)}, expected one of: "
                f"full-key={sorted(expected_keys)}, "
                f"canonical-only={sorted(canonical_keys)}"
            )
        expected_file_keys = expected_keys if legacy_full_key_format else canonical_keys
    elif historical_save_model_format:
        expected_file_keys = canonical_keys
    else:
        if not isinstance(metadata, dict) or set(metadata) != {
            _SAFETENSORS_SCHEMA_KEY,
            _SAFETENSORS_ALIASES_KEY,
        }:
            raise RuntimeError("checkpoint safetensors metadata schema mismatch")
        if metadata[_SAFETENSORS_SCHEMA_KEY] != PORTABLE_SAFETENSORS_SCHEMA:
            raise RuntimeError(
                "unsupported checkpoint safetensors schema: "
                f"{metadata[_SAFETENSORS_SCHEMA_KEY]!r}"
            )
        try:
            aliases = json.loads(metadata[_SAFETENSORS_ALIASES_KEY])
        except json.JSONDecodeError as exc:
            raise RuntimeError("invalid checkpoint shared-tensor metadata") from exc
        if aliases != expected_aliases:
            raise RuntimeError(
                "checkpoint shared-tensor aliases mismatch: "
                f"expected={expected_aliases}, actual={aliases}"
            )
        expected_file_keys = canonical_keys

    if checkpoint_keys != expected_file_keys:
        raise RuntimeError(
            "checkpoint key mismatch: "
            f"missing={sorted(expected_file_keys - checkpoint_keys)}, "
            f"unexpected={sorted(checkpoint_keys - expected_file_keys)}"
        )
    state_dict = load_file(str(source), device=str(device))
    for name, tensor in state_dict.items():
        expected = expected_state[name]
        if tensor.shape != expected.shape or not _portable_dtype_compatible(
            tensor.dtype,
            expected.dtype,
        ):
            raise RuntimeError(
                f"checkpoint tensor shape/dtype mismatch: {name}: "
                f"saved={tensor.dtype}, runtime={expected.dtype}"
            )
    if legacy_full_key_format:
        for alias, canonical in expected_aliases.items():
            alias_tensor = state_dict[alias]
            canonical_tensor = state_dict[canonical]
            if (
                alias_tensor.shape != canonical_tensor.shape
                or alias_tensor.dtype != canonical_tensor.dtype
                or not torch.equal(alias_tensor, canonical_tensor)
            ):
                raise RuntimeError(f"legacy checkpoint shared aliases differ: {alias} -> {canonical}")
    else:
        for alias, canonical in expected_aliases.items():
            canonical_tensor = state_dict[canonical]
            expected_alias = expected_state[alias]
            if canonical_tensor.shape != expected_alias.shape or not _portable_dtype_compatible(
                canonical_tensor.dtype,
                expected_alias.dtype,
            ):
                raise RuntimeError(
                    f"checkpoint shared alias shape/dtype mismatch: {alias} -> {canonical}: "
                    f"saved={canonical_tensor.dtype}, runtime={expected_alias.dtype}"
                )
            state_dict[alias] = canonical_tensor
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise ValueError(f"checkpoint contains non-finite tensor: {name}")


@dataclass(frozen=True)
class ActionHeadDefaults:
    hidden_size: int
    num_layers: int
    ffn_multiplier: int


ACTION_HEAD_DEFAULTS = {
    "qwenpi_v3": ActionHeadDefaults(hidden_size=1024, num_layers=16, ffn_multiplier=4),
    "qwengr00t": ActionHeadDefaults(hidden_size=1024, num_layers=16, ffn_multiplier=4),
    "qwenoft": ActionHeadDefaults(hidden_size=1024, num_layers=16, ffn_multiplier=4),
}


@dataclass
class QwenSonicFMConfig:
    model_architecture: str = "qwenpi_v3"
    action_hidden_size: int | None = None
    num_action_layers: int | None = None
    num_attention_heads: int | None = None
    action_ffn_multiplier: int | None = None
    action_dropout: float = 0.1
    num_future_tokens: int = 32
    state_dropout_prob: float = 0.2
    flow_noise_beta_alpha: float = 1.5
    flow_noise_beta_beta: float = 1.0
    flow_noise_s: float = 0.999
    action_horizon: int = ACTION_HORIZON
    state_dim: int = PROPRIO_DIM
    action_dim: int = ACTION_DIM
    num_inference_steps: int = 4
    base_vlm_path: str = "resource/model/Qwen3-VL-2B-Instruct"
    use_qwen: bool = True
    freeze_qwen: bool = False
    qwen_lora: bool = False
    train_qwen_lora: bool = False
    qwen_lora_r: int = 8
    qwen_lora_alpha: int = 16
    qwen_lora_dropout: float = 0.0
    qwen_lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    vlm_projector_type: str = "per_layer"
    oft_action_token: str = "\U0001f50d"
    oft_use_proprio: bool = True
    oft_proprio_bins: int = 256
    max_text_tokens: int | None = 48
    model_max_length: int = 8192
    char_max_length: int = 256
    qwen_image_size: tuple[int, int] = (256, 256)
    image_crop_fraction: float = 0.95
    image_color_jitter: bool = True

    @classmethod
    def from_checkpoint_config(cls, config: Mapping[str, Any]) -> "QwenSonicFMConfig":
        """Load config while preserving pre-proprio QwenOFT checkpoint behavior."""
        allowed_keys = {field.name for field in fields(cls)}
        values = {key: value for key, value in config.items() if key in allowed_keys}
        if (
            str(values.get("model_architecture", "qwenpi_v3") or "qwenpi_v3").lower() == "qwenoft"
            and "oft_use_proprio" not in config
        ):
            values["oft_use_proprio"] = False
        return cls(**values)


class QwenSonicFM(nn.Module):
    def __init__(self, config: QwenSonicFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or QwenSonicFMConfig()
        self.qwen = None
        if not self.config.use_qwen:
            raise ValueError("Huge now requires use_qwen=True; char-token fallback has been removed")
        self.action_head: nn.Module | None = None
        self.vlm_projector = nn.Identity()
        self.vlm_projectors = nn.ModuleList()
        self._load_qwen()

    def _load_qwen(self) -> None:
        from transformers import AutoModelForImageTextToText

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        impls = ["flash_attention_2", "sdpa", None] if torch.cuda.is_available() else ["sdpa", None]
        last_err: Exception | None = None
        for impl in impls:
            try:
                kwargs: dict[str, Any] = {"local_files_only": True, "dtype": dtype}
                if impl is not None:
                    kwargs["attn_implementation"] = impl
                self.qwen = AutoModelForImageTextToText.from_pretrained(
                    Path(self.config.base_vlm_path),
                    **kwargs,
                )
                break
            except Exception as exc:
                last_err = exc
        if self.qwen is None:
            raise RuntimeError("failed to load Qwen3VL with flash-attn/sdpa fallback") from last_err
        text_config = getattr(self.qwen.config, "text_config", self.qwen.config)
        hidden = getattr(self.qwen.config, "hidden_size", None) or getattr(text_config, "hidden_size", None)
        if hidden is None:
            raise ValueError("loaded Qwen model does not expose hidden_size")
        hidden = int(hidden)
        num_vl_layers = getattr(text_config, "num_hidden_layers", None)
        if num_vl_layers is None:
            raise ValueError("loaded Qwen model does not expose text_config.num_hidden_layers")
        self._init_action_structure(hidden, int(num_vl_layers))
        if self.config.train_qwen_lora and not self.config.qwen_lora:
            raise ValueError("train_qwen_lora requires qwen_lora")
        if self.config.qwen_lora:
            self._apply_qwen_lora()
        elif self.config.freeze_qwen:
            for p in self.qwen.parameters():
                p.requires_grad_(False)

    def _apply_qwen_lora(self) -> None:
        if self.qwen is None:
            raise RuntimeError("cannot apply LoRA before Qwen is loaded")
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("Qwen LoRA requires the `peft` package; run `uv sync` after updating dependencies") from exc

        for p in self.qwen.parameters():
            p.requires_grad_(False)
        lora_config = LoraConfig(
            r=self.config.qwen_lora_r,
            lora_alpha=self.config.qwen_lora_alpha,
            lora_dropout=self.config.qwen_lora_dropout,
            target_modules=list(self.config.qwen_lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        if not self.config.train_qwen_lora:
            for p in self.qwen.parameters():
                p.requires_grad_(False)

    def _architecture(self) -> str:
        architecture = str(self.config.model_architecture or "qwenpi_v3").lower()
        if architecture not in ACTION_HEAD_DEFAULTS:
            raise ValueError(f"unsupported model_architecture {self.config.model_architecture!r}")
        self.config.model_architecture = architecture
        return architecture

    @staticmethod
    def _make_projector(qwen_hidden_size: int, action_hidden_size: int) -> nn.Module:
        if qwen_hidden_size == action_hidden_size:
            return nn.Identity()
        return nn.Sequential(
            nn.LayerNorm(qwen_hidden_size),
            nn.Linear(qwen_hidden_size, action_hidden_size),
        )

    def _init_action_structure(self, qwen_hidden_size: int, num_vl_layers: int) -> None:
        architecture = self._architecture()
        defaults = ACTION_HEAD_DEFAULTS[architecture]
        action_hidden_size = int(self.config.action_hidden_size or defaults.hidden_size)
        num_action_layers = int(self.config.num_action_layers or defaults.num_layers)
        action_ffn_multiplier = int(self.config.action_ffn_multiplier or defaults.ffn_multiplier)
        if action_hidden_size % 64 != 0:
            raise ValueError(f"action_hidden_size {action_hidden_size} is not divisible by 64")
        num_heads = int(self.config.num_attention_heads or (action_hidden_size // 64))
        if num_action_layers <= 0:
            raise ValueError(f"num_action_layers must be positive, got {num_action_layers}")
        if action_ffn_multiplier <= 0:
            raise ValueError(f"action_ffn_multiplier must be positive, got {action_ffn_multiplier}")
        if architecture == "qwenpi_v3" and num_action_layers > num_vl_layers:
            raise ValueError(
                f"num_action_layers={num_action_layers} exceeds available VLM layers {num_vl_layers}"
            )
        if action_hidden_size % num_heads != 0:
            raise ValueError(f"action_hidden_size {action_hidden_size} must be divisible by num_heads {num_heads}")
        self.config.action_hidden_size = action_hidden_size
        self.config.num_action_layers = num_action_layers
        self.config.num_attention_heads = num_heads
        self.config.action_ffn_multiplier = action_ffn_multiplier
        if architecture == "qwenpi_v3":
            projector_type = str(self.config.vlm_projector_type or "per_layer").lower()
            if projector_type != "per_layer":
                raise ValueError(f"unsupported vlm_projector_type {self.config.vlm_projector_type!r}")
            self.config.vlm_projector_type = projector_type
            self.vlm_projectors = nn.ModuleList(
                [
                    self._make_projector(qwen_hidden_size, action_hidden_size)
                    for _ in range(num_action_layers)
                ]
            )
            self.action_head = StarVLAPIActionHead(
                hidden_size=action_hidden_size,
                action_dim=self.config.action_dim,
                state_dim=self.config.state_dim,
                action_horizon=self.config.action_horizon,
                num_layers=num_action_layers,
                num_heads=num_heads,
                ffn_multiplier=action_ffn_multiplier,
                dropout=self.config.action_dropout,
                num_inference_steps=self.config.num_inference_steps,
                noise_beta_alpha=self.config.flow_noise_beta_alpha,
                noise_beta_beta=self.config.flow_noise_beta_beta,
                noise_s=self.config.flow_noise_s,
                state_dropout_prob=self.config.state_dropout_prob,
            )
            return
        self.vlm_projector = self._make_projector(qwen_hidden_size, action_hidden_size)
        if architecture == "qwengr00t":
            self.action_head = StarVLAGR00TActionHead(
                hidden_size=action_hidden_size,
                action_dim=self.config.action_dim,
                state_dim=self.config.state_dim,
                action_horizon=self.config.action_horizon,
                num_layers=num_action_layers,
                num_heads=num_heads,
                ffn_multiplier=action_ffn_multiplier,
                dropout=self.config.action_dropout,
                num_future_tokens=self.config.num_future_tokens,
                num_inference_steps=self.config.num_inference_steps,
                noise_beta_alpha=self.config.flow_noise_beta_alpha,
                noise_beta_beta=self.config.flow_noise_beta_beta,
                noise_s=self.config.flow_noise_s,
                state_dropout_prob=self.config.state_dropout_prob,
            )
            return
        self.action_head = StarVLAOFTActionHead(
            hidden_size=action_hidden_size,
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            dropout=self.config.action_dropout,
        )

    def _qwen_grad_enabled(self) -> bool:
        return self.training and (self.config.train_qwen_lora or not self.config.freeze_qwen)

    def _encode_qwen(
        self,
        *,
        qwen_inputs: dict[str, torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if qwen_inputs is None:
            raise ValueError("Huge requires qwen_inputs from the Qwen processor")
        if self.qwen is None:
            raise RuntimeError("Huge requires a loaded Qwen model")
        qwen_inputs = {k: v.to(next(self.qwen.parameters()).device) if torch.is_tensor(v) else v for k, v in qwen_inputs.items()}
        with torch.set_grad_enabled(self._qwen_grad_enabled()):
            out = self.qwen(**qwen_inputs, output_hidden_states=True, return_dict=True, logits_to_keep=1)
        attention_mask = qwen_inputs.get("attention_mask")
        return list(out.hidden_states), attention_mask

    def encode_vlm_layers(
        self,
        *,
        qwen_inputs: dict[str, torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        hidden_states, attention_mask = self._encode_qwen(qwen_inputs=qwen_inputs)
        num_action_layers = int(self.config.num_action_layers or 0)
        if len(hidden_states) < num_action_layers:
            raise ValueError(
                f"num_action_layers={num_action_layers} exceeds available hidden states {len(hidden_states)}"
            )
        selected = hidden_states[-num_action_layers:]
        if attention_mask is not None and selected:
            attention_mask = attention_mask[:, : selected[0].shape[1]]
        return selected, attention_mask

    def _project_vlm_layers(self, vl_embs_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(vl_embs_list) != len(self.vlm_projectors):
            raise ValueError(f"expected {len(self.vlm_projectors)} VLM layers, got {len(vl_embs_list)}")
        out: list[torch.Tensor] = []
        for projector, hidden in zip(self.vlm_projectors, vl_embs_list, strict=True):
            if isinstance(projector, nn.Identity):
                out.append(hidden)
                continue
            target_dtype = next(projector.parameters()).dtype
            out.append(projector(hidden.to(dtype=target_dtype)))
        return out

    def _project_vlm_layer(self, hidden: torch.Tensor) -> torch.Tensor:
        if isinstance(self.vlm_projector, nn.Identity):
            return hidden
        target_dtype = next(self.vlm_projector.parameters()).dtype
        return self.vlm_projector(hidden.to(dtype=target_dtype))

    def _gather_action_queries(
        self,
        hidden: torch.Tensor,
        action_token_mask: torch.Tensor | None,
        *,
        horizon: int,
    ) -> torch.Tensor:
        if action_token_mask is None:
            raise ValueError("qwenoft requires action_token_mask from the Huge collator/policy")
        mask = action_token_mask.to(device=hidden.device, dtype=torch.bool)
        if mask.shape != hidden.shape[:2]:
            raise ValueError(f"action_token_mask shape {tuple(mask.shape)} does not match hidden {tuple(hidden.shape[:2])}")
        batch_size, seq_len, hidden_size = hidden.shape
        counts = mask.sum(dim=1)
        if (counts < horizon).any():
            bad = (counts < horizon).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"qwenoft requires at least {horizon} action tokens per sample; bad samples={bad} counts={counts.tolist()}")
        idx = torch.arange(seq_len, device=hidden.device).unsqueeze(0).expand(batch_size, seq_len)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))
        selected_pos = masked_pos.topk(k=horizon, dim=1).values.sort(dim=1).values
        gather_idx = selected_pos.unsqueeze(-1).expand(-1, -1, hidden_size)
        return hidden.gather(dim=1, index=gather_idx)

    def forward(
        self,
        *,
        normalized_actions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        qwen_inputs: dict[str, torch.Tensor] | None = None,
        action_token_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del input_ids, images
        if self.action_head is None:
            raise RuntimeError("action head is not initialized")
        architecture = self._architecture()
        if architecture == "qwenpi_v3":
            vl_embs_list, attention_mask = self.encode_vlm_layers(qwen_inputs=qwen_inputs)
            vl_embs_list = self._project_vlm_layers(vl_embs_list)
            return self.action_head(
                vl_embs_list=vl_embs_list,
                actions=normalized_actions.to(vl_embs_list[0].dtype),
                state=state,
                state_mask=state_mask,
                encoder_attention_mask=attention_mask,
                noise=noise,
                t=t,
            )
        hidden_states, attention_mask = self._encode_qwen(qwen_inputs=qwen_inputs)
        last_hidden = self._project_vlm_layer(hidden_states[-1])
        if attention_mask is not None:
            attention_mask = attention_mask[:, : last_hidden.shape[1]]
        if architecture == "qwengr00t":
            return self.action_head(
                vl_embs=last_hidden,
                actions=normalized_actions.to(last_hidden.dtype),
                state=state,
                state_mask=state_mask,
                encoder_attention_mask=attention_mask,
                noise=noise,
                t=t,
            )
        horizon = min(int(self.config.action_horizon), normalized_actions.shape[1])
        action_queries = self._gather_action_queries(last_hidden, action_token_mask, horizon=horizon)
        return self.action_head(
            action_queries=action_queries,
            actions=normalized_actions.to(action_queries.dtype),
        )

    @torch.no_grad()
    def predict_action(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        qwen_inputs: dict[str, torch.Tensor] | None = None,
        action_token_mask: torch.Tensor | None = None,
        horizon: int | None = None,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del input_ids, images
        if self.action_head is None:
            raise RuntimeError("action head is not initialized")
        architecture = self._architecture()
        if architecture == "qwenpi_v3":
            vl_embs_list, attention_mask = self.encode_vlm_layers(qwen_inputs=qwen_inputs)
            vl_embs_list = self._project_vlm_layers(vl_embs_list)
            return self.action_head.predict(
                vl_embs_list=vl_embs_list,
                state=state,
                state_mask=state_mask,
                encoder_attention_mask=attention_mask,
                horizon=horizon or self.config.action_horizon,
                num_inference_steps=num_inference_steps or self.config.num_inference_steps,
                generator=generator,
            )
        hidden_states, attention_mask = self._encode_qwen(qwen_inputs=qwen_inputs)
        last_hidden = self._project_vlm_layer(hidden_states[-1])
        if attention_mask is not None:
            attention_mask = attention_mask[:, : last_hidden.shape[1]]
        if architecture == "qwengr00t":
            return self.action_head.predict(
                vl_embs=last_hidden,
                state=state,
                state_mask=state_mask,
                encoder_attention_mask=attention_mask,
                horizon=horizon or self.config.action_horizon,
                num_inference_steps=num_inference_steps or self.config.num_inference_steps,
                generator=generator,
            )
        action_queries = self._gather_action_queries(
            last_hidden,
            action_token_mask,
            horizon=horizon or self.config.action_horizon,
        )
        return self.action_head.predict(action_queries)

    def save_checkpoint(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        from dataclasses import asdict

        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)
        _save_portable_safetensors(
            self,
            self.state_dict() if state_dict is None else state_dict,
            dst / "model.safetensors",
        )
        (dst / "config.json").write_text(
            json.dumps({"config": asdict(self.config), "metadata": metadata or {}}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> "QwenSonicFM":
        ckpt_path = Path(checkpoint_dir)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "model.safetensors"
        config_path = ckpt_path.parent / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        config_raw = payload.get("config", {})
        if not isinstance(config_raw, dict):
            raise ValueError(f"checkpoint config must be an object: {config_path}")
        config = QwenSonicFMConfig.from_checkpoint_config(config_raw)
        model = cls(config)
        load_portable_safetensors_strict(model, ckpt_path, device=map_location)
        return model
