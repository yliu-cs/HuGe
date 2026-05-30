# Kappa PI Upgrade Plan

本文档给出 Kappa 从当前轻量 `QwenSonicFM` 升级到 starVLA `QwenPI` 路线的实现方案。目标是解决当前模型的两个主要瓶颈：

1. action expert 容量过小。当前 Kappa 的 `LayerwiseFMActionHead` 约 13.82M 参数，实际是一个小型 TransformerEncoder。
2. VLM 信息结合过弱。当前实现将 Qwen 最后一层 hidden states 做 attention-mask mean pooling，压成单个 latent，再与 state latent 相加；动作头看不到 token 级、多层级视觉语言表征。

升级目标不是直接照搬 starVLA 原始 `QwenPI`，而是实现一版适配 Kappa 66-D SONIC action schema、34-D G1 proprio、当前训练脚本和 checkpoint 体系的 `KappaPI`。

## 1. 当前 Kappa 架构问题

当前模型路径：

- `src/kappa/models/qwen_sonicfm.py`
- `src/kappa/models/action_heads/layerwise_fm.py`
- `src/kappa/training/losses.py`
- `src/kappa/training/runner.py`

当前信息流：

```text
Qwen3-VL hidden_states[-1]
  -> attention-mask mean pooling
  -> qwen_proj: qwen_hidden -> 512

state: 34D
  -> state_encoder -> 512

condition = qwen_latent + state_latent + fallback_image_latent

noisy 64-D SONIC motion tokens
  -> 4-layer TransformerEncoder
  -> velocity head: 64-D motion velocity
  -> hand head: 2-D hand logits
```

主要问题：

- Qwen 输出被池化成单向量，token 级空间信息、视觉 token 信息、语言 token 对齐信息都被压掉。
- action expert 不执行 cross-attention，只是把 condition 通过加法注入 action hidden。
- 当前 action expert depth 固定为 4，hidden 固定为 512，容量明显低于 starVLA 的 GR00T/PI action heads。
- hand binary 是旁路 BCE head，和 motion flow matching 的耦合较弱。

## 2. 目标架构：KappaPI

KappaPI 采用 starVLA `QwenPI` 的核心思想：

```text
Qwen hidden_states[-N:]
  -> select/project each layer
  -> layer-wise cross-attention DiT

normalized action chunk
  -> flow matching noising
  -> action encoder + timestep encoder
  -> layer-wise cross-DiT
  -> velocity prediction
```

Kappa 适配点：

- action dim 保持 `ACTION_DIM=66`，其中前 64 维是 SONIC motion token，后 2 维是 hand binary。
- 对 66-D action 全量做 flow matching velocity prediction，不额外加 hand BCE。
- state dim 保持 `PROPRIO_DIM=34`，通过 state encoder 形成 state tokens 注入 action sequence。
- Qwen 继续使用 LoRA 策略；Stage 1/2 训练 LoRA，Stage 3 冻结 Qwen/LoRA。
- hidden size 已固定为 `1024`，层数固定为 `28`。

目标信息流：

```text
Qwen3-VL output_hidden_states=True
  -> hidden_states[-num_action_layers:]
  -> per-layer projector: qwen_hidden -> action_hidden
  -> vl_embs_list: list[(B, seq_len, action_hidden)]

state: (B, 34)
  -> StateEncoder
  -> state tokens: (B, num_state_tokens, action_hidden)

actions: (B, 50, 66)
  -> sample t and noise
  -> noisy_action = (1 - t) * noise + t * target_action
  -> ActionEncoder(noisy_action, t)
  -> action tokens: (B, 50, action_hidden)

sequence = [state tokens, future tokens, action tokens]
  -> DiT block 0 cross-attends vl_embs_list[0]
  -> DiT block 1 cross-attends vl_embs_list[1]
  -> ...
  -> action decoder
  -> pred_velocity: (B, 50, 66)
```

## 3. 采用 QwenPI 还是 QwenPI_v3 思路

推荐采用 `QwenPI_v3` 风格，而不是 raw `QwenPI`：

- raw `QwenPI` 让 Action DiT hidden 等于 Qwen hidden。Qwen3-VL-2B 的 text hidden 是 2048，如果 action DiT 也用 2048，action expert 会非常大。
- `QwenPI_v3` 引入 per-layer projectors，将 Qwen hidden 压到 `action_hidden_size`，再送入 layer-wise cross-DiT。
- Kappa 目前训练资源和数据规模更适合压缩版 PI。

建议命名：

- 模型配置：`architecture="pi"`
- 模型类：继续使用 `QwenSonicFM` 并通过配置切换，或新增 `QwenSonicPI`
- action head：`LayerwiseCrossDiTFMActionHead`
- 文件：`src/kappa/models/action_heads/layerwise_cross_dit_fm.py`

推荐先新增类，不直接替换旧 `LayerwiseFMActionHead`。保留旧模型用于 checkpoint 兼容、ablation 和 smoke test。

## 4. 关键配置项

在 `QwenSonicFMConfig` 中新增：

```python
architecture: str = "lite"  # "lite" | "pi"
action_hidden_size: int = 1024
num_action_layers: int = 28
num_attention_heads: int = 16
attention_head_dim: int = 64
action_ffn_multiplier: int = 4
action_dropout: float = 0.1
final_dropout: bool = True
interleave_self_attention: bool = True
num_future_tokens: int = 32
num_state_tokens: int = 1
flow_noise_beta_alpha: float = 1.5
flow_noise_beta_beta: float = 1.0
flow_noise_s: float = 0.999
num_timestep_buckets: int = 1000
train_all_action_dims_fm: bool = True
lambda_hand_bce: float = 0.0
lambda_hand_velocity: float = 1.0
vlm_layer_selection: str = "last_n"  # reserved
project_vlm_layers: bool = True
```

说明：

- `action_hidden_size` 和 `num_action_layers` 已固定为 `1024 / 28`。
- Kappa 采用 full PI，对齐 Qwen3-VL-2B 的 28 层 hidden states。
- 如果 `num_action_layers < num_qwen_layers`，默认取最后 N 层 hidden states。
- `num_future_tokens` 沿用 starVLA 思路，为 action expert 提供 learnable planning slots。
- `train_all_action_dims_fm=True` 表示 66 维全部做 flow matching，不额外加 hand BCE。

## 5. Action Head 设计

### 5.1 模块组成

新增 `LayerwiseCrossDiTFMActionHead`，包含：

- `ActionEncoder(action_dim=66, hidden_size=action_hidden_size)`
- `StateEncoder(state_dim=34, output_dim=action_hidden_size)`
- `future_tokens: nn.Embedding(num_future_tokens, action_hidden_size)`
- `position_embedding: nn.Embedding(max_seq_len, action_hidden_size)`
- `timestep_encoder`
- `transformer_blocks: nn.ModuleList[CrossDiTBlock]`
- `action_decoder: MLP(action_hidden_size -> 66)`

### 5.2 CrossDiTBlock

建议先实现 Kappa 本地简化版，不直接 import starVLA，以减少包路径和配置耦合。

每个 block：

```text
hidden_states
  -> self-attention over [state, future, action] tokens
  -> cross-attention Q=hidden_states, K/V=vl_embs_list[layer_idx]
  -> timestep-conditioned MLP / AdaLN
```

第一版可以简化 AdaLN：

- timestep embedding 加到 hidden states 或作为 MLP bias conditioning。
- 先使用 PreNorm Transformer block，保证稳定。
- 后续再对齐 starVLA 的 `ada_norm`。

建议 block 接口：

```python
class CrossDiTBlock(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        ...
```

### 5.3 Flow Matching 目标

建议改为 starVLA 风格：

```python
noise = torch.randn_like(actions)
t = sample_beta_time(batch)
noisy = (1 - t) * noise + t * actions
target_velocity = actions - noise
```

当前 Kappa 是：

```python
noisy = t * noise + (1 - t) * motion
target_velocity = noise - motion
```

两者方向相反，但只要训练和推理一致都能工作。升级时建议统一到 starVLA 方向：

```text
t=0: noise
t=1: data
dx/dt = data - noise
inference: actions = actions + dt * pred_velocity
```

这样便于和 starVLA 实现对齐。

## 6. Qwen 输出接口改造

当前 `encode_latent` 返回 `(B, 512)`。PI 需要返回 hidden state list。

建议新增方法：

```python
def encode_vlm_layers(
    self,
    *,
    qwen_inputs: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    ...
```

逻辑：

```python
out = self.qwen(
    **qwen_inputs,
    output_hidden_states=True,
    return_dict=True,
)
hidden_states = list(out.hidden_states[-self.config.num_action_layers:])
attention_mask = qwen_inputs.get("attention_mask")
projected = [
    self.vlm_projectors[i](h.to(projector_dtype))
    for i, h in enumerate(hidden_states)
]
return projected, attention_mask
```

如果 `num_action_layers` 大于 Qwen 层数，直接报错。

如果 `qwen_inputs is None`，PI 模式应报错，不再走 char-token fallback。PI 模式必须使用 Qwen processor。

## 7. State 注入方式

第一版推荐 state 作为显式 token，而不是拼进文本。

```text
state: (B, 34)
  -> MLP
  -> (B, 1, action_hidden)
  -> concat before future/action tokens
```

原因：

- Kappa 的 proprio 是连续控制状态，数值精度对动作影响强。
- 直接离散化进语言会增加 tokenizer 和 prompt 复杂度。
- Stage 1 EgoDex 没有真实 state，可以用 `state_mask=0` 将 state token 置零。

后续可选 `PI_v3` 的 state text injection：

```text
[STATE] <discretized proprio bins> [ACTION]
```

但这应作为第二阶段实验，不建议在第一版同时引入。

## 8. Hand 维度建模方案

当前 action 是 66D：

- `0:64` 是 SONIC motion token
- `64:66` 是 hand binary

最终选择：

1. 对 66D 全量做 flow matching velocity loss。
2. 不额外加 hand BCE。
3. 推理时直接使用 66D FM 输出。

这样做的原因：

- motion/hand 同轨迹建模更一致。
- PI action expert 容量已经足够，不需要再引入手部分支。
- 训练目标更干净，避免 FM/BCE 双目标冲突。

## 9. 训练阶段策略

### Stage 1

数据：EgoDex，只有 64D SONIC motion，hand masked。

建议：

- 训练 action expert 和 VLM projectors。
- Qwen LoRA 可训练。
- state token 使用 mask 置零。
- 目标是建立视觉语言到 SONIC motion latent 的基础映射。

风险：

- EgoDex 缺少 proprio，PI action head 会学不到 state 依赖。
- 但这可以作为视觉语言 action prior 初始化。

### Stage 2

数据：Humanoid-Everyday + G1-WBT。

建议：

- 加强 proprio-conditioned whole-body action learning。
- 默认训练 action expert + state encoder + VLM projectors。
- Qwen LoRA 开启，继续训练以适配 whole-body 数据。

### Stage 3

数据：proprietary real adaptation。

建议：

- 冻结 Qwen 和 LoRA。
- 训练 action expert、state encoder、projectors。
- 使用较低 LR。
- 保持 Qwen/LoRA 冻结，仅训练 action expert、projectors 和 state 路径。

## 10. 参数量预估

基于 starVLA action head 结构，在 Kappa 66D action、34D state、50 horizon 下，粗略 action expert 规模：

| action hidden | num layers | 预计 action expert |
| ---: | ---: | ---: |
| 512 | 28 | ~106M |
| 768 | 28 | ~238M |
| 1024 | 28 | ~421M |
| 2048 | 28 | ~1.68B |

当前默认选择是 `1024 / 28`，已经接近 full PI。若后续显存压力过大，再回退到 `768 / 16`。

## 11. 需要修改的文件

### 11.1 新增

```text
src/kappa/models/action_heads/layerwise_cross_dit_fm.py
```

包含：

- `TimestepEmbedder`
- `ActionEncoder`
- `StateEncoder`
- `CrossDiTBlock`
- `LayerwiseCrossDiTFMActionHead`

### 11.2 修改

```text
src/kappa/models/qwen_sonicfm.py
```

修改点：

- `QwenSonicFMConfig` 增加 PI 配置项。
- `__init__` 根据 `architecture` 选择 lite head 或 PI head。
- `_load_qwen` 后构建 `vlm_projectors`。
- 新增 `encode_vlm_layers`。
- `forward` 在 `architecture=="pi"` 时走 PI 分支。
- `predict_action` 在 `architecture=="pi"` 时走 PI sampling。
- checkpoint config 继续保存所有新增字段。

```text
src/kappa/training/losses.py
```

修改点：

- 支持 `pred["velocity"]` shape 为 `(B,T,66)`。
- 支持 `pred["target_velocity"]` shape 为 `(B,T,66)`。
- 不再使用 hand BCE。
- loss mask 从 `sonic_mask` 扩展到 66D。

```text
src/kappa/training/runner.py
```

修改点：

- 增加 CLI 参数：
  - `--architecture lite|pi`
  - `--action-hidden-size`
  - `--num-action-layers`
  - `--num-attention-heads`
  - `--num-future-tokens`
  - `--flow-noise-beta-alpha`
  - `--flow-noise-beta-beta`
  - `--flow-noise-s`
- PI 模式强制 `use_qwen=True` 和 `use_qwen_processor=True`。
- 不再保留 char-token fallback，lite/debug 也不保留。
- 训练日志打印 action expert 参数量、VLM projector 参数量、Qwen LoRA 参数量。

```text
src/kappa/data/collator.py
```

修改点：

- PI 模式必须返回 `qwen_inputs`。
- 保留 `attention_mask`，不要只截断 `input_ids` 而遗漏相关 mask。
- 如果后续支持多图，应保持 Qwen processor 的 image/video token 对齐。

```text
src/kappa/inference/policy.py
```

修改点：

- PI 模式 inference 必须构建 Qwen processor inputs。
- 调用 `model.predict_action(qwen_inputs=..., state=..., state_mask=...)`。
- hand dims 使用 hand logits 或 decoded hand override。

## 12. Checkpoint 兼容策略

旧 checkpoint 无法直接加载到 PI action head，因为 head 结构不同。

建议：

- `architecture="lite"` 的 checkpoint 仍用旧逻辑加载。
- `architecture="pi"` 的 checkpoint 加载新 head。
- 从旧 checkpoint 初始化 PI 时，只迁移：
  - Qwen/LoRA 权重
  - state encoder 可选择迁移，若维度一致
  - 不迁移旧 action head
- 提供 `strict=False` 加载，并在日志中打印 missing/unexpected keys。

推荐新增工具：

```text
scripts/train/convert_lite_to_pi_init.py
```

功能：

- 读取 lite checkpoint。
- 创建 PI config。
- 迁移 Qwen/LoRA 相关权重。
- 写出 PI 初始化 checkpoint。

## 13. 验证计划

### 13.1 单元 smoke

用随机输入验证：

- `LayerwiseCrossDiTFMActionHead.forward`
- `LayerwiseCrossDiTFMActionHead.predict_action`
- mask shape 正确
- output shape：
  - `velocity: (B,50,66)`
  - `hand_logits: (B,50,2)`

### 13.2 训练 smoke

```bash
scripts/train/run_kappa.sh stage1 \
  --architecture pi \
  --max-steps 1 \
  --max-episodes 1 \
  --batch-size 1
```

同样跑 stage2/stage3。

### 13.3 参数统计

训练启动时打印：

```text
[stageX] parameters:
  total
  trainable
  qwen_total
  qwen_trainable
  vlm_projectors
  state_encoder
  action_head
```

### 13.4 过拟合小集

用每阶段 1-4 个 episode 跑到 loss 明显下降：

- motion velocity loss 应快速下降。
- 66-D velocity loss 应在 stage2/stage3 继续下降。
- 如果 loss 不下降，优先检查 flow direction、mask、action normalization。

## 14. 训练稳定性建议

- action head LR 可以高于 Qwen LoRA LR。
- 推荐分组 optimizer：
  - Qwen LoRA: `1e-5` 到 `5e-5`
  - VLM projectors: `5e-5` 到 `1e-4`
  - action head: `1e-4`
  - state encoder: `1e-4`
- 使用 gradient clipping，保留 `max_grad_norm=1.0`。
- Stage 2 开启 LoRA，Stage 3 冻结 Qwen/LoRA。
- 如果出现 hand 抖动，推理时可沿用当前 `SonicAdapter` 的状态平滑。

## 15. 推荐实施顺序

### Step 1: 新 action head

实现 `layerwise_cross_dit_fm.py`，只用随机 tensor 验证 shape 和 loss。

### Step 2: 接入 QwenSonicFM

加入 `architecture="pi"` 分支，实现 Qwen hidden state list 提取和 projection。

### Step 3: 接入训练

修改 runner CLI、loss 和日志，跑 1-step smoke。

### Step 4: 小数据过拟合

Stage 2 选择少量 HE/WBT episode，验证 action expert 能拟合。

### Step 5: 全阶段训练

先跑：

```text
stage1: PI + LoRA
stage2: PI + LoRA
stage3: PI, freeze Qwen/LoRA
```

再做 ablation：

```text
num_action_layers 12 vs 16 vs 28
action_hidden_size 512 vs 768 vs 1024
FM 64D vs FM 66D
```

## 16. 关键风险

### 显存风险

PI 会保留多个 Qwen hidden states，并让 action DiT cross-attend token sequence。显存主要受以下因素影响：

- `batch_size`
- Qwen token length
- `num_action_layers`
- `action_hidden_size`
- 是否训练 Qwen LoRA

缓解：

- 降低 per-process batch。
- 控制 `max_text_tokens` 和 image resolution。
- 先用 `num_action_layers=12/16`。
- 使用 gradient checkpointing。
- 冻结 Qwen，减少反向图。

### 训练不稳定

flow matching direction 必须和 sampling 一致。建议完全对齐 starVLA：

```text
noisy = (1 - t) * noise + t * actions
target_velocity = actions - noise
sampling: x = x + dt * pred_velocity
```

### 数据阶段不一致

Stage 1 无 proprio，Stage 2/3 有 proprio。state token 需要 mask：

```text
state_mask=0 -> zero state token
state_mask=1 -> encoded state token
```

### checkpoint 断裂

PI head 和 lite head 不兼容。必须通过 `architecture` 明确区分。

## 17. 默认配置

最终默认配置：

```text
architecture=pi
action_hidden_size=1024
num_action_layers=28
num_attention_heads=16
attention_head_dim=64
num_future_tokens=32
num_state_tokens=1
num_inference_steps=4
flow_noise_beta_alpha=1.5
flow_noise_beta_beta=1.0
flow_noise_s=0.999
train_all_action_dims_fm=true
lambda_hand_bce=0.0
```

如果显存压力过大，再回退到：

```text
action_hidden_size=768
num_action_layers=16
```

## 18. 最终判断

Kappa 当前问题的核心不是 Qwen 不够强，而是 action expert 与 VLM 之间的信息瓶颈太窄。PI 路线应优先解决：

- 从单 latent 条件升级到 token-level 条件。
- 从最后层条件升级到 multi-layer 条件。
- 从小 TransformerEncoder 升级到 cross-DiT flow-matching action expert。

推荐实现路线是 `KappaPI_v3-lite`：

```text
Qwen3-VL hidden_states[-N:]
  -> per-layer projectors to action_hidden_size
  -> layer-wise cross-DiT FM action head
  -> 66D SONIC action chunk
```

这条路线保留 starVLA `QwenPI` 的核心建模能力，同时通过 hidden 压缩和层数裁剪控制工程成本，是 Kappa 当前最合理的升级方向。
