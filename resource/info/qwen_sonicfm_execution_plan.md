# QwenSonicFM VLA 执行计划

本文档总结当前项目从零重构后的 VLA 技术路线，并给出可执行的实现、训练、验证和部署计划。

## 1. 总目标

构建一个面向人形机器人全身控制的 VLA 模型，内部基于 starVLA 的模块化设计实现 `QwenSonicFM`，外部兼容 Isaac-GR00T / GR00T-WholeBodyControl 的 SONIC 控制链路。

核心目标：

- 使用 `Qwen3VL-2B` 作为 VLM backbone。
- 使用 flow-matching DiT action head 每次预测 50 步 future action chunk。
- 训练目标不是 Isaac-GR00T 默认的 78D schema，而是项目数据中实际存在的 66D schema：
  - SONIC universal latent token: 64D
  - 左手闭合 binary: 1D
  - 右手闭合 binary: 1D
- 保留扩展多个下游任务和多个数据源的能力。
- 使用纯 `uv` 管理 Python 环境、依赖和项目元数据。

## 2. 当前关键决策

### 2.1 内部实现和外部接口分层

采用“starVLA 内部研发 + Isaac-GR00T/SONIC 外部兼容”的方案。

内部：

```text
QwenSonicFM = Qwen3VL-2B backbone + Layerwise Flow-Matching DiT action head
```

外部：

```text
输出通过 adapter 转换为 GR00T-WholeBodyControl 可消费的 SONIC latent protocol
```

这意味着项目不直接复刻 Isaac-GR00T 的 `UNITREE_G1_SONIC` 78D schema，而是实现自定义 66D schema，并在部署边界做协议适配。

### 2.2 `NEW_EMBODIMENT` 的定位

Isaac-GR00T 中的 `NEW_EMBODIMENT` 不是已经实现好的 66D SONIC schema。它只是自定义机器人和自定义 modality config 的通用标签。

本项目应借鉴 `NEW_EMBODIMENT` 的思想：

- 自己定义 modality schema。
- 自己保存 schema 和 stats。
- 自己实现部署 adapter。

不要把本项目直接等同于 Isaac-GR00T 的原生 `UNITREE_G1_SONIC`，因为后者默认 action keys 更接近：

```text
motion_token: 64
left_hand_joints: 7
right_hand_joints: 7
total: 78
```

本项目目标是：

```text
motion_token: 64
left_hand_closed: 1
right_hand_closed: 1
total: 66
```

### 2.3 资源和参考实现边界

本轮重构只把项目源码重新组织到当前分支；数据、模型和参考项目的边界固定如下：

```text
数据根目录:
  resource/dataset

Qwen3VL-2B 本地模型:
  resource/model/Qwen3-VL-2B-Instruct

SONIC / WBC 相关模型:
  resource/model/sonic
  resource/model/sonic/model_encoder.onnx

参考实现:
  git master 分支: backup before full rewrite
  starVLA: 架构参考
  Isaac-GR00T: policy API / modality 思路参考
  GR00T-WholeBodyControl: SONIC 部署协议参考
```

约束：

- 所有数据读写都必须位于 `resource/dataset` 下；不要把 cache、索引或中间 token 写到仓库外部路径。
- 非 cache 的 WBT 数据位于 `resource/dataset/G1_WBT`，该路径是符号链接；cache 路径使用 `resource/dataset/G1_WBT-cache`。
- `humanoid-everyday`、`G1_WBT`、`EgoDex`、`proprietary` 的预处理、训练配置、推理部署细节优先参考 `master` 备份分支对应实现。
- 当前 `rewrite-from-scratch` 分支只保留新计划和资源；不要把 `master` 的旧源码原样复制回来，应把其中已经验证过的接口契约转写为新架构下的模块和测试。

## 3. VLA 输入输出定义

### 3.1 模型输入

每个训练样本的模型输入应统一为：

```text
image:
  ego_view
  可扩展 left_wrist_view
  可扩展 right_wrist_view
  部署兼容: GR00T-WholeBodyControl 中 right_wrist 可能映射为 wrist_view

language:
  task instruction

state:
  34D G1 proprioception
  EgoDex 阶段允许缺失

state_mask:
  1 表示 state 有效
  0 表示 state 缺失

dataset_id:
  egodex / humanoid_everyday / g1_wbt / proprietary_real / downstream_task_x

task_id:
  下游任务标识
```

### 3.2 模型输出

模型一次推理输出一个 future action chunk：

```text
normalized_actions:
  shape: [B, 50, 66]
```

其中：

```text
0:64   -> SONIC motion token
64:65  -> left_hand_closed logit / binary
65:66  -> right_hand_closed logit / binary
```

推荐外部 dict 形式：

```python
{
    "action.motion_token": FloatTensor[B, T, 64],
    "action.hand_binary": FloatTensor[B, T, 2],  # [left_closed, right_closed]
    # 可选兼容别名: "action.left_hand_closed" / "action.right_hand_closed"
}
```

### 3.3 部署端转换

部署时不应把 66D 原样发给低层控制器，而应转换为 SONIC latent protocol。

如果控制端接受 binary 手部闭合状态：

```text
motion_token[64] + left_closed[1] + right_closed[1]
```

如果控制端仍需要 7D 手部关节：

```text
left_closed = 1  -> 左手闭合预设 7D 姿态
left_closed = 0  -> 左手打开预设 7D 姿态
right_closed = 1 -> 右手闭合预设 7D 姿态
right_closed = 0 -> 右手打开预设 7D 姿态
```

部署 adapter 必须清晰隔离在 inference/deployment 层，不污染训练 schema。训练和部署统一使用 `closed=1/open=0` 极性；不要在代码里使用 `left_hand_open/right_hand_open` 作为主键，避免和数据元信息中的 `left_hand_closed/right_hand_closed` 相反。

### 3.4 SONIC token 和 hand binary 极性

66D action 的后两维固定为：

```text
action[..., 64] = left_hand_closed
action[..., 65] = right_hand_closed
```

本项目数据和备份分支实现均使用 `[left_closed, right_closed]`：

```text
resource/dataset/proprietary/*/meta/info.json: action.hand_binary names
src/tokenizer/sonic_encoder.py: binarize_hands / encode_episode_with_bits
```

语义：

```text
0 -> open
1 -> closed
```

HE 的 hand bits 来自 `observation.hand_joints` 的双阈值和 reverse causal scan；WBT 的 hand bits 来自 `hand_state + hand_type`；proprietary real 优先使用 parquet 中已经存在的 `action.hand_binary`。

## 4. 本体状态 schema 和归一化

### 4.1 34D proprio layout

当前项目应使用 `unitree_g1_sonic v1` 的 34D proprio layout：

```text
left_leg           6
right_leg          6
waist              3
left_arm           7
right_arm          7
left_hand          1
right_hand         1
projected_gravity  3
total             34
```

### 4.2 统计文件

本项目按训练阶段使用不同的本体状态归一化统计。

Stage 2 后训练使用 **HE + WBT** 计算出的 stats：

```text
resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json
或等价的 HE+WBT union stats 文件
```

每个下游任务 fine-tune 时，重新计算：

```text
HE + WBT + 当前下游任务
```

例如 `proprietary/real/2026-05-28` 的下游任务 stats：

```text
resource/dataset/proprietary/real/2026-05-28/meta/proprio_stats_union_sonic.json
```

该文件来自：

```text
HE + WBT + proprietary/real/2026-05-28
```

本地检查显示其中包含：

```text
min
max
mean
std
q01
q99
```

并且元信息显示：

```text
g1_proprio_dim = 34
layout = unitree_g1_sonic v1
```

### 4.3 归一化公式

本项目使用 percentile bounds 归一化，而不是 mean/std 标准化。

配置含义：

```json
{
  "use_percentiles": true,
  "clip_outliers": true
}
```

每个 group、每一维分别计算：

```python
lo = stats[group]["q01"]
hi = stats[group]["q99"]

x_clip = clip(x, lo, hi)
x_norm = 2 * (x_clip - lo) / (hi - lo + eps) - 1
```

输出范围：

```text
[-1, 1]
```

反归一化：

```python
x = (x_norm + 1) * 0.5 * (hi - lo) + lo
```

### 4.4 注意事项

- 同一个 checkpoint 的训练、验证、部署必须使用该 checkpoint metadata 记录的同一份 proprio stats。
- Stage 2 使用 HE+WBT stats；Stage 3 每个下游任务使用 HE+WBT+task stats。
- Stage 2 到 Stage 3 切换 stats 后，模型看到的同一物理姿态会变成不同 normalized value。
- Stage 3 必须训练 `state_encoder`，必要时也训练 action head 的输入投影层，用于重新校准新的 normalized state 空间。
- 不要把 missing state 的全零输入当作真实 proprio。归一化后的 0 通常代表统计中心附近的真实姿态。

## 5. 数据阶段设计

### 5.1 Stage 1: EgoDex 预训练

路径目标：

```text
resource/dataset/EgoDex
resource/dataset/EgoDex/egodex_repacked_simple
```

所有需要的训练数据、索引和 cache 都应位于 `resource/dataset` 下。备份分支默认使用 `resource/dataset/EgoDex/egodex_repacked_simple`，并允许通过 `cache_root` 自动解析到可写 cache 目录。

EgoDex 的核心价值：

- 学习视觉语义。
- 学习手物交互。
- 学习任务指令和动作意图。
- 为后续 G1/SONIC action learning 提供更好的 VLM 表征。

EgoDex 没有 G1 本体状态，因此不能直接当作完整 G1 proprio 分布训练。

推荐输入：

```text
image + language
state = zeros(34)
state_mask = 0
```

模型侧必须做：

```python
state_feat = state_encoder(state) * state_mask
```

不要只传全零 state 而不加 mask。

EgoDex 的训练目标以视觉语言到 SONIC latent 为主。备份分支实现中 EgoDex 只有 64D SONIC token，没有可靠 hand binary；因此 Stage 1 使用统一 66D action 容器，但后两维右补零，并通过 `sonic_mask` 屏蔽 hand binary loss。

```text
image + language -> sonic_token[64]
action shape        = [T, 66]
action[..., 0:64]   = SONIC token
action[..., 64:66]  = 0 padding
sonic_mask[..., 0:64]  = 1
sonic_mask[..., 64:66] = 0
```

注意：EgoDex 仍然没有 G1 本体状态，必须使用 `state_mask = 0`，不能把全零 state 当作真实站立姿态。Stage 1 不训练真实 proprio-conditioned 控制能力，也不监督 hand binary。

### 5.2 Stage 2: humanoid-everyday + G1_WBT 后训练

路径目标：

```text
resource/dataset/humanoid-everyday
resource/dataset/G1_WBT
```

当前数据路径约定：

```text
resource/dataset/humanoid-everyday
resource/dataset/humanoid-everyday-cache
resource/dataset/G1_WBT          # ln -s 原始 WBT 数据
resource/dataset/G1_WBT-cache
```

实现 dataloader 时必须支持：

- 原始目录模式。
- cache 目录模式。
- `cache_root=None` 时优先写入 `data_root`，不可写时使用 `<data_root>-cache`。
- `G1_WBT` 是符号链接，预处理和索引产物应写入 `G1_WBT-cache`，避免污染只读或外部挂载数据。
- 某些数据源暂缺 `sonic_tokens/`、`g1_proprio/`、`meta/episodes_meta.parquet` 时给出明确报错，而不是静默跳过。

Stage 2 直接混合 HE + WBT 训练，不拆成 HE-only 和 HE+WBT 两个阶段。

Stage 2 本体状态归一化：

```text
只使用 HE + WBT 计算出的 q01/q99 bounds
```

Stage 2 训练目标：

```text
image + language + normalized 34D proprio -> 66D action chunk
```

建议混合数据：

```text
humanoid-everyday: 日常人体/人形交互动作
G1_WBT: G1 whole-body tracking / body control 分布
```

WBT 数据来自 LeRobot 风格子集，预处理后需要包含：

```text
G1_WBT-cache/meta/episodes_meta.parquet
G1_WBT-cache/sonic_tokens/<subset>/episode_*.npy
G1_WBT-cache/g1_proprio/<subset>/episode_*.npy
```

如果某些 WBT 子集缺图像或语言，需要定义专门的 sampling policy：

- 有图像语言的数据用于完整 VLA 训练。
- 只有 proprio/action 的数据用于 action head / state-conditioned training。
- 必须使用 dataset_id 区分数据来源。

Stage 2 混合比例默认使用自然帧比例，而不是强行 1:1；备份分支 HE-G1 与 WBT 约为 `1 : 1.46`。如需任务均衡，应显式在 mixture config 中记录采样权重。

### 5.3 Stage 3: proprietary real 微调

路径：

```text
resource/dataset/proprietary/real/2026-05-28
```

本地检查结果：

```text
sonic_tokens: 388 episodes, sample shape (1512, 66)
g1_proprio:   388 episodes, sample shape (1512, 34)
videos:       exists
meta:         stats, modality, tasks, episodes, info
action.hand_binary names: [left_hand_closed, right_hand_closed]
```

Stage 3 目标：

- 适配真实机器人。
- 适配真实相机。
- 适配实际任务语言分布。
- 保留 Stage 2 学到的全身运动能力。

每个下游任务 fine-tune 前都要重新计算该任务自己的三源合并 stats：

```text
HE + WBT + downstream_task
```

例如：

```text
HE + WBT + proprietary/real/2026-05-28
```

该阶段使用新的 merged stats 重新归一化 proprio，并从 Stage 2 checkpoint 开始微调。

建议：

- 小学习率。
- 冻结大部分 VLM。
- 必须训练 state encoder；同时训练 action head、必要的 LoRA。
- 使用 replay/mix 防止忘记 HE/WBT 的基础运动能力。

### 5.4 预处理流水线

重构后保留备份分支已经验证过的预处理边界，但重新组织源码。所有预处理产物仍写入 `resource/dataset`。

共同约束：

```text
目标 SONIC 时间轴: 50Hz
SONIC encoder: resource/model/sonic/model_encoder.onnx
SONIC token 文件: episode_*.npy, float32, shape [T_50hz, 66]
proprio 文件: episode_*.npy, float32, shape [T_50hz, 34]
写文件方式: 临时文件 + os.replace 原子替换
恢复方式: skip-if-exists，但必须校验 metadata 中的 fps / encoder id
```

HE：

```text
1. 仅保留 robot_type=g1；不要把 H1 关节布局送入同一个 G1 SONIC encoder。
2. 先计算 hand_threshold.json。
3. 将原始帧率重采样到 50Hz。
4. 根据 hand joints 双阈值 + reverse causal scan 生成 [left_closed, right_closed]。
5. 生成 sonic_tokens/ 和 g1_proprio/。
```

WBT：

```text
1. 从 resource/dataset/G1_WBT 符号链接读取各 subset。
2. 默认 source_fps=30，target_fps=50。
3. 根据 hand_state 和 hand_type 生成 hand bits。
4. 生成 G1_WBT-cache/meta/episodes_meta.parquet。
5. 生成 G1_WBT-cache/sonic_tokens/<subset>/ 和 g1_proprio/<subset>/。
```

Proprietary real：

```text
1. 读取 LeRobot v2.1 dump。
2. 优先读取 parquet 中已有 action.motion_token 和 action.hand_binary。
3. 生成本地 sonic_tokens/ 和 g1_proprio/ cache。
4. 每个下游任务重新计算 HE + WBT + task union proprio stats。
```

重采样必须由共享工具实现：连续量使用 linear interpolation；WXYZ quaternion 使用符号对齐后的 lerp + normalize。不要在每个 dataset 中分别实现不同版本。

### 5.5 时间对齐和 chunk 构造

统一在 50Hz token 时间轴采样：

```text
observation image/state at t
-> target action chunk [t, t + 49]
-> shape [50, 66]
```

约束：

- 图像原始帧率可能是 30Hz，使用 `frame_idx = min(int(t * video_len / token_len), video_len - 1)` 映射到对应视频帧。
- proprio、SONIC token 和 hand binary 必须位于同一条 50Hz 时间轴。
- episode 尾部不足 50 步时允许右侧 padding，但必须输出 `sonic_mask`，loss 只统计有效帧和有效维度。
- split 必须按 episode 划分，不能按 frame 随机划分，避免相邻帧泄漏。
- 索引文件要记录生成版本、源路径、episode 数量、fps 和 SONIC encoder id，参数变化后不得复用旧 index。

## 6. 模型架构

### 6.1 Backbone

选择：

```text
Qwen3VL-2B
```

理由：

- 比 4B/7B 更容易快速迭代。
- 对当前从零重构阶段更务实。
- VLA 控制能力主要由 action head 和数据质量决定。
- 后续可以替换成 Qwen3VL-4B、Qwen3.5VL 或 Cosmos-Reason2，而不改变 action schema。

初始训练策略：

```text
Stage 1: Qwen attention LoRA + projector + action head
Stage 2: 保留 Stage 1 LoRA 结构但冻结 Qwen/LoRA，训练 action head + state encoder
Stage 3: 保留 Stage 1 LoRA 结构但冻结 Qwen/LoRA，训练 action head + state encoder
```

### 6.2 Action Head

基于 starVLA 的 `QwenPI` 思路实现：

```text
Layerwise Flow-Matching DiT
```

原因：

- SONIC token 是连续 latent，不适合自回归文本 token 生成。
- 64D latent + 长 horizon 更像连续轨迹生成。
- Layerwise cross-DiT 能利用 VLM 多层 hidden states，比只用最后一层更适合高维控制。

### 6.3 `QwenSonicFM` 推荐结构

第一版目标结构：

```text
images + language + optional normalized proprio
       |
Qwen3VL-2B
       |
multi-layer hidden states / latent prefix
       |
Layerwise Flow-Matching DiT / MoT-style action expert
       |
sonic_token_head: 64D continuous
hand_binary_head: 2D closed logits
```

备份分支已有 MoT-style Qwen action expert 可作为实现参考：

```text
src/models/mot_kappa.py
sequence = [image+text | optional proprio | timestep + sonic action]
proprio_embedder: 34D -> hidden
action_embedder: 66D -> hidden
velocity_head: hidden -> 66D
```

新重构如果采用 starVLA Layerwise FM DiT，应保留同等接口：`encode_latent` / `denoise_step` / `predict_action`，并把 proprio 放在可缓存的 latent prefix 中。

建议不要把 66 维全部作为普通连续 flow-matching target。

推荐拆分：

```text
action[0:64]  -> flow-matching velocity loss
action[64:66] -> BCEWithLogitsLoss / focal BCE，目标为 [left_closed, right_closed]
```

这样可以避免把离散手部开合当作连续扩散变量，提升稳定性。

### 6.4 Horizon

建议默认对齐 SONIC/GR00T 推理节奏：

```text
action_horizon = 50
control_freq = 50Hz
每次预测 50 步 action chunk
```

本地数据是 50Hz 序列，应从每个 timestep 采样未来 50 帧作为 action chunk，对应约 1 秒的未来动作。

如果显存或训练不稳定，可以只在最小 smoke test 中临时用更短 horizon：

```text
action_horizon = 16 或 32
```

但正式训练和 checkpoint metadata 必须固定为 50。GR00T-WholeBodyControl 旧脚本可能默认 `action_horizon=40`；Kappa server 必须显式声明 `action_horizon=50`，部署端只允许在 50 步 chunk 内做 latency skip，不允许训练 50、部署 40 的隐式不一致。

### 6.5 图像和文本预处理

训练和部署必须使用同一套 Qwen3VL processor 与图像尺寸约束：

```text
base_vlm_path = resource/model/Qwen3-VL-2B-Instruct
processor     = AutoProcessor.from_pretrained(base_vlm_path, local_files_only=True)
chat template = Qwen3VL processor.apply_chat_template
```

备份分支默认图像处理：

```text
Resize(short_side=224)
CenterCrop(H=224, W=320)
输出尺寸: 224 x 320
Qwen merged vision tokens: 约 70
```

部署端相机帧如果是 640x480，必须先 resize 到训练尺寸，再送入 Qwen processor；否则 Qwen 看到的 vision token 数和训练分布不一致。文本输入统一为 `annotation.human.task_description` 或 `task`，缺失时 fallback 到 `perform the task`，但数据审计必须统计 fallback 比例。

### 6.6 训练可调参数参考

备份分支可作为默认超参起点，新实现应在配置中显式记录：

```text
model_max_length: 8192
mixed_precision: bf16
gradient_checkpointing: true
num_inference_steps: 4
train_batch_size: 128
num_workers: 8
prefetch_factor: 2
block_shuffle_size: 64
frame_cache_size: 2
Stage 1 learning_rate: 5e-5
Stage 2 learning_rate: 3e-5
Stage 3 learning_rate: 2e-5
Stage 1 checkpointing_steps: 30000
Stage 2 checkpointing_steps: 10000
Stage 3 checkpointing_steps: 1000
```

这些不是不可变常量，但每次实验必须落到 run config 和 checkpoint metadata 中。

## 7. Loss 设计

### 7.1 SONIC token loss

Flow matching 训练目标可以使用以下等价约定之一，但训练和部署 sampler 必须完全一致。推荐沿用备份分支的 denoise 方向：

```python
noise = randn_like(action)
t = sample_beta_time()          # t=1 表示纯噪声，t=0 表示干净 action
noisy = t * noise + (1 - t) * target
velocity = noise - target
pred_velocity = model(noisy, t, condition)
loss_token = mse(pred_velocity, velocity)
```

推理时从 `x_1 = noise` 开始，按 Euler 积分：

```python
x = x - dt * pred_velocity
```

如果实现采用 starVLA 的反向 t 定义，也必须在 `action_head.forward()`、`predict_action()`、server sampler 和文档中显式记录，禁止训练和部署使用不同 convention。

只对前 64D 使用：

```text
target = action[..., :64]
```

### 7.2 手部 binary loss

输出 logits：

```text
hand_logits = model output[..., 64:66]
hand_target = action[..., 64:66]  # [left_closed, right_closed]
```

训练：

```python
loss_hand = BCEWithLogitsLoss(hand_logits, hand_target)
```

EgoDex Stage 1 没有 hand target，应通过 per-dim `sonic_mask` 或专用 `hand_mask=0` 跳过该项。

可选：

- 如果 open/close 类别不平衡，使用 positive weight 或 focal BCE。
- 评估时记录 left/right accuracy、precision、recall 和 transition accuracy。

### 7.3 总 loss

建议：

```python
loss = loss_token + lambda_hand * loss_hand
```

初始：

```text
lambda_hand = 0.1 到 1.0
```

后续根据 hand loss scale 调整。

### 7.4 Action token scale

SONIC token 是预计算 latent feature，备份分支使用 identity field transform，不再对 action 做额外 q01/q99 归一化。训练目标命名为 `normalized_actions` 时仅表示模型内部统一张量，并不意味着对 64D SONIC token 再做 percentile normalization。

部署前必须检查：

```text
motion_token finite
abs(motion_token).max() <= 1.25  # 参考 GR00T-WholeBodyControl run_vla_inference.py
hand_binary logits/prob 能稳定阈值化
```

如后续引入 action normalizer，必须把 action stats、公式、反归一化和 metadata 写入 checkpoint，并同步修改部署 adapter。第一版不引入额外 action normalizer。

### 7.5 可选辅助 loss

可后续添加：

- token temporal smoothness loss
- hand transition loss
- chunk overlap consistency loss
- state reconstruction / proprio embedding auxiliary loss
- language-conditioned task classification loss
- token distribution bound penalty / OOD penalty

不要在第一版同时加入过多辅助 loss，先建立可跑通的最小闭环。

## 8. 代码结构建议

项目已经进入从零重构分支，应在 `src` 下建立清晰边界。

建议目录：

```text
src/
  kappa/
    data/
      schemas.py
      proprio_normalizer.py
      sonic_dataset.py
      egodex_dataset.py
      mixture_dataset.py
      transforms.py
      tokenized_video_dataset.py
      video_reader.py
      preprocess.py
      sampler.py
    models/
      qwen_sonicfm.py
      action_heads/
        layerwise_fm.py
        hand_binary_head.py
      vlm/
        qwen3vl.py
    training/
      train_stage1_egodex.py
      train_stage2_posttrain.py
      train_stage3_finetune.py
      losses.py
      checkpointing.py
    inference/
      policy.py
      sonic_adapter.py
      server.py
      run_kappa_server.py
    config/
      data.py
      model.py
      train.py
      train/
        pretrain_egodex.py
        posttrain_he_wbt.py
        finetune_prop.py
scripts/
  preprocess/
    humanoid_everyday/
    g1_wbt/
    proprietary/
  eval/
  train/
```

注意：

- `starVLA` 当前被 `.gitignore` 忽略，可作为参考库，不应直接混入项目源码。
- `Isaac-GR00T` 和 `GR00T-WholeBodyControl` 也应作为参考/外部依赖，不要把本项目强耦合到它们的内部训练代码。
- 项目 Python 使用纯 `uv` 管理。

## 9. uv 环境管理

当前已经完成：

```text
.venv
.python-version
pyproject.toml
```

`pyproject.toml` 当前最小内容：

```toml
[project]
name = "kappa"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []
```

后续规则：

- 不使用 conda 管理本项目 Python 依赖。
- 不手动 pip install 到系统环境。
- 新依赖使用：

```bash
uv add <package>
```

- 运行命令使用：

```bash
uv run python ...
```

- 生成锁文件：

```bash
uv lock
```

需要尽早添加基础依赖：

```text
torch
torchvision
torchaudio
transformers
accelerate
deepspeed
numpy
pillow
opencv-python
pyyaml
omegaconf
tqdm
einops
safetensors
qwen-vl-utils
polars
pydantic
tyro
scipy
onnxruntime-gpu
decord
pyzmq
msgpack
msgpack-numpy
fastapi
uvicorn[standard]
websockets
gear_sonic  # editable path: GR00T-WholeBodyControl/gear_sonic
flash-attn  # 集群 CUDA / torch 版本匹配时启用
```

具体版本应根据 Qwen3VL-2B、CUDA 和集群环境确定。备份分支曾使用 PyTorch CUDA 12.8 索引、`torch==2.7.1`、`torchvision==0.22.1`、`transformers>=5.5.4`、`qwen-vl-utils>=0.0.14`、`lerobot==0.3.3`；新分支应重新验证后写入 `uv.lock`。

## 10. 训练执行计划

### 10.1 Phase 0: 数据和接口验证

目标：

- 跑通 dataloader。
- 确认 shape。
- 确认 state normalization。
- 确认 66D action slicing。

检查项：

```text
image shape = 224 x 320 x 3 after transform
language not empty
state shape = [34]
action shape = [T, 66]
T = 50
action[..., :64] finite
EgoDex: sonic_mask[..., 64:66] = 0
HE/WBT/PROP: action[..., 64:66] in {0, 1} or close to binary
hand polarity = [left_closed, right_closed]
state_mask correct
cache path remains under resource/dataset
```

还要验证：

```text
HE robot_type filter excludes H1 by default
WBT original path resolves through resource/dataset/G1_WBT symlink
WBT cache files exist under resource/dataset/G1_WBT-cache
video frame index maps token timeline correctly
SONIC encoder path exists under resource/model/sonic/model_encoder.onnx
Qwen processor loads from resource/model/Qwen3-VL-2B-Instruct with local_files_only=True
```

产物：

```text
resource/info/data_audit_report.md
```

### 10.2 Phase 1: EgoDex 预训练

目标：

- 建立视觉语言表征。
- 避免污染 proprio 分布。
- 学习 EgoDex 的 `image + language -> 64D SONIC latent` 映射，并在 66D 容器中 mask 掉 hand dims。

输入：

```text
image + language
state_mask = 0
```

训练模块：

```text
VLM adapter / projector
QwenSonicFM action head 的 64D token 分支
```

训练标签：

```text
sonic_token[64] + zero_padding[2]
sonic_mask 屏蔽 hand dims
```

本阶段不训练真实 proprio-conditioned 控制能力；`state_mask=0` 只表示 state 缺失。hand binary head 可初始化但不计算监督损失。

### 10.3 Phase 2: HE + WBT 后训练

目标：

- 学习 G1/SONIC 66D action。
- 学习 proprio-conditioned whole-body latent control。

输入：

```text
image + language + normalized 34D state
state_mask = 1
```

归一化：

```text
HE + WBT stats only
```

训练模块：

```text
Layerwise FM action head
state encoder
必要 adapter
```

推荐冻结：

```text
大部分 Qwen3VL backbone
```

### 10.4 Phase 3: proprietary real 微调

目标：

- 真实机器人任务适配。
- 真实相机/环境适配。
- 保持基础运动能力。

策略：

```text
重新计算 HE + WBT + 当前下游任务 stats
用新 stats 重新归一化 proprio
低学习率
短周期验证
混入少量 HE/WBT replay
保存多个 checkpoint
```

需要重点监控：

```text
token loss
hand closed binary accuracy
open-loop token L2 / cosine
chunk temporal smoothness
closed-loop safety
```

### 10.5 Checkpoint metadata 和 sidecar

每个 checkpoint 必须能独立恢复训练、验证和部署所需上下文。至少保存：

```text
model_name_or_path
qwen_processor_path
qwen_config_hash / tokenizer_hash
action_schema_version
action_dim = 66
motion_token_dim = 64
hand_dim = 2
hand_polarity = closed_1_open_0
action_horizon = 50
control_freq = 50
num_inference_steps
sonic_checkpoint_id
sonic_encoder_path
sonic_encoder_hash
proprio_layout = unitree_g1_sonic v1
proprio_dim = 34
proprio_stats_path
proprio_stats_hash
proprio_use_percentiles
proprio_clip_outliers
data_manifest / split_manifest
image_transform = resize_short_224_center_crop_224x320
server_protocol = kappa_policy_server_msgpack_numpy
sonic_zmq_protocol = latent_v4_pose
source_git_commit / branch
```

sidecar 文件建议：

```text
model.safetensors
config.json
training_config.json
action_schema.json
proprio_stats_sonic.json
proprio_norm_config.json
data_manifest.json
processor_manifest.json
```

部署时优先读取 checkpoint sidecar，不依赖用户手工传入 stats 路径。CLI override 只用于调试，并必须在日志中标明。

## 11. 验证计划

### 11.1 Dataloader smoke test

每个数据源至少检查：

```text
episode count
frame count
state shape
action shape
image availability
language availability
NaN / Inf
hand binary distribution
```

### 11.2 Model forward smoke test

最小 batch：

```text
batch_size = 1 或 2
action_horizon = 50
```

检查：

```text
loss finite
backward passes
grad norm finite
predict_action returns [B, T, 66]
```

### 11.3 Open-loop evaluation

指标：

```text
motion_token L2
motion_token cosine similarity
per-dim q01/q99 out-of-bound ratio
hand binary accuracy
hand transition accuracy
chunk overlap consistency
```

### 11.4 Closed-loop evaluation

先仿真，再真机。

闭环步骤：

```text
1. 静态站立 token replay
2. 低速短 chunk replay
3. policy open-loop playback
4. teleop-like task
5. real downstream task
```

安全要求：

- 初期限制动作幅度。
- 检查 motion token 是否超出训练分布。
- 控制端保留 stop / pause / initial pose。
- hand binary 转 7D 手姿态时必须使用安全姿态。

## 12. 部署计划

### 12.1 Inference 输出

`QwenSonicFM.predict_action()` 内部输出统一为：

```python
{
    "action.motion_token": np.ndarray[B, T, 64],
    "action.hand_binary": np.ndarray[B, T, 2],  # [left_closed, right_closed]
}
```

兼容旧部署脚本时也可以返回无前缀 key：

```python
{
    "motion_token": np.ndarray[B, T, 64],
    "hand_binary_action": np.ndarray[B, T, 2],
}
```

不要返回 `left_hand_open/right_hand_open` 作为主接口；如果为了临时兼容保留别名，必须在 adapter 中立即转换为 closed polarity。

### 12.2 Kappa PolicyServer

部署端推荐使用 Kappa 自己的 PolicyServer 替代 Isaac-GR00T server，但保持同一套 REQ/REP wire format，方便复用 GR00T-WholeBodyControl Terminal 3：

```text
默认 host: 0.0.0.0
默认 port: 5550
serializer: msgpack + msgpack-numpy
endpoints: ping, get_action, reset, get_modality_config, kill
```

server 启动参数必须显式记录：

```text
ckpt_path
base_vlm_path = resource/model/Qwen3-VL-2B-Instruct
action_horizon = 50
num_inference_steps = 4
deploy_proprio = true/false
state_tap_host = localhost
state_tap_port = 5557
HF_HUB_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1
```

`get_action(observation)` 输入从 GR00T-WholeBodyControl 侧接收：

```text
video.ego_view: uint8, [B, T, H, W, 3]
language.annotation.human.task_description 或 language.task
state.left_leg / right_leg / waist / left_arm / right_arm
state.hand_binary_state: [left_closed, right_closed]
state.base_quat: WXYZ，可选
```

如果 checkpoint 使用 proprio，server 必须在部署端组装 34D raw proprio：

```text
[left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7),
 left_hand_closed(1), right_hand_closed(1), projected_gravity(3)]
```

`projected_gravity` 优先由 `state.base_quat` 计算；缺失时可从 C++ `g1_debug` ZMQ topic 读取最新 `base_quat`；仍缺失时只能用 upright fallback `[0, 0, -1]` 并记录 warning。

### 12.3 SONIC adapter / latent protocol v4

实现：

```text
src/kappa/inference/sonic_adapter.py
```

职责：

- 取当前 chunk index。
- 取 motion token `[64]`。
- 将 hand closed logits/prob 转 binary。
- 如果控制端需要 7D，将 `[left_closed, right_closed]` 映射为 `left_hand_joints[7]` 和 `right_hand_joints[7]`。
- 打包为 GR00T-WholeBodyControl ZMQ latent protocol v4。

Protocol v4 单帧 message 字段：

```text
topic: pose
version: 4
pose_data:
  token_state:       float32 [1, 64]
  frame_index:       int64   [1]
  left_hand_joints:  float32 [1, 7]
  right_hand_joints: float32 [1, 7]
```

frame counter 必须单调递增；发送 initial pose 时可以从 0 开始，恢复 policy loop 前应清空 cached chunk 并重置 chunk index。

### 12.4 Latency compensation

沿用 GR00T-WholeBodyControl 的思路：

```text
action_publish_rate = 50Hz
action_horizon = 50
inference latency -> skip stale prefix
current_idx = round(inference_delay * action_publish_rate)
current_idx = clip(current_idx, 0, action_horizon - 1)
```

需要确保：

- 新 chunk 和旧 chunk 拼接平滑。
- 手部 binary 不因阈值抖动频繁开合。
- closed hysteresis 可以使用 close threshold 0.6、open threshold 0.4。
- 如果 motion token `abs(max) > 1.25`、出现 NaN/Inf、推理超时、相机/状态缺失，应跳过该 chunk 或切回 initial pose，而不是发送异常 token。

### 12.5 部署日志和回放

部署 server 应支持记录每次 `get_action` 请求和响应：

```text
.runs/deploy_log/<timestamp>/manifest.jsonl
.runs/deploy_log/<timestamp>/calls/call_<idx>.msgpack
.runs/deploy_log/<timestamp>/calls/call_<idx>.jpg  # 可选首帧
```

日志必须包含 prompt、视频 shape、proprio raw/normalized、输出 action shape、是否启用 proprio inference，便于和训练集做 open-loop 对比。

## 13. 主要风险和对策

### 13.1 EgoDex 无 proprio

风险：

```text
模型把 zero state 当作真实站立状态
```

对策：

```text
使用 state_mask = 0
state feature 显式置零
Stage 1 只监督 64D SONIC token，hand dims 用 mask 跳过；不能训练真实 proprio-conditioned 控制能力
```

### 13.2 66D schema 与 Isaac-GR00T 78D schema 不一致

风险：

```text
直接复用 UNITREE_G1_SONIC 部署接口时 shape 不匹配
```

对策：

```text
训练内部固定 66D
部署 adapter 做 hand binary 到控制端格式的转换
不要在训练中混用 78D 和 66D
```

### 13.3 SONIC token checkpoint-specific

风险：

```text
更换 SONIC tokenizer / WBC checkpoint 后，64D latent 语义变化
```

对策：

```text
stats、initial token、训练标签必须绑定 SONIC checkpoint 版本
checkpoint metadata 记录 sonic_checkpoint_id
切换 SONIC checkpoint 必须重新生成 token 或重新微调
```

### 13.4 本体状态 stats 切换

风险：

```text
同一物理姿态 normalized value 变化
```

对策：

```text
每个 checkpoint 绑定自己的 proprio_stats_path
Stage 2 使用 HE+WBT stats
Stage 3 使用 HE+WBT+task stats
切换 stats 后必须训练 state encoder 做重新校准
```

### 13.5 手部 binary 极性或信息不足

风险：

```text
无法表达细粒度手指姿态
```

对策：

```text
短期接受 [left_closed, right_closed]，服务抓放类任务
数据审计强制检查 closed=1/open=0 极性
部署 adapter 中所有 open/closed 转换必须集中实现
保留扩展 action schema 的能力
后续可升级为 64 + 14 或 64 + hand latent
```

### 13.6 cache / symlink / 版本复用错误

风险：

```text
G1_WBT 是符号链接，预处理产物写入错误位置；或参数变化后复用旧 index / token cache
```

对策：

```text
所有 cache 固定在 resource/dataset/*-cache
index metadata 记录 source path、fps、encoder hash、schema version
参数变化时强制重建 index/cache
数据审计输出每个源的 cache_root 和 token/proprio 文件数量
```

### 13.7 Qwen3VL-2B 表征能力不足

风险：

```text
复杂语义、小物体识别或长任务泛化弱于 4B/7B
```

对策：

```text
先用 2B 快速闭环
保留 backbone registry
需要时替换为 Qwen3VL-4B 或 Cosmos-Reason2
```

## 14. 里程碑

### Milestone 1: 项目基础设施

- [x] 创建重构分支。
- [x] 创建 `.gitignore`。
- [x] 创建 uv `.venv`。
- [x] 创建 `pyproject.toml`。
- [ ] 添加基础依赖并生成 `uv.lock`。
- [ ] 建立 `src/kappa` 包结构。

### Milestone 2: 数据层

- [ ] 实现 schema 定义，固定 66D action 和 closed hand polarity。
- [ ] 实现 proprio normalizer。
- [ ] 实现共享 resample / projected gravity / atomic npy preprocess 工具。
- [ ] 实现 HE cache dataset，默认过滤 G1。
- [ ] 实现 G1_WBT symlink + cache dataset。
- [ ] 实现 proprietary real dataset。
- [ ] 实现 EgoDex dataset，支持 64D token pad 到 66D 并输出 mask。
- [ ] 实现 mixture dataset 和自然帧比例采样。
- [ ] 输出 data audit report。

### Milestone 3: 模型层

- [ ] 接入 Qwen3VL-2B wrapper。
- [ ] 实现 QwenSonicFM。
- [ ] 实现 Layerwise FM action head。
- [ ] 实现 hand binary head。
- [ ] 实现 state mask 逻辑。
- [ ] 跑通 forward/backward smoke test。

### Milestone 4: 训练

- [ ] Stage 1 EgoDex 预训练脚本，hand dims masked。
- [ ] Stage 2 HE + WBT 后训练脚本。
- [ ] Stage 3 proprietary real 微调脚本。
- [ ] checkpoint metadata 保存 action_horizon、stats、schema、sonic checkpoint id、Qwen processor hash、hand polarity、image transform。
- [ ] checkpoint sidecar bundle 保存 proprio_stats_sonic.json 和 proprio_norm_config.json。
- [ ] 训练日志记录 token loss、hand loss、grad norm、hand closed accuracy。

### Milestone 5: 评估

- [ ] Open-loop evaluation。
- [ ] Chunk overlap consistency evaluation。
- [ ] Hand binary metrics。
- [ ] SONIC token distribution check。
- [ ] 生成评估报告。

### Milestone 6: 部署

- [ ] 实现 `sonic_adapter.py`。
- [ ] 实现 Kappa PolicyServer，兼容 Isaac PolicyClient 的 msgpack-numpy REQ/REP。
- [ ] 接入 ZMQ latent protocol v4: token_state + frame_index + 7D hands。
- [ ] 实现 state tap / projected_gravity fallback / deploy logger。
- [ ] 仿真闭环测试。
- [ ] 真机低风险测试。

## 15. 第一批具体执行任务

建议从以下任务开始：

1. 添加 uv 依赖和 lock 文件。
2. 创建 `src/kappa` 包结构。
3. 实现 `ProprioNormalizer`，支持读取 Stage 2 的 HE+WBT stats 和 Stage 3 的 HE+WBT+task stats。
4. 实现共享预处理工具和 data audit，不执行大规模 token 生成前先检查路径、fps、schema、encoder 文件。
5. 实现 `SonicTokenDataset`，先支持 `humanoid-everyday-cache`、`G1_WBT-cache` 和 `proprietary/real/2026-05-28`。
6. 写一个 `uv run python src/kappa/data/audit.py`，输出所有数据源 shape、mask、closed hand 分布、cache_root、video/token 对齐。
7. 实现最小版 `QwenSonicFM`，可以先 mock Qwen hidden states 跑通 action head。
8. 再接入真实 Qwen3VL-2B。
9. 最后实现 Kappa PolicyServer + SONIC adapter 的无模型 replay 测试。

## 16. 最终原则

- 内部训练 schema 固定为 66D，不和 78D 混用。
- 正式训练 `action_horizon` 固定为 50。
- 外部部署通过 Kappa PolicyServer + adapter 兼容 SONIC 控制链路。
- hand binary 极性固定为 closed=1/open=0。
- EgoDex 无 proprio 时必须使用 `state_mask = 0`，且 hand dims 必须 mask 掉。
- Proprio normalization 使用 percentile bounds stats；Stage 2 用 HE+WBT，Stage 3 用 HE+WBT+task。
- 第一版追求可验证闭环，不追求一次性加入所有高级技巧。
- 所有 Python 依赖和命令通过 `uv` 管理。
