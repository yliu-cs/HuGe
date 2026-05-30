# GR00T N1.7 微调可学习参数说明

> 针对 `gr00t/experiment/launch_finetune.py` 微调流程,梳理可学习(可训练)参数有哪些、分别多少。
> Base model 路径:`/inspire/qb-ilm2/project/robot-body/liujiacheng-240108120088/Isaac-GR00T/models`(含 `GR00T-N1.7-3B` 与 `Cosmos-Reason2-2B`)。

## 1. 微调入口暴露的可训练开关

`launch_finetune.py` 通过 `FinetuneConfig`(`gr00t/configs/finetune_config.py`)向 `config.model` 写入 **4 个**开关来控制可训练参数(见 `launch_finetune.py:76-79`):

| CLI 开关 | 默认值 | 对应模型模块 |
|---|---|---|
| `tune_llm` | **False** | `backbone.language_model`(LLM) |
| `tune_visual` | **False** | `backbone.visual`(ViT) |
| `tune_projector` | **True** | action head 的 state/action 编码器、解码器、位置嵌入 |
| `tune_diffusion_model` | **True** | action head 的扩散模型(DiT) |

> ⚠️ **第 5 个组 `tune_vlln`**(控制 `vlln` + `vl_self_attention`)**未在 `FinetuneConfig` 中暴露**,launcher 不会覆盖它,因此始终沿用 `get_default_config()` 的默认值 **True**。也就是说,默认微调时它**也会被训练**。

冻结/解冻的具体逻辑:
- backbone:`gr00t/model/modules/qwen3_backbone.py:99-112`(`set_trainable_parameters`)
- action head:`gr00t/model/gr00t_n1d7/gr00t_n1d7.py:108-136`(`set_trainable_parameters`)
- 训练启动时 `gr00t/model/gr00t_n1d7/setup.py:155-160` 会打印实际的 `Total parameters` / `Trainable parameters`,可与本文核对。

## 2. backbone 为何是 Qwen3-VL —— GR00T 确实用到了 Qwen

GR00T N1.7 的 VLM backbone 本质上就是一个 **Qwen3-VL** 模型:

1. **默认 backbone 是 Cosmos-Reason2-2B**(`finetune_config.py:41`:`backbone_model_path = "nvidia/Cosmos-Reason2-2B"`),而 `models/Cosmos-Reason2-2B/config.json` 明确为 Qwen3-VL:
   ```json
   "architectures": ["Qwen3VLForConditionalGeneration"],
   "model_type": "qwen3_vl"
   ```
   NVIDIA Cosmos-Reason2-2B 是在 Qwen3-VL 基础上做的推理型 VLM,权重结构(text_config / vision_config)完全是 Qwen3-VL 那套。
2. **代码直接加载 Qwen3-VL**:`get_backbone_cls`(`gr00t_n1d7.py:477-507`)对 `cosmos-reason2` / `qwen3-vl` 返回 `Qwen3Backbone`,否则报 `Unsupported model name`——N1.7 目前**只支持** Qwen3-VL 系 backbone。`Qwen3Backbone`(`qwen3_backbone.py:80`)内部即 `Qwen3VLForConditionalGeneration.from_pretrained(...)`,并访问 `self.model.language_model` 与 `self.model.visual`。
3. **checkpoint 张量名也是 Qwen3 命名**:如 `...language_model.layers.N.self_attn.q_norm/k_norm`、`mlp.gate_proj/up_proj/down_proj`、`visual.blocks.N...`,正是 Qwen3 结构(带 q_norm/k_norm、SwiGLU MLP)。

**一句话:GR00T N1.7 的视觉语言 backbone(Cosmos-Reason2-2B)就是 Qwen3-VL。**

## 3. 各组参数量(B / M)

- 模型总参数 = **3,144,016,000**(≈3.14B),取自 `models/GR00T-N1.7-3B/model.safetensors.index.json` 元数据 `total_parameters`。
- **action head 各组与总量为实测精确值**(用配置实例化 `Gr00tN1d7ActionHead` 统计)。
- **backbone 的 LLM / ViT 拆分为按 Qwen3-VL 配置推算值**(标 ≈):LLM 28 层在 `select_layer=16` 截断后保留 16 层 + 词表 151936;ViT 保留全部 24 个 block。

| 开关 | 模块 | 参数量 (B) | 参数量 (M) | 精确值 |
|---|---|---:|---:|---:|
| `tune_llm` (默认 False) | backbone.language_model (16层+词嵌入) | ≈1.12B | ≈1116.54M | ≈1,116,542,976 |
| `tune_visual` (默认 False) | backbone.visual (24-block ViT) | ≈0.41B | ≈406.96M | ≈406,957,056 |
| `tune_projector` (默认 True) | state_encoder | 0.05B | 54.74M | 54,738,944 |
| `tune_projector` (默认 True) | action_encoder | 0.23B | 233.13M | 233,127,936 |
| `tune_projector` (默认 True) | action_decoder | 0.04B | 37.92M | 37,916,800 |
| `tune_projector` (默认 True) | position_embedding | 0.00B | 1.57M | 1,572,864 |
| `tune_projector` | **小计** | **0.33B** | **327.36M** | **327,356,544** |
| `tune_diffusion_model` (默认 True) | action_head.model (DiT) | 1.09B | 1091.72M | 1,091,722,240 |
| `tune_vlln` (CLI 未暴露,默认 True) | vlln + vl_self_attention | 0.20B | 201.44M | 201,437,184 |
| | **总计** | **3.14B** | **3144.02M** | **3,144,016,000** |

## 4. 默认微调设置下的冻结 / 可训练汇总

默认值:`tune_llm=False`、`tune_visual=False`、`tune_projector=True`、`tune_diffusion_model=True`,加上隐含的 `tune_vlln=True`。

| 状态 | 组成 | 参数量 (B) | 参数量 (M) | 占比 |
|---|---|---:|---:|---:|
| 冻结 | backbone (LLM + ViT) | ≈1.52B | ≈1523.50M | 48.5% |
| 可训练 | action head (projector + DiT + vlln) | 1.62B | 1620.52M | 51.5% |
| | **合计** | **3.14B** | **3144.02M** | 100% |

即默认情况下整个 backbone 被冻结,只训练整个 action head(约 1.62B,占 51.5%)。

## 5. 备注

- 本机 `models/GR00T-N1.7-3B` 下的 safetensors 为指向 `/root/.cache/huggingface` 的失效软链接(无实体权重),因此 backbone 内部 LLM↔ViT 拆分采用 Qwen3-VL 配置推算;如需逐张量精确统计,需先把权重 blob 下载到本地再读取 safetensors header。
- action head 与总参数量不依赖权重文件,均为精确值。
