# Ego Cola 仿真回放生成伪 Proprio — 进度交接 (Handoff)

> ⚠️ **发现**：写这份文档时检查 `resource/info/` 发现之前的两份底稿
> (`ego_cola_full_stack_verification_and_headless_render.md`、
> `ego_cola_sim_replay_pseudo_proprio_pipeline.md`) 已经**不在磁盘上了**
> (`resource/` 整个目录被 gitignore，git 里查不到任何删除记录，原因不明——
> 可能是其他会话/进程清理的)。本文档把两份原文档里仍然生效的关键内容都
> 折叠合并进来了，做到自包含，不再依赖那两个已消失的文件。下次会话直接读
> 这一份即可。

---

## 0. 硬约束 (必须遵守，贯穿全程)

1. **不修改 `GR00T-WholeBodyControl/` 内任何代码/文件。** 所有定制逻辑放在
   `scripts/preprocess/proprietary_ego/` 下的外部脚本里，直接调用已编译二进制
   / 现有仓库脚本 + CLI flag。
2. **Python 环境用 `uv` 管理，调用绑定固定**：
   - Huge 侧 Python → `.venv/bin/python`
   - 仿真环境 Python → `GR00T-WholeBodyControl/.venv_sim/bin/python`（必须在
     `GR00T-WholeBodyControl/` 目录下运行）
   - **禁止使用裸 `python3`**（本会话中一次用 `python3` 检查端口被用户当场
     驳回，此后全程改用上面两个 venv 绑定）。
3. 目标：仿真回放出的伪 proprio 与真机训练**完全对齐**，能直接
   `state_mask=1` 参与训练，不做任何训练侧特殊处理。
4. **范围：单样本端到端验证，不做 400 episode 批处理。**（用户已明确要求
   "先不批量跑，先对一个特定样本做验证"，下次会话若要扩展到批量需重新征求
   确认。）

激活环境：`source /inspire/qb-ilm2/project/robot-dna/liujiacheng-240108120088/liuyang/Huge/.venv/bin/activate`

---

## 1. 任务背景

用 cola episode 的 SONIC latent token 回放（token → WBC deploy → MuJoCo
仿真），采集仿真里的机器人状态，重排成 Huge 34D 伪 proprio，使其能像真机
`PropDataset` 一样直接参与训练。整条链路已经过端到端验证（详见第 2 节），
当前的核心难点是：**仿真采集到的状态帧如何与原始 token 的时间步精确
对齐**（第 3 节），这也是本次会话的主要工作内容。

数据现状（`resource/dataset/proprietary/ego/cola/` 下）：
- 共 **400 个 episode 目录**（`0` ~ `399`），每个目录结构一致：
  ```
  trackingData_..._latent_tokens_64d.npy   # (T,64) 直接可回放，喂 player
  trackingData_..._latent_tokens_66d.npy   # (T,66) = 64D + 左右手 binary，落盘用
  CameraRecord_..._left.mp4 / _right.mp4   # 1080x810 @30fps，真机录制视频
  llm_description.txt                       # 任务语言描述
  ```
- 已验证的样本：`episode 0`
  (`trackingData_20260612_154443_latent_tokens_64d.npy`, shape `(979,64)`
  float32)。819 行唯一，160 行重复实例（111 组重复，其中 156 是相邻重复）。
- 其余 399 个 episode 结构与 `episode 0` 一致，理论上同一套流程都适用，但
  **只验证过 `episode 0`**。

---

## 2. 完整链路拓扑（已验证）

```
cola token (.npy, T×64 float32)
  → sonic_encoder_input_player.py   [ZMQ PUB pose@5556, topic "pose"]
  → g1_deploy_onnx_ref (C++ WBC)    [ZMQ SUB pose@5556; DDS(lo, domain 0) ↔ MuJoCo LowState/LowCmd; ZMQ PUB g1_debug@5557]
  → 无头 MuJoCo 仿真 (launch_headless_sim.py, EGL 离屏渲染可选)
  → g1_debug 输出 (30 个字段：body_q/token_state/left_hand_q/right_hand_q/base_quat 等)
```

端口对照：

| 进程 | 角色 | 端口 |
|------|------|------|
| player (`sonic_encoder_input_player.py`) | ZMQ PUB | 5556 (`pose`) |
| C++ WBC (`g1_deploy_onnx_ref`) | SUB 5556 / PUB 5557 | 5557 (`g1_debug`) |
| sim 图像发布（可选，仅 QA 渲染时需要） | image server | 5555 |

`g1_debug` 消息 wire 格式：`[topic_str(g1_debug)][msgpack payload]`，剥掉
`len(topic)` 字节后 `msgpack.unpackb(payload, raw=False)`。已知字段（30个）：
`control_loop_type, index, ros_timestamp, base_quat, base_ang_vel,
body_torso_quat, body_torso_ang_vel, body_q, body_dq, last_action,
left_hand_q, left_hand_dq, right_hand_q, right_hand_dq,
last_left_hand_action, last_right_hand_action, token_state,
motor_temperature, init_base_quat, delta_heading, base_quat_measured,
base_quat_target, base_trans_measured, base_trans_target, body_q_measured,
body_q_target, left_hand_q_measured, right_hand_q_measured,
vr_3point_compliance, vr_3point_orientation, vr_3point_position`。

**完整栈验证结论（直接抓包确认，非仅靠日志推断）**：15s 回放窗口收到
789 帧 `g1_debug`；`body_q` 非零非默认值 → WBC 策略在真实驱动机器人；
deploy 日志交叉验证 `frame_index` 连续递增、`Protocol v4: Received 64D
token`、`operator_state.start=true` 状态转换、`Program exiting normally`。

---

## 3. ⭐ 关键根因 1：Elastic Band 导致机器人悬空静止

### 现象
第一版渲染 mp4 里机器人**始终悬空静止**（帧间像素差均值仅 ~0.008，575 帧
中只有 1 帧变化 > 0.5 → 冻结画面，非慢动作）。

### 根因
`gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml:20`：
`ENABLE_ELASTIC_BAND: True`（虚拟弹簧，`unitree_sdk2py_bridge.py:366-411`
的 `ElasticBand` 类），把机器人骨盆钉在空中固定点。设计本意是有屏 viewer
里按键盘 `9` 手动关闭，无头环境没有键盘，弹簧一直生效，覆盖了 token 驱动
的物理。

### ⚠️ 修复的坑
**不能**把 yaml 里 `ENABLE_ELASTIC_BAND` 改 `False`（而且也不允许改仓库
文件）：`base_sim.py` 的 `else` 分支根本不初始化 `self.elastic_band`
属性，会在 `sim_step()` 抛 `AttributeError`。

**正确做法**（已内置于 `launch_headless_sim.py`）：保持 yaml 不动，在
`BaseSimulator` 构造后**运行时**翻转：
```python
if sim.sim_env.elastic_band is not None:
    sim.sim_env.elastic_band.enable = False
```

### 修复后结果（`render2.mp4`，已在 `/tmp` 生成过，现已不在磁盘）
帧间差均值 **1.38**（vs 冻结时 0.008），602 帧中 437 帧变化 > 0.5；视觉上
脚踩地面、腿部行走步态、身体旋转平移 → 真实 token 驱动动作。deploy 日志
0 条 `LowState is not available` → DDS↔sim 链路健康。渲染窗口前 ~11 秒会
看到机器人反复前倾摔倒被重置——这是 WBC/TRT 冷启动 + 一次性安全复位的
**正常现象**，不是 bug，用第 4 节的锚点算法丢弃即可。

---

## 4. ⭐ 关键根因 2：token_state 逐帧对齐不可靠 → 锚点+位置对齐算法

### 问题
如果要用这条仿真链路合成伪 proprio，必须让采集到的每一帧与原始 token 的
时间步**精确对齐**（否则违反 pipeline 契约 C2，见第 5 节）。

### 排查过程
1. 最初假设：用 `g1_debug` 里的 `token_state` 字段（64维，理论上就是当前
   驱动用的 token 向量）逐帧和原始 npy 做内容匹配（全局 argmin / 序列双
   指针）。结果：非单调匹配，出现「~246 行缺失」的假象，且在 npy 第 224
   行附近彻底崩溃。
2. 根因定位：检查 `body_q`（连续物理量）在同一段跳变处的变化——发现完全
   平滑连续（~0.001-0.002 rad/tick，零跳变）。说明物理/控制循环完整无
   丢帧，`token_state` 字段本身在逐帧粒度上有**上报抖动**（很可能是 ZMQ
   接收线程与 debug 发布线程之间的竞争），并不真实反映"当前正在执行第几个
   token"。
3. 额外验证：控制/物理循环稳定在 **50.00Hz**（978 帧连续 tick 间隔均值
   0.020000s，标准差仅 0.000183s，零大间隙）；player 对一整段 token 也是
   精确 50Hz 发送一遍 —— 说明真实的"采集帧 ↔ 原始 token 序号"对应关系是
   简单确定的**顺序/位置关系**，不需要（也不能可靠依赖）逐帧内容匹配。

### 最终算法：锚点 + 位置对齐（已验证、已写成正式工具，见第 6 节）
1. **找锚点**：唯一一次内容匹配——找到第一帧，其 `token_state` 与原始
   npy 第 0 行完全一致（max abs diff < 1e-5）。这一帧标志着真正开始被
   token 驱动控制（跳过前面 TRT 加载/安全复位的噪声期）。
2. **位置对齐**：从锚点起，第 k 帧 → 原始 token 第 k 个索引（纯顺序位置，
   不再做内容重新匹配）。
3. **完整性校验**：锚点到运行结束的总帧数应约等于总 token 数；尾部有
   ~1 帧的缺口是预期且可接受的（deploy 进程在最后一个 token 处理完后立即
   退出，来不及 flush 最后一次 debug 发布）。
4. `token_state` 内容匹配**只用作找锚点和抽检诊断**，绝不作为逐帧对齐的
   主键。

**实测验证（episode 0，979 tokens）**：锚点在第 40 帧（`anchor_idx=40`），
总共 1018 帧被抓到，对齐后 **978/979 匹配**（`missing_tail=1`，落在预期
阈值内），对齐后的 `token_state` 数组逐行与原始 npy 完全一致（第 0 行、
第 500 行 max diff 均为 0.0）。

### ⭐ 关键根因 3（小批量验证时发现）：锚点算法只试 token 第 0 行，遇到
### 「首个 token 被 ZMQ 丢弃」的 episode 会误报失败

**现象**：跑 10-episode 小批量验证（episode 1-15）时，episode 7 在
`align` 阶段抛 `ValueError: no frame matched token row 0 within tol=1e-05`，
capture/deploy/player 日志本身完全正常（deploy 顺利播完 1002 个 token 并
`Program exiting normally`，capture 抓到 1041 帧）。

**根因**：`sonic_encoder_input_player.py` 通过 ZMQ PUB 发第一条 pose 消息
时，deploy 二进制的 SUB socket 很可能还没完成 subscribe（经典 ZMQ
slow-joiner），导致 token 序号 0 那一条真实被机器人执行的消息**从未发生
过**——deploy 实际是从 token 序号 1 开始执行的。之前跑通的 episode
0/1-6 之所以没暴露这个问题，是因为这批 cola 数据本身重复行很多（第1节
提到过"819 行唯一，160 行重复"），恰好这几条 episode 的 token 第 0/1 行
内容相同，掩盖了"第 0 行其实没被执行"这个事实；episode 7 恰好第 0/1 行
有一维差 0.0625，才把这个问题暴露出来（`np.diff` 定位：捕获帧里没有任何
一帧和 token 第 0 行的 max diff < 1e-5，最小是 0.0625；但第 40 帧和 token
第 1 行完全相等，diff=0.0）。

**修复**（`align_sim_replay.py`）：`find_anchor` 不再只试 token 第 0 行，
改成 `find_anchor_with_offset` 依次尝试第 `0,1,2,...,4` 行（`ANCHOR_OFFSET_WINDOW=5`），
取第一个能匹配上的行号作为 `token_offset`，对齐结果新增
`token_offset`/`missing_head` 两个字段（原来只有 `missing_tail`）。
`export_pseudo_proprio.py` 的 `forward_fill_tail` 相应泛化为「头尾都能
补」：头部缺口用第一帧真实数据**向前**补（back-fill），尾部缺口沿用原来
的向后补（forward-fill），中间如果出现 NaN 仍然直接报错（说明对齐本身
断裂，不是这个已知坑）。**复测 episode 7**：`token_offset=1`，
`matched=1001/1002`，export 后 self-check 全部通过。

---

## 5. 「完全对齐」的硬契约 C1-C6（来自已消失的 pipeline 文档，折叠保留）

要让伪 proprio 和真机走同一条加载路径
（`SonicTokenDataset._load_state`，`src/huge/data/sonic_dataset.py:225`），
必须逐条满足：

| # | 契约 | 依据 | 违反后果 |
|---|------|------|----------|
| C1 | **存 raw 物理量，不预归一化** | Real `g1_proprio/episode_*.npy` 经查为 raw：proj_gravity 模长=1、hand∈{0,1}、关节弧度。归一化由 `StateNormalizer` 在加载时现场做（`sonic_dataset.py:230`） | 预归一化会被二次归一化，分布全错 |
| C2 | **形状严格 `[T, 34]`，且 T == token 帧数** | `sonic_dataset.py:147-150` 强校验 | 加载即崩 |
| C3 | **34D 子顺序逐维对齐 PROPRIO_LAYOUT** | `schemas.py:21`：left_leg6/right_leg6/waist3/left_arm7/right_arm7/left_hand1/right_hand1/proj_gravity3 | 维度错位 |
| C4 | **waist 子顺序 = yaw,roll,pitch** | G1 `body_actuated_joints`（`g1_supplemental_info.py:62-64`）。**已验证 Real 同序**（见下方"C4 验证结果"） | waist 3 维内部错位 |
| C5 | **hand 极性 = closed_1_open_0** | `schemas.py:17`；exporter `_hand_binary_state` 同极性 | 手部语义反向 |
| C6 | **归一化用同一 stats 文件** | 默认 `resource/.../cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json`（`runner.py:339`） | stats 不匹配则分布不在同一尺度 |

**C6 深层风险**：该 stats 是从 Real 数据统计的 q01/q99。伪 proprio 若分布
偏离 Real，同一 stats 归一化后会大量被 clip 到 ±1 边界
（`state_normalizer.py:74` `clip_outliers=True`）——形状对齐是确定性的，
**分布对齐不是**，只能逐 episode 校验、激进过滤。

### 文件落盘约定
Real 布局：
```
cola_merged_cleaned_0530/
  sonic_tokens/episode_000000.npy   # [T,66] 动作 token
  g1_proprio/episode_000000.npy     # [T,34] raw proprio，同名同帧数
  meta/proprio_stats_union_sonic.json
```
伪 proprio 数据集须复刻此布局：`sonic_tokens/` 放 cola 的 66D token（直接
复制/软链），`g1_proprio/` 放生成的 raw 34D，文件名一一对应，`PropDataset`
即可零改动加载。

### whole_q → raw 34D 组装映射表

**关键发现**：G1 `body_q` 前 29 维与 Huge 34D 前 29 维同序同长。

| Huge 34D 维度 | 来源 g1_debug 字段 | 处理 |
|----------|---------------------|------|
| 0:29 (leg/leg/waist/arm/arm) | `body_q[:29]` | 直接取（已是弧度 raw） |
| 29 (left_hand) | `left_hand_q` | `_hand_binary_state(...)[0]`，closed=1 |
| 30 (right_hand) | `right_hand_q` | `[1]` |
| 31:34 (proj_gravity) | `base_quat` | `compute_projected_gravity(base_quat)`，模长=1 |

`align_sim_replay.py` 的 `align` 子命令产出的 `.npz` 里已经包含
`body_q`/`left_hand_q`/`right_hand_q`/`base_quat` 四个字段，正好是这张
表需要的全部原料——组装脚本（第 7 节遗留项）可以直接消费它的输出。

必带 self-check：
```python
assert out.shape[1] == 34 and out.shape[0] == token_len          # C2
assert np.allclose(np.linalg.norm(out[:,31:34],axis=1),1.0,atol=1e-2)  # C1 proj_gravity raw
assert np.isin(out[:,29:31],[0.0,1.0]).all()                     # C5 hand 极性
norm = StateNormalizer.from_json("resource/.../proprio_stats_union_sonic.json").normalize(out)
assert (np.abs(norm) >= 0.999).mean() < 0.2   # C6 抽检，阈值待调
```

---

## 6. 当前进度：已实现并验证的工具

`scripts/preprocess/proprietary_ego/` 下（真实仓库文件，非 `/tmp`）：

| 文件 | 状态 | 说明 |
|------|------|------|
| `build_index.py` | 原有 | 与本任务无关的既有工具 |
| `align_sim_replay.py` | **新增，已验证；小批量验证时修过锚点 bug（见第4节根因3）** | 抓包 g1_debug + 锚点(带 offset 窗口)/位置对齐 |
| `launch_headless_sim.py` | **新增，已跑通** | 无头仿真启动器（含 elastic band 修复） |
| `export_pseudo_proprio.py` | **新增，已用真实 episode 0 数据端到端验证** | 消费 `align` 产出的 `.npz`，按第5节映射表组装 raw `[T,34]`，用 66D token 的 hand_binary 事后订正手部状态维（见第7节"手部状态根因+事后订正"），落盘 `g1_proprio/`+`sonic_tokens/`，跑 C1/C2/C5/C6 self-check |
| `run_episode_pipeline.py` | **新增，已跑通** | 单 episode 一键编排（sim→deploy→capture+player→align→export），见第8节"已完成"条目 |

### `align_sim_replay.py` 用法

两个子命令：

```bash
# 1. 抓包（仿真+deploy+player 都跑起来之后，在 player 启动前先起）
.venv/bin/python scripts/preprocess/proprietary_ego/align_sim_replay.py capture \
  --duration <token数/50 + 启动期~11s + 余量> \
  --out /path/to/ep_frames.pkl
# --host/--port/--topic 默认 localhost:5557/"g1_debug"，一般不用改

# 2. 对齐（capture 结束后，纯 numpy 离线计算，可独立跑）
.venv/bin/python scripts/preprocess/proprietary_ego/align_sim_replay.py align \
  --frames /path/to/ep_frames.pkl \
  --tokens resource/dataset/proprietary/ego/cola/0/trackingData_*_latent_tokens_64d.npy \
  --out /path/to/ep_aligned.npz
# 产出 ep_aligned.npz（含 token_state/body_q/left_hand_q/right_hand_q/base_quat，
# 按 token 序号对齐，缺帧用 NaN 填）+ ep_aligned.report.json（anchor_idx/matched/missing_tail）
```
`missing_tail` 超过 1 会打印 WARNING，需要人工排查该 episode 再决定是否
采信。

可复用的核心函数（供后续组装脚本 import）：`find_anchor()`、
`align_frames_to_tokens()`、`build_aligned_arrays()`。

### `launch_headless_sim.py` 用法

```bash
cd GR00T-WholeBodyControl
# 只驱动 g1_debug 状态流，不发布图像（对齐/组装伪 proprio 只需要这个）
.venv_sim/bin/python /path/to/Huge/scripts/preprocess/proprietary_ego/launch_headless_sim.py &

# 需要人工肉眼核验渲染视频时，加 --enable-image-publish（需要 MUJOCO_GL=egl）
MUJOCO_GL=egl .venv_sim/bin/python /path/to/Huge/scripts/preprocess/proprietary_ego/launch_headless_sim.py \
  --enable-image-publish --camera-port 5555 &
```

---

## 7. 单样本完整执行流程（截至目前，供下次会话直接照做）

以 `episode 0` 为例：
```bash
EP=resource/dataset/proprietary/ego/cola/0
TOK64=$EP/trackingData_20260612_154443_latent_tokens_64d.npy   # 喂 player + 锚点匹配
TOK66=$EP/trackingData_20260612_154443_latent_tokens_66d.npy   # 最终落盘 sonic_tokens/
```

**阶段 0 — 端口检查**：5555（图像，可选）/ 5556（pose）/ 5557（g1_debug）
必须空闲；用 `.venv/bin/python` 做 socket bind 检测（**不要用 `python3`**）。
本机无 `lsof`/`fuser`。杀残留进程用 `pkill -9 -f <script>`（注意：本环境
`pkill` 的退出码上报有异常，即使成功也可能报非 0，不要用退出码判断成败，
改用 `ps aux | grep` 确认）。

**阶段 1 — 启动无头仿真**（`cd GR00T-WholeBodyControl`）：
```bash
.venv_sim/bin/python ../scripts/preprocess/proprietary_ego/launch_headless_sim.py &
```

**阶段 2 — 启动 C++ WBC deploy 二进制**：
```bash
cd gear_sonic_deploy
nohup ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
  --input-type zmq_manager --output-type all --hand-type dex3 --disable-crc-check \
  --obs-config policy/release/observation_config.yaml \
  --encoder-file policy/release/model_encoder.onnx \
  --planner-file planner/target_vel/V2/planner_sonic.onnx &
cd ..
```
位置参数 `<policy_file>` 必须是 `policy/release/model_decoder.onnx`
（即 `${CHECKPOINT}_decoder.onnx`），不是裸 `policy/release/model`。等日志
出现 `Init Done`（TRT 引擎已缓存约 20s；首次冷启动 planner 约 108s），确认
`LowState is not available` 计数为 0。

**阶段 3 — 启动抓帧（必须早于 player，避免 ZMQ slow-joiner 漏帧）**：
```bash
.venv/bin/python scripts/preprocess/proprietary_ego/align_sim_replay.py capture \
  --duration 25 --out /tmp/cola_ep0_frames.pkl &
sleep 2
```

**阶段 4 — 前台跑 player，驱动真实动作**：
```bash
.venv_sim/bin/python gear_sonic/scripts/sonic_encoder_input_player.py \
  --latent-input-file "../$TOK64" --hz 50 --source-hz 50 --duration 0
```
`--duration 0` = 把整段 token 完整播一遍。

**阶段 5 — 对齐**：
```bash
.venv/bin/python scripts/preprocess/proprietary_ego/align_sim_replay.py align \
  --frames /tmp/cola_ep0_frames.pkl --tokens $TOK64 \
  --out /tmp/cola_ep0_aligned.npz
cat /tmp/cola_ep0_aligned.report.json   # 确认 missing_tail <= 1
```

**阶段 6 — 组装 34D**：`export_pseudo_proprio.py`（见第 8 节，已写并已用
episode 0 真实数据跑通）：
```bash
.venv/bin/python scripts/preprocess/proprietary_ego/export_pseudo_proprio.py \
  --aligned /tmp/cola_ep0_aligned.npz --episode-num 0 \
  --out-root resource/dataset/proprietary/ego/cola_pseudo_proprio
```

### ⭐ episode 0 真实端到端验证结果（本次会话，非合成数据）

完整重跑了一遍阶段 0-6（sim + deploy 二进制 + 抓包 + player + align +
export），**抓包时机是关键坑**：第一次抓包和 player 启动之间因为隔了多次
工具调用（跨消息延迟），只抓到 284 帧（远低于预期），锚点/对齐会失败；
把「抓包启动 → `sleep 2` → 前台跑 player」放进**同一次** shell 调用里连续
执行后，稳定抓到 1018 帧。**教训：capture 和 player 必须在同一个 shell
会话/同一条命令链里背靠背执行，不能跨多轮工具调用分开做，否则抓包窗口
会被无关延迟吃掉。**

对齐结果与文档第 4 节此前记录的验证值完全一致（可复现）：
`anchor_idx=40, num_frames=1018, matched=978/979, missing_tail=1`；
`token_state` 逐行（第0/500/977行）与原始 npy max diff 均为 0.0。

`export_pseudo_proprio.py` 产出的真实 `g1_proprio/episode_000000.npy`
`[979,34]` self-check 结果：
- C1（proj_gravity 模长）：`max_err ≈ 4.4e-16`（机器精度，非常干净）
- C5（hand 极性 ∈ {0,1}）：通过。
- C6（clip_frac，真实分布 vs Real stats）：**8.7%**，远低于合成随机数据
  测试时的 22.7%，也低于 0.2 的 WARNING 阈值——说明仿真回放出的真实伪
  proprio 分布和 Real 数据统计量吻合得不错，不是靠运气对齐形状。
- 用 `SonicTokenDataset` + `StateNormalizer` 实际加载了这份数据：
  `state.shape=(34,)`, `state_mask=[1.]`, `normalized_actions.shape=(50,66)`，
  零改动加载成功。

### ⭐ 手部状态根因 + 事后订正（本次会话新增）

**发现**：episode 0 最初产出的 hand 维（29/30）全程恒为 0，但对应的 66D
token 动作维（64/65，hand_binary）并不恒为 0。排查（非猜测，读代码确认）
定位到根因：`sonic_encoder_input_player.py` 的 `publish_token_stream()`
发给 WBC deploy 二进制的 ZMQ pose 消息只包含
`token_state`/`frame_index`/`body_quat_w`，**从不传手部动作字段**；deploy
二进制的手部实际由独立的键盘遥操作（`g/h`/`b/v`）驱动，无头环境没有键盘
输入，手部状态整场保持默认值不变。也就是说仿真根本没有模拟手部动力学，
采集到的 `left_hand_q`/`right_hand_q` 对手部状态而言不是有效信号。

**用户决策**：不改仿真输入端（"这样的改动太复杂了"），改为**事后订正**：
伪 proprio 生成后，直接用同一 episode 的 66D token 动作里的 hand_binary
（维度 64:66，顺序 left,right，与 `run_data_exporter.py` 的
`_hand_binary_action(left,right)` 一致，和 PROPRIO_LAYOUT 的
`left_hand(29)/right_hand(30)` 顺序天然吻合）**逐帧覆盖**伪 proprio 的
[29:31] 维——两者本来就按 token 序号严格 1:1 对齐（同一构造关系），不需要
额外插值/滞后建模。把"指令动作"当作"瞬时达成的状态"，是仿真拿不到真实
手部状态时能拿到的最佳近似。

**实现**：新增 `correct_hand_state_from_action(out, tokens)`（
`export_pseudo_proprio.py`），在 `build_pseudo_proprio()` 之后、
`self_check()` 之前调用，用 `tokens[:,64:66]` 覆盖 `out[:,29:31]`，report
里新增 `hand_state_source="action"`/`hand_frames_corrected`/
`hand_frames_total` 三个字段。

**用真实 episode 0 数据重新跑过**（复用已有的 `/tmp/cola_ep0_aligned.npz`，
未重跑仿真/deploy）：`hand_frames_corrected=163/979`（约 16.6% 的帧原来
被仿真误判为 open，订正后变为与动作一致的 closed）；订正后
`proprio[:,29:31]` 与 `tokens[:,64:66]` 逐帧完全相等（已验证）；
`left_hand` 全程共 13 帧 closed，`right_hand` 共 156 帧 closed——和此前
"全程 open" 的错误结论形成对比，说明订正前的自检结论（"本 episode 任务
不涉及抓握动作"）是错的，是仿真没驱动手部造成的假象，不是真实任务内容。
C1/C5/C6 self-check 全部照常通过（C6 clip_frac 数值未受明显影响，因为
hand 维本身是二值，落在 stats 的 [0,1] 范围内不会被 clip）。

**追加验证：state/action 是否存在 1 步时序偏差？**（用户提问，已用真机
数据实证，非推测）用户质疑：`sonic_tokens[k]` 的 hand_binary 和
`g1_proprio[k]` 的 hand 状态是否应该有 1 帧滞后（即 action 先发生，state
隔一帧才体现）。实测方法：扫描 `cola_merged_cleaned_0530` 60 个真实
episode，找 `sonic_tokens` hand_binary 每次跳变（open↔closed）的位置，
统计 `g1_proprio` 对应维度真正变成新值滞后几帧。结果：114 次跳变里
**113 次滞后 = 0 帧**，仅 1 次滞后 1 帧——绝大多数情况下真机的
`g1_proprio[t]`/`sonic_tokens[t]` 本来就是**同索引配对**，没有系统性偏移。
根因（读 `run_data_exporter.py` 确认）：`action.hand_binary` 取自
`proprio["last_left_hand_action"]`——deploy 二进制自己在同一条 g1_debug
消息里把"当前状态"和"上一次执行动作"一起上报，控制时延已经折进
`last_*` 字段语义里，文件层面不需要再额外错位。**结论：本次
`correct_hand_state_from_action` 用同索引直接赋值（无偏移）与真机配对
惯例一致，不需要改。**

### ⭐ C4 验证结果：waist 顺序在 Real 数据上同样成立

**证据链（代码级，非推测）**：`run_data_exporter.py`（Real 采集用的同一个
exporter）用 `g1_rm = get_g1_robot_model()`（`features_sonic_vla.py:418`）
拿机器人模型 → `get_g1_robot_model()` 内部调用
`instantiate_g1_robot_model()`（`data/robot_model/instantiation/g1.py`）→
构造 `G1SupplementalInfo(waist_location=..., elbow_pose=...)`。这是**唯一
一个** `G1SupplementalInfo` 实例化路径，Real 采集和本次仿真回放的
`body_actuated_joints` 顺序（`waist_yaw_joint, waist_roll_joint,
waist_pitch_joint`，`g1_supplemental_info.py:62-64`）来自同一份硬件描述，
不存在"仿真专用顺序"和"真机专用顺序"分叉的可能——顺序由机器人本体结构
决定，与数据来源（真机/仿真）无关。

**数值抽查（辅助，非决定性）**：取 `cola_merged_cleaned_0530` 前 30 个真实
episode 的 `g1_proprio[:,12:15]`，统计极值：dim0 `[-0.358, 0.227]`，dim1
`[-0.231, 0.075]`，dim2 `[-0.046, 0.112]`——三维绝对值都远小于 yaw 的关节
限位 `±2.618 rad`，也都落在 roll/pitch 各自的 `±0.52 rad` 限位内，**无法
仅凭数值区分 yaw/roll/pitch 顺序**（这批真实 episode 里腰部动作幅度本身
很小，三维数值区间接近）。这一项只能证明"数值没有违反任何一维限位"，
真正定序的证据是上面的代码链路，不是这个数值抽查。

**结论**：C4 视为已验证，依据是"Real/仿真共用同一个 `G1SupplementalInfo`
实例"这一代码事实，而非数值巧合。

产出文件已落盘在仓库内（非 `/tmp`，但 `resource/` 本身 gitignore）：
`resource/dataset/proprietary/ego/cola_pseudo_proprio/g1_proprio/episode_000000.npy`
`resource/dataset/proprietary/ego/cola_pseudo_proprio/sonic_tokens/episode_000000.npy`

---

### ⭐ 用真机数据反向验证管线（本次会话新增）——比自检更强的正确性证据

**思路**：`resource/dataset/proprietary/real/cola_merged_cleaned_0530` 是真机
采集数据，每个 episode 同时有 `sonic_tokens/episode_*.npy`（[T,66]，真机
teleop 时 deploy 二进制内置 encoder 实时产出的 `token_state`，和 ego/cola
消费的是同一个 decode 契约，只是 encoder mode 不同：真机是 g1/teleop，
人类演示是 smpl——但这只影响编码输入，不影响我们只用到的 decode 侧）和
`g1_proprio/episode_*.npy`（真机记录的 ground truth）。把真机 token 喂回
sim replay 管线，生成的伪proprio可以直接和这份 ground truth 做数值对比，
不用像 ego/cola 那样只能靠自检。

**新增两个小脚本**（都不改动任何已有文件）：
- `scripts/preprocess/proprietary_ego/prepare_real_validation_tokens.py`：
  把真机 `sonic_tokens/episode_NNNNNN.npy`（66d）切出 `[:, :64]`，镜像成
  ego/cola 的目录布局（`<out_root>/<ep>/*_latent_tokens_{64,66}d.npy`），
  这样 `run_episode_pipeline.py`/`export_pseudo_proprio.py` 不用改一行就能
  直接复用。
- `scripts/preprocess/proprietary_ego/compare_pseudo_vs_real_proprio.py`：
  按 `PROPRIO_LAYOUT` 分组算逐维 Pearson r（lag=0）+ RMSE + 滞后互相关
  扫描（±20 帧 = ±400ms，用于区分"正常 WBC 跟踪延迟/ZMQ 抖动"和"真实
  对齐/维度顺序 bug"）。手部维度被排除在"信号有效性"判断之外——两边的
  手部状态都是从同一份 `action.hand_binary` 字段抄来的（同源），比对不
  出 sim 回放本身准不准。

**踩过一个坑**：`run_episode_pipeline.py --work-dir` 传相对路径会炸——
脚本内部会先 `cd GR00T-WholeBodyControl` 再跑 player，capture/player 的
日志重定向路径如果是相对路径，`cd` 之后就解析到错误的目录，报
`No such file or directory`。**必须传绝对路径**（`"$(pwd)/resource/tmp/..."`）。

**第一次尝试（episode 385，146 帧≈2.9s）是个坏样本**：真机这一段几乎
原地不动（腿部关节 std 只有 0.001-0.008 rad），而 `run_episode_pipeline.py`
里 `DEPLOY_STARTUP_STARTUP_S=11.0` 的冷启动摔倒/复位期比整个 episode
还长——对比窗口几乎全被 sim 自身的复位抖动占满（sim 腿部 std 反而是真机
的 10-40 倍），相关系数低且杂乱（mean_r 从 -0.25 到 0.28 不等），不能说明
任何问题，纯粹是选错了样本。

**改用 episode 0（1916 帧≈38.3s，"走路捡可乐扔垃圾桶"完整任务）**，
`anchor_idx=40, matched=1916/1916`，对比结果（lag=0 时的 Pearson r，
排除手部）：
```
left_leg:  mean_r=0.86  (单维最高 0.994/0.995，最低 0.565)
right_leg: mean_r=0.86  (单维最高 0.992/0.979，最低 0.722)
left_arm:  mean_r=0.86  (单维最高 0.986/0.962，最低 0.646)
right_arm: mean_r=0.89  (单维最高 0.982/0.978，最低 0.714)
waist:     dim12(yaw)=0.579, dim13(roll)=0.943, dim14(pitch)=0.379
gravity:   dim31=0.739, dim32=0.405, dim33=-0.051
```
多数维度在**零滞后**就有 >0.8、甚至 >0.99 的相关性——如果维度顺序/符号
接反了会看到接近 0 或负相关，不会看到 0.99。这是迄今为止最强的一份
"管线整体接线是对的"证据，比自检强得多。

少数维度（waist pitch、部分手臂/腿部维度、gravity z）在扩大滞后窗口
（±20 帧=±400ms）后相关性才明显提升（如 waist dim14: 0.379→0.831 at
lag=14），判断为正常的 WBC 跟踪延迟/摔倒-复位后的动力学恢复过程 + 该
episode 有周期性步态（互相关在周期倍数处容易出现虚假峰值），不是对齐
bug 的特征（真正的对齐 bug 会表现为**所有**维度在同一个固定滞后上一起
改善，而不是像现在这样有的 lag=-2 有的 lag=20 各自分散）。gravity z
（dim33，机器人直立时本身接近 -1、方差极小）相关性弱是低信噪比导致，
不是结构性问题。

**结论**：管线维度顺序/符号/单位没有系统性错误，真机反向验证通过；
sim2real 数值差异（RMSE 普遍在 0.03-0.13 rad）符合预期的物理差异
（质量/摩擦/控制器增益/接触模型不同），不需要进一步排查。

产出/中间文件都在 `resource/tmp/`（不进正式 `cola_pseudo_proprio/` 产出
目录，避免和 ego/cola 的正式产出混在一起）：
`resource/tmp/real_validation_cola_root/`（适配后的 token 目录）
`resource/tmp/real_validation_pseudo_proprio/g1_proprio/episode_000000.npy`
`resource/tmp/real_validation_work/ep_0/compare_report.txt`（完整对比报告）

---

## 8. 遗留 / 下一步

1. ~~`export_pseudo_proprio.py`~~ **已完成**：见第7节"阶段6"，已用 episode 0
   真实数据端到端验证通过（self-check + `SonicTokenDataset` 实际加载）。
2. ~~单 episode 编排脚本~~ **已完成**：
   `scripts/preprocess/proprietary_ego/run_episode_pipeline.py`，一条命令
   串起阶段 0-6（起 sim → 起 deploy 二进制等 `Init Done` → capture+player
   背靠背 → align → export+hand-correct）：
   ```bash
   .venv/bin/python scripts/preprocess/proprietary_ego/run_episode_pipeline.py \
     --episode-num 0 --work-dir resource/tmp/ego_pipeline/ep_0
   ```
   **踩过一个坑**：第一版脚本里 capture 用 `nohup ... &` 丢进后台，但
   `bash -c` 整条命令只会阻塞到前台的 player 跑完（~20s），capture 自己的
   固定时长窗口（`num_tokens/50 + 启动期 + 尾部余量` ≈ 45.6s）还没跑完，
   `bash -c` 就已经返回，导致 `frames.pkl` 还没写出来就去跑 align，报
   `FileNotFoundError`。修复：在 capture 那行记下 `CAPTURE_PID=$!`，player
   跑完后加一行 `wait $CAPTURE_PID`，让整条 `bash -c` 命令阻塞到 capture
   真正写完文件为止。修复后端到端跑通 episode 0：
   `anchor_idx=39, matched=979/979, missing_tail=0`（比手工分步验证时的
   `978/979` 还干净一点），self-check 全部通过。
3. ~~C4 待验证~~ **已完成**：见第7节"C4 验证结果"，代码级证据（Real/仿真
   共用同一个 `G1SupplementalInfo` 实例）确认 waist 顺序对 Real 同样成立。
4. **批量 400 episode**：小批量验证（episode 1-15）已完成，见第4节
   "关键根因3"。之后用户自行/后台跑了全量 400 episode 中的 0-163
   （某次会话中途中断，进程已不存在），158/164 成功，6 个失败/中断：
   - episode 0/10/22/100/104：deploy 日志里是全新的一类失败——
     `[ERROR] Lost LowState data connection from robot! / Safety check
     failed, cannot start control.`，是 sim/deploy 二进制刚启动握手时
     DDS 连接偶发丢失导致的安全退出，纯粹是启动时序 flake，和第4节的
     ZMQ slow-joiner 锚点 bug是两回事，重跑一次通常能过（episode 0 后来
     单独重跑验证通过，见上面"用真机数据反向验证管线"一节，虽然是
     从 real 数据集跑的，说明重跑本身没问题）。
   - episode 163：`capture.log` 为空，说明进程正好在这一集刚开始
     capture 时被中断，不算真正失败。
   - episode 164-399（236 集）尚未跑过。
   `scripts/preprocess/proprietary_ego/run_batch_pipeline.py`：小批量顺序
   跑 `run_episode_pipeline.py`，单 episode 失败不中断整批，跑完打印汇总
   （每 episode 的 clip_frac/missing_tail_filled/hand_corrected 或
   FAILED）。补跑命令：
   `--episodes 0 10 22 100 104 163`（失败/中断的）和
   `--episodes $(seq 164 399)`（未跑的）。扩展到全量前按惯例仍需再和
   用户确认一次。
5. **`/tmp/replay_test/` 里仍未搬进仓库、且已知有用的东西**：无（视频
   渲染/抓帧脚本 `capture_frames.py` 目前判断优先级低，核心链路不需要
   图像；`debug_probe*.py` 是一次性诊断脚本，无需搬）。`/tmp/replay_test/`
   本身内容会在环境重启/换机器后丢失，不要依赖其中任何文件。

---

## 9. 踩过的坑（避免下次重复踩）

- **cyclonedds `[ChannelFactory] create domain error`**：偶发，原因未完全
  定位（怀疑与 `/dev/shm/psm_*`、`/dev/shm/sem.mp-*` 遗留文件有关，发现过
  但未清理）。表现：机器人始终 "Robot has fallen"，零 `g1_debug` 帧。
  处理：确认残留进程已 kill、端口已空闲后，原样重跑一次即可，通常第二次
  就成功。
- **`pkill -9 -f ...` 退出码不可信**：即使成功杀掉进程，也可能报 Exit
  code 1 且无输出，即使用 `|| true`/`|| echo` 包裹也一样（怀疑是本沙箱
  环境对 `pkill` 的特殊处理，非正常 shell 语义问题）。不要用退出码判断
  是否杀成功，改用 `ps aux | grep` 或 socket bind 测试端口。
- **禁止裸 `python3`**：本环境必须用 `.venv/bin/python`（Huge 侧）或
  `.venv_sim/bin/python`（sim 侧），哪怕只是做个端口检测这种小事。
- **`ENABLE_ELASTIC_BAND` 不能在 yaml 里改 `False`**：会导致
  `AttributeError`，必须构造完 `BaseSimulator` 后运行时翻转
  `enable` 属性（已内置进 `launch_headless_sim.py`，无需再手动处理）。
- **`token_state` 不能用作逐帧对齐主键**：见第 4 节，只能用来找锚点。
- **`resource/` 整个目录 gitignore，无版本控制**：里面的东西如果被删除
  /丢失，git 完全找不回。本文档以及未来在 `resource/info/` 下的任何笔记，
  务必额外考虑是否要在别处（如 commit message、PR 描述、或仓库内非
  `resource/` 路径）留一份关键结论的备份。
- **capture 和 player 不能跨多轮工具调用分开执行**：`align_sim_replay.py
  capture --duration N` 是从自己进程内 `time.time()` 计时的固定窗口，不会
  等你。如果「启动 capture」和「启动 player」分成两次独立的工具调用，中间
  的模型思考/往返延迟会实打实吃掉窗口时间，导致抓到的帧数远低于预期（本次
  会话实测：分开执行只抓到 284 帧，纪录里预期是 1018 帧），进而让对齐失败
  或对不上。正确做法：把 `capture 启动（nohup 后台）+ sleep 2 + player（前台
  阻塞）` 写进**同一次** Bash 调用里连续执行。
- **deploy 二进制播完一整段 token 后会自动退出**（`frame_index` 递增到最后
  一个后打印 `Program exiting normally`），不是常驻服务。如果要在同一个
  episode 上重新抓包/重跑 player，必须先重新起一次 deploy 二进制（仿真本身
  不需要重启，`launch_headless_sim.py` 是常驻的）。第二次启动时 TRT 引擎
  已经缓存，`Init Done` 会快很多，不用再等冷启动的 ~108s。

---

## 10. 关键文件/路径速查

| 项 | 路径 |
|----|------|
| WBC yaml 配置 | `GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml` |
| ElasticBand 类 | `GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:366-411` |
| sim 主循环/弹簧施力 | `GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/base_sim.py`（L185构造 / L395-413施力） |
| 官方 sim 入口（参考） | `GR00T-WholeBodyControl/gear_sonic/scripts/run_sim_loop.py` |
| deploy 二进制 | `GR00T-WholeBodyControl/gear_sonic_deploy/target/release/g1_deploy_onnx_ref` |
| token player | `GR00T-WholeBodyControl/gear_sonic/scripts/sonic_encoder_input_player.py` |
| g1_debug 输出格式定义 | `GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp` |
| cola token 样本(已验证) | `resource/dataset/proprietary/ego/cola/0/trackingData_20260612_154443_latent_tokens_64d.npy` (979×64) |
| cola 数据集根目录 | `resource/dataset/proprietary/ego/cola/`（400 个 episode，`0`~`399`） |
| Real proprio 数据集（stats 来源） | `resource/dataset/proprietary/.../cola_merged_cleaned_0530/`（`sonic_tokens/`、`g1_proprio/`、`meta/proprio_stats_union_sonic.json`） |
| 34D 布局定义 | `src/huge/data/schemas.py`（`PROPRIO_LAYOUT` 附近） |
| proprio 加载校验逻辑 | `src/huge/data/sonic_dataset.py:147-150`（形状校验）、`:225`（`_load_state`）、`:230`（归一化） |
| 归一化实现 | `src/huge/data/state_normalizer.py`（`clip_outliers=True`，L74） |
| 抓包+对齐工具 | `scripts/preprocess/proprietary_ego/align_sim_replay.py` |
| 无头仿真启动器 | `scripts/preprocess/proprietary_ego/launch_headless_sim.py` |
| 组装 34D 伪 proprio 工具 | `scripts/preprocess/proprietary_ego/export_pseudo_proprio.py` |
| episode 0 伪 proprio 产出（已生成） | `resource/dataset/proprietary/ego/cola_pseudo_proprio/{g1_proprio,sonic_tokens}/episode_000000.npy` |
