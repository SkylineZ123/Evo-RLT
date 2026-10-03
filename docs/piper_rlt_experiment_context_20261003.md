# Piper RLT 实验记录与代码审查上下文

整理日期：2026-10-03。用途：将本文件整体交给 ChatGPT，作为分析 Evo-RLT 代码、训练目标和实验设计的上下文。本文汇总已经完成的实验；后续建议单独标为“未验证”，不把建议写成实验结果。

仓库：`/home/boss/zzq/Evo-RLT`。除完整绝对路径外，下文代码与产物路径均相对此仓库。阅读本文件不要求访问原始对话；本地链接用于追溯证据，外部 ChatGPT 未必能打开。

## 1. 任务、症状与当前判断

任务是 Piper 机械臂的采血管插入。用户观察到的 rollout 问题是：**接近目标后插不准，需要人工微调**。希望弄清：RL token、actor/critic、Q 项、动作表示、训练参数及执行路径，分别对问题有何影响。

最初排查对象：

```text
output/checkpoint/actor_critic/piper_blood_gas_20261001_mix_r4_openpi/rl_checkpoint.pt
configs/train/actor_critic_piper.yaml
```

之后用户要求采用 10 月 2 日训练的 RL token，统一数据集和配套 backbone，重新对比 Evo-RLT 与 openpi-RLT。主结论以这轮新实验为准，旧实验保留为历史证据。

目前最有依据的判断：

1. 本地 Evo actor 的 Q 最大化项损害了本批数据上的人类动作拟合：新一轮删除 Q 项后，测试人类动作 MSE 降低 44.06%。
2. Critic 的轨迹回报拟合与动作优化方向是两种能力；现有 Q 的细微动作排序、局部纠偏梯度尚未表现出稳定可靠性。应优先排查 Q 梯度及其相对 BC 的强度。
3. 问题不能全归于 critic：Evo BC 的关节 MAE 仍高于同一 backbone 的原始 VLA。动作表示、reference dropout、压缩特征、训练停止点的贡献尚未完全分离。
4. openpi delta 配方改善了离线模仿，但 AC 与删除 Q 的版本基本相同，尚无证据把收益归因于 Q。
5. 以上不等于“Q 降低了真机成功率”。这些实验未执行机器人，缺少闭环成功率、接管率与微调次数的配对验证。

**版本注意：** 配置文件会继续被用户修改。例如整理时 `actor_critic_piper.yaml` 的输出目录已指向 `piper_blood_gas_20261003_mix_r4_openpi`。不能用当前 YAML 反推历史 checkpoint 的实际训练条件；应读取对应实验快照、checkpoint 元数据和 manifest。

## 2. 变量、单位与数据流

### 2.1 网络与动作变量

| 符号/字段 | 定义与维度 | 容易混淆的地方 |
|---|---|---|
| `z_rl` | 冻结 RL token 编码器输出，2048 维 | RL 阶段不更新该编码器 |
| `p_t` / proprio | 当前 6 个关节位置与夹爪，7 维 | Evo 输入归一化 proprio；正式 openpi 组输入物理 proprio |
| `s_t` / `state_vec` | Evo 中为 `[z_rl, proprio]`，2055 维 | 是网络可见状态表示，不保证包含真实系统全部状态 |
| `C` | RL 动作 chunk 长度，10 帧 | 不等于 VLA 原始 horizon 30；也不自动等于实机每次执行帧数 |
| `D` | 每帧动作维数，7 | 前六维关节位置使用 rad，夹爪使用 m |
| `A_ref` / `ref_chunk_flat` | 同一状态下 backbone 生成的 VLA 参考动作，取前 10 帧，展平为 70 维 | 本轮保持 VLA reference，不替换为人类动作 |
| `A_exec` / `exec_chunk_flat` | 记录中实际执行的动作 chunk | Critic 的 TD 回归使用该动作 |
| `h[t,j]` / human mask | chunk 内第 j 帧是否由人类接管 | BC target 按帧选择，不按整段成功与否选择 |
| `A_bc` / `bc_target_flat` | 人类帧取 `A_exec`，其他帧取 `A_ref` | 自动执行帧的 BC target 不是必然等于实际 actor 动作 |
| `μθ(s,A_ref)` | Actor 输出的 70 维动作均值 | μ 是 actor 输出；本地实现不是 `A_ref + residual` |
| `σ` / `fixed_std` | 固定高斯采样标准差，在各配方的归一化动作坐标中定义 | 不是学习得到的不确定性；相同 σ 不代表相同物理噪声 |
| `Q1,Q2` | 两个 critic，对状态与整个动作 chunk 输出标量 | Q 不是自动校准好的任务成功概率 |
| `Qmin` | `min(Q1,Q2)` | Evo actor 优化它；上游 actor 优化 Q1 |
| `k` / `actual_steps` | chunk 中有效执行帧数，尾部可能小于 10 | TD 折扣使用 γ 的 k 次方；评估排除 padding |
| `done` | transition 是否终止 | 终止时不 bootstrap |

数据流：图像 → 冻结 pi0.5 前缀特征 → 冻结 RL token → `z_rl`；`[z_rl, proprio, A_ref]` → actor → `μ`；`[z_rl, proprio, A]` → critic → Q。

本地 `arch: openpi` 已采用与上游对应的 head：z/proprio/chunk 分别投影到 256/64/256，拼接后经过两层 256 宽的 LN/GELU 网络。配置里的 `residual` 指普通 MLP 的隐藏层残差连接，且不作用于 `arch: openpi`；它不表示相对 VLA 的动作残差。

### 2.2 三种动作坐标

绝对动作归一化的主要尺度关系是：

```text
scale_d = (q99_d - q01_d) / 2
a_norm,d = 2 * (a_raw,d - q01_d) / (q99_d - q01_d) - 1
Δa_raw,d = scale_d * Δa_norm,d
```

上游 adapter 的分母还带数值稳定 epsilon，应以保存的实现为准。分位数变换不意味着所有样本都严格落在 [-1,1]；动作裁剪是另一个操作。

| 表示 | 训练动作内容 | 分位数来源 |
|---|---|---|
| absolute / SFT stats | 物理绝对关节、夹爪位置的归一化值 | 配套 backbone 的 SFT normalizer |
| delta / train stats | `delta[j,d] = a_raw[j,d] - p_t[d]`，仅前六关节；夹爪保持绝对值 | 仅 train split 的有效执行帧 |
| absolute / train stats | 绝对动作，保持表示不变 | 同样仅 train split 的有效执行帧 |

Delta 是相对 chunk 开始时当前 proprio 的位移，**不是相邻动作帧差，也不是相对 VLA 的残差**。改变表示和归一化会同时改变物理空间的 BC 权重与噪声幅度。

### 2.3 奖励、critic loss 与 actor loss

奖励是终止成功时 1，其余时刻为 0；没有直接的对准、接触或插入深度奖励。人类完成的真实任务成功同样是成功，不应仅因接管就自动改成失败奖励。

本地 Evo TD 目标与损失：

```text
R_chunk = Σ(j=0..k-1) γ^j * r[t+j],  γ = 0.99
a_next = clamp(μθ(s_next, A_ref,next))
y = R_chunk + γ^k * (1-done) * clip(Qmin_target(s_next,a_next), -100, 100)
L_critic = mean((Q1(s,A_exec)-y)^2) + mean((Q2(s,A_exec)-y)^2)
```

目标动作来自当前 actor，不是 target actor；目标 critic 使用软更新。

本地 actor loss：

```text
L_BC = 0.5 * mean_batch[Σ(j=1..10,d=1..7) (μ[j,d]-A_bc[j,d])²]
L_Q = -mean_batch[Qmin(s,μ)]
L_actor = L_BC + L_Q = 35 * MSE_over_70_coordinates - mean(Qmin)
```

训练时以 0.5 概率将一个样本的整段 reference 置零，BC target 仍保留原目标。Actor loss 查询未裁剪 μ；TD 目标和部署路径使用裁剪动作。

上游 openpi 组在 actor 更新时实际采样 `a = μ + 0.002 * ε`，`ε ~ N(0,I)`，目标为：

```text
前 20k 更新：L_actor = 10 * MSE(a,A_bc) - 0.1 * mean(Q1(s,a)) + 10 * L_motion
之后：      L_actor =  5 * MSE(a,A_bc) - 0.1 * mean(Q1(s,a)) + 10 * L_motion
L_motion = mean[(Δtime a_abs[:, :6] - Δtime A_bc,abs[:, :6])²]
```

`L_motion` 匹配物理关节动作的帧间增量，目标不是简单让机械臂不动。将 Q 系数归一为 1，上游显式 BC 系数为 warmup 100、之后 50，本地为 35；还必须考虑各自动作尺度，不能直接比较配置中的 `beta=0.5` 与 `bc_weight=5`。

## 3. 实验经过：各轮模型与统计口径不能混用

### 3.1 早期 52.5% 与原模型完整诊断（10 月 1 日）

最早对 560 个选定锚点得到 `Q(human)>Q(VLA)` 为 52.5%。随后扩大到原 final 模型全缓存的 11,766 个完整人类 chunk，得到 47.60%，episode bootstrap 95% 区间 43.6%–51.4%。这是评估集合变化，不是模型性能从 52.5% 退化到 47.6%。

原模型诊断还确认：

- Final `.pt` 为 outer step 20,000，对应 100,000 次 critic、50,000 次 actor 更新，只用 epoch4 数据。
- 当时录制配置指向的 LeRobot export 为 step 10,000，训练混合 warmup/epoch4=10%/90%；它与 final 不只是步数不同。
- 原缓存只有 train，没有独立 val/test；每条 transition 平均被抽样约 1,519 次。
- 13,046 条 transition 存在独立 BC target，按人类帧重建误差为 0。没有发现该 r4 缓存的“人类动作替换 VLA 输入”泄漏。
- 所有 bootstrap 链能连到终止 transition，没有发现本次缓存丢失尾部成功奖励。
- 12 个录制帧的训练/部署 eager 路径核验：权重、normalizer 一致；六关节路径差 MAE 约 0.0216°。未覆盖 compile、实时相机和控制延迟。
- 全缓存人类动作 MAE：export 0.487°，final 0.440°；final 单关节绝对误差 P95=1.351°。这是训练内描述性指标。

| 原 final 的 Q 诊断 | 数值 |
|---|---:|
| Q 人类偏好 | 47.60% |
| 平均 Q(human) / Q(VLA) | 0.490609 / 0.490841 |
| mean absolute Q gap / Q(human) 跨状态标准差 | 0.01596 / 0.18723 |
| VLA 处 Q 梯度朝人类方向 | 52.18% |
| 该方向平均 cosine | 0.00394 |
| Q(actor)>Q(human) | 98.27% |
| 记录结束前 30 帧的人类偏好 | 63.0% |

结束前窗口没有接触阶段标注，不能自动当作插入阶段。`Q(actor)>Q(human)` 本身也不证明 Q 错误，因为 actor 就在最大化 Q。

固定原 critic、同一起始 actor，3 seeds，各做 600 次 actor 更新、batch 256。每个 seed 分 120 episodes 做额外更新、30 留出；但这些 episode 都参与过原模型训练，不能称独立泛化测试。

| 冻结 critic 的 actor 消融 | 人类 MSE | 六关节 MAE |
|---|---:|---:|
| 原目标 BC+minQ | 0.0002906 | 0.457° |
| 只删除 Q | 0.0001484 | 0.286° |
| 只关闭 reference dropout | 0.0003072 | 0.465° |
| 只降 LR 到 1e-4 | 0.0002743 | 0.439° |
| 移植上游 online actor loss | 0.0002243 | 0.387° |
| 上游 actor loss + LR 1e-4 | 0.0002136 | 0.382° |

删除 Q 的 MSE 改善为 48.9%。另有从头训练的 12 次短预算实验：每组 3,000 critic 更新机会、1,500 actor 更新；原目标/BC-only/target 机制/组合机制的 MAE 分别为 1.477°/0.680°/0.883°/1.045°。短预算仅为原 critic 更新量的 3%，不用于判定收敛后的排名。

来源：[初轮诊断](../output/diagnostics/openpi_ablation_20261001/README.md)。其中早期关于 HIL 数据的机制解释，应结合本文第 8 节的纠正理解。

### 3.2 旧 token/backbone 的正式对照（10 月 2 日完成）

目录：`output/experiments/piper_openpi_benchmark_20261001`。6 组 × 3 seeds，100k 更新机会、50k actor 更新；固定 train/val/test，验证选模型。使用旧 r4 token 与旧 0925 backbone 的配套缓存，不是后来 1002 token 的结果。

| 配方 | 测试人类 MSE | 六关节 MAE | Q 人类偏好 |
|---|---:|---:|---:|
| Evo AC | 0.0010767 | 0.799° | 27.5% |
| Evo BC | 0.0007313 | 0.566° | — |
| openpi absolute / SFT stats | 0.0008103 | 0.609° | 39.4% |
| openpi delta / train stats | 0.0006135 | 0.469° | 66.0% |
| openpi delta BC | 0.0006125 | 0.468° | — |
| openpi absolute / train stats | 0.0009697 | 0.652° | 29.8% |

Evo 去 Q 后 MSE 降低 32.1%，95% 区间 24.5%–40.7%。Delta 去 Q 仅降低 0.16%，区间 −0.15%–0.46%。Delta 的 Q 偏好 66.0% 在三个 seed 中分别为 70.6%、77.5%、49.8%，区间约 50.0%–78.8%；不能据均值认定稳定可靠。

旧 VLA reference 在同一旧测试集合上的 MAE 为 1.539°。原 final/export 在此集合分别为 0.423°/0.474°，但它们已经训练过这些 episode，不能与从头训练的六组作公平泛化排名。

来源：[旧正式实验](../output/experiments/piper_openpi_benchmark_20261001/README.md)。新旧轮 token、backbone、VLA reference、SFT 归一化共同变化，不能将跨轮改善单独归给 token；跨轮归一化 MSE 也不直接可比。

### 3.3 新 1002 token 的正式对照（10 月 3 日完成，主实验）

以下第 4–7 节均指此轮，目录简写为 `P`：

```text
P = output/experiments/piper_openpi_benchmark_r4_1002_20261003
```

完成 18 次训练；`completion.json` 记录输入一致、导出数值核验通过。实验没有操作机器人、切换生产策略或覆盖原 checkpoint。

## 4. 主实验固定输入、划分与训练变量

### 4.1 固定输入与可追溯版本

| 项目 | 实际输入 |
|---|---|
| Token | `output/checkpoint/rl_token/piper_blood_gas_perceiver_r4_1002/demo_adapt_checkpoint.pt`，step 20,000 |
| Token 快照 | `P/inputs/demo_adapt_checkpoint.pt` |
| Backbone | `output/checkpoint/pi05/blood_gas_teleop_1001/040000/pretrained_model` |
| 数据集 | `/home/boss/lerobot_datasets/piper/1001_piper_segment/eval_rlt_segment_144623` |
| 统一预处理结果 | `P/dataset.npz` |
| Split seed | 20261001 |
| 训练 seeds | 11、22、33 |
| 编码/reference 随机种子 | 每 episode 为 20261003 + episode_id，reference 生成一次供所有组共享 |

SHA256：

```text
token:   55e9ee77819fbf439b8746c64deb3bd0453b479056b4f7e8a5b8bb13f0d1c1f7
backbone: ddbe262341b32d25b1e35bb0a9877621955109ffc7fdf1878e0925abae72a235
dataset: b1b5d9d9a9d767926bcf04a8c19b1ac01aa8842e951be9dc3932638e5f19eae8
```

全部图像特征与 VLA reference 用新配套模型重建，未复用旧 backbone 缓存；相机、prompt、等比缩放加黑边、image_only 和 512 个图像前缀 token 与 token 元数据保持一致。

| Split | Episodes | 成功/失败 | Transitions |
|---|---:|---:|---:|
| train | 104 | 98 / 6 | 11,661 |
| val | 23 | 21 / 2 | 2,602 |
| test | 23 | 21 / 2 | 2,585 |
| 合计 | 150 | 140 / 10 | 16,848 |

140 条成功都发生过人工接管并由人工结束，10 条失败都未接管。没有记录到自主成功。Chunk 窗口重叠，同一 episode 不跨 split。缺少自主成功是覆盖范围的描述，不是判定 HIL 协议无效的依据。

Token 预训练验证 episode 为 `[10,66,98,107]`，其中 3 个落在本轮 head train、1 个在 head val，head test 没有。这意味着本轮 23 个 test episode 已被 token 预训练使用过。**只保证 heads 的 episode 留出，不能宣称整个视觉策略未见测试数据。** Backbone 的 SFT 数据同样不能假定与这批数据完全独立。

### 4.2 六组配方与对齐范围

| 项目 | Evo AC / Evo BC | openpi 四组 |
|---|---|---|
| Head 架构 | z/p/chunk 投影 256/64/256，2×256 LN/GELU | 相同对应结构 |
| Actor / critic LR | 3e-4 / 3e-4 | 1e-4 / 1e-4 |
| Batch | 256 | 128 |
| 优化器 | Adam，global grad clip=1 | 上游 Adam |
| γ / τ | 0.99 / 0.005 | 0.99 / 0.005 |
| Actor 更新频率 | 每 2 次 critic 更新机会 | 每 2 次 |
| Target 更新 | 每 5 次更新 target critic | 每 2 次更新 target actor 与 target critic |
| TD 下一动作 | 当前 actor μ，按动作范围裁剪 | target actor 采样，σ=0.002 |
| Actor 目标 | 35MSE−minQ | warmup 10MSE−0.1Q1+10motion，之后 5MSE−0.1Q1+10motion |
| Reference dropout | 0.5 | 0.5 |
| Proprio | SFT 归一化 | 物理量 |

六组具体变量：

1. `evo_ac`：本地 absolute/SFT 配方。
2. `evo_bc`：同一 actor 配方删除 Q 项，跳过 critic 更新，actor 更新数保持一致。
3. `openpi_abs`：上游训练器 + absolute/SFT 动作。
4. `openpi_delta`：上游训练器 + delta/train stats。
5. `openpi_delta_bc`：第 4 组的 actor Q 权重设为 0，仍保留 critic 训练、采样和 motion loss。
6. `openpi_abs_trainstats`：上游训练器 + absolute/train stats，控制“只换统计量”的影响。

每组 100,000 次更新机会，50,000 次 actor 更新。Evo 与 openpi 的样本抽取量约为 25.6M 与 12.8M，**对齐的是更新次数，不是总抽样数**。跨 Evo/openpi 是整套配方比较，不是单一变量因果实验。

正式实验中的选中步数、20k warmup 和 100k 预算均按训练更新机会计数，actor 每两步更新一次；最初原 checkpoint 的 20k 则是 outer step，乘 UTD=5 才得到 100k critic 更新，二者不能混用。

正式 openpi 组直接调用保存的未修改上游 `trainer.train_step`、networks、action adapter；Evo 为保持计算效率移植到 JAX，已与本地 PyTorch 验证前向、loss 和梯度。该实验没有复现上游在线采集系统；20k 后“online 权重”仍在同一固定 replay 上使用。

每 1,000 步评估，以 **val 的 episode 平均人类 MSE** 选择 checkpoint，test 不参与选择；仍训练到 100k 并保留最终模型。旧正式实验每 2,000 步评估，早期最佳步数的分辨率不同。

### 4.3 指标定义

| 指标 | 精确定义与边界 |
|---|---|
| Episode 平均人类 MSE | 预测先转回物理绝对动作，再用统一新 SFT scale 计算 `(pred-human)/scale` 的平方误差；只取有效人类帧、7 维。先在 episode 内平均，再对有人类帧的 episode 等权平均 |
| 六关节 MAE（°） | 有效人类帧的前六维物理绝对误差平均，rad 转 degree；按帧/出现次数汇总，不是 episode 等权 |
| Q 人类偏好 | 在完整人类 chunk 上统计 `Qmin(s,A_human)>Qmin(s,A_ref)` 比例；不是成功率、最优动作准确率或人工必优标签 |
| VLA 处 Q 方向 | `∇a Qmin(s,A_ref) · (A_human-A_ref)>0` 的比例 |
| Actor 处 Q 方向 | `∇a Qmin(s,μ) · (A_human-μ)>0` 的比例；与上一指标的求导位置不同 |
| TD loss | 双 Q 对各自配方 bootstrap target 的 MSE 之和；跨配方目标不同，不能直接按绝对值排名 |
| 终止 Q RMSE | 终止 transition 无 bootstrap，Q 对其折扣 chunk 回报的 RMSE |
| Q gap | `mean(abs(Qhuman-Qref))`；同时报告 Qhuman 跨状态标准差 |
| ± | 3 个训练 seed 的标准差，不是置信区间 |
| 相对 MSE 改善 | `1 - MSE_右/MSE_左`；负数表示变差 |
| 95% 区间 | 固定划分/编码器下，对 seed 和完整 episode 配对 bootstrap 3,000 次；不是闭环成功率区间 |

测试集有 21 个包含人类帧的 episode，1,743 个完整人类 chunk。窗口大量重叠，不能将 1,743 当独立试验次数；同一原始帧可能出现在多个 chunk 中。

正式离线评估均使用完整 reference 和确定性 actor mean，不采样探索噪声。Evo 按动作边界裁剪并反归一化；openpi 经各自动作 adapter 转回物理绝对动作。上游训练时 σ=0.002 不代表结果表中的评估动作也加入了噪声。

## 5. 主实验结果

### 5.1 验证选中 checkpoint 的测试表现

| 配方 | Episode 人类 MSE（均值±seed标准差） | 六关节 MAE | Q 人类偏好 | 选中步数，seeds 11/22/33 |
|---|---:|---:|---:|---|
| 同一 backbone 的 VLA reference | 0.0001373 | 0.352° | — | 无 head 训练 |
| 整个 chunk 保持当前 proprio | 0.0023041 | 0.656° | — | 静态基线 |
| Evo AC | 0.0003012 ± 0.0000196 | 0.519° | 51.3% | 41000 / 68000 / 41000 |
| Evo BC | 0.0001685 ± 0.0000017 | 0.398° | — | 7000 / 12000 / 9000 |
| openpi absolute / SFT stats | 0.0002311 ± 0.0000070 | 0.482° | 54.2% | 20000 / 20000 / 32000 |
| openpi delta / train stats | 0.0001287 ± 0.0000004 | 0.317° | 53.3% | 5000 / 2000 / 3000 |
| openpi delta BC | 0.0001285 ± 0.0000004 | 0.316° | — | 5000 / 2000 / 3000 |
| openpi absolute / train stats | 0.0002747 ± 0.0000079 | 0.521° | 48.5% | 43000 / 20000 / 82000 |

| 左 → 右 | 人类 MSE 相对改善 | 95% 区间 |
|---|---:|---:|
| Evo AC → Evo BC | 44.06% | 35.75%–51.16% |
| Evo AC → openpi delta | 57.26% | 51.20%–62.54% |
| openpi abs/SFT → delta | 44.29% | 36.64%–51.17% |
| openpi abs/train → delta | 53.14% | 47.10%–58.65% |
| openpi delta → delta BC | 0.20% | −0.01%–0.38% |
| VLA → delta | 6.24% | −1.63%–13.48% |
| VLA → delta BC | 6.42% | −1.47%–14.19% |
| VLA → Evo BC | −22.72% | −39.78%–−8.72% |

解释：Evo Q 项损害模仿的证据较强；delta AC 与 BC 的差异不能确认；delta 相对 VLA 的均值改善也不能确认稳定胜出。静态基线差很多，delta 的低误差不只是“保持当前姿态”造成。

### 5.2 Critic 与动作方向

| 配方 | train / test TD loss | VLA 处 Q 方向朝人类 | Q(exec)<0 | 终止 Q RMSE |
|---|---:|---:|---:|---:|
| Evo AC | 0.000050 / 0.005638 | 51.2% | 0.0% | 0.167 |
| openpi abs/SFT | 0.000169 / 0.006120 | 54.4% | 0.4% | 0.165 |
| openpi delta | 0.004500 / 0.016418 | 58.4% | 54.6% | 0.190 |
| openpi abs/train | 0.000193 / 0.006861 | 57.0% | 0.0% | 0.175 |

| 配方 | Q 人类偏好及 95% 区间 | mean absolute Q gap / Qhuman 跨状态标准差 |
|---|---:|---:|
| Evo AC | 51.3% [47.7%,54.8%] | 0.002487 / 0.200154 |
| openpi abs/SFT | 54.2% [50.4%,57.8%] | 0.002837 / 0.208927 |
| openpi delta | 53.3% [47.9%,59.0%] | 0.039876 / 0.399851 |
| openpi abs/train | 48.5% [38.5%,58.6%] | 0.017451 / 0.192011 |

这些现象支持“对状态/轨迹变化的拟合强于细微动作区分”的解释，但不是其因果证明。人类动作本身也未被证明处处优于 VLA，偏好率接近 50% 不等于 critic 准确率为随机水平。

上述偏好和方向诊断统一使用 minQ，上游 actor 实际优化 Q1，因此表中的 minQ 方向指标不能直接视为上游 actor 的实际优化方向。

Delta 的 actor 在 2k–5k warmup 被选中；所选早期 checkpoint 的 critic 存在明显的绝对值校准误差，其后是否改善未由本表证明。任务回报非负，而大量 Q 为负，不能把其 Q 当可靠成功概率。不能由早期最佳 actor 的结果推出在线采集后成熟 critic 仍无收益。

### 5.3 继续训练到 100k

| 配方 | train MSE：选中 → 100k | test MSE：选中 → 100k |
|---|---:|---:|
| Evo AC | 0.0000999 → 0.0000914 | 0.0003012 → 0.0003242 |
| Evo BC | 0.0000850 → 0.0000107 | 0.0001685 → 0.0002617 |
| openpi abs/SFT | 0.0001386 → 0.0001654 | 0.0002311 → 0.0002915 |
| openpi delta | 0.0000983 → 0.0000078 | 0.0001287 → 0.0002055 |
| openpi delta BC | 0.0000980 → 0.0000074 | 0.0001285 → 0.0002066 |
| openpi abs/train | 0.0000729 → 0.0000427 | 0.0002747 → 0.0002967 |

多数配方训练误差下降、测试误差上升，表明固定数据反复优化存在泛化退化。不能将在线不断新增数据时的训练预算直接套到固定 replay。

## 6. 后续代码与梯度诊断

### 6.1 BC 与 Q 的梯度冲突

只加载主实验三个验证选中 Evo AC checkpoint，在 CPU 计算梯度，没有优化器更新。测试完整人类 chunk 全部 1,743 个，训练集最多抽取 2,048 个；batch 不超过 256。先按样本数加权 batch 指标，再对 seeds 等权平均。

令 `g_BC=∇θ L_BC`，`g_Q=∇θ[-mean Qmin(s,μθ)]`。注意是两个损失的参数梯度，不是 Q 值大小之比。

| 测试指标 | 完整 reference | reference dropout=0.5 |
|---|---:|---:|
| cosine(g_BC,g_Q) | −0.9116 | −0.9085 |
| norm(g_Q)/norm(g_BC) | 1.2705 | 1.2809 |
| Actor 处 Q 梯度朝人类 | 15.22% | 15.99% |
| 输出越界的坐标比例 | 0.0331% | 0.0284% |
| Clamp 后 BC loss 相对变化 | +0.00689% | +0.00495% |

负 cosine 表示单独沿 `-g_Q` 做无穷小参数更新会增加该批 BC loss。梯度在 Adam 和总梯度裁剪之前计算，不代表每个实际 Adam 更新都如此。联合目标平衡点出现梯度冲突并不自动构成算法错误；结合去 Q 的受控消融，才能支持“当前 Q 项妨碍模仿”的解释。

**15.22% 是 actor 动作处，51.2% 是 VLA 动作处，不能互换。** 当前 clamp 对模仿误差影响很小，不支持把越界裁剪当成主要解释；未因此排除真机控制限幅等其他问题。

来源：[梯度诊断](../output/experiments/piper_openpi_benchmark_r4_1002_20261003/code_analysis/README.md)。

### 6.2 RL token 是否已被证明有问题

1002 token 使用 Perceiver：两层编码、两层解码，8 heads，FFN 4096，1 个 2048 维 RL token，从两相机共 512 个图像前缀 token 重建输入。预训练 20k 步，batch 16，Adam LR 峰值 2e-4，warmup 2k，余弦衰减到 5e-6，grad clip 1。

其重建目标为 `mean[((pred-target) * std^(-norm_gamma))²]`，`norm_gamma=0.5`；这里的 norm_gamma 与 RL 折扣 γ=0.99 无关。RL 阶段冻结 token，所以加入 actor Q 项没有通过反向传播改变 token。

20k 最后一次验证日志：

| Token 指标 | 数值 |
|---|---:|
| reconstruction loss | 0.1577447 |
| batch 内打乱 z_rl 后 reconstruction loss | 0.6005150 |
| z_rl_usage，`1-recon/recon_shuffled` | 0.7373176 |
| z_rl 样本间平均 cosine | 0.7847694 |

来源：`output/checkpoint/rl_token/piper_blood_gas_perceiver_r4_1002/metrics.jsonl`。这些数据不支持“解码器完全不使用 token”这一解释，也不能证明 token 保留了插入所需的微小几何信息。重建通用视觉特征并不直接监督对准误差、接触状态或动作价值；这一点是待验证的表示瓶颈，尚未做 token 专项因果消融。

## 7. μ、噪声与实际执行路径

本地 `ChunkActor.sample()` 已实现 `μ + σ*randn`；但当前 actor loss、critic 的下一动作，以及 LeRobot `RLTActionModifier` 执行路径都取 μ，其中后两者裁剪动作。因此只修改 `fixed_std` 不会改变这些路径。通用 `RLTPolicy.select_action` 则支持随机采样，不能笼统说整个仓库没有随机策略。

按主实验 `dataset.npz` 的 q01/q99 换算，裁剪前噪声尺度为：

| 动作表示 / 归一化 σ | 六关节标准差范围 | 夹爪标准差 |
|---|---:|---:|
| absolute/SFT，0.05（当前配置值） | 1.37°–3.53° | 1.848 mm |
| absolute/SFT，0.002 | 0.0549°–0.1411° | 0.07392 mm |
| delta/train，0.002 | 0.00331°–0.00629° | 此处未列 |

这些是尺度换算，不是机器人测量；关节角标准差不能直接换算为针尖位移。Clamp 和平滑还会改变最终执行扰动。

当前讨论形成的建议，均尚未闭环验证：

- 固定离线数据上给 actor 的 Q 查询加噪声，没有提供新的真实动作后果；不能指望自动修正 critic 的偏差。
- 在线采集时的小幅、限幅、时间平滑扰动可能补充附近动作结果，应记录实际执行动作及对应后继状态，再训练 critic。
- 不能给既有 transition 的动作加噪后，假定原 reward 和 next_state 仍是该新动作的真实结果。
- 精细插入评估保留确定性 μ 基线；不要直接将 absolute 坐标下 σ=0.05 全面开启。不同表示的 σ 不能照数值复制。
- 本轮没有单独隔离“actor 采样”“target 动作采样”“在线探索”三种噪声的收益；openpi 组合结果不能证明任意一项单独有效。

## 8. HIL 采集方式的讨论与结论纠正

用户提出两种方式：A）人类纠偏并一直操作到成功；B）人类纠到更有利的状态，再交回 actor 自动完成。

**现有实验没有证明 B 优于 A，也没有证明 A 导致 critic 失效。** 人类完成的成功轨迹是合法的 off-policy 数据；相同 `(s,a,r,s_next)` 的 TD 学习规则不因操作者身份改变。本地 TD 目标在人类采集的 transition 上也用 actor 的下一动作，并非简单拟合“后续永远由人类完成”的回报。

两种采集方式可能改变状态、动作、后续结果的覆盖分布。B 能增加当前 actor 在恢复状态后的真实结果记录，A 能提供更完整的后段人类 BC 监督；净效果未知。当前 BC target 在人类帧取执行动作，自动帧取 VLA reference，提前交回也会减少后段的人类动作标签。

没有自主成功样本是潜在覆盖限制，不是 RL 能工作的必要条件。每个状态都有成对的人类/VLA 反事实结果，也不是 off-policy RL 的必要前提。不能因此给数据硬造 `Q(human)>Q(VLA)` 标签，或把所有人类成功改为失败。

更稳妥的后续方向是保留完整人类成功示教，同时补充恢复状态后的 actor 成功和失败尝试，并单独评价自主完成能力；此建议尚未实验验证。

## 9. 已知代码行为、证据等级与待排查项

| 项目 | 当前证据 | 可得结论 / 尚缺的验证 |
|---|---|---|
| Q 项与 BC 冲突 | 两轮正式去 Q 对照及梯度诊断 | 优先检查 Q 方向与权重；未证明真实回报降低 |
| Critic 留出泛化 | Evo train/test TD 差距，动作偏好与方向弱 | 不能用低训练 TD 判断纠偏能力已形成；尚缺闭环/真实动作后果校验 |
| 固定 replay 过度训练 | 多组 train 下降、test 上升 | 需要 actor 验证选模型；本地 native loop 目前主要验证 critic loss 并最终保存最后模型 |
| Actor 直接预测绝对动作 | `mu=self.net(...)`，非 reference 残差 | 强 VLA 的细节可能未被保留；待单独验证 residual、dropout 等设计 |
| 表示及尺度 | delta 优于 absolute；仅换 train stats 不足 | 不能将中心化、尺度、物理 BC 强度完全分离解释 |
| Target 机制差异 | 本地无 target actor，target critic 每 5 次更新 | 短预算移植有收益；尚未以主实验预算逐项隔离 |
| Token 的任务相关性 | 重建使用了 z_rl，但无几何/价值监督 | 不能认定塌缩；也不能确认精细信息充分 |
| 训练/执行一致性 | 旧模型 eager 核验通过；新导出数值对齐 | compile、相机时序、chunk 实际执行长度和控制延迟仍未闭环排除 |
| Checkpoint 身份 | 原 10k export 与 20k final 不同 | 评估必须固定实际部署权重，不能混合报告 |

优先代码入口（本文件整理时的实现，后续修改需重新核对）：

| 文件 | 审查内容 |
|---|---|
| `src/evo_rlt/core/actor.py` | `OpenpiMLP`、reference dropout、直接 μ 输出、固定 std、sample、动作范围 |
| `src/evo_rlt/core/critic.py` | 双 Q 的状态/动作输入与 minQ |
| `src/evo_rlt/core/losses.py` | `critic_loss` 的当前 actor bootstrap、γ^k；`actor_loss` 的 summed BC 和未裁剪 μ |
| `src/evo_rlt/core/trainer.py` | `offline_rl_loop` 的 outer step/UTD、target 更新、验证和最终保存 |
| `src/evo_rlt/core/rl_token.py` | `encode`、`reconstruction_loss`、shuffle diagnostics |
| `src/evo_rlt/cli/train_rl_token.py` | 冻结 backbone、图像处理、token 训练目标 |
| `src/evo_rlt/cli/train_chunk_actor_critic.py` | Token 冻结、缓存和 head 训练装配 |
| `src/evo_rlt/cli/build_bucket_cache.py` | `bc_target` 模式的逐帧人类标记与 VLA reference 保留 |
| `src/evo_rlt/core/policy.py` | 通用随机/确定性选择路径 |
| `src/evo_rlt/adapters/lerobot/policies/action_modifier.py` | LeRobot 使用 μ 并 clamp 的执行入口 |
| `P/benchmark.py` | 六组配方、共同数据转换、统一评估指标 |
| `P/upstream/rlt_online_rl/trainer.py` | 冻结快照的上游 Q1 actor loss、motion、target 更新 |
| `P/upstream/rlt_online_rl/action_representation.py` | Delta 的定义、归一化、反归一化 |

## 10. 数值核验与产物索引

主实验已完成的核验：重建人类 BC target 误差为 0；反归一化 state/action 对原始数据最大误差约 2.4e-7 / 3.6e-7；bootstrap 断链为 0。所有非终止 chunk 的有效长度均为 10，因此上游 γ^C 与本地 γ^k 在本批非终止样本上一致；终止样本不 bootstrap。

Evo JAX/PyTorch 前向、loss、梯度最大差约 3.8e-6。训练使用 FP32 最高矩阵乘法精度，backbone 特征提取使用 bfloat16。18 个 Torch bundle 与对应 JAX 测试预测最大物理动作差约 1.9e-6。

环境为隔离 JAX/Flax 训练环境；其中 NumPy 2.5.3 不满足上游原 pyproject 的 numpy<2 限制。上游函数源码被冻结并记录 hash，但不能声称软件环境完全一致。

| 产物 | 内容 |
|---|---|
| [主实验 README](../output/experiments/piper_openpi_benchmark_r4_1002_20261003/README.md) | 完整正式报告、图表与复现入口 |
| `P/summary.json` | 正式结果汇总 |
| `P/input_manifest.json`、`source_manifest.json`、`upstream_sha256.json` | 输入、项目源码、上游源码溯源 |
| `P/numerical_validation.json` | Evo JAX 与 PyTorch 数值对照 |
| `P/completion.json` | 18 次完成、输入一致、导出核验状态 |
| `P/baseline_controls.json`、`vla_baseline.json` | 静态与 VLA 基线 |
| `P/code_analysis/gradient_conflict.json`、`gradient_summary.json` | 梯度逐 batch 结果与汇总 |
| `P/<variant>/seed<seed>/actor_critic_bundle.pt` | 各组导出候选 |
| `P/candidate.py` | `CandidateActor` 读取候选并处理各自坐标 |
| `P/inputs/rl_token_piper.yaml` | 实验时 token 配置快照 |

Delta/raw-proprio bundle 不能直接当现有 LeRobot `rlt_ac` checkpoint 加载，须使用配套动作变换。实验未切换生产策略。`setup_experiment.py` 是首次初始化脚本，已有结果目录不要重跑初始化覆盖。

## 11. 可随本文件一起发送给 ChatGPT 的分析请求

> 请基于上述事实审查我的 RLT 实现。先区分已验证的问题、机制假设和证据不足的结论，再提出最小且有区分力的实验。若不能访问本地代码，请明确需要补充哪个函数，不要假装已经阅读。
>
> 重点分析：
> 1. 为什么 critic 可以降低训练 TD loss，却没有形成稳定的局部动作纠偏方向？如何分别检验状态表示、动作覆盖、bootstrap 偏差、Q 权重和 critic 泛化？
> 2. summed BC、动作归一化和物理尺度是否导致 Q 相对约束过强？如何避免只比较原始 loss 数字？
> 3. 为什么去 Q 的 Evo actor 仍差于同 backbone 的 VLA？如何单独检验 absolute/residual、reference dropout、token 压缩和早停？
> 4. 原版 openpi 训练函数在固定 replay 上没有显示明确 Q 收益，这能支持什么结论，不能外推到哪些在线场景？
> 5. 如何分别对照 actor 训练采样、target 动作噪声和在线采集探索？使用什么物理幅度与记录字段才能解释结果？
> 6. 如何设计闭环配对实验，评价自主成功、人工微调次数、接管率、完成时间，并控制目标位姿和执行时序？
>
> 请保留以下限制：人类动作不保证处处最优；Q 偏好不是准确率；偏离人类不自动等于任务失败；HIL 人类完成是有效数据；没有自主成功不等于 RL 必然失败；当前实验只留出 heads 的 episode；新旧轮不只更换了 token。对每个建议标明要改的变量、固定的变量、评价指标及可证伪条件，避免一次修改多个因素后无法归因。
