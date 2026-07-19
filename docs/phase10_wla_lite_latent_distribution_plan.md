# Phase 10: WLA-Lite Latent Distribution Rollout

## 背景

Phase 6B 目前仍是 SteadySky 的最优主线：

| Arm | 120mo tos RMSE | 120mo tos ACC | 120mo tos CRPS |
|---|---:|---:|---:|
| Phase 6B `distribution_rollout` | **0.7375** | **9.4922** | **0.4818** |

Phase 9 将 Phase 6B 的全局 feature-MMD 扩展为 regional/coupled feature-MMD，但 120 个月 rollout 没有超过 Phase 6B：

| Arm | 120mo tos RMSE | 120mo tos ACC | 120mo tos CRPS |
|---|---:|---:|---:|
| Phase 9A regional | 0.8786 | 8.3969 | 0.5961 |
| Phase 9B coupled regional | 0.8031 | 9.0533 | 0.5389 |

这说明继续手工堆区域均值、协方差等统计量不一定能解决长期平滑化问题。更合理的下一步是让数据自己学习一个“什么样的天气状态是有细节、可信、在真实系统分布内”的 latent 表征。

## WLA 启发

参考官方源码 [`NJU-LHRS/Weather-Latent-Autoencoder`](https://github.com/NJU-LHRS/Weather-Latent-Autoencoder)，WLA 的核心不是简单加一个 loss，而是先学习 pixel-to-latent 的 weather autoencoder：

- encoder 将天气场压到 latent tokens；
- decoder 从 latent tokens 重建天气场；
- 下游任务在 latent 表征中获得更高效、更不容易过度平滑的表示；
- 官方实现使用 transformer encoder/decoder、binary spherical quantization、感知/对抗重建损失。

官方仓库目前公开了模型代码，但训练代码、预训练权重和 ERA5-latent 数据仍在 TODO 中。因此本项目不直接依赖官方 WLA pipeline，而是做一个保守、可复现实验：

> 训练一个四变量 Walker 数据上的轻量 autoencoder，冻结 encoder，用其 latent feature 替代 Phase 6B 的手工 feature-MMD。

这保持了核心实验原则：**SFNO 架构不变，数据注入和训练约束方式改变。**

## Phase 10A

| 项 | 设置 |
|---|---|
| Arm | `latent_distribution_rollout` |
| Run | `phase10_latent_distribution_rollout_edim384` |
| 模型 | Makani SFNO `sfno_walker_1deg_edim384_layers8` |
| 参数量 | 147,776,272 |
| 数据 schedule | 与 Phase 6B 完全一致 |
| Rollout schedule | `1,1,1,3,6,12` |
| Batch schedule | `16,16,16,8,4,2` |
| Epoch schedule | `10,15,20,25,35,45` |
| Loss | `L = L_field + lambda_mmd L_latent_mmd` |
| MMD 权重 | stage 4/5/6 为 `0.020/0.035/0.050` |

`L_latent_mmd` 的 feature map：

- 原 Phase 6B global stats 作为兜底；
- frozen WLA-lite encoder latent mean；
- frozen WLA-lite encoder latent log variance；
- frozen WLA-lite encoder latent 6x12 coarse grid。

## Phase 10B

| 项 | 设置 |
|---|---|
| Arm | `latent_recon_distribution_rollout` |
| Run | `phase10_latent_recon_distribution_rollout_edim384` |
| 与 Phase 10A 差异 | 加一个很弱的 autoencoder reconstruction-manifold penalty |
| Loss | `L = L_field + lambda_mmd (L_latent_mmd + 0.02 L_recon)` |

`L_recon = ||D(E(x_pred)) - x_pred||^2`。

直觉是：如果预测状态偏离 autoencoder 在真实数据上学到的状态流形，重建误差会变大，因此这个项会温和地惩罚“跑出真实系统状态分布”的预测。但它只加在 Phase10B，方便和 Phase10A 做干净消融。

## WLA-Lite Autoencoder 训练

新增脚本：

```bash
python scripts/train_wla_lite_autoencoder.py
```

默认配置：

| 项 | 默认值 |
|---|---:|
| Epochs | 80 |
| Batch size | 16 |
| Latent channels | 32 |
| Hidden channels | 64 |
| Optimizer | AdamW |
| LR | 2e-4 |
| Spectral shape weight | 0.02 |
| Checkpoint | `${STEADYSKY_WORK}/latent/wla_lite/wla_lite_best.pt` |

autoencoder 输入为四变量场的 z-score 标准化版本。弱 spectral-shape 项用于避免 autoencoder 自身只学到过度平滑重建。

## 启动方式

双卡启动脚本：

```bash
bash scripts/launch_phase10_pair.sh
```

它会：

1. 如果不存在 `${STEADYSKY_WORK}/latent/wla_lite/wla_lite_best.pt`，先训练 WLA-lite autoencoder；
2. 在两张卡上并行启动 Phase10A 和 Phase10B；
3. 保持 Phase 6B 的训练预算、数据 schedule、rollout schedule 和正式评测协议。

## 正式评测

训练完成后仍使用 Makani 120-month autoregressive rollout：

```bash
bash scripts/run_phase1_long_rollout_eval.sh latent_distribution_rollout 120
bash scripts/run_phase1_long_rollout_eval.sh latent_recon_distribution_rollout 120
```

核心判据仍是：

- `validation RMSE tos(87600)` 越低越好；
- `validation ACC tos(87600)` 越高越好；
- `validation CRPS tos(87600)` 越低越好；
- 同时检查 Nino3.4 曲线、空间误差图、谱图、漂移曲线，防止数值指标掩盖平滑化副作用。

## 预期与风险

预期：

- 比 Phase 9 的手工 regional features 更有希望，因为 latent feature 是从真实天气状态重建任务中学到的；
- 比直接频段 loss 更稳，因为它不强迫 exact Fourier phase；
- 比单纯 MMD 更可能抑制平滑化，因为 latent encoder 可以携带细节结构。

风险：

- 如果 WLA-lite autoencoder 自身重建偏平滑，latent-MMD 仍可能继承平滑偏置；
- 如果 reconstruction penalty 太强，Phase10B 可能退化为短期状态流形约束，伤害长期动力学；
- 如果 AE latent 太弱，只会退化成 Phase6B 的另一个手工统计替代品。

因此 Phase10A 是主实验，Phase10B 是额外消融。
