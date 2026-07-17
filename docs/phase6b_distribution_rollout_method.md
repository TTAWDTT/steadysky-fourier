# Phase 6B 分布 Rollout 方法说明

日期：2026-07-17

## 研究定位

Phase 6B 是目前实验中表现最好的方法，也应该被视为本项目新的主线。

最初的想法是“傅里叶逐层数据注入”：先让模型学习缓慢的低频结构，再逐步加入更高频的成分。这个想法确实带来了稳定化效果，但后续实验也暴露了一个关键问题：低频优先训练会让模型更平稳，却也可能让预测变得过度平滑、异常振幅被压低、长期相位技巧不足。

Phase 6B 将核心问题从“如何按频率注入数据”转向：

> 长期 rollout 稳定性本质上是一个闭环分布问题。模型不应该只最小化一步场误差；它在自回归预测中生成的状态，也应该持续停留在真实系统的状态分布附近。

换句话说，Phase 6B 关注的是模型自己的闭环轨迹会不会跑到错误吸引子上。

## 方法概述

Phase 6B 由三个组件组成：

```text
residual-soft 数据课程
+ scheduled rollout exposure
+ batch-level feature-MMD distribution matching
```

它不改变模型架构，改进完全来自训练方式、数据注入方式和损失函数。

## 固定模型与数据设置

| 项目 | 设置 |
|---|---|
| 架构 | Makani SFNO `sfno_walker_1deg_edim384_layers8` |
| 可训练参数量 | 147,776,272 |
| 变量 | `tauu`, `tauv`, `tos`, `zos` |
| 数据 | 四变量 Walker/ocean 1-degree 数据 |
| 训练长度 | 150 epochs |
| 正式评测 | 120-month autoregressive rollout |
| 主要指标 | Makani `tos` RMSE, ACC, CRPS |

## 训练日程

Phase 6B 沿用了 Phase 6 之前引入的 residual-soft curriculum。训练数据先从更容易的低通/软残差目标开始，再逐步回到 raw data。和更早阶段相比，Phase 6B 的关键区别是：后期显式加入越来越长的 rollout exposure。

| Stage | 训练数据 | Epochs | Multistep count | Batch size |
|---|---|---:|---:|---:|
| 1 | `train_residual_soft_lp004_l020` | 10 | 1 | 16 |
| 2 | `train_residual_soft_lp008_l030` | 15 | 1 | 16 |
| 3 | `train_residual_soft_lp016_l045` | 20 | 1 | 16 |
| 4 | `train_residual_soft_lp032_l060` | 25 | 3 | 8 |
| 5 | `train_residual_soft_lp064_l080` | 35 | 6 | 4 |
| 6 | `train_raw` | 45 | 12 | 2 |

Makani 配置中使用的是累计 epoch，但实际增量训练预算是：

```text
10 + 15 + 20 + 25 + 35 + 45 = 150 epochs
```

## 损失函数

在 stage 1-3，Phase 6B 使用标准 Makani 场损失：

```text
L = L_field
```

在 stage 4-6，Phase 6B 加入一个较弱的 feature-MMD 分布损失：

```text
L = L_field + lambda_mmd L_feature_mmd
```

| Stage | `lambda_mmd` |
|---|---:|
| 4 | 0.020 |
| 5 | 0.035 |
| 6 | 0.050 |

其中 `L_field` 是标准 Makani L2 field loss，使用 constant channel weights 和 temperature-difference normalization。

## Feature-MMD 在衡量什么

MMD loss 比较的是一批预测 rollout 状态和一批真实目标状态。它并不强迫每一个预测样本都精确匹配对应的未来样本，而是判断：

> 预测 batch 和真实 batch 是否像是来自相似的粗粒度状态分布。

Feature vector 包括：

| 特征 | 作用 |
|---|---|
| Field mean | 控制大尺度状态位置，避免整体状态漂移过远 |
| Log variance | 惩罚方差塌缩和过度平滑 |
| Low-pass mean | 保留粗尺度空间结构 |
| RBF MMD kernel | 匹配 batch 分布，而不是逐样本轨迹 |

实现中会在 batch 内对 feature 维度做归一化，再计算 RBF MMD。这样可以避免某一类特征仅仅因为数值尺度更大就主导整个损失。

## 为什么它比纯傅里叶课程更有效

纯 Fourier curriculum 先教模型低频结构。这个过程有助于稳定，但也可能让模型学到一个低能量吸引子：模型不容易爆炸，却会变得过度平滑，异常被压低，长期相位技巧仍然弱。

Phase 6B 直接针对这个后续失败模式：

1. **Rollout exposure** 让模型在训练时见到自己生成的状态，而不仅仅是 teacher-forced 的一步状态。
2. **Distribution matching** 让塌缩的、过度平滑的、不真实的 rollout batch 变得有代价。
3. **Batch-level matching** 避免过度惩罚长期相位偏移，因为在长期混沌或气候式预测中，逐点未来本来就很难严格对齐。

这里最重要的概念转变是：

> 目标不是精确命中长期逐点轨迹，而是让模型诱导出的闭环分布持续接近真实数据分布。

## 正式结果

Phase 6B 是目前最好的正式 120-month 结果。

| Arm | 120-month `tos` RMSE | 120-month `tos` ACC | 120-month `tos` CRPS |
|---|---:|---:|---:|
| Raw baseline | 1.2134 | 6.4593 | 0.8532 |
| Phase 1 Fourier curriculum | 1.0546 | 8.3921 | 0.6513 |
| Phase 4 residual+rollout | 0.7416 | 9.3691 | 0.4900 |
| **Phase 6B distribution rollout** | **0.7375** | **9.4922** | **0.4818** |

相对 raw baseline：

| 指标 | 变化 |
|---|---:|
| `tos` RMSE | -39.2% |
| `tos` CRPS | -43.5% |
| `tos` ACC | +47.0% relative improvement |

Phase 6B 相比 Phase 4 的 RMSE 提升不大，但这个提升仍然重要：Phase 4 已经通过 rollout exposure 找到了强稳定性，而 Phase 6B 在此基础上进一步改善了稳定 rollout 的分布真实性。

## Phase 6B 还没有解决什么

Phase 6B 不应该被过度包装。它是目前最好的机制，但不是最终答案。

已知限制包括：

| 限制 | 含义 |
|---|---|
| 全局特征仍然粗糙 | mean/variance/low-pass 特征可能漏掉区域结构 |
| 长期相位技巧仍不确定 | 分布匹配不保证 ENSO/Nino3.4 相位正确 |
| 目前主要是单架构证据 | 证据最强的是当前 SFNO 设置 |
| 目前是单一正式数据族 | 还不是完整 ERA5 风格 benchmark |
| 还没有 multi-seed 不确定性 | 仍不知道 run-to-run variance |

## 为什么 Phase 9 从 Phase 6B 出发

Phase 9 保持 Phase 6B 的训练包络，只改变 MMD 内部的 feature map。

| Phase | 相比 Phase 6B 的变化 |
|---|---|
| Phase 9A | 在 MMD feature 中加入粗区域均值 |
| Phase 9B | 加入区域均值，并额外加入跨变量 covariance feature |

这是一个干净的延续实验：它检验“闭环分布匹配”这个当前最强机制，是否会因为更好地表达区域结构和变量耦合而进一步增强。

## 建议名称

Phase 6B 的工作名称可以是：

```text
Closed-loop Attractor Distribution Matching
```

更具体一点，也可以叫：

```text
Rollout Feature-MMD Training
```

中文可以暂称为：

```text
闭环吸引子分布匹配训练
```
