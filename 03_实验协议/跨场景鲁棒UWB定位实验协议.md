# 跨场景鲁棒 UWB 定位实验协议

更新时间：2026-07-19

## 1. 实验目标

本实验协议服务于拟定论文：

> Signal-Informed and Uncertainty-Aware UWB Localization under Cross-Environment Degradation

核心问题不是“平均定位误差能不能降低一点”，而是：

1. 测距误差是否可以通过信号特征被解释和补偿；
2. 每条 UWB 测距是否能输出可信度和不确定性；
3. 可信度和不确定性能否改善鲁棒几何定位；
4. 方法是否能跨环境、跨布局、跨数据集泛化；
5. 在 NLOS、anchor dropout 和几何退化下是否能减少长尾失败。

## 2. 总体实验结构

第一篇论文建议分为五个实验层次：

| 层次 | 实验名称 | 目的 |
|---|---|---|
| E1 | 单链路测距误差建模 | 验证信号特征能否解释 ranging error |
| E2 | 同环境定位实验 | 验证基础定位精度 |
| E3 | 跨环境泛化实验 | 验证模型不依赖单一场景 |
| E4 | 退化鲁棒性实验 | 验证 NLOS、anchor dropout、几何退化 |
| E5 | 外部数据集验证 | 验证跨数据集泛化 |

## 3. 数据集划分协议

### 3.1 主数据集 D1 划分

D1 包含四个室内环境，建议使用两类划分。

第一类：同环境随机划分。

```text
Train: 70%
Validation: 15%
Test: 15%
Group constraint: 同一 tag position 的重复测量不能同时出现在 train 和 test
```

该划分用于验证模型基本能力，但不能作为论文主结论。

第二类：leave-one-environment-out。

```text
Fold 1: train env1, env2, env3; test env0
Fold 2: train env0, env2, env3; test env1
Fold 3: train env0, env1, env3; test env2
Fold 4: train env0, env1, env2; test env3
```

该划分应作为论文最重要实验，因为它直接对应“cross-environment degradation”。

### 3.2 外部数据集 D2 划分

D2 不建议参与主训练，建议作为外部测试。

```text
Train: D1 only
Test: D2 all trajectories
Few-shot: D2 10% trajectory segment for adaptation, remaining 90% for test
```

需要报告：

- zero-shot；
- few-shot adaptation；
- full retrain upper bound。

这样可以区分“真实泛化能力”和“重新训练后的上限”。

### 3.3 户外数据集 D5 划分

D5 建议用于 domain shift 分析。

```text
Train: D1 indoor
Test: D5 outdoor static/dynamic
Optional adaptation: D5 LOS static subset only
Final test: D5 NLOS + dynamic subset
```

重点看模型是否把室内多径规律错误迁移到户外场景。

## 4. 模型与方法设置

### 4.1 单链路输入

对每条 tag-anchor 测距构造输入：

```text
x_i = [range_i, signal_i, temporal_i, geometry_i]
```

其中：

- `range_i`：原始测距；
- `signal_i`：CIR/RSS/SNR/power difference 等；
- `temporal_i`：前后测距差分、滑动统计；
- `geometry_i`：anchor-tag 相对距离、anchor 方位、anchor 可见数、GDOP、condition number。

### 4.2 模型输出

每条测距输出：

```text
b_i      : ranging bias correction
sigma_i  : measurement uncertainty
q_i      : low-trust / outlier probability
```

修正测距：

```text
d_i = r_i - b_i
```

定位权重：

```text
w_i = (1 - q_i) / (sigma_i^2 + eps)
```

### 4.3 定位后端

主后端：

```text
Robust uncertainty-weighted WLS
```

优化目标：

```text
p* = argmin_p sum_i w_i * rho(d_i - ||p - a_i||)
```

其中 `rho` 使用 Huber 或 Cauchy robust kernel。

扩展后端：

- LS；
- WLS；
- robust WLS；
- EKF；
- adaptive-noise EKF；
- factor graph，可作为后续扩展。

第一篇中不必把所有后端都做得很复杂。核心是证明 learned trustworthiness and uncertainty 可以稳定改善几何定位。

## 5. 基线方法

必须包含以下基线：

| 编号 | Baseline | 作用 |
|---|---|---|
| B1 | Raw LS | 最基础下限 |
| B2 | Raw WLS | 常规加权定位 |
| B3 | Robust WLS | 验证鲁棒核作用 |
| B4 | CIR-based ranging correction only | 验证误差补偿 |
| B5 | Trustworthiness-weighted WLS | 验证可信度 |
| B6 | Uncertainty-weighted WLS | 验证不确定性 |
| B7 | Proposed full model | 完整方法 |

如果时间允许，可补充：

- EKF fixed-R；
- EKF adaptive-R；
- existing NLOS classifier + WLS；
- direct coordinate regression neural network。

直接坐标回归只作为反例基线，不作为主路线。

## 6. 消融实验

建议至少做以下消融：

| 消融项 | 去掉内容 | 目的 |
|---|---|---|
| A1 | no CIR/signal features | 验证信号层贡献 |
| A2 | no bias correction | 验证测距补偿贡献 |
| A3 | no uncertainty sigma | 验证不确定性贡献 |
| A4 | no trustworthiness q | 验证可信度贡献 |
| A5 | no robust kernel | 验证鲁棒几何后端贡献 |
| A6 | no geometry features | 验证几何退化感知 |
| A7 | same-environment only training | 验证跨环境训练必要性 |

消融结论要服务于论文叙事：

> 信号特征解释误差，误差补偿改善 range，可信度和不确定性改善定位权重，鲁棒核减少长尾失败，跨环境训练提升泛化。

## 7. 退化实验设计

### 7.1 NLOS/困难样本分组

如果数据集提供 LOS/NLOS 标签，直接分组。

如果没有显式标签，可用以下代理指标构造困难样本：

```text
abs(ranging_error) > P75
abs(ranging_error) > P90
RSS below threshold
CIR peak spread above threshold
residual after raw WLS above threshold
```

报告：

- LOS mean/median/P90 error；
- NLOS mean/median/P90 error；
- hard samples failure rate；
- severe error recall。

### 7.2 Anchor dropout

对每个位置随机或按策略移除 anchor：

```text
8 anchors -> 7 anchors
8 anchors -> 6 anchors
8 anchors -> 5 anchors
8 anchors -> 4 anchors
```

两种 dropout：

1. random dropout；
2. worst-geometry dropout，即保留几何条件更差的 anchor 组合。

报告：

- error vs anchor number；
- failure rate vs anchor number；
- uncertainty calibration vs anchor number。

### 7.3 几何退化

对每个样本计算几何指标：

```text
GDOP
condition number of geometry matrix
minimum anchor angle spread
visible anchor count
anchor convex hull coverage
```

分组：

```text
Good geometry: bottom 33%
Medium geometry: middle 33%
Poor geometry: top 33%
```

报告不同几何条件下的定位误差和失败率。

### 7.4 跨环境退化

使用 leave-one-environment-out：

```text
train environments != test environment
```

报告每个 held-out environment 的性能，并分析哪个环境最难。

### 7.5 跨数据集退化

至少做一个外部数据集：

```text
Train on D1
Test on D2
```

如果 D5 处理顺利，再补充：

```text
Train on D1
Test on D5
```

跨数据集实验不要求绝对误差最低，重点是证明：

- 方法退化可控；
- 不确定性会随 domain shift 增大；
- risk-aware selection 可以减少高风险定位失败。

## 8. 指标体系

### 8.1 定位误差

必须报告：

```text
Mean error
Median error
RMSE
P75
P90
P95
Max error
```

平均值不能单独使用，因为 UWB 在 NLOS 下有明显长尾。

### 8.2 失败率

建议报告：

```text
Failure@0.5m
Failure@1.0m
Failure@2.0m
```

如果场景尺度较小，可以使用：

```text
Failure@0.3m
Failure@0.5m
Failure@1.0m
```

### 8.3 测距误差

对 corrected range 报告：

```text
range MAE
range RMSE
range bias mean
range error std
range P90
```

必须画出：

- correction before/after histogram；
- error CDF；
- per-anchor error distribution。

### 8.4 不确定性校准

如果模型输出 `sigma_i`，必须评价它是否可信。

建议指标：

```text
NLL
ECE
coverage probability
interval width
risk-coverage curve
```

其中 risk-coverage 最契合“可信 UWB 定位”：

> 当模型拒绝或降权低可信测距后，剩余样本的定位风险是否下降。

### 8.5 异常检测

如果输出 `q_i`，可以评价其识别困难观测的能力：

```text
AUROC
AUPRC
F1
Recall@high-error
Precision@high-error
```

困难观测标签可定义为：

```text
abs(ranging_error) > P90
```

或：

```text
localization error contribution causes failure
```

### 8.6 计算效率

报告：

```text
inference time per range
inference time per position
model parameters
CPU/GPU setting
```

UWB 定位具有实时属性，不能完全忽略效率。

## 9. 图表计划

建议论文至少包含以下图表：

| 图表 | 内容 |
|---|---|
| Fig. 1 | 方法总体框架 |
| Fig. 2 | 单链路误差补偿与可信度建模 |
| Fig. 3 | 数据集和实验划分 |
| Fig. 4 | ranging error before/after correction |
| Fig. 5 | 同环境与跨环境定位误差 CDF |
| Fig. 6 | anchor dropout 下的误差曲线 |
| Fig. 7 | 几何退化分组误差 |
| Fig. 8 | 不确定性校准或 risk-coverage curve |
| Table 1 | 数据集字段对照 |
| Table 2 | baselines 对比 |
| Table 3 | ablation study |
| Table 4 | cross-dataset generalization |

## 10. 论文主结论应如何组织

实验结论建议按以下顺序写：

1. 测距误差确实与 CIR/RSS/几何特征相关；
2. 误差补偿显著改善单链路 range error；
3. 单纯补偿还不够，可信度和不确定性可以进一步降低定位长尾；
4. proposed full model 在同环境下优于传统 LS/WLS/robust WLS；
5. 在 leave-one-environment-out 下仍保持较稳定性能；
6. anchor dropout 和几何退化下，方法的 failure rate 更低；
7. 外部数据集上，误差会增加，但模型不确定性也同步升高，说明具备风险感知能力。

## 11. 第一阶段最小实验闭环

为了避免工程范围失控，第一阶段只做以下闭环：

1. D1 数据解析；
2. LS/WLS/robust WLS baseline；
3. CIR/RSS/range/geometry 特征提取；
4. 单链路 ranging bias prediction；
5. uncertainty/trustworthiness 输出；
6. robust uncertainty-weighted WLS；
7. 同环境测试；
8. leave-one-environment-out；
9. anchor dropout；
10. ablation。

达到这个闭环后，就已经有第一篇 SCI 的主体实验。

## 12. 第二阶段扩展

第一阶段完成后，再做：

1. D2 Industrial AMR 外部泛化；
2. D5 Outdoor UWB 户外泛化；
3. dynamic EKF/adaptive-R；
4. MILUV 的 UAV/CIR 扩展；
5. UTIL 的 TDoA 扩展。

## 13. 当前执行清单

下一步应进入工程准备：

1. 下载 D1 数据集；
2. 建立 `SCI-UWB/data/raw/D1_indoor_uwb/`；
3. 建立统一 loader；
4. 输出首份数据审计报告；
5. 复现 raw LS/WLS；
6. 检查 CIR 字段维度和可用性；
7. 生成第一版特征表；
8. 开始 baseline notebook。

## 14. 最关键的边界

第一篇论文不要过早变成 UWB+IMU+Vision 大融合。

更清晰的第一篇边界是：

> 只研究 UWB 测距本身的可信度、误差补偿和几何定位鲁棒性。

IMU、vision、odometry 可作为动态后端或外部数据集辅助验证，但不作为核心贡献。这样论文会更聚焦，也更容易讲清楚创新。

