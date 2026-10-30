# 公开 UWB 数据集筛选与字段对照

更新时间：2026-07-19

## 1. 筛选目标

本阶段的目标不是罗列所有 UWB 数据集，而是筛选能够支撑第一篇 SCI 的公开数据集。

围绕前期确定的研究主线，数据集必须尽量支持以下问题：

1. 是否能学习单链路测距误差；
2. 是否有 CIR、RSS、SNR、power difference 等信号层特征；
3. 是否有可靠 ground truth；
4. 是否包含 LOS/NLOS、多路径、遮挡或困难样本；
5. 是否支持跨环境、跨布局、跨动态状态测试；
6. 是否可以构造 anchor dropout 和几何退化实验；
7. 是否便于复现实验。

第一篇 SCI 的主线应围绕：

> 信号感知测距误差补偿 + 测距可信度/不确定性建模 + 鲁棒几何定位 + 跨场景验证。

因此，优先级最高的数据集应具有 `range + CIR/RSS + anchor position + tag ground truth + multi-environment`。

## 2. 候选数据集总表

| 编号 | 数据集 | 年份 | 场景 | UWB 类型 | 信号特征 | Ground truth | 推荐角色 |
|---|---:|---:|---|---|---|---|---|
| D1 | Indoor UWB Positioning and Position Tracking Data Set | 2023 | 多室内环境 | ADS-TWR range | CIR, RSS, DW1000寄存器特征 | 预定义路径坐标 | 主数据集 |
| D2 | UWB Localization for an Industrial AMR | 2026 | 工业仓库 + 室内外过渡 | TWR range | 主要为 range | LiDAR参考位姿 | 外部泛化验证 |
| D3 | MILUV | 2026 | 多 UAV 室内飞行 | range + raw timestamp | CIR, received power, clock skew | Vicon motion capture | 高阶扩展 |
| D4 | UTIL | 2024 | UAV 室内 TDoA | UWB TDoA | SNR, power difference | Vicon motion capture | TDoA扩展 |
| D5 | Comprehensive Outdoor UWB Dataset | 2025 | 户外 LOS/NLOS | CC-SS-TWR range | RSS, CIR | 静态激光测距 + 动态RTK-GNSS | 户外泛化/信号验证 |
| D6 | MURP Dataset | 2024 | 多机器人相对定位 | 多标签 range | 主要为 range | 实验参考位姿 | 相对定位扩展 |
| D7 | RCV UWB Dataset | 2024 | 相对位姿影响分析 | range | 姿态因素相关信息 | 精确 UWB pose | 测距误差机理分析 |
| D8 | Aalborg Industrial UWB/Ultrasound Dataset | 2022 | 工业实验室 | UWB TDoA/position | 主要为系统输出 | ground truth点/轨迹 | 低优先级对照 |

## 3. 第一优先级：D1 Indoor UWB Positioning and Position Tracking Data Set

来源：

- 论文页面：[Scientific Data, 2023](https://www.nature.com/articles/s41597-023-02639-5)
- 数据仓库：[Zenodo](https://doi.org/10.5281/zenodo.8280736)

### 3.1 数据集特点

该数据集来自 Scientific Data 2023，包含四个不同室内环境、八个固定 anchor 和一个移动 tag。每个路径位置采集多次 UWB range 和 CIR 数据，并提供 anchor 坐标、tag 路径坐标、floorplan、raw csv、预处理 json 和技术验证脚本。

论文明确指出，数据集中每个 tag-anchor pair 都采集了 measured range 和 CIR，且每个链路可能处于 LOS 或 NLOS 状态。这一点非常契合我们要做的单链路测距质量建模。

### 3.2 可用字段

可预期字段：

- environment id；
- anchor id；
- anchor position；
- tag position；
- measured range；
- true range；
- ranging error；
- CIR；
- RSS；
- rxpacc 等 DW1000 相关寄存器特征；
- raw csv；
- harmonized json；
- floorplan。

### 3.3 适合本研究的原因

它最适合作为第一篇论文的主数据集，原因是：

1. 包含多环境，能做 leave-one-environment-out；
2. 包含 CIR，能支撑信号特征创新；
3. 包含 measured range 和 true range，能构造测距误差补偿标签；
4. 包含 anchor 和 tag 坐标，能做几何定位；
5. 有技术验证和代码仓库，复现成本相对可控；
6. 它本身就面向 NLOS-induced ranging error mitigation。

### 3.4 局限

主要局限：

- 路径是预定义等距位置，不是真正连续运动；
- 动态效果是通过路径点近似，不包含真实速度和惯性扰动；
- 数据来自 DW1000，跨硬件泛化仍需谨慎；
- environment2 中 A6 anchor 数据被移除，锚点数量不完全一致。

### 3.5 本研究用法

推荐用作：

- 单链路误差补偿模型训练；
- CIR 特征贡献分析；
- 不确定性建模；
- 同环境和跨环境定位；
- anchor dropout；
- 几何退化分组；
- LOS/NLOS 或困难链路分析。

## 4. 第二优先级：D2 UWB Localization for an Industrial AMR

来源：

- 数据页面：[University of Klagenfurt](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/datasets/uwb-localization-for-an-industrial-amr/)

### 4.1 数据集特点

该数据集记录于 AGILOX ONE Autonomous Mobile Robot 在工业仓库和相邻户外区域中的运动，包含 13 条轨迹、车载 LiDAR-based reference poses、wheel odometry、body velocity estimates 和 UWB range measurements。三个位于车辆上的 UWB tags 与十二个静态 UWB anchors 进行测距。

数据覆盖工业仓库运行、室内外过渡、局部 LOS 阻挡和稀疏 anchor 配置，并提供 robot 参数、tag 安装位置、anchor ground-truth positions 和 anchor calibration results。

### 4.2 可用字段

可预期字段：

- timestamp；
- trajectory id；
- vehicle reference pose；
- wheel odometry；
- body velocity；
- tag id；
- anchor id；
- measured range；
- anchor ground-truth position；
- anchor calibration result；
- tag extrinsic。

### 4.3 适合本研究的原因

它非常适合作为外部泛化验证集：

1. 场景从室内静态路径扩展到真实工业 AMR；
2. 有动态轨迹和 LiDAR 参考位姿；
3. 存在稀疏 anchor 和遮挡条件；
4. 可检验模型在非同源数据上的退化程度。

### 4.4 局限

主要局限：

- 目前公开描述中未强调 CIR，可能主要是 range 层数据；
- License 为 CC BY-NC-SA 4.0，科研可用但商业不可用；
- 2026 CASE 论文相关，引用时要确认最终出版信息；
- 与 D1 的硬件和字段差异大，直接训练-测试可能需要统一特征子集。

### 4.5 本研究用法

推荐用作：

- cross-dataset zero-shot；
- few-shot adaptation；
- dynamic AMR test；
- sparse-anchor test；
- industrial-domain robustness test。

不建议第一阶段用它作为主训练集，因为缺少 CIR 会削弱信号特征创新。

## 5. 第三优先级：D3 MILUV

来源：

- 论文页面：[IJRR Online First, 2026](https://journals.sagepub.com/doi/10.1177/02783649251405898)
- 项目页面：[MILUV Dataset](https://decargroup.github.io/miluv/)
- 数据 DOI：[Figshare](https://doi.org/10.25452/figshare.plus.28386041.v1)
- Devkit：[GitHub](https://github.com/decargroup/miluv)

### 5.1 数据集特点

MILUV 是 2026 年 IJRR 的多 UAV 室内定位数据集，包含 36 个实验、217 分钟飞行、三架 quadcopters、最多 12 个 UWB transceivers。它提供 UWB ranging、raw timestamps、CIR、received signal power、clock skew、passive listening、vision、IMU、laser rangefinder、magnetometer 和 Vicon ground truth。

### 5.2 适合本研究的原因

MILUV 的学术价值很高，尤其适合支撑高阶实验：

1. 数据质量高，期刊层次高；
2. 有 CIR 和 received power；
3. 有不同 anchor constellation；
4. 有 obstacles 和 LOS/NLOS；
5. 有 Vicon ground truth；
6. 支持多机器人、多标签和动态场景。

### 5.3 局限

主要局限：

- 数据量约 173 GB，第一阶段处理成本高；
- 多模态结构复杂，容易把第一篇论文拖成“大系统融合”；
- 多 UAV 和视觉数据会引入额外变量；
- 若只做 UWB range+CIR，需明确裁剪范围。

### 5.4 本研究用法

推荐作为：

- 第二阶段扩展；
- reviewer 要求更强动态验证时的补充；
- multi-agent 或 UAV 场景扩展；
- CIR trustworthiness 的高水平外部验证。

不建议一开始作为主数据集。它很强，但太重。第一篇需要稳，不需要一上来把自己卷进 173 GB 的工程泥潭。

## 6. 第四优先级：D4 UTIL

来源：

- 项目页面：[UTIL UWB Dataset](https://utiasdsl.github.io/util-uwb-dataset/)
- 论文页面：[IJRR, 2024](https://journals.sagepub.com/doi/10.1177/02783649241230640)

### 6.1 数据集特点

UTIL 是 IJRR 2024 的 UWB TDoA 室内定位数据集，基于 Decawave DWM1000。它包含 identification experiments 和 flight experiments。Identification 部分提供 LOS/NLOS 下的 SNR、power difference 和 raw UWB TDoA；flight 部分包含约 150 分钟真实飞行，四种 anchor constellations，并提供 UWB TDoA、IMU、optical flow、ToF laser、barometer 和 Vicon ground truth。

### 6.2 适合本研究的原因

UTIL 非常适合研究：

- TDoA 测量误差建模；
- anchor constellation 对定位性能的影响；
- 动态 NLOS；
- UWB-inertial fusion；
- cluttered indoor environment。

### 6.3 局限

主要局限：

- TDoA 与 D1/D2 的 TWR range 模型不同；
- 不能直接并入同一个 range-error compensation 框架，除非扩展为通用 measurement residual；
- signal features 是 SNR 和 power difference，不是完整 CIR；
- 更适合作为第二条技术路线，而不是第一篇主线。

### 6.4 本研究用法

推荐作为：

- TDoA 扩展实验；
- anchor constellation 泛化；
- dynamic UAV benchmark；
- 审稿时证明方法可迁移到不同 UWB measurement type。

但第一阶段不优先。

## 7. 第五优先级：D5 Comprehensive Outdoor UWB Dataset

来源：

- 论文页面：[Scientific Data, 2025](https://www.nature.com/articles/s41597-025-05887-9)
- 数据仓库：[Figshare](https://doi.org/10.6084/m9.figshare.29605775)
- 代码仓库：[GitHub](https://github.com/cucudasluv/UWB-dataset)

### 7.1 数据集特点

这是 2025 年 Scientific Data 的户外 UWB 数据集，包含 static 和 dynamic 两类实验。静态部分覆盖不同 antenna height、LOS/NLOS、距离和遮挡；动态部分使用移动机器人，包含 UWB、RSS、IMU、GNSS，并以 RTK-GNSS 作为参考。

论文说明该数据集提供 UWB range、RSS、CIR，距离可达 60 m，并有不同 anchor configurations 和 PDOP 条件。

### 7.2 适合本研究的原因

它适合补强两个方向：

1. 信号特征和测距误差之间的关系；
2. 从室内到户外的 domain shift。

### 7.3 局限

主要局限：

- 场景是户外，不完全对应室内 UWB 定位；
- 动态部分 anchors 数量较少；
- 单 tag 设置限制泛化；
- 与 D1 的室内多径机理不同，直接合并训练可能导致域偏移过大。

### 7.4 本研究用法

推荐作为：

- 户外 zero-shot test；
- RSS/CIR 特征迁移实验；
- antenna height 与 ground reflection 分析；
- long-range robustness 补充。

## 8. 其他候选数据集

### 8.1 D6 MURP

来源：

- 论文 DOI：`10.1109/LRA.2024.3451393`
- 数据/视频：[GitHub](https://github.com/mit-acl/murp-datasets)

MURP 面向多智能体 UWB 相对 3D pose estimation。它的价值是多标签、多机器人、约束通信和 relative localization。对我们第一篇绝对定位主线不是最直接，但可以作为第二篇“基础设施弱依赖/多智能体相对定位”的延展方向。

### 8.2 D7 RCV UWB Dataset

来源：

- 数据仓库：[GitHub](https://github.com/cjhhalla/RCV_uwb_dataset)

该数据集关注 relative pose factors 对 UWB ranging 的影响，适合分析测距误差与姿态、相对位置之间的机理关系。它不适合作为主定位数据集，但适合作为测距误差解释性实验的补充。

### 8.3 D8 Aalborg Industrial UWB/Ultrasound Dataset

来源：

- 论文页面：[Sensors, 2022](https://www.mdpi.com/1424-8220/22/8/2927)
- 数据 DOI：[Figshare](https://doi.org/10.6084/m9.figshare.17429654)

该数据集面向工业环境下 UWB 与 ultrasonic localization 对比，包含静态点和动态移动机器人测试。它有工业场景价值，但更偏系统定位输出对比，信号层和测距误差建模支撑较弱，作为低优先级对照即可。

## 9. 推荐数据集组合

### 9.1 第一篇 SCI 的推荐组合

最稳组合：

| 角色 | 数据集 | 用途 |
|---|---|---|
| 主训练/主验证 | D1 Indoor UWB Positioning and Tracking | range+CIR 单链路误差补偿、可信度、不确定性、跨室内环境 |
| 外部验证 | D2 Industrial AMR UWB | 工业动态场景、稀疏锚点、室内外过渡 |
| 补充验证 | D5 Outdoor UWB 2025 | 户外 LOS/NLOS、RSS/CIR、long-range domain shift |
| 高阶扩展 | D3 MILUV | 动态 UAV、多标签、CIR、多模态 |

### 9.2 不推荐一开始纳入的组合

不建议第一阶段把 D4 UTIL 与 D1/D2 直接合并，因为 TDoA 与 TWR range 的测量方程不同。若纳入，论文主线要从 range correction 扩展为 generalized measurement correction，复杂度会显著上升。

不建议第一阶段直接以 D3 MILUV 为主数据集，因为数据量和多模态复杂度过高，容易让贡献从“可信 UWB 定位”变成“多传感器工程集成”。

## 10. 统一字段设计

后续代码应统一为以下最小 schema：

```text
sample_id
dataset_id
environment_id
trajectory_id
timestamp
tag_id
anchor_id
range_measured
range_true
ranging_error
anchor_x
anchor_y
anchor_z
tag_x
tag_y
tag_z
los_label
nlos_label
signal_feature_vector
geometry_feature_vector
temporal_feature_vector
```

对于没有 `range_true` 的动态数据集，应通过 tag ground truth 与 anchor position 计算：

```text
range_true = || tag_position - anchor_position ||
ranging_error = range_measured - range_true
```

对于 TDoA 数据，应另建扩展 schema：

```text
tdoa_measured
tdoa_true
tdoa_error
anchor_i_position
anchor_j_position
tag_position
```

第一阶段先实现 TWR/range schema，TDoA schema 暂缓。

## 11. 公开数据集使用顺序

建议执行顺序：

1. D1：下载、解析、复现 WLS/CIR baseline；
2. D1：建立单链路误差补偿标签；
3. D1：完成同环境和跨环境实验；
4. D1：完成 anchor dropout 和几何退化实验；
5. D2：只使用 range + geometry + temporal 特征做外部验证；
6. D5：使用 RSS/CIR 做户外迁移验证；
7. D3：在主线稳定后作为高阶扩展；
8. D4：如需要写第二篇或扩展到 TDoA，再处理。

## 12. 当前结论

第一篇 SCI 的数据集策略应定为：

> 以 D1 为主数据集建立信号感知测距误差补偿与可信度建模方法，以 D2 和 D5 作为外部泛化验证，D3/MILUV 作为高阶扩展储备，D4/UTIL 暂不进入第一阶段主实验。

这个策略能兼顾 SCI 深度和工程可完成性。D1 保证信号层创新，D2 保证工业动态泛化，D5 保证户外长距离和 LOS/NLOS 泛化，D3 作为高水平机器人数据集储备。

## 13. 本地 D1 数据审计补充

本地 `SCI-UWB/数据集论文/data_set` 已完成首次结构审计。四个环境的 raw csv 覆盖率均为 100%：

| 环境 | walking points | raw csv files | coverage | CIR sample length |
|---|---:|---:|---:|---:|
| environment0 | 85 | 4080 | 100% | 1016 |
| environment1 | 80 | 3840 | 100% | 1016 |
| environment2 | 84 | 4032 | 100% | 1016 |
| environment3 | 81 | 3888 | 100% | 1016 |

已确认原始 CSV 字段可直接构造：

```text
range_true = ||tag_position - anchor_position||
ranging_error = range_measured - range_true
```

因此 D1 可以作为第一阶段主数据集。更详细的审计结果见：

```text
SCI-UWB/02_公开数据集/D1数据集首次审计报告.md
SCI-UWB/02_公开数据集/D1样本特征抽取与初步误差剖析.md
```
