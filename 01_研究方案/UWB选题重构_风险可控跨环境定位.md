# UWB 选题重构：从“可信度预测”到“风险可控定位”

日期：2026-08-01  
目标期刊优先级：Measurement / IEEE Transactions on Instrumentation and Measurement（TIM）；具备边缘部署或系统验证后再考虑 IEEE Internet of Things Journal（IoTJ）。

## 1. 本轮结论

不再将“预测单链路测距偏差、不确定性和可信度，再做鲁棒 WLS”作为论文主创新。

原因是该主张已被相邻工作明显占据：

- Yang 等在 *Measurement*（2025）已经提出 UWB 测距可信度评估、细粒度误差分类和误差缓解；
- Sun 与 Gao 在 *Measurement Science and Technology*（2026）已经提出同时进行 NLOS 识别、偏差校正和动态不确定性估计的物理约束多任务框架；
- 2026 年的预印本 Phy-CDAN 已将无标签跨环境适配、异方差不确定性和 CIR 结合，虽然其正式出版状态仍须追踪。

因此，直接复刻“可信度 + 不确定性 + 误差补偿”会面临明显的新颖性风险。

## 2. 研究脉络

```text
LOS/NLOS 二分类
    → 按类别或信号特征补偿测距偏差
    → 鲁棒优化 / 自适应滤波降低异常测量影响
    → 单链路可信度与不确定性预测
    → 跨环境域适配
    → 【候选突破】经校准的风险控制 + 几何感知测量选择 + 安全定位输出
```

前四步主要回答“测距是否可能错误、错多少”。候选突破回答更接近工程部署的问题：

> 在未见环境中，系统能否知道何时不应相信某些测距，并在给定风险预算下选择兼顾测量质量和定位几何的锚点子集？

## 3. 推荐主选题

### 中文题目

**面向未知环境退化的几何条件保序 UWB 测距区间与风险可控定位方法**

### 英文工作题目

**Geometry-Conditioned Conformal Measurement Sets for Risk-Controlled UWB Localization under Unseen Environments**

### 精确问题定义

给定每条 UWB 链路可在线取得的测距、RSS/首径/噪声/CIR 派生特征、信道和锚点坐标，系统不只给出校正距离或高低可信标签，而是：

1. 为每条测距产生经校准的误差区间；
2. 利用留一链路几何一致性估计在线风险，而不使用标签真值坐标；
3. 在“区间宽度（测量风险）—几何精度衰减（GDOP）”之间选择锚点子集；
4. 输出位置、位置风险指标以及“接受/拒绝/降级”决策。

核心不是把不确定性显示出来，而是使其能够约束定位决策。

## 4. 方法骨架

```text
Online UWB link features + leave-one-link-out geometric consistency
                ↓
Quantile residual predictor
                ↓
Geometry-/feature-stratified split conformal calibration
                ↓
Per-link measurement interval and risk score
                ↓
Risk–geometry anchor subset selection
                ↓
Robust weighted localization + position protection region + abstention
```

其中第 i 条链路的部署期几何残差必须由不包含该链路的临时解构造：

\[
g_i=r_i-\|\hat p^{(-i)}-a_i\|.
\]

这样 `g_i` 可在线计算，不能以真实标签坐标、真实距离或实际测距误差作为输入。模型输出残差区间 `[-q_i^-, q_i^+]`；区间宽度和留一一致性共同组成链路风险。锚点选择要同时优化预期测量风险与局部几何条件数，而不是仅删除最低可信的锚点。

## 5. 可写入论文的贡献边界

1. 提出面向 UWB 单链路残差的几何条件区间建模，将信号质量与多锚点留一几何一致性共同用于风险估计。
2. 提出风险—几何联合的锚点选择与鲁棒定位后端，使“稳定但严重偏置”的链路不会仅因方差较小而被高权重使用。
3. 建立 D1 上严格的 leave-one-environment-out、锚点丢失、几何退化和选择性定位协议，同时报告准确度、覆盖率、区间效率、风险—覆盖曲线和定位失败率。

不能写成“在任意未知环境中提供无条件数学覆盖保证”。常规 conformal prediction 的有限样本覆盖前提是可交换性；跨环境分布漂移会破坏这一前提。论文应报告：

- 同分布/混合环境下的目标覆盖率；
- leave-one-environment-out 下的**经验覆盖率与风险控制能力**；
- 如使用无标签目标环境样本进行加权校准，明确其依赖协变量漂移等假设，而非夸大为无条件保证。

## 6. 候选方向对比

| 方向 | 新颖性风险 | D1 适配性 | 目标期刊匹配 | 决策 |
|---|---:|---:|---:|---|
| 偏差 + 不确定性 + 可信度 + WLS | 高：已有 2025/2026 直接重叠 | 高 | Measurement/TIM | 放弃作为主线 |
| 无标签跨环境域适配 | 中高：已有 2026 Phy-CDAN 预印本 | 中 | TIM/IoTJ | 作为后续增强，不作首篇主线 |
| 轨迹预测/延迟补偿 | 中高：已有近期工作，且 D1 时序真实性有限 | 低 | IoTJ/机器人 | 暂不选 |
| 未知锚点自标定 | 中 | 低：D1 不直接支持真实自标定 | TIM/IoTJ | 需新数据/仿真后再选 |
| **区间校准 + 风险—几何锚点选择 + 选择性定位** | **中低：需全文核验，但与已有工作区分清楚** | **高：8 锚点和四环境可做退化实验** | **Measurement/TIM** | **推荐** |

## 7. D1 实验协议

### 主协议

- 四个环境 leave-one-environment-out：每次 3 个训练/校准，1 个完全留出测试；
- 每个环境逐信道、逐锚点报告测距和定位结果；
- 不使用 `NLOS`、真值位置、真值距离、真实误差作为部署特征；
- 同时做零样本跨环境与“有短暂无标签目标窗口”的加权校准两个设定。

### 退化协议

- 锚点数：8、6、5、4；
- 采样三类锚点缺失：随机、最高风险链路缺失、几何最差子集；
- 用环境 2 和环境 3 的长尾异常点作为困难案例；
- 将“总是定位”与“风险阈值下选择性定位”对比。

### 指标

| 层级 | 指标 |
|---|---|
| 测距 | MAE、RMSE、P90、区间覆盖率、平均区间宽度、校准误差 |
| 链路筛选 | risk–coverage、AURC、异常链路召回率、误拒绝率 |
| 定位 | mean/median/RMSE、P90/P95、Fail@0.5m、Fail@1m、接受后风险 |
| 工程 | 推理时间、选锚开销、锚点数量与性能曲线 |

### 必要基线

- LS、WLS、Huber/鲁棒 WLS；
- 仅信号特征的误差/不确定性模型；
- 仅几何残差的筛选模型；
- 不做 conformal 校准的分位数或 evidential 模型；
- 仅按预测风险删链路；
- 仅按 GDOP 选锚；
- 完整的风险—几何联合方法。

## 8. 取刊建议

首选 **Measurement**：论文核心是测距质量、区间校准和误差传播，定位后端承担验证作用。

备选 **IEEE TIM**：需要补强量测模型、校准定义、可追溯的不确定性报告和重复性。

暂不以 **IEEE IoT Journal** 为首投：仅有 D1 离线数据时，IoTJ 的系统、边缘部署或网络级证据不够强；除非后续新增轻量部署、在线选择策略和外部数据验证。

## 9. 第一阶段的可否定实验

先不实现完整大模型，完成以下四件事：

1. 从 D1 聚合表构建不泄漏的链路特征及 leave-one-link-out 几何残差；
2. 建立分位数回归 + split conformal 基线；
3. 在四次 leave-one-environment-out 中比较未校准与校准的覆盖率/宽度；
4. 验证风险—几何选锚是否降低 P95 和 Fail@1m，而不是只改善平均误差。

若第 4 项不能在困难环境中稳定改善尾部风险，则不应继续堆叠网络；应转向“外部数据集上的环境漂移检测与安全降级”课题。

## 10. 证据来源与待核验项

- Yang et al., *Measurement*, 2025, 256, 117721, DOI: 10.1016/j.measurement.2025.117721。
- Sun and Gao, *Measurement Science and Technology*, 2026, DOI: 10.1088/1361-6501/ae7c20。
- Cohen et al., *Calibrating AI Models for Wireless Communications via Conformal Prediction*, 2022：说明校准与准确度不同，且常规覆盖结论依赖相应数据假设。
- Phy-CDAN（2026 SSRN 预印本）：追踪其是否正式发表；在结论中只可称其为公开预印本，不能作为已同行评议工作表述。

在正式写 Related Work 前，应下载并精读 Yang 2025 与 Sun 2026 的全文，逐项确认它们是否已经进行：conformal calibration、风险—覆盖评估、几何条件锚点选择和定位拒绝机制。
