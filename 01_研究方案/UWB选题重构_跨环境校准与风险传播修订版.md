# UWB 选题修订版：跨环境校准与不确定性传播

## 一、为什么修订

近期检索显示，以下方案已存在直接竞争者：

- 测距可信度/credibility：IEEE TIM 2025、Measurement 2025；
- 概率不确定性和动态校准：Measurement Science and Technology 2026；
- CIR 域适应：IEEE IoTJ 2025 AdaLOS；
- 跨环境迁移学习：IEEE IoTJ 2024；
- 残差或链路质量锚点选择：IEEE TIM 2025、IEEE Sensors Letters 2024；
- CIR 直接位置修正：IEEE TWC 2026 Transformer。

因此，不能再把“预测可信度 + 加权 WLS + 锚点选择”写成主要创新。

## 二、推荐研究问题

**在训练环境之外，UWB 测距误差区间是否仍然校准？若不校准，如何检测失效并将测量风险传播到定位结果，进而执行接受、降级或拒绝决策？**

推荐英文题目：

> *Cross-Environment Calibration and Uncertainty Propagation for Risk-Controlled UWB Localization*

研究对象从“预测一个质量分数”改为三个可审计对象：

1. 每条链路的测距误差预测区间；
2. 测距区间在锚点几何下形成的位置保护区域；
3. 面向尾部误差的 risk–coverage 和 selective localization 决策。

## 三、方法骨架

```text
Range/RSS/FP/noise/CIR-statistics + short history + anchor geometry
        -> quantile or evidential interval predictor
        -> split/weighted calibration under leave-one-environment-out
        -> per-link coverage/risk and calibration-failure diagnosis
        -> uncertainty-to-position propagation
        -> robust WLS + accept/degrade/reject decision
```

如果保留原始 CIR，可增加 CIR encoder；如果进行 CIR 消融，主结论必须仍由 D1 的 Range/RSS/FP/噪声/几何特征成立。部署时不得使用真值坐标、真值距离、真实测距误差或 NLOS 标签。

## 四、D1 实验协议

1. 四个环境轮流 leave-one-environment-out；随机点划分仅作容易场景对照。
2. 训练/校准/测试严格按环境和轨迹分离，避免同一测量点或相邻时刻泄漏。
3. 基线：LS、WLS、Huber/robust WLS、AUKF、AR-PNN、未校准 quantile/evidential、pooled conformal、RAS、启发式链路质量选择。
4. 链路缺失：随机缺失、最高风险链路缺失、几何最差子集缺失；锚点数至少 8/6/5/4。
5. 指标：MAE、RMSE、P90/P95、Fail@0.5 m、Fail@1 m、区间 coverage、区间宽度、校准误差、risk–coverage、拒绝率、推理时间。

## 五、统计表述边界

不要声称对任意未知环境提供无条件 conformal guarantee。交换性在环境分布漂移下可能失效。论文中应报告：

- 混合环境的 coverage；
- 每个留出环境的经验 coverage、区间宽度和最差环境结果；
- 校准失效检测的召回/误报；
- 若使用加权或无标签目标域校准，明确协变量漂移和权重估计假设。

## 六、投稿定位

- **Measurement：** 主张测量区间、校准误差、尾部风险和误差传播。
- **IEEE TIM：** 增强量测模型、可追溯不确定度预算、严格环境留出和基线公平性。
- **IEEE IoTJ：** 只有在加入在线推理、资源开销、链路调度或边缘部署实验后再考虑。

## 七、阶段门

第一阶段只实现区间预测和 coverage 审计。如果 leave-one-environment-out 下校准明显失效，论文主线转为“跨环境校准失效诊断与安全降级”；只有在风险传播稳定降低 P95/Fail@1 m 后，才继续实现完整联合决策模型。
