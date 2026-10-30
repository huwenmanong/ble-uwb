# D1 原始 LS/WLS 定位基线结果

更新时间：2026-07-19

## 1. 实验设置

本次 baseline 从 D1 `raw_data` 直接读取每个 `environment + position + channel + anchor` 的原始测距，聚合为 mean range 和 range std，然后在每个 `environment + position + channel` 下使用可见 anchors 进行 2D 定位。tag 高度使用数据集中给定的 `Z_TAG`，估计目标为 `(x, y)`。

当前实现包含四个基线：

- `LS`：原始 mean range 非加权最小二乘；
- `WLS_range_std`：用同一文件内 range 均值的方差估计权重；
- `Robust_LS_Huber`：非加权 Huber 鲁棒最小二乘；
- `Robust_WLS_Huber`：range std 加权 + Huber 鲁棒核。
- `Oracle_Range_LS`：使用几何真实距离替代原始 range 的理论上限。

输出文件：

```text
SCI-UWB/data/interim/d1_anchor_range_aggregates.csv
SCI-UWB/data/results/d1_positioning_baseline_results.csv
SCI-UWB/data/results/d1_positioning_baseline_summary.csv
```

## 2. 数据规模

- anchor-level 聚合行数：15840
- positioning groups：1980
- baseline 结果行数：9900

| 环境 | anchor-level aggregates |
|---|---:|
| environment0 | 4080 |
| environment1 | 3840 |
| environment2 | 4032 |
| environment3 | 3888 |

## 3. 环境级定位误差

| 环境 | 方法 | N | Mean | Median | RMSE | P90 | P95 | Max | Fail@0.5m | Fail@1.0m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| environment0 | LS | 510 | 0.344 | 0.299 | 0.398 | 0.605 | 0.777 | 1.174 | 18.0% | 1.0% |
| environment0 | Oracle_Range_LS | 510 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| environment0 | Robust_LS_Huber | 510 | 0.393 | 0.344 | 0.443 | 0.680 | 0.780 | 1.052 | 27.5% | 0.2% |
| environment0 | Robust_WLS_Huber | 510 | 0.391 | 0.344 | 0.440 | 0.690 | 0.780 | 0.988 | 26.5% | 0.0% |
| environment0 | WLS_range_std | 510 | 0.339 | 0.296 | 0.391 | 0.600 | 0.776 | 1.181 | 17.1% | 0.8% |
| environment1 | LS | 480 | 0.262 | 0.219 | 0.313 | 0.526 | 0.638 | 0.799 | 11.5% | 0.0% |
| environment1 | Oracle_Range_LS | 480 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| environment1 | Robust_LS_Huber | 480 | 0.308 | 0.254 | 0.371 | 0.657 | 0.734 | 0.863 | 18.5% | 0.0% |
| environment1 | Robust_WLS_Huber | 480 | 0.308 | 0.254 | 0.372 | 0.660 | 0.738 | 0.863 | 18.8% | 0.0% |
| environment1 | WLS_range_std | 480 | 0.262 | 0.218 | 0.314 | 0.528 | 0.640 | 0.799 | 11.7% | 0.0% |
| environment2 | LS | 504 | 0.734 | 0.581 | 0.885 | 1.476 | 1.614 | 2.963 | 58.9% | 22.8% |
| environment2 | Oracle_Range_LS | 504 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| environment2 | Robust_LS_Huber | 504 | 0.610 | 0.543 | 0.722 | 1.022 | 1.204 | 2.946 | 55.8% | 11.1% |
| environment2 | Robust_WLS_Huber | 504 | 0.573 | 0.504 | 0.689 | 0.972 | 1.150 | 2.936 | 51.4% | 9.3% |
| environment2 | WLS_range_std | 504 | 0.691 | 0.537 | 0.866 | 1.413 | 1.601 | 4.689 | 55.2% | 17.7% |
| environment3 | LS | 486 | 0.520 | 0.417 | 1.066 | 0.721 | 0.833 | 8.771 | 34.0% | 2.3% |
| environment3 | Oracle_Range_LS | 486 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0% | 0.0% |
| environment3 | Robust_LS_Huber | 486 | 0.541 | 0.429 | 1.063 | 0.762 | 0.844 | 8.786 | 39.3% | 2.5% |
| environment3 | Robust_WLS_Huber | 486 | 0.536 | 0.422 | 1.062 | 0.747 | 0.843 | 8.786 | 38.7% | 3.1% |
| environment3 | WLS_range_std | 486 | 0.515 | 0.398 | 1.064 | 0.716 | 0.826 | 8.771 | 33.1% | 2.9% |

## 4. 初步解读

这份 baseline 的作用不是追求最终精度，而是建立后续研究的地板线：如果后续的误差补偿、可信度权重和不确定性建模有效，必须在这些原始 LS/WLS 指标上稳定改进，尤其是 P90/P95 和 Fail@1.0m。

需要特别注意：`WLS_range_std` 只是一个朴素权重，使用同一 anchor-channel 文件内的 range std 估计均值不确定性。它不是最终可信度模型，反而适合作为一个简单可解释 baseline。

`Oracle_Range_LS` 的误差接近 0，说明当前 2D 几何定位求解本身没有明显问题。主要瓶颈来自测距偏差和观测可信度，而不是定位方程不可解。

## 5. 关键异常与研究价值

### 5.1 Environment2 是主困难环境

在四个环境中，`environment2` 的原始 LS 定位误差最高：

```text
LS mean = 0.734 m
LS P90  = 1.476 m
LS Fail@1.0m = 22.8%
```

加入 Huber 鲁棒核后，`Robust_WLS_Huber` 明显改善：

```text
Robust_WLS_Huber mean = 0.573 m
Robust_WLS_Huber P90  = 0.972 m
Robust_WLS_Huber Fail@1.0m = 9.3%
```

这说明 `environment2` 中存在典型长尾测距误差，鲁棒定位后端确实有价值。但鲁棒核只能被动降低异常残差影响，还不能主动判断哪条测距不可信。

### 5.2 Environment3 存在极端位置级异常

`environment3` 的整体 P90 不算最高，但最大误差达到约 8.77 m。最严重位置为：

```text
position_id = 9.87_8.32_1.20
```

该位置在多个 channel 下都产生严重定位失败。进一步查看 anchor-level 聚合发现，多条测距存在 5 m 到 9 m 级别偏差，而且 `range_std` 很小。

这说明一个非常关键的问题：

> 测距值可能非常稳定，但稳定地错。

因此，仅使用同一文件内测距方差作为 WLS 权重是不够的。后续必须引入 CIR、RSS、FP/RXPACC、几何和历史特征来估计 measurement trustworthiness。

### 5.3 Anchor-level 长尾来源

按环境和 anchor 汇总，P90 绝对测距误差最高的组合包括：

| 组合 | Mean abs ranging error | P90 abs ranging error | Mean NLOS ratio |
|---|---:|---:|---:|
| environment2, A6 | 1.474 | 8.461 | 0.417 |
| environment2, A8 | 0.897 | 2.042 | 0.881 |
| environment2, A7 | 0.868 | 1.818 | 0.929 |
| environment2, A5 | 0.829 | 1.765 | 0.869 |
| environment3, A4 | 0.758 | 1.428 | 0.691 |

其中 `environment2, A6` 特别值得注意。它的 NLOS ratio 并不是最高，但 P90 绝对测距误差却最大。这进一步说明：

> NLOS 标签本身不足以解释测距风险。

### 5.4 Channel 差异也存在

以原始 LS 为例，部分环境下 channel 差异明显：

| 环境 | 最优 channel mean error | 最差 channel mean error |
|---|---:|---:|
| environment0 | ch2: 0.247 m | ch7: 0.509 m |
| environment1 | ch1: 0.145 m | ch7: 0.415 m |
| environment2 | ch4: 0.516 m | ch3: 0.963 m |
| environment3 | ch5: 0.459 m | ch3: 0.571 m |

这说明 channel 也是测距质量建模的重要上下文变量。后续模型不应只输入 range，还应保留 `channel_number`、`preamble_code`、RSS 和 CIR 统计特征。

## 6. 对后续方法的直接启发

当前 baseline 已经给出三个明确研究理由：

1. **鲁棒后端有必要**  
   `environment2` 中 Huber robust WLS 明显降低长尾失败。

2. **朴素 WLS 不够**  
   `environment3` 极端位置中，错误测距的 `range_std` 很小，说明仅靠测距方差加权会误信稳定异常值。

3. **二值 NLOS 标签不够**  
   `environment2, A6` 的 NLOS ratio 不是最高，但长尾误差最严重，说明需要连续可信度和不确定性。

因此，我们的方法应聚焦：

```text
signal/geometry/temporal features
        -> per-link bias, sigma, trustworthiness
        -> robust uncertainty-weighted WLS
```

这不是为了比 LS 平均误差低一点，而是要压低：

```text
P90/P95 error
Failure@1.0m
极端位置级失败
```

## 7. 下一步

下一步建议做两件事：

1. 画出 LS/WLS 的 CDF、按 channel 的误差曲线、按 anchor-level ranging error 的分布；
2. 基于 oracle 上限结果，训练第一版 `features -> ranging_error` 测距误差补偿模型。
