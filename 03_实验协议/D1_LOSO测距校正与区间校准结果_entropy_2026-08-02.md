# D1 LOSO 测距校正与区间校准结果

## Target profile

Measurement / IEEE TIM；本实验是跨环境测距不确定性建模的第一版可复现实验。

## Manuscript claim

检验部署可用信号特征能否在未见 environment 上校正测距偏差，并通过独立 calibration 组构造具有覆盖率证据的误差区间。

## Evidence available

- 输入：483,692 条有效测量；四个 LOSO 折；训练最多 150,000 行/折；运行耗时 14.71 s。
- 模型输入：Range、RSS、RSS_FP、FP 指标、噪声/接收诊断、锚点坐标、信道和锚点 ID，以及窗口化熵特征。
- 未使用：真实位置、真实距离、测距误差标签（仅训练目标）、NLOS 标签和错位文件夹。

### 测距校正与区间覆盖

| fold | raw MAE | corrected MAE | raw P90 | corrected P90 | q90 (m) | test coverage q90 | q95 (m) | test coverage q95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| environment0 | 0.4270 | 0.3036 | 0.8348 | 0.7186 | 0.5148 | 82.63% | 0.6985 | 89.49% |
| environment1 | 0.5074 | 0.2016 | 0.9182 | 0.3895 | 0.5392 | 97.25% | 0.7316 | 98.89% |
| environment2 | 0.8048 | 0.6739 | 1.4409 | 1.2959 | 0.4345 | 53.25% | 0.5945 | 64.14% |
| environment3 | 0.4689 | 0.4100 | 0.9377 | 0.9540 | 0.4484 | 64.03% | 0.6235 | 75.45% |

### 定位影响

| fold | method | N | mean (m) | RMSE (m) | P90 (m) | P95 (m) | Fail@0.5m | Fail@1m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| environment0 | corrected_LS | 510 | 0.3249 | 0.3997 | 0.6176 | 0.7482 | 20.00% | 1.18% |
| environment0 | raw_LS | 510 | 0.3440 | 0.3995 | 0.6107 | 0.7876 | 17.65% | 0.98% |
| environment1 | corrected_LS | 480 | 0.2506 | 0.2729 | 0.3781 | 0.4107 | 1.88% | 0.00% |
| environment1 | raw_LS | 480 | 0.3111 | 0.3862 | 0.7124 | 0.8081 | 18.75% | 0.00% |
| environment2 | corrected_LS | 504 | 0.6569 | 0.8297 | 1.3604 | 1.5491 | 49.40% | 24.01% |
| environment2 | raw_LS | 504 | 0.7341 | 0.8849 | 1.4813 | 1.6257 | 58.93% | 22.82% |
| environment3 | corrected_LS | 480 | 0.3654 | 0.3957 | 0.5741 | 0.6443 | 17.29% | 0.00% |
| environment3 | raw_LS | 480 | 0.4169 | 0.4729 | 0.7098 | 0.7909 | 33.54% | 1.04% |

## Evidence still required

- 当前是第一版全局误差模型，不等于最终方法；需要与不校正 LS/WLS、分层模型、CIR 特征模型和真正的风险传播模型比较。
- calibration coverage 只在固定 D1 split 上成立；需要报告跨折均值/方差，并进行 bootstrap 或重复种子稳定性检查。
- 后续必须增加 CIR 特征和 uncertainty-aware WLS，验证区间是否能降低定位尾部风险。
