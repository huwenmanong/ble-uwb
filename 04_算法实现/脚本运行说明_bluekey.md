# SCI-UWB 脚本运行规范

## 阶段3：当前推荐入口（数据协议审计）

在 PyCharm 选择 **D1 V3 Protocol Audit**，或直接运行 `scripts/d1_v3_protocol_audit.py`。环境仍为 `D:\anaconda\envs\bluekey\python.exe`，无需安装新包。

默认流式核对四个官方 JSON，再运行移除 E2-A6 的单因素对照和固定权重强度敏感性实验。此次完整运行约112秒，实际时间随磁盘和机器负载变化。输出在 `data/results/d1_v3_protocol_20261029` 的新重复目录中，旧结果不覆盖。

当前已验证结果为 `data/results/d1_v3_protocol_20261029_repeat01`。未带 repeat 的第一次尝试未通过复现断言，已标为未完成，不能使用。判断成功应检查输出目录 `protocol.json` 的 `completed: true`。

`--skip-json-audit` 仅用于快速算法复跑，会省略原始数据来源审计。`--figures-only --run-dir <完整输出目录>` 只更新图表，不重新训练。

旧 V2 入口保留用于追溯，但包含原论文已说明的故障锚点，不能当成当前正式研究结果。

## 2026-10-29：V2历史入口

在PyCharm选择 `D1 V2 Research`，或直接运行 `scripts/run_d1_v2_pipeline.py`。解释器为 `D:\anaconda\envs\bluekey\python.exe`。该入口依次执行独立校准实验、结果复核、大误差可识别性诊断，并生成CSV、研究记录和图表。核心实验本轮约110秒，完整流程时间取决于机器负载。

结果写入 `SCI-UWB/data/results/d1_v2_20261029_seed20261029`；再次运行会创建 `_repeat01` 等目录，不覆盖历史实验。当前新版采用静态完整窗口评价，与旧版逐测量训练结果不可直接作单一因素比较。

仅检查数学实现可传入 `--self-test`。最终研究决策见 `SCI-UWB/研究进展与决策_2026-10-29.md`。

后续数据分析和算法实验优先写入 `SCI-UWB/04_算法实现/scripts/`，不在对话中反复展开大段原始数据。所有脚本使用 Conda 环境 `bluekey`（用户称 blue-key）。

## 单个脚本

在 `E:\陕汽实习\数据集` 下执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\SCI-UWB\04_算法实现\run_bluekey.ps1 `
  -ScriptPath '04_算法实现/scripts/check_bluekey_environment.py'
```

脚本路径相对于 `SCI-UWB` 根目录；也可以传绝对路径。

## 阶段流水线

```powershell
powershell -ExecutionPolicy Bypass -File .\SCI-UWB\04_算法实现\run_stage0_pipeline.ps1
```

该流水线依次执行环境记录、D1 全量审计、environment3 错位核查和 LOSO 冻结划分。

## 复现约束

- 大文件使用流式读取；不得将 `data.json` 或全部 CIR 一次性读入内存。
- 每个脚本将结果写入 `SCI-UWB/02_公开数据集/` 或 `SCI-UWB/04_算法实现/outputs/`。
- 所有正式实验记录数据版本、随机种子、split 清单和 Python 可执行文件。
- Jupyter 仅用于展示和复核；正式计算逻辑必须保留在可独立运行的 `.py` 脚本中。

## 可视化输出

阶段性结果使用 `D1 Research Figures` 配置运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\SCI-UWB\04_算法实现\run_bluekey.ps1 `
  -ScriptPath '04_算法实现/scripts/plot_d1_research_figures.py'
```

图像输出到 `SCI-UWB/data/figures/d1_stage1/`，包括误差分布、基线 P90 热图、校正/覆盖率总览和特征关联图。后续每个新实验脚本都应同时生成 CSV/JSON 数值结果和至少一张可检查图形。
