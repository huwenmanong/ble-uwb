"""Review a completed v2 run; no model fitting or hyperparameter selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def main(run_dir=None):
    if run_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--run-dir', type=Path, default=ROOT / 'data/results/d1_v2_20261029_seed20261029')
        run_dir = parser.parse_args().run_dir
    out = Path(run_dir)
    info = json.loads((out / 'run_manifest.json').read_text(encoding='utf-8'))
    assert info['completed']
    win = pd.read_csv(out / 'window_features.csv')
    loc = pd.read_csv(out / 'localization.csv')
    summary = pd.read_csv(out / 'summary.csv')
    risk = pd.read_csv(out / 'intervals.csv')
    raw = loc[(loc.variant == 'raw') & (loc.method == 'raw_LS')]
    diag = win.merge(raw[['fold', 'raw_folder', 'channel', 'x', 'y']],
        left_on=['environment_id','raw_folder','channel'], right_on=['fold','raw_folder','channel'], validate='many_to_one')
    geometric_range = np.sqrt((diag.x - diag.anchor_x)**2 + (diag.y-diag.anchor_y)**2 + (diag.tag_z-diag.anchor_z)**2)
    diag['geometry_residual_m'] = diag.range_measured - geometric_range
    diag['stable_consistent'] = (diag.range_std <= .05) & (diag.geometry_residual_m.abs() <= .20)
    diag['large_error'] = diag.error_label.abs() >= .50
    diag['hidden_bad_link'] = diag.stable_consistent & diag.large_error
    # Fixed descriptive thresholds, not used for model fitting or deployment.
    diag.to_csv(out / 'hidden_bias_links.csv', index=False)
    rows = []
    for env, g in diag.groupby('environment_id'):
        rows.append(dict(fold=env, all_links=len(g), stable_consistent_links=int(g.stable_consistent.sum()),
            large_error_links=int(g.large_error.sum()), hidden_bad_links=int(g.hidden_bad_link.sum()),
            hidden_fraction_all=float(g.hidden_bad_link.mean()),
            hidden_fraction_large=float(g.hidden_bad_link.sum()/max(1,g.large_error.sum())),
            hidden_positions=int(g.loc[g.hidden_bad_link, 'position_id'].nunique())))
    hidden = pd.DataFrame(rows)
    hidden.to_csv(out / 'hidden_bias_summary.csv', index=False)
    rng = np.random.default_rng(20261029)
    feature_ci = []
    for fold, g in loc[loc.variant.isin(['cir','cir_entropy'])].groupby('fold'):
        for method in ['corrected_Huber','risk_Huber']:
            paired = g[g.method.eq(method)].pivot(index=['raw_folder','channel'], columns='variant', values='error_m')
            groups = paired.index.get_level_values(0)
            clusters = [np.flatnonzero(groups == p) for p in groups.unique()]
            a, b = paired.cir_entropy.to_numpy(), paired.cir.to_numpy()
            differences = []
            for _ in range(1000):
                ids = np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
                differences.append(np.quantile(a[ids],.9)-np.quantile(b[ids],.9))
            feature_ci.append(dict(fold=fold, method=method, contrast='CIR+entropy minus CIR',
                delta_p90_m=np.quantile(a,.9)-np.quantile(b,.9),
                ci_low_m=np.quantile(differences,.025), ci_high_m=np.quantile(differences,.975)))
    feature_ci = pd.DataFrame(feature_ci)
    feature_ci.to_csv(out / 'entropy_increment_bootstrap.csv', index=False)
    fig, ax = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    names = ['E0','E1','E2','E3']
    ax[0].bar(names,100*hidden.hidden_fraction_large,color='#3568a8')
    ax[0].set(ylabel='Hidden subset / large-error links (%)',title='Stable, geometrically consistent bad links',ylim=(0,None))
    for i,r in enumerate(hidden.itertuples()):
        ax[0].text(i,100*r.hidden_fraction_large+.3,f'{r.hidden_bad_links}/{r.large_error_links}',ha='center',fontsize=9)
    ci = feature_ci[feature_ci.method.eq('corrected_Huber')].reset_index(drop=True)
    for i,r in enumerate(ci.itertuples()):
        ax[1].plot([r.ci_low_m,r.ci_high_m],[i,i],color='#65758b',lw=2)
        ax[1].scatter(r.delta_p90_m,i,color='#3568a8',s=40,zorder=3)
    ax[1].axvline(0,ls='--',color='black',lw=1)
    ax[1].set_yticks(range(4),names)
    ax[1].invert_yaxis()
    ax[1].set(xlabel='P90 difference (m); negative favors added entropy',title='Entropy increment: 95% position-bootstrap intervals')
    for a in ax:
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='x',alpha=.15)
    fig.savefig(out / 'v2_hidden_bias_and_entropy.png',dpi=180)
    plt.close(fig)
    lines = ['# D1 第二版实验记录与研究决策（2026-10-29）','',
        '## Target profile','', 'Measurement / IEEE TIM 方法研究；本轮为开发性实验，不构成投稿就绪判断。','',
        '## Manuscript claim','', '待验证：部署可得的信号信息能否识别传统几何/波动指标遗漏的高误差链路，并在未见环境改善定位尾部风险。','',
        '## Evidence available','',
        f'- 数据：{info["valid_rows"]:,} 条有效测量，{info["windows"]:,} 个静态链路窗口，{info["positions"]} 个位置。',
        '- 评价单位改为完整窗口，最多31条有效测量。新版与旧版训练单位及模型设定不同，不将跨版本差异归因为单一修复。',
        '- 偏差拟合、风险尺度拟合、区间校准、外层测试按位置互斥。原始外层LOSO划分不变。',
        '- 高度作为已知安装条件：E0/E1/E2/E3分别为1.20/0.95/1.32/1.20 m。仅研究静态窗口二维定位。',
        '- 原始标量文件已排除已知坐标冲突文件夹；本轮继续排除非正或非有限测距。运行总耗时包含本轮拟合与解算，不包含之前CIR/熵提取。',
        '- 风险尺度预测绝对窗口误差，明确不称为标准差。区间半宽与定位权重分开，定位使用未乘校准分位数的相对尺度。',
        '- Huber门限统一固定0.30 m，尚未做源域内部调参或灵敏度验证。两初值TRF按目标函数值选择解，不使用真实坐标选解。',
        '- 原始同目标函数解算审计覆盖3,948组。新解目标函数均不高于旧解（容差1e-5），但成功标志不构成全局最优证明。',
        '- 全部37,506个新定位结果有限且求解器报告成功；仍存在较大定位误差。','',
        '### CIR特征版本的定位P90（m）','',
        '| 环境 | 原始LS | 原始Huber | 校正LS | 校正Huber | 风险WLS | 风险Huber |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for fold in sorted(summary.fold.unique()):
        vals=[]
        for variant,method in [('raw','raw_LS'),('raw','raw_Huber'),('cir','corrected_LS'),('cir','corrected_Huber'),('cir','risk_WLS'),('cir','risk_Huber')]:
            vals.append(float(summary[(summary.fold==fold)&(summary.variant==variant)&(summary.method==method)].p90_m.iloc[0]))
        lines.append('| '+fold+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines += ['', '### CIR区间覆盖率与宽度','', '| 环境 | 区间方法 | 单链路覆盖 | 同位置全部链路覆盖 | 平均全宽(m) |', '|---|---|---:|---:|---:|']
    for r in risk[risk.variant.eq('cir')].itertuples():
        lines.append(f'| {r.fold} | {r.interval} | {r.link_coverage:.2%} | {r.whole_position_coverage:.2%} | {r.mean_full_width_m:.3f} |')
    lines += ['', '### 稳定但有偏的链路','',
        '描述性门限事先固定为：测距标准差≤0.05 m、原始LS几何残差绝对值≤0.20 m、真实窗口测距误差绝对值≥0.50 m。真值仅用于离线标记，不能直接部署这一判据。门限敏感性未验证。', '',
        '| 环境 | 全部链路 | 大误差链路 | 同时稳定且几何一致的大误差链路 | 涉及位置 |','|---|---:|---:|---:|---:|']
    for r in hidden.itertuples():
        lines.append(f'| {r.fold} | {r.all_links} | {r.large_error_links} | {r.hidden_bad_links} | {r.hidden_positions} |')
    lines += ['', '## 本轮决策','',
        '1. 保留“误差补偿+稳定鲁棒定位”为强基线。E2中简单校正Huber已达到约0.708 m P90，新增风险加权为约0.734 m；不能宣称风险模块在该环境胜出。',
        '2. 直接逆尺度加权仍会损害E3。CIR版本风险Huber相对校正Huber的P90增加约0.140 m，位置分组bootstrap区间为正。该结论是本轮D1内部探索证据，未做多重比较校正。',
        '3. 独立校准后欠覆盖仍存在，证明当前实现不能提供目标域风险保证；不能由本实验单独分离环境偏移、特征不足与模型失配的因果贡献。',
        '4. position_max区间虽提升单链路覆盖，平均全宽约3–4.4 m；E2同位置全部链路覆盖仍只有约52%。单链路覆盖不等于位置保护区域覆盖，也不应混用两个目标。',
        '5. 熵保持为候选消融。查看entropy_increment_bootstrap.csv判断相对CIR的边际贡献，不以单环境最佳值选主方法。',
        '6. 下一项核心任务为对几何一致但有偏链路做交叉环境可识别性实验，并审计风险权重如何削弱有用锚点几何约束；在机制确认前不增加深度网络。','',
        '## Evidence still required','',
        '- 基于源环境内部的嵌套分组验证选择模型/鲁棒阈值，外层测试不能作为调参依据。',
        '- 对尺度失配与几何退化进行受控消融，区分误差排序、偏差补偿、权重分配的独立贡献。',
        '- 补充无偏差补偿但使用同类风险模型的消融，完成全因子比较。',
        '- 分组bootstrap只能反映D1位置采样不确定性，不能代表新环境总体；4个环境不足以支持广泛泛化结论。',
        '- 全部D1外层折此前已被反复查看，需独立外部数据作最终验证。',
        '- 非正测距排除率、样本不足窗口、熵完整窗口等待延迟、完整特征提取成本应在最终工程评价中单独报告。','',
        '## 复现','',
        '- 主脚本：04_算法实现/scripts/d1_v2_research.py；解释器：D:/anaconda/envs/bluekey/python.exe。',
        '- 复核脚本：04_算法实现/scripts/d1_v2_review.py --run-dir <本次输出目录>。',
        '- 数据、角色划分、预测、解算结果、区间与bootstrap均保存在本报告同目录。',
        '- run_manifest.json记录输入哈希、软件版本、种子和当次执行源码哈希；experiment_source.py为执行版本快照。','']
    (out / '研究记录.md').write_text('\n'.join(lines),encoding='utf-8')
    print(hidden.to_string(index=False))
    print(feature_ci.round(5).to_string(index=False))
    print(out / '研究记录.md')


if __name__ == '__main__':
    main()
