"""Descriptive source and fold diagnostics; no retraining or selection."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[3]/'SCI-UWB'
def main():
    out=ROOT/'data/results/vehicle_source_comparison_20261102';out.mkdir(exist_ok=False)
    a=pd.read_csv(ROOT/'data/results/vehicle_model_complexity_control_20261030/predictions.csv')
    b=pd.read_csv(ROOT/'data/results/vehicle_rbf_control_20261102/predictions.csv');b['seed']=0
    p=pd.concat([a[a.model.isin(['full','simple_cnn'])],b],ignore_index=True)
    keys=['source_file','window_start','window_end']
    base=p[(p.model=='full')&(p.seed==20261030)][keys].reset_index(drop=True)
    for _,g in p.groupby(['model','seed']):
        assert not g.duplicated(keys).any()
        assert set(map(tuple,g[keys].values))==set(map(tuple,base.values))
    foldmap={}
    for fold,(_,test) in enumerate(GroupKFold(5).split(base,groups=base.source_file),1):
        for s in base.iloc[test].source_file:foldmap[s]=fold
    p['fold']=p.source_file.map(foldmap)
    def summarize(cols):
        return p.groupby(cols).distance_error_cm.agg(mean_cm='mean',p90_cm=lambda s:s.quantile(.9),over100_pct=lambda s:100*(s>100).mean()).reset_index()
    src=summarize(['model','seed','source_file']);src.to_csv(out/'source_metrics.csv',index=False)
    summarize(['model','seed','fold']).to_csv(out/'fold_metrics.csv',index=False)
    avg=src.groupby(['source_file','model']).mean_cm.mean().unstack()
    avg['full_minus_rbf']=avg.full-avg.rbf_mean3
    counts=base.groupby('source_file').size();avg['windows']=counts
    avg['weighted_gain_cm']=-avg.full_minus_rbf*counts/counts.sum()
    avg.to_csv(out/'source_comparison.csv')
    seedcmp=src[src.model=='full'].pivot(index='source_file',columns='seed',values='mean_cm').sub(avg.rbf_mean3,axis=0)
    gain=avg.weighted_gain_cm.sort_values(ascending=False)
    report=dict(source_count=len(avg),mean_source_equal_errors=avg[['full','simple_cnn','rbf_mean3','rbf_stats24']].mean().to_dict(),full_better_sources=int((avg.full_minus_rbf<0).sum()),full_always_better_sources=int((seedcmp.max(axis=1)<0).sum()),full_always_worse_sources=int((seedcmp.min(axis=1)>0).sum()),overall_weighted_gain_cm=float(gain.sum()),top3_gain_contribution_cm=float(gain.head(3).sum()),remaining_gain_cm=float(gain.iloc[3:].sum()),limitations='Descriptive existing-test analysis. CNN averages per-run errors, not ensemble coordinates. No exclusions or new hyperparameter tuning.')
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2));print(avg.to_string())
if __name__=='__main__':main()

