"""Same-filter descriptive comparison; all seeds retained."""
from vehicle_ppt_gated_filter_control import *
import numpy as np
def run():
    root=ROOT/'SCI-UWB/data/results';out=root/'ppt_gated_filter_summary_20261105';out.mkdir(exist_ok=False)
    frames=[];protocols=[]
    for seed in [20261030,20261040,20261050]:
        folder=root/('ppt_gated_filter_grouped_20261105' if seed==20261030 else f'ppt_gated_filter_grouped_20261105_seed{seed}')
        protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'));assert protocol['completed'];protocols.append(protocol)
        for filename,stage in [('raw_predictions.csv','raw'),('filtered_predictions.csv','filtered')]:
            p=pd.read_csv(folder/filename);p['seed']=seed;p['method']='gated';p['stage']=stage;frames.append(p)
    assert all(p['splits']==protocols[0]['splits'] for p in protocols)
    cfg=protocols[0]['config'];kw={k:cfg[k] for k in ['process_var','measurement_var','init_var','robust_c']}
    rbf=pd.read_csv(root/'vehicle_rbf_control_20261102/predictions.csv');rbf=rbf[rbf.model=='rbf_mean3'].copy()
    keys=['source_file','window_start','window_end']
    assert set(map(tuple,rbf[keys].values))==set(map(tuple,frames[0][keys].values))
    for p,stage in [(rbf,'raw'),(apply_filter(rbf,**kw),'filtered')]:
        p['seed']=0;p['method']='rbf_mean3';p['stage']=stage;frames.append(p)
    allp=pd.concat(frames,ignore_index=True);allp.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for (method,seed,stage),g in allp.groupby(['method','seed','stage']):
        e=g.distance_error_cm.values
        rows.append(dict(method=method,seed=seed,stage=stage,mean_cm=e.mean(),rmse_cm=np.sqrt((e**2).mean()),p90_cm=np.quantile(e,.9),p95_cm=np.quantile(e,.95),over100_pct=100*np.mean(e>100)))
    d=pd.DataFrame(rows);d.to_csv(out/'metrics.csv',index=False)
    summary=d.groupby(['method','stage'])[['mean_cm','p90_cm','p95_cm','over100_pct']].agg(['mean','std']);summary.to_csv(out/'summary.csv')
    print(d.to_string(index=False));print(summary.to_string())
if __name__=='__main__':run()
