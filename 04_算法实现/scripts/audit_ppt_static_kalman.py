"""Recompute saved PPT static results and verify filter label independence."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from run_uwb_pqg_robust_kalman_postprocess import apply_filter
def main():
    folder=ROOT/'UWB测距数据/uwb_pqg_cw_cnn_robust_kalman_tuning_results'
    out=ROOT/'SCI-UWB/data/results/ppt_static_kalman_audit_20261105_repeat01';out.mkdir(exist_ok=False)
    p=pd.read_csv(folder/'best_tuned_predictions.csv');cfg=json.loads((folder/'best_tuned_config.json').read_text())
    raw=p.copy();raw['pred_x_cm']=p.raw_pred_x_cm;raw['pred_y_cm']=p.raw_pred_y_cm;raw['model']='raw'
    kw={k:cfg[k] for k in ['process_var','measurement_var','init_var','robust_c']}
    fresh=apply_filter(raw,**kw)
    cols=['pred_x_cm','pred_y_cm'];delta=float(np.max(np.abs(fresh[cols].values-p[cols].values)))
    fake=raw.copy();fake['x_cm']=123456.;fake['y_cm']=-78901.
    changed=apply_filter(fake,**kw)
    independence=float(np.max(np.abs(fresh[cols].values-changed[cols].values)))
    rows=[]
    for name,frame in [('raw',raw),('filtered',p)]:
        for scene,g in [('ALL',frame),*list(frame.groupby('scene'))]:
            e=np.linalg.norm(g[cols].values-g[['x_cm','y_cm']].values,axis=1)
            rows.append(dict(stage=name,scene=scene,n=len(g),mean_cm=e.mean(),median_cm=np.median(e),p90_cm=np.quantile(e,.9),max_cm=e.max()))
    pd.DataFrame(rows).to_csv(out/'recomputed_metrics.csv',index=False)
    grid=pd.read_csv(folder/'pqg_cw_cnn_robust_kalman_tuning_grid.csv')
    report=dict(n_windows=len(p),n_sources=p.source_file.nunique(),filter_reproduction_max_cm=delta,label_perturbation_prediction_max_cm=independence,saved_grid_rows=len(grid),saved_grid_models=grid.model.unique().tolist(),initialization='first predicted coordinate per source',grouping='model and source_file; ordered by window_start/window_end')
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(report)
if __name__=='__main__':main()
