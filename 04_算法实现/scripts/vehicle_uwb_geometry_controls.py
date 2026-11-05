"""Same-window geometric controls; all units cm, existing planar geometry."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m
from run_uwb_range_error_correction_study import ANCHORS,ANCHOR_NAMES,wls_from_ranges

def main():
    prior=ROOT/'SCI-UWB/data/results/vehicle_uwb_input_ablation_20261030'
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    meta,q,r=m.build_window_auxiliary(pd.read_csv(ROOT/cfg['input']),cfg['window'],cfg['step'])
    a=np.array([ANCHORS[k] for k in ANCHOR_NAMES],float)
    init=wls_from_ranges(r)
    predictions=[]
    for name in ['linear_LS','linear_WLS','nonlinear_LS','nonlinear_WLS','Huber_30cm']:
        print(name,flush=True)
        xy=[];ok=[]
        linear=wls_from_ranges(r,np.maximum(q[:,3:6],1)) if name=='linear_WLS' else init
        for i,ranges in enumerate(r):
            if name.startswith('linear'):
                xy.append(linear[i]);ok.append(True);continue
            weights=1/np.maximum(q[i,3:6],1)**2 if name=='nonlinear_WLS' else np.ones(3)
            weights=weights/weights.mean()
            def residual(x):return (np.linalg.norm(x-a,axis=1)-ranges)*np.sqrt(weights)
            candidates=[least_squares(residual,start,loss='huber' if name.startswith('Huber') else 'linear',f_scale=30,max_nfev=500) for start in [init[i],a.mean(axis=0)]]
            best=min(candidates,key=lambda z:z.cost);xy.append(best.x);ok.append(best.success)
        frame=meta.copy();xy=np.asarray(xy)
        frame['model']=name;frame['pred_x_cm']=xy[:,0];frame['pred_y_cm']=xy[:,1]
        frame['distance_error_cm']=np.linalg.norm(xy-meta[['x_cm','y_cm']].to_numpy(),axis=1);frame['solver_success']=ok
        predictions.append(frame)
    neural=pd.read_csv(prior/'predictions.csv')
    keys=['source_file','window_start','window_end']
    assert len(neural[neural.model=='full'].merge(meta,on=keys,validate='one_to_one'))==len(meta)
    allpred=pd.concat(predictions+[neural],ignore_index=True)
    out=ROOT/'SCI-UWB/data/results/vehicle_uwb_geometry_controls_20261030';out.mkdir(exist_ok=False)
    allpred.to_csv(out/'predictions.csv',index=False)
    def metrics(g):
        e=g.distance_error_cm.to_numpy();assert np.isfinite(e).all()
        return dict(n=len(e),mean_cm=e.mean(),rmse_cm=np.sqrt(np.mean(e**2)),p90_cm=np.quantile(e,.9),p95_cm=np.quantile(e,.95),over_100cm=np.mean(e>100))
    summary=pd.DataFrame([dict(model=k,**metrics(g)) for k,g in allpred.groupby('model')]);summary.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame([dict(model=k,source_file=s,**metrics(g)) for (k,s),g in allpred.groupby(['model','source_file'])]).to_csv(out/'per_source_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'anchors_cm':ANCHORS,'huber_delta_cm':30,'caveats':['processed data retained','planar anchor model inherited, not independently surveyed','geometric methods untrained; neural methods grouped out-of-fold','Huber threshold fixed, not target-tuned','no significance claim from single seed'],'completed':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(out)

if __name__=='__main__':main()

