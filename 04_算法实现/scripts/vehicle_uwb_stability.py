"""Fixed source splits; vary network training seeds, cluster bootstrap sources."""
from pathlib import Path
import sys,os,json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m

def main():
    os.chdir(ROOT)
    out=ROOT/'SCI-UWB/data/results/vehicle_uwb_stability_20261030';i=0
    while out.exists():
        i+=1;out=out.with_name(f'vehicle_uwb_stability_20261030_repeat{i:02d}')
    out.mkdir(parents=True)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    sys.argv=['stability'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:setattr(args,k,cfg[k])
    wide=pd.read_csv(ROOT/cfg['input']);seq,meta=m.build_sequence_windows(wide,args.window,args.step)
    aux,q,r=m.build_window_auxiliary(wide,args.window,args.step);wls=m.compute_wls_features(r,q)
    assert meta[['source_file','window_start','window_end']].equals(aux[['source_file','window_start','window_end']])
    groups=meta.source_file.to_numpy()
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]))
        assert not(set(groups[ids[a]])&set(groups[ids[b]]));return ids[a],ids[b]
    m.split_train_val=split
    rows=[];pred=[]
    for seed in [20260704,20260705,20260706]:
        for name,quality,physical in [('full',q,wls),('sequence_only_information',np.zeros_like(q),np.zeros_like(wls))]:
            print(seed,name,flush=True);m.set_seed(seed)
            result,p=m.evaluate_model(name,seq,meta,quality,physical,args,'direct')
            rows.append(dict(seed=seed,model=name,**result));p['seed']=seed;pred.append(p)
            pd.DataFrame(rows).to_csv(out/'seed_metrics.csv',index=False)
    predictions=pd.concat(pred,ignore_index=True);predictions.to_csv(out/'predictions.csv',index=False)
    keys=['source_file','window_start','window_end']
    # Average errors over seeds, NOT coordinates: evaluates expected single-run
    # performance, not an ensemble. Bootstrap uncertainty is source sampling only.
    avg=predictions.groupby(keys+['model']).distance_error_cm.mean().unstack('model').reset_index()
    geo=pd.read_csv(ROOT/'SCI-UWB/data/results/vehicle_uwb_geometry_controls_20261030/predictions.csv')
    geo=geo[geo.model.eq('nonlinear_LS')][keys+['distance_error_cm']].rename(columns={'distance_error_cm':'nonlinear_LS'})
    avg=avg.merge(geo,on=keys,validate='one_to_one');assert len(avg)==len(meta)
    sources=avg.source_file.unique();clusters=[np.flatnonzero(avg.source_file.eq(s)) for s in sources]
    rng=np.random.default_rng(20261030);stats=[]
    for baseline in ['sequence_only_information','nonlinear_LS']:
        a=avg['full'].to_numpy();b=avg[baseline].to_numpy()
        for metric,fun in [('mean_cm',np.mean),('p90_seedmean_error_cm',lambda x:np.quantile(x,.9))]:
            values=[]
            for _ in range(2000):
                ids=np.concatenate([clusters[k] for k in rng.integers(0,len(clusters),len(clusters))])
                values.append(fun(a[ids])-fun(b[ids]))
            stats.append(dict(baseline=baseline,metric=metric,delta=fun(a)-fun(b),ci_low=np.quantile(values,.025),ci_high=np.quantile(values,.975)))
    pd.DataFrame(stats).to_csv(out/'paired_source_bootstrap.csv',index=False)
    avg.groupby('source_file')[['full','sequence_only_information','nonlinear_LS']].mean().to_csv(out/'per_source_seedmean_errors.csv')
    (out/'protocol.json').write_text(json.dumps({'completed':True,'training_seeds':[20260704,20260705,20260706],'split_seed':cfg['seed'],'bootstrap':'2000 paired source resamples; fixed fitted models, no multiplicity correction','p90_caveat':'bootstrap P90 is quantile of seed-mean window errors, not mean of per-seed P90; see seed_metrics for ordinary P90','data':'historical processed data retained; source groups not independent vehicles'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows)[['seed','model','mae_distance_cm','p90_distance_error_cm']].to_string(index=False))
    print(pd.DataFrame(stats).to_string(index=False));print(out)

if __name__=='__main__':main()

