"""Structural CNN ablation and train-only tuned ridge baselines."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m

class SimpleCNN(nn.Module):
    def __init__(self,quality_dim=24):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(3,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,sequence,quality,wls_xy):
        pred=self.head(self.encoder(sequence))
        return pred,pred,torch.zeros_like(pred),torch.ones((len(pred),1),device=pred.device)

def main():
    os.chdir(ROOT);out=ROOT/'SCI-UWB/data/results/vehicle_model_complexity_control_20261030'
    out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    sys.argv=['complexity'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:setattr(args,k,cfg[k])
    source=ROOT/cfg['input'];wide=pd.read_csv(source)
    seq,meta=m.build_sequence_windows(wide,args.window,args.step)
    aux,q,r=m.build_window_auxiliary(wide,args.window,args.step)
    assert meta[['source_file','window_start','window_end']].equals(aux[['source_file','window_start','window_end']])
    wls=m.compute_wls_features(r,q);groups=meta.source_file.to_numpy();state={};split_log=[]
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]))
        state['fold']=seed-cfg['seed'];return ids[a],ids[b]
    m.split_train_val=split
    original=m.DirectQualityPhysicsCNN
    counts={'full':sum(p.numel() for p in original(24).parameters()),'simple_cnn':sum(p.numel() for p in SimpleCNN().parameters())}
    def factory(*a,**kw):
        m.set_seed(state['seed']+state['fold']);return SimpleCNN(*a,**kw)
    m.DirectQualityPhysicsCNN=factory;torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    frames=[]
    for seed in [20261030,20261040,20261050]:
        state['seed']=seed;print('simple_cnn',seed,flush=True)
        _,p=m.evaluate_model('simple_cnn',seq,meta,q,wls,args,'direct');p['seed']=seed;frames.append(p)
    y=meta[['x_cm','y_cm']].to_numpy(dtype=np.float32);weights=m.sample_weights(meta)
    tuning=[]
    for name,x in [('ridge_mean3',r),('ridge_stats24',q)]:
        pred=np.zeros_like(y)
        for fold,(train,test) in enumerate(GroupKFold(5).split(x,y,groups),1):
            core,val=split(train,cfg['seed']+fold)
            sx=StandardScaler().fit(x[core]);sy=StandardScaler().fit(y[core])
            xt=sx.transform(x[core]);yt=sy.transform(y[core]);xv=sx.transform(x[val]);yv=sy.transform(y[val])
            candidates=[]
            for alpha in [.01,.1,1.,10.,100.,1000.]:
                model=Ridge(alpha=alpha).fit(xt,yt,sample_weight=weights[core])
                loss=float(np.mean((model.predict(xv)-yv)**2));candidates.append((loss,alpha,model))
                tuning.append(dict(model=name,fold=fold,alpha=alpha,val_mse=loss))
            loss,alpha,model=min(candidates,key=lambda t:(t[0],t[1]))
            pred[test]=sy.inverse_transform(model.predict(sx.transform(x[test])))
            split_log.append(dict(model=name,fold=fold,selected_alpha=alpha,train_sources=sorted(set(groups[core])),val_sources=sorted(set(groups[val])),test_sources=sorted(set(groups[test]))))
        p=meta.copy();p['model']=name;p['seed']=0;p['pred_x_cm']=pred[:,0];p['pred_y_cm']=pred[:,1];p['distance_error_cm']=np.linalg.norm(pred-y,axis=1);frames.append(p)
        counts[name]=2*x.shape[1]+2
    # Reuse unchanged completed controls, without pretending they were retrained here.
    oldpath=ROOT/'SCI-UWB/data/results/vehicle_deterministic_input_ablation_20261030'
    oldprotocol=json.loads((oldpath/'protocol.json').read_text())
    assert oldprotocol['input_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    old=pd.read_csv(oldpath/'predictions.csv');frames.append(old[old.model.isin(['full','sequence_only'])])
    p=pd.concat(frames,ignore_index=True);p.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for (model,seed),g in p.groupby(['model','seed']):
        e=g.distance_error_cm.to_numpy();rows.append(dict(model=model,seed=seed,mean_cm=e.mean(),rmse_cm=np.sqrt(np.mean(e**2)),p90_cm=np.quantile(e,.9),p95_cm=np.quantile(e,.95),over100_pct=100*np.mean(e>100)))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'ridge_tuning.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,counts=counts,ridge_splits=split_log,input_sha256=oldprotocol['input_sha256'],torch=torch.__version__,device='cpu',ridge_seed='deterministic, one run only',ridge_refit=False,limitations='Fixed processed dataset and repeated exploratory tests. Ridge uses same inner core, weights and normalized validation MSE as CNN; no refit on validation. Existing full and sequence-only information controls reused from completed prior experiment.'),indent=2,ensure_ascii=False),encoding='utf-8')
    print(metrics.to_string(index=False));print(counts);print(out)

if __name__=='__main__':main()

