"""Adaptive anchor gate versus identity gate under the fixed original split."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m

class IdentityGateCNN(nn.Module):
    """Preserves quality context; fixes only three anchor weights at one."""
    def __init__(self,quality_dim):
        super().__init__()
        self.context_net=nn.Sequential(nn.Linear(quality_dim,32),nn.ReLU(),nn.Dropout(.08),nn.Linear(32,16),nn.ReLU())
        self.context=nn.Linear(16,16)
        self.encoder=nn.Sequential(nn.Conv1d(3,32,5,padding=2),nn.BatchNorm1d(32),nn.ReLU(),nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.physics_encoder=nn.Sequential(nn.Linear(2,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(96,64),nn.ReLU(),nn.Dropout(.10),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
    def forward(self,sequence,quality,wls_xy):
        context=torch.relu(self.context(self.context_net(quality)))
        pred=self.head(torch.cat([self.encoder(sequence),context,self.physics_encoder(wls_xy)],dim=1))
        return pred,pred,wls_xy,torch.ones((len(sequence),1),device=sequence.device,dtype=sequence.dtype)

def main():
    os.chdir(ROOT);out=ROOT/'SCI-UWB/data/results/vehicle_gate_ablation_20261104_checkpoint';out.mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    sys.argv=['gate'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:setattr(args,k,cfg[k])
    source=ROOT/cfg['input'];wide=pd.read_csv(source);seq,meta=m.build_sequence_windows(wide,args.window,args.step);aux,q,r=m.build_window_auxiliary(wide,args.window,args.step);wls=m.compute_wls_features(r,q)
    groups=meta.source_file.to_numpy();state={};cls=m.DirectQualityPhysicsCNN
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]));state['fold']=seed-cfg['seed'];return ids[a],ids[b]
    def factory(*a,**kw):
        m.set_seed(state['training_seed']+state['fold']);return IdentityGateCNN(*a,**kw)
    m.split_train_val=split;m.DirectQualityPhysicsCNN=factory;torch.use_deterministic_algorithms(True);assert not torch.cuda.is_available()
    requested=int(os.environ.get('GATE_TRAINING_SEED','0'))
    seeds=[requested] if requested else [20261030,20261040,20261050]
    frames=[]
    prior_file=out/'predictions.csv'
    if prior_file.exists():
        old=pd.read_csv(prior_file)
        frames.append(old[old.model=='identity_gate'])
    done=set(pd.concat(frames).seed.tolist()) if frames else set()
    for seed in seeds:
        if seed in done:
            print('already saved',seed,flush=True);continue
        state['training_seed']=seed;print('identity gate',seed,flush=True)
        _,p=m.evaluate_model('identity_gate',seq,meta,q,wls,args,'direct');p['seed']=seed;frames.append(p)
    prior=pd.read_csv(ROOT/'SCI-UWB/data/results/vehicle_deterministic_input_ablation_20261030/predictions.csv')
    full=prior[prior.model=='full'].copy();full['model']='adaptive_gate';frames.append(full)
    p=pd.concat(frames,ignore_index=True);p.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for (model,seed),g in p.groupby(['model','seed']):
        e=g.distance_error_cm.values;rows.append(dict(model=model,seed=seed,mean_cm=e.mean(),rmse_cm=np.sqrt(np.mean(e**2)),p90_cm=np.quantile(e,.9),p95_cm=np.quantile(e,.95),over100_pct=100*np.mean(e>100)))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    counts={'adaptive_gate':sum(x.numel() for x in cls(24).parameters()),'identity_gate':sum(x.numel() for x in IdentityGateCNN(24).parameters())}
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),seeds=[20261030,20261040,20261050],split_seed=cfg['seed'],fold_seed='training_seed+fold',parameter_counts=counts,adaptive_predictions='reused exact verified full predictions from vehicle_deterministic_input_ablation_20261030',identity_definition='Quality context retained; raw three-anchor sequence multiplied by [1,1,1] instead of learned [0.5,1.5] weights.',limitations='Structural ablation changes parameter initialization order and removes 51 learned anchor-head parameters. Fixed processed dataset; exploratory repeated data.'),indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(counts)
if __name__=='__main__':main()

