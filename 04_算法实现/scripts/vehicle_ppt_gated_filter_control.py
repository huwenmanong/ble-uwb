"""Grouped validation of PPT gated structure with frozen historical filter."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import sys,json
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m
from run_uwb_pqg_robust_kalman_postprocess import apply_filter,metrics_frame
def main():
    os.chdir(ROOT)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_pqg_cw_cnn_robust_kalman_tuning_results/best_tuned_config.json').read_text())
    training_seed=int(os.environ.get('PPT_TRAINING_SEED','20261030'))
    directory='ppt_gated_filter_grouped_20261105' if training_seed==20261030 else f'ppt_gated_filter_grouped_20261105_seed{training_seed}'
    out=ROOT/'SCI-UWB/data/results'/directory;out.mkdir(exist_ok=False)
    sys.argv=['gated-control'];args=m.parse_args()
    for key in ['window','step','epochs','patience','batch_size','lr','weight_decay','seed']:setattr(args,key,cfg[key])
    data=pd.read_csv(args.input);seq,meta=m.build_sequence_windows(data,args.window,args.step);aux,q,r=m.build_window_auxiliary(data,args.window,args.step)
    groups=meta.source_file.to_numpy();state={};splits=[]
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]));state['fold']=seed-args.seed
        splits.append(dict(fold=state['fold'],train=sorted(set(groups[ids[a]])),validation=sorted(set(groups[ids[b]]))))
        return ids[a],ids[b]
    cls=m.PhysicsQualityGuidedCNN
    def factory(*a,**kw):
        m.set_seed(training_seed+state['fold']);return cls(*a,**kw)
    m.split_train_val=split;m.PhysicsQualityGuidedCNN=factory;torch.use_deterministic_algorithms(True)
    _,raw=m.evaluate_model('ppt_gated',seq,meta,q,m.compute_wls_features(r,q),args,'gated')
    filtered=apply_filter(raw,**{k:cfg[k] for k in ['process_var','measurement_var','init_var','robust_c']})
    raw.to_csv(out/'raw_predictions.csv',index=False);filtered.to_csv(out/'filtered_predictions.csv',index=False)
    rows=[metrics_frame(name,'source_grouped_inner_and_outer',frame) for name,frame in [('gated_raw',raw),('gated_frozen_filter',filtered)]]
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,config=cfg,splits=splits,training_seed=training_seed,fold_seed='training_seed+fold',limitations='One seed; frozen filter previously selected on existing data; exploratory, not independent confirmation.'),indent=2,ensure_ascii=False),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
