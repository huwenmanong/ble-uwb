"""Grouped input ablation with identical architecture and reset seeds.
Zero-input variants test information availability, not architecture removal.
"""
from pathlib import Path
import sys,os,json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m

def main():
    os.chdir(ROOT)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    out=ROOT/'SCI-UWB/data/results/vehicle_uwb_input_ablation_20261030';i=0
    while out.exists():
        i+=1;out=out.with_name(f'vehicle_uwb_input_ablation_20261030_repeat{i:02d}')
    out.mkdir(parents=True)
    sys.argv=['ablation']
    args=m.parse_args()
    for key in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:setattr(args,key,cfg[key])
    data=pd.read_csv(ROOT/cfg['input'])
    sequences,meta=m.build_sequence_windows(data,args.window,args.step)
    aux,q,r=m.build_window_auxiliary(data,args.window,args.step)
    assert meta[['source_file','window_start','window_end']].equals(aux[['source_file','window_start','window_end']])
    wls=m.compute_wls_features(r,q)
    groups=meta.source_file.to_numpy()
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]))
        assert not(set(groups[ids[a]])&set(groups[ids[b]]))
        return ids[a],ids[b]
    m.split_train_val=split
    rows=[];predictions=[]
    for name,quality,physical in [('full',q,wls),('no_quality_information',np.zeros_like(q),wls),('no_physical_information',q,np.zeros_like(wls)),('sequence_only_information',np.zeros_like(q),np.zeros_like(wls))]:
        print('Training',name,flush=True)
        m.set_seed(args.seed)
        result,pred=m.evaluate_model(name,sequences,meta,quality,physical,args,'direct')
        rows.append({'model':name,**result});predictions.append(pred)
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
        pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'completed':True,'input':cfg['input'],'seed':args.seed,'description':'Same direct-regression architecture; unavailable auxiliary inputs set to zero before scaling; constant gates/biases retained. Not a parameter-count ablation. Source-grouped inner/outer validation. Historical processed data retained.'},indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(out)

if __name__=='__main__':main()

