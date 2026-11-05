"""Controlled rerun: same processed data/models, source-grouped inner validation.
Historical source is copied into the run directory; no original code is edited.
"""
from pathlib import Path
import sys,json,hashlib,os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as model

def main():
    os.chdir(ROOT)
    old=ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results'
    cfg=json.loads((old/'physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    stem=ROOT/'SCI-UWB/data/results/vehicle_uwb_grouped_control_20261030'
    out=stem;i=0
    while out.exists():
        i+=1;out=stem.with_name(stem.name+f'_repeat{i:02d}')
    out.mkdir(parents=True)
    source=Path(model.__file__)
    (out/'original_model_source.py').write_bytes(source.read_bytes())
    data=pd.read_csv(ROOT/cfg['input'],encoding='utf-8-sig')
    _,meta=model.build_sequence_windows(data,cfg['window'],cfg['step'])
    groups=meta.source_file.astype(str).to_numpy()
    splits=[]
    def grouped_split(train_idx,seed,val_fraction=.2):
        split=GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed)
        a,b=next(split.split(train_idx,groups=groups[train_idx]))
        tr,va=train_idx[a],train_idx[b]
        assert not(set(groups[tr])&set(groups[va]))
        for role,ids in [('train',tr),('validation',va)]:
            d=meta.iloc[ids].copy();d['role']=role;d['split_seed']=seed;splits.append(d)
        pd.concat(splits).to_csv(out/'inner_split_audit.csv',index=False)
        return tr,va
    model.split_train_val=grouped_split
    argv=['grouped-control','--input',str(ROOT/cfg['input']),'--output',str(out),'--baseline',str(out/'no_historical_baseline.csv')]
    for key in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:
        argv += ['--'+key.replace('_','-'),str(cfg[key])]
    sys.argv=argv
    (out/'control_protocol.json').write_text(json.dumps({'completed':False,'intervention':'Inner source-grouped validation only','data_caveat':'Existing repaired/processed data retained; not a raw-measurement benchmark','historical_config':cfg,'original_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf-8')
    print('CONTROL OUTPUT:',out,flush=True)
    model.main()
    new=pd.read_csv(out/'physics_quality_guided_cnn_metrics.csv')
    prior=pd.read_csv(old/'physics_quality_guided_cnn_metrics.csv')
    prior.merge(new,on='model',suffixes=('_historical','_grouped'),validate='one_to_one').to_csv(out/'historical_vs_grouped.csv',index=False)
    p=json.loads((out/'control_protocol.json').read_text(encoding='utf-8'));p['completed']=True
    (out/'control_protocol.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETED:',out,flush=True)

if __name__=='__main__':main()

