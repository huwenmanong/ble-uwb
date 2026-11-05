"""Audit historical PQG inner-window overlap; no training or raw edits."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from run_uwb_static_deep_learning import build_sequence_windows
from run_uwb_physics_quality_guided_cnn import split_train_val

def main():
    config_path=ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json'
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    source=ROOT/cfg['input']
    data=pd.read_csv(source)
    _,meta=build_sequence_windows(data,cfg['window'],cfg['step'])
    groups=meta.source_file.astype(str).to_numpy()
    out=ROOT/'SCI-UWB/data/results/vehicle_uwb_split_audit_20261030'
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    manifests=[]
    for fold,(train,test) in enumerate(GroupKFold(5).split(meta,groups=groups),1):
        oldtrain,oldval=split_train_val(train,cfg['seed']+fold)
        inner=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=cfg['seed']+fold)
        a,b=next(inner.split(train,groups=groups[train]))
        newtrain,newval=train[a],train[b]
        for mode,tr,va in [('historical_random_windows',oldtrain,oldval),('proposed_source_grouped',newtrain,newval)]:
            overlap=0
            for idx in va:
                row=meta.iloc[idx]
                candidates=meta.iloc[tr]
                hits=candidates[(candidates.source_file==row.source_file)&(candidates.window_start<=row.window_end)&(candidates.window_end>=row.window_start)]
                overlap+=int(len(hits)>0)
            shared=len(set(groups[tr])&set(groups[va]))
            assert not (set(groups[train])&set(groups[test]))
            if mode=='proposed_source_grouped':
                assert shared==0 and overlap==0
            records.append(dict(fold=fold,mode=mode,train_windows=len(tr),validation_windows=len(va),test_windows=len(test),shared_train_validation_sources=shared,validation_windows_overlapping_train=overlap,overlap_fraction=overlap/len(va)))
            for role,ids in [('train',tr),('validation',va),('test',test)]:
                m=meta.iloc[ids].copy();m['fold']=fold;m['mode']=mode;m['role']=role
                manifests.append(m)
    result=pd.DataFrame(records)
    result.to_csv(out/'split_audit.csv',index=False)
    pd.concat(manifests).to_csv(out/'window_split_manifest.csv',index=False)
    data.groupby(['source_file','scene','position_label']).size().rename('rows').to_csv(out/'source_inventory.csv')
    (out/'provenance.json').write_text(json.dumps({'input':str(source),'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'configuration':cfg,'source_count':len(set(groups)),'windows':len(meta),'limitation':'Source grouping does not prove augmented-parent independence; no models retrained.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(result.to_string(index=False));print('Sources',len(set(groups)),'Windows',len(meta));print(out)

if __name__=='__main__':main()

