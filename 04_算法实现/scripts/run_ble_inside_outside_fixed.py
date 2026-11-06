"""PyCharm: frozen single-interior-position feasibility experiment, not cross-seat validation."""
from pathlib import Path
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ble_research'))
import run_ble_static_learning_baselines as features
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix

OUT=ROOT/'SCI-UWB/data/results/ble_inside_outside_fixed_20261105'
ANCHORS=['3DD5','208A','230D','B5D7']
WINDOW,STEP,GAP=16,4,32

def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=False)
    data=pd.read_csv(ROOT/'ble_research/data/ble_static_ukf_86files_dataset.csv')
    sources=data[['source_path','direction']].drop_duplicates()
    assignment={}
    for direction,part in sources[sources.direction!='I'].groupby('direction'):
        ordered=sorted(part.source_path,key=lambda x:hashlib.sha256(('20261105|'+x).encode()).hexdigest())
        a,b=int(.6*len(ordered)),int(.8*len(ordered))
        for i,source in enumerate(ordered):assignment[source]='train' if i<a else ('validation' if i<b else 'test')
    records=[];xs=[];segments=[]
    for source,part in data.groupby('source_path',sort=True):
        part=part.sort_values('sample_index').reset_index(drop=True)
        assert part.sample_index.is_unique
        inside=part.direction.iloc[0]=='I'
        if inside:
            n=len(part);a,b=int(.6*n),int(.8*n)
            spans=[('train',0,a),('validation',a+GAP,b),('test',b+GAP,n)]
        else:spans=[(assignment[source],0,len(part))]
        for split,begin,end in spans:
            segments.append(dict(source=source,direction=part.direction.iloc[0],split=split,begin=begin,end_exclusive=end))
            for start in range(begin,end-WINDOW+1,STEP):
                block=part.iloc[start:start+WINDOW]
                xs.append(features.window_features(block[ANCHORS].to_numpy(dtype=float)))
                records.append(dict(source_path=source,direction=part.direction.iloc[0],split=split,
                                    inside=int(inside),start=int(block.sample_index.iloc[0]),end=int(block.sample_index.iloc[-1])))
    meta=pd.DataFrame(records);x=pd.DataFrame(xs)
    for split in ['train','validation','test']:assert set(meta.loc[meta.split==split,'inside'])=={0,1}
    outside_sets=[set(meta.loc[(meta.split==s)&(meta.inside==0),'source_path']) for s in ['train','validation','test']]
    assert not(outside_sets[0]&outside_sets[1] or outside_sets[0]&outside_sets[2] or outside_sets[1]&outside_sets[2])
    for left,right in [('train','validation'),('validation','test')]:
        assert meta.loc[(meta.split==left)&(meta.inside==1),'end'].max()<meta.loc[(meta.split==right)&(meta.inside==1),'start'].min()
    protocol=dict(completed=False,interpreter=sys.executable,seed=20261105,window=WINDOW,stride=STEP,gap_rows=GAP,
                  interior='User confirmed one fixed interior position, one source. Index blocks 60/20/20 with 32-row gaps.',
                  exterior='Whole source files assigned by fixed SHA256 order within direction; ratios60/20/20.',
                  preprocessing='Existing UKF-processed dataset; no new filtering. Historical full-sequence preprocessing dependence remains.',
                  restrictions='Exploratory same-record interior test, not independent sessions/seats/vehicles; no timestamp-based duration claims.',
                  features=list(x.columns),segments=segments,
                  counts=meta.groupby(['split','inside']).size().reset_index(name='windows').to_dict('records'))
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    meta.to_csv(OUT/'window_assignments.csv',index=False)
    train=meta.split=='train';metrics=[]
    models={'WKNN_k5':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5,weights='distance')),
            'RF_balanced':RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=20261105,n_jobs=-1)}
    for name,model in models.items():
        model.fit(x.loc[train],meta.loc[train,'inside'])
        for split in ['validation','test']:
            ids=meta.split==split;y=meta.loc[ids,'inside'].to_numpy();pred=model.predict(x.loc[ids])
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
            row=dict(model=name,split=split,n=len(y),accuracy=float(accuracy_score(y,pred)),
                     balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),
                     outside_as_inside=float(fp/(tn+fp)),inside_as_outside=float(fn/(tp+fn)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
            metrics.append(row)
            result=meta.loc[ids].copy();result['pred_inside']=pred
            result.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
            print(row,flush=True)
    (OUT/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    protocol['completed']=True
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
