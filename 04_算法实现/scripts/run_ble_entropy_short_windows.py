"""PyCharm: matched-endpoint W8/W12/W16 sensitivity, fixed entropy and models."""
from pathlib import Path
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ble_cabin_entropy_study as base
from ble_entropy_features import dispersion_entropy, spde, self_test

ROOT=base.ROOT
OUT=ROOT/'SCI-UWB/data/results/ble_entropy_short_windows_20261105'
WINDOWS=[8,12,16]
GROUPS=['Stats','Stats_DE','Stats_SPDEraw','Stats_SPDEdc','EntropyOnly']


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=False)
    dump('unit_tests.json',self_test())
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    audit=subprocess.run([sys.executable,'E:/codex_skill/nature-figure/scripts/validate_figure.py',__file__,'--json'],
                         capture_output=True,encoding='utf-8',env=env,check=False)
    preflight=json.loads(audit.stdout);dump('figure_preflight.json',preflight)
    assert preflight['summary']['counts']['FAIL']==0,preflight
    assignment=base.prior.PRIOR/'window_assignments.csv'
    meta=pd.read_csv(assignment)
    data_path=ROOT/'ble_research/data/ble_static_ukf_86files_dataset.csv'
    data=pd.read_csv(data_path)
    lookup={s:p.set_index('sample_index')[base.prior.ANCHORS] for s,p in data.groupby('source_path')}
    tr=(meta.split=='train').to_numpy();va=(meta.split=='validation').to_numpy();y=meta.inside.to_numpy()
    protocol=dict(completed=False,interpreter=sys.executable,windows=WINDOWS,c=3,m=2,lag=1,stride=4,
                  alignment='Same source, original W16 endpoint and label; take trailing W records. Same12531 decisions each length.',
                  counts=meta.groupby(['split','inside']).size().reset_index(name='windows').to_dict('records'),
                  assignment_sha256=base.sha(assignment),dataset_sha256=base.sha(data_path),
                  runner_sha256=base.sha(Path(__file__)),entropy_sha256=base.sha(Path(__file__).with_name('ble_entropy_features.py')),
                  candidates='3 lengths x5 feature groups x2 fixed classifiers=30 fits. No optimization.',
                  selection='Validation Macro-F1 then balanced accuracy then lower outside FP; predefined order ties.',
                  limits=['Same-record single interior position; existing full-sequence UKF dependency.',
                          'Historical test already viewed; this is exploratory sensitivity, not blind confirmation.',
                          'W8 has7 DE embeddings for9 possible patterns; small-sample estimates remain uncertain.',
                          'Matched endpoints omit earlier startup decisions; not a standalone stream benchmark.',
                          'No timestamp conversion or full-cabin claim.'])
    dump('protocol.json',protocol)
    cache={};fitted={};metrics=[];predictions={};diagnostics=[]
    for width in WINDOWS:
        rich=[];entropy=[]
        aligned=meta.copy();aligned['start']=aligned.end-width+1;aligned['window']=width
        assert (aligned.start>=meta.start).all()
        aligned.to_csv(OUT/f'W{width}_assignments.csv',index=False)
        for number,row in enumerate(aligned.itertuples()):
            raw=lookup[row.source_path].loc[row.start:row.end].to_numpy(dtype=float)
            assert raw.shape==(width,4) and np.isfinite(raw).all()
            rich.append(base.prior.original.window_features(raw));record={}
            for j,anchor in enumerate(base.prior.ANCHORS):
                for method,fn in [('DE',dispersion_entropy),('SPDEraw',spde),('SPDEdc',lambda z:spde(z,demean_fft=True))]:
                    value,occupied,count=fn(raw[:,j]);record[f'{method}_{anchor}']=value
                    diagnostics.append(dict(window=width,row=number,split=row.split,inside=row.inside,
                                            anchor=anchor,method=method,occupied=occupied,pattern_rows=count,entropy=value))
            entropy.append(record)
            if (number+1)%4000==0:print('FEATURES',width,number+1,flush=True)
        rich=pd.DataFrame(rich);ent=pd.DataFrame(entropy)
        assert np.isfinite(ent.to_numpy()).all()
        ent.to_csv(OUT/f'W{width}_entropy.csv',index=False)
        de=[c for c in ent if c.startswith('DE_')];sr=[c for c in ent if c.startswith('SPDEraw_')];sd=[c for c in ent if c.startswith('SPDEdc_')]
        frames={'Stats':rich,'Stats_DE':pd.concat([rich,ent[de]],axis=1),
                'Stats_SPDEraw':pd.concat([rich,ent[sr]],axis=1),'Stats_SPDEdc':pd.concat([rich,ent[sd]],axis=1),
                'EntropyOnly':ent[de+sd]}
        cache[width]=frames
        for family in ['HGB','RF']:
            for group,frame in frames.items():
                name=f'W{width}_{family}_{group}'
                if family=='HGB':
                    model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,max_leaf_nodes=31,
                             min_samples_leaf=20,l2_regularization=0.,early_stopping=False,random_state=20261105)
                    model.fit(frame.loc[tr],y[tr],sample_weight=compute_sample_weight('balanced',y[tr]))
                else:
                    model=RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=20261105,n_jobs=2)
                    model.fit(frame.loc[tr],y[tr])
                p=model.predict_proba(frame.loc[va])[:,1]
                result=base.prior.measure(name,'validation',y[va],p)
                result.update(window=width,family=family,feature_group=group);metrics.append(result)
                fitted[name]=(model,width,group);predictions[name]={'validation':p}
                print('VALIDATION',name,'F1',result['macro_f1'],'FP/FN',result['fp'],result['fn'],flush=True)
        dump('validation_progress.json',metrics)
    rank=sorted(metrics,key=lambda r:(-r['macro_f1'],-r['balanced_accuracy'],r['outside_as_inside']))
    selected=rank[0]['model'];dump('selection_locked_before_test.json',dict(selected=selected,ranking=rank))
    te=(meta.split=='test').to_numpy()
    for name,(model,width,group) in fitted.items():
        p=model.predict_proba(cache[width][group].loc[te])[:,1]
        row=base.prior.measure(name,'test',y[te],p);row.update(window=width,family=name.split('_')[1],feature_group=group)
        metrics.append(row);predictions[name]['test']=p
        print('TEST',name,'F1',row['macro_f1'],'FP/FN',row['fp'],row['fn'],flush=True)
        for split,score in predictions[name].items():
            output=meta.loc[meta.split==split].copy();output.start=output.end-width+1
            output['inside_score']=score;output['pred_inside']=(score>=.5).astype(int)
            output.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
    joblib.dump(fitted[selected][0],OUT/'validation_selected_model.joblib')
    table=pd.DataFrame(metrics);table.to_csv(OUT/'metrics.csv',index=False);dump('metrics.json',metrics)
    old=pd.read_csv(base.OUT/'metrics.csv')
    for family in ['HGB','RF']:
        for group in GROUPS:
            for split in ['validation','test']:
                a=old[(old.model==family+'_'+group)&(old.split==split)].iloc[0]
                b=table[(table.model=='W16_'+family+'_'+group)&(table.split==split)].iloc[0]
                assert all(int(a[k])==int(b[k]) for k in ['tn','fp','fn','tp'])
    effects=[]
    for width in WINDOWS:
        for family in ['HGB','RF']:
            for split in ['validation','test']:
                anchor_row=table[(table.window==width)&(table.family==family)&(table.feature_group=='Stats')&(table.split==split)].iloc[0]
                base_pred=predictions[f'W{width}_{family}_Stats'][split]>=.5
                for group in GROUPS[1:]:
                    r=table[(table.window==width)&(table.family==family)&(table.feature_group==group)&(table.split==split)].iloc[0]
                    effects.append(dict(window=width,family=family,split=split,feature_group=group,
                                         delta_f1=float(r.macro_f1-anchor_row.macro_f1),
                                         changed_labels=int(np.sum((predictions[f'W{width}_{family}_{group}'][split]>=.5)!=base_pred)),
                                         delta_fp=int(r.fp-anchor_row.fp),delta_fn=int(r.fn-anchor_row.fn)))
    pd.DataFrame(effects).to_csv(OUT/'paired_entropy_effects.csv',index=False)
    diag=pd.DataFrame(diagnostics)
    diag.groupby(['window','split','inside','anchor','method']).agg(n=('entropy','size'),
         median_entropy=('entropy','median'),median_occupied=('occupied','median'),
         min_occupied=('occupied','min'),max_occupied=('occupied','max'),pattern_rows=('pattern_rows','first')).reset_index().to_csv(OUT/'entropy_occupancy_summary.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    test=table[table.split=='test'];fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
    colors=['#5F6D79','#86B0BA','#477F91','#235965','#C8AA78']
    for ax,family in zip(axes,['HGB','RF']):
        for group,color in zip(GROUPS,colors):
            part=test[(test.family==family)&(test.feature_group==group)].sort_values('window')
            ax.plot(part.window,part.macro_f1,marker='o',label=group,color=color)
        ax.set(title=family,xlabel='Trailing window length (samples)',ylabel='Test Macro-F1',xticks=WINDOWS,ylim=(0,1.02))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
    save(fig,'01_window_sensitivity')
    eff=pd.DataFrame(effects);eff=eff[(eff.split=='test')&(eff.feature_group!='EntropyOnly')]
    limit=max(.1,float(np.abs(eff.delta_f1).max()*100))
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.),layout='constrained')
    for ax,family in zip(axes,['HGB','RF']):
        matrix=eff[eff.family==family].pivot(index='window',columns='feature_group',values='delta_f1')
        matrix=matrix.reindex(index=WINDOWS,columns=GROUPS[1:4])*100
        ax.imshow(matrix.to_numpy(),vmin=-limit,vmax=limit,cmap='RdBu',aspect='auto')
        ax.set_xticks(range(3),['+DE','+SPDEraw','+SPDEdc']);ax.set_yticks(range(3),WINDOWS)
        ax.set(title=family,ylabel='Window length')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix.iloc[i,j]:+.2f}',ha='center',va='center')
    fig.suptitle('Test Macro-F1 change versus Stats, percentage points',fontsize=10)
    save(fig,'02_paired_entropy_gain')
    protocol.update(completed=True,selected=selected,fits=len(fitted),decisions_per_window=len(meta),w16_replay_passed=True)
    dump('protocol.json',protocol);print('COMPLETE',OUT,'SELECTED',selected,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
