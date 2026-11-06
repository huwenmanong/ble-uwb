"""PyCharm: ordinary histogram Shannon entropy, two fixed binning definitions."""
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

ROOT=base.ROOT
OUT=ROOT/'SCI-UWB/data/results/ble_shannon_entropy_20261105'
WINDOWS=[8,12,16]
GROUPS=['Stats','Stats_LocalShannon','Stats_TrainShannon','ShannonOnly']


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def count_entropy(counts,normalizer=4):
    counts=np.asarray(counts,dtype=float)
    if counts.sum()<=0 or np.any(counts<0):raise ValueError('Invalid counts')
    p=counts[counts>0]/counts.sum()
    return float(-np.sum(p*np.log2(p))/np.log2(normalizer))


def entropy(x,edges=None):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid signal')
    if edges is None:
        if np.ptp(x)<=1e-12:return 0.
        counts=np.histogram(x,bins=4,range=(float(x.min()),float(x.max())))[0]
        return count_entropy(counts)
    counts=np.histogram(x,bins=edges)[0]
    if len(counts)==1:return 0.
    return count_entropy(counts,normalizer=len(counts))


def self_test():
    assert np.isclose(count_entropy([1,1,1,1]),1.)
    assert np.isclose(count_entropy([4,0,0,0]),0.)
    assert np.isclose(entropy(np.full(8,-70.)),0.)
    # Deterministic unit-test fixtures only, never added to observed data.
    fixture=np.array([-80.,-79.,-76.,-75.,-72.,-71.,-68.,-67.])
    assert np.isclose(entropy(fixture),entropy(fixture[::-1]))
    assert np.isclose(entropy(fixture),entropy(2*fixture+10))
    assert np.isclose(entropy(fixture,np.array([-np.inf,-77.,-73.,-69.,np.inf])),1.)
    return dict(passed=True,checks=['Uniform four-bin entropy1','Single occupied bin entropy0',
          'Constant signal0','Order invariance','Local equal-width positive affine invariance',
          'Fixed-bin endpoint coverage'],note='Normalized histogram Shannon entropy, not differential or spectral entropy.')


def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)


def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=False);dump('unit_tests.json',self_test())
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    check=subprocess.run([sys.executable,'E:/codex_skill/nature-figure/scripts/validate_figure.py',__file__,'--json'],
                         capture_output=True,encoding='utf-8',env=env,check=False)
    qa=json.loads(check.stdout);dump('figure_preflight.json',qa)
    assert qa['summary']['counts']['FAIL']==0,qa
    assignment=base.prior.PRIOR/'window_assignments.csv';meta=pd.read_csv(assignment)
    data_path=ROOT/'ble_research/data/ble_static_ukf_86files_dataset.csv';data=pd.read_csv(data_path)
    lookup={s:p.set_index('sample_index')[base.prior.ANCHORS] for s,p in data.groupby('source_path')}
    tr=(meta.split=='train').to_numpy();va=(meta.split=='validation').to_numpy();y=meta.inside.to_numpy()
    # Fit shared bin edges on unique raw rows covered by original training windows only.
    train_rows=[]
    for source,part in meta.loc[tr].groupby('source_path'):
        indices=np.unique(np.concatenate([np.arange(r.start,r.end+1) for r in part.itertuples()]))
        train_rows.append(lookup[source].loc[indices].to_numpy(dtype=float))
    fit_values=np.vstack(train_rows);edges=[]
    for j in range(4):
        cuts=np.unique(np.quantile(fit_values[:,j],[.25,.5,.75]))
        edges.append(np.concatenate([[-np.inf],cuts,[np.inf]]))
    dump('training_bins.json',dict(unique_training_rows=len(fit_values),
         boundaries={a:dict(interior_cuts=e[1:-1].tolist(),outer_edges='-inf,+inf',bins=len(e)-1)
                     for a,e in zip(base.prior.ANCHORS,edges)}))
    protocol=dict(completed=False,interpreter=sys.executable,windows=WINDOWS,seed=20261105,
                  assignments_sha256=base.sha(assignment),dataset_sha256=base.sha(data_path),script_sha256=base.sha(Path(__file__)),
                  method='H=-sum(p log2p)/log2(B). Local4 equal-width per window; shared train quantile bins, duplicate cuts collapse.',
                  alignment='Same original W16 endpoints and12531 decisions for each length; stride4.',
                  bins_fit='Only unique raw rows covered by original train windows; no validation/test values.',
                  fits_planned=24,parameters='Unchanged fixed HGB and RF; threshold.5; no optimization.',
                  selection='Validation Macro-F1, balanced accuracy, lower outside FP, then predefined order.',
                  limits=['Single interior position and same-record splits; upstream full-sequence UKF dependence.',
                          'Repeated exploratory use of historical test data; not blinded validation.',
                          'Histogram entropy ignores ordering and depends on chosen bins.'])
    dump('protocol.json',protocol);cache={};models={};predictions={};metrics=[];diagnostics=[]
    for width in WINDOWS:
        rich=[];features=[]
        for i,row in enumerate(meta.itertuples()):
            raw=lookup[row.source_path].loc[row.end-width+1:row.end].to_numpy(dtype=float)
            assert raw.shape==(width,4) and np.isfinite(raw).all()
            rich.append(base.prior.original.window_features(raw));record={}
            for j,anchor in enumerate(base.prior.ANCHORS):
                record['Local_'+anchor]=entropy(raw[:,j]);record['Train_'+anchor]=entropy(raw[:,j],edges[j])
            features.append(record)
        rich=pd.DataFrame(rich);ent=pd.DataFrame(features)
        assert np.isfinite(ent.to_numpy()).all() and ((ent.to_numpy()>=0)&(ent.to_numpy()<=1+1e-12)).all()
        ent.to_csv(OUT/f'W{width}_shannon.csv',index=False)
        for split in ['train','validation','test']:
            for col in ent:
                vals=ent.loc[meta.split==split,col]
                diagnostics.append(dict(window=width,split=split,feature=col,n=len(vals),zero_fraction=float((vals==0).mean()),
                                        median=float(vals.median()),unique_values=int(vals.nunique())))
        frames={'Stats':rich,'Stats_LocalShannon':pd.concat([rich,ent.filter(regex='^Local_')],axis=1),
                'Stats_TrainShannon':pd.concat([rich,ent.filter(regex='^Train_')],axis=1),'ShannonOnly':ent}
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
                p=model.predict_proba(frame.loc[va])[:,1];models[name]=(model,width,group,family)
                row=base.prior.measure(name,'validation',y[va],p);row.update(window=width,family=family,feature_group=group)
                metrics.append(row);predictions[name]={'validation':p}
                print('VALIDATION',name,row['macro_f1'],'FP/FN',row['fp'],row['fn'],flush=True)
        dump('validation_progress.json',metrics)
    ranking=sorted(metrics,key=lambda r:(-r['macro_f1'],-r['balanced_accuracy'],r['outside_as_inside']))
    selected=ranking[0]['model'];dump('selection_locked_before_test.json',dict(selected=selected,ranking=ranking))
    te=(meta.split=='test').to_numpy()
    for name,(model,width,group,family) in models.items():
        p=model.predict_proba(cache[width][group].loc[te])[:,1];predictions[name]['test']=p
        row=base.prior.measure(name,'test',y[te],p);row.update(window=width,family=family,feature_group=group);metrics.append(row)
        print('TEST',name,row['macro_f1'],'FP/FN',row['fp'],row['fn'],flush=True)
        for split,score in predictions[name].items():
            output=meta.loc[meta.split==split].copy();output.start=output.end-width+1
            output['inside_score']=score;output['pred_inside']=(score>=.5).astype(int)
            output.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
    table=pd.DataFrame(metrics);table.to_csv(OUT/'metrics.csv',index=False);dump('metrics.json',metrics)
    pd.DataFrame(diagnostics).to_csv(OUT/'entropy_diagnostics.csv',index=False)
    old=pd.read_csv(ROOT/'SCI-UWB/data/results/ble_entropy_short_windows_20261105/metrics.csv')
    for name in models:
        if name.endswith('_Stats'):
            for split in ['validation','test']:
                a=old[(old.model==name)&(old.split==split)].iloc[0]
                b=table[(table.model==name)&(table.split==split)].iloc[0]
                assert all(int(a[k])==int(b[k]) for k in ['tn','fp','fn','tp'])
    effects=[]
    for width in WINDOWS:
        for family in ['HGB','RF']:
            baseline=predictions[f'W{width}_{family}_Stats']['test']>=.5
            a=table[(table.window==width)&(table.family==family)&(table.feature_group=='Stats')&(table.split=='test')].iloc[0]
            for group in GROUPS[1:]:
                b=table[(table.window==width)&(table.family==family)&(table.feature_group==group)&(table.split=='test')].iloc[0]
                effects.append(dict(window=width,family=family,feature_group=group,delta_f1=float(b.macro_f1-a.macro_f1),
                                    changed_labels=int(np.sum((predictions[f'W{width}_{family}_{group}']['test']>=.5)!=baseline)),
                                    delta_fp=int(b.fp-a.fp),delta_fn=int(b.fn-a.fn)))
    pd.DataFrame(effects).to_csv(OUT/'paired_effects.csv',index=False)
    joblib.dump(models[selected][0],OUT/'validation_selected_model.joblib')
    meta.to_csv(OUT/'original_endpoint_assignments.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained');test=table[table.split=='test']
    colors=['#697784','#609DAA','#245E6E','#C2A16E']
    for ax,family in zip(axes,['HGB','RF']):
        for group,color in zip(GROUPS,colors):
            part=test[(test.family==family)&(test.feature_group==group)].sort_values('window')
            ax.plot(part.window,part.macro_f1,marker='o',color=color,label=group)
        ax.set(title=family,xlabel='Trailing window length',ylabel='Test Macro-F1',xticks=WINDOWS,ylim=(0,1.02))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,'01_shannon_comparison')
    protocol.update(completed=True,fits=len(models),selected=selected,decisions_per_window=len(meta),baseline_replay_passed=True)
    dump('protocol.json',protocol);print('COMPLETE',OUT,'SELECTED',selected,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
