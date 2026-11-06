"""PyCharm: frozen feature ablations, no hyperparameter or window search."""
from pathlib import Path
import json
import hashlib
import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ble_cabin_literature_study as prior
from ble_entropy_features import dispersion_entropy, spde, self_test

ROOT = prior.ROOT
OUT = ROOT/'SCI-UWB/data/results/ble_cabin_entropy_20261105_r1'


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2),encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=False)
    dump('unit_tests.json',self_test())
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,'E:/codex_skill/nature-figure/scripts/validate_figure.py',
                           __file__,'--json'],capture_output=True,encoding='utf-8',env=env,check=False)
    audit=json.loads(result.stdout);dump('figure_preflight.json',audit)
    print('FIGURE_PREFLIGHT',audit['summary'],flush=True)
    for finding in audit['findings']:
        if finding['level']=='FAIL':print('PREFLIGHT_FAIL',finding,flush=True)
    assert audit['summary']['counts']['FAIL']==0, 'Figure source preflight failed; see saved JSON'
    assignments=prior.PRIOR/'window_assignments.csv'
    dataset=ROOT/'ble_research/data/ble_static_ukf_86files_dataset.csv'
    meta=pd.read_csv(assignments);data=pd.read_csv(dataset)
    lookup={s:p.set_index('sample_index')[prior.ANCHORS] for s,p in data.groupby('source_path')}
    protocol=dict(completed=False,interpreter=sys.executable,seed=20261105,window=16,stride=4,c=3,m=2,lag=1,
                  assignments_sha256=sha(assignments),dataset_sha256=sha(dataset),
                  entropy_module_sha256=sha(Path(__file__).with_name('ble_entropy_features.py')),
                  runner_sha256=sha(Path(__file__)),
                  matlab_sha256={p.name:sha(p) for p in (ROOT/'岚的熵').glob('*.m')},
                  split='Exact prior window list, no new grouping.',
                  variants=['Stats','Stats_DE','Stats_SPDEraw','Stats_SPDEdc','EntropyOnly'],
                  classifiers='HGB fixed100,.1,31,20,l2=0; RF400 balanced; both seed20261105.',
                  selection='Validation Macro-F1 then balanced accuracy then lower FP, fixed order ties; test after lock.',
                  restrictions=['Single interior position, same-record index splits.',
                                'Full-sequence UKF preprocessing dependency remains.',
                                'No multiscale, entropy parameter, threshold or classifier parameter search.',
                                'No MATLAB parity run; source-inspired Python implementation.',
                                'Previously explored test set, not fresh blinded validation.'])
    dump('protocol.json',protocol)
    rich=[];entropy=[];diagnostics=[];elapsed=[]
    for number,row in enumerate(meta.itertuples()):
        raw=lookup[row.source_path].loc[row.start:row.end].to_numpy(dtype=float)
        assert raw.shape==(16,4) and np.isfinite(raw).all()
        rich.append(prior.original.window_features(raw));record={}
        begin=time.perf_counter_ns()
        for j,anchor in enumerate(prior.ANCHORS):
            for name,fn in [('DE',dispersion_entropy),('SPDEraw',spde),
                            ('SPDEdc',lambda z:spde(z,demean_fft=True))]:
                value,occupied,patterns=fn(raw[:,j])
                record[f'{name}_{anchor}']=value
                diagnostics.append(dict(window=number,anchor=anchor,method=name,entropy=value,
                                        occupied_patterns=occupied,pattern_rows=patterns,
                                        constant=bool(np.std(raw[:,j],ddof=1)<=1e-12)))
        elapsed.append((time.perf_counter_ns()-begin)/1e6);entropy.append(record)
        if (number+1)%2500==0: print('FEATURE_WINDOWS',number+1,'/',len(meta),flush=True)
    rich=pd.DataFrame(rich);ent=pd.DataFrame(entropy)
    assert np.isfinite(ent.to_numpy()).all() and ((ent.to_numpy()>=0)&(ent.to_numpy()<=1+1e-12)).all()
    ent.to_csv(OUT/'entropy_features.csv',index=False)
    meta.to_csv(OUT/'window_assignments.csv',index=False)
    diag=pd.DataFrame(diagnostics);diag.to_csv(OUT/'entropy_diagnostics.csv',index=False)
    de=[c for c in ent if c.startswith('DE_')];spraw=[c for c in ent if c.startswith('SPDEraw_')]
    spdc=[c for c in ent if c.startswith('SPDEdc_')]
    variants={'Stats':rich,'Stats_DE':pd.concat([rich,ent[de]],axis=1),
              'Stats_SPDEraw':pd.concat([rich,ent[spraw]],axis=1),
              'Stats_SPDEdc':pd.concat([rich,ent[spdc]],axis=1),'EntropyOnly':ent[de+spdc]}
    tr=(meta.split=='train').to_numpy();va=(meta.split=='validation').to_numpy();y=meta.inside.to_numpy()
    weights=compute_sample_weight('balanced',y[tr]);models={};metrics=[];predictions={}
    for family in ['HGB','RF']:
        for variant,frame in variants.items():
            name=f'{family}_{variant}'
            if family=='HGB':
                model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.1,max_leaf_nodes=31,
                         min_samples_leaf=20,l2_regularization=0.,early_stopping=False,random_state=20261105)
                model.fit(frame.loc[tr],y[tr],sample_weight=weights)
            else:
                model=RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=20261105,n_jobs=2)
                model.fit(frame.loc[tr],y[tr])
            p=model.predict_proba(frame.loc[va])[:,1];models[name]=(model,variant)
            row=prior.measure(name,'validation',y[va],p);metrics.append(row)
            predictions[name]={'validation':p};print('VALIDATION',row,flush=True)
    ranked=sorted(metrics,key=lambda r:(-r['macro_f1'],-r['balanced_accuracy'],r['outside_as_inside']))
    selected=ranked[0]['model'];dump('selection_locked_before_test.json',dict(selected=selected,validation=metrics))
    te=(meta.split=='test').to_numpy()
    for name,(model,variant) in models.items():
        p=model.predict_proba(variants[variant].loc[te])[:,1]
        row=prior.measure(name,'test',y[te],p);metrics.append(row);predictions[name]['test']=p
        print('TEST',row,flush=True);joblib.dump(model,OUT/f'{name}.joblib')
    for name,split_scores in predictions.items():
        for split,p in split_scores.items():
            output=meta.loc[meta.split==split].copy();output['inside_score']=p;output['pred_inside']=(p>=.5).astype(int)
            output.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
    table=pd.DataFrame(metrics);table.to_csv(OUT/'metrics.csv',index=False);dump('metrics.json',metrics)
    for family,folder,old_name in [('HGB','ble_cabin_boost_bayes_20261105','HGB_fixed'),
                                   ('RF','ble_cabin_literature_20261105','Balanced_RF')]:
        old=pd.read_csv(ROOT/f'SCI-UWB/data/results/{folder}/metrics.csv')
        for split in ['validation','test']:
            a=old[(old.model==old_name)&(old.split==split)].iloc[0]
            b=table[(table.model==family+'_Stats')&(table.split==split)].iloc[0]
            assert all(int(a[k])==int(b[k]) for k in ['tn','fp','fn','tp'])
    # Train-only univariate association; never used for feature selection.
    assoc=[]
    for col in ent:
        auc=roc_auc_score(y[tr],ent.loc[tr,col])
        assoc.append(dict(feature=col,train_auc=float(auc),direction_free_auc=float(max(auc,1-auc)),
                          train_inside_median=float(ent.loc[tr&(y==1),col].median()),
                          train_outside_median=float(ent.loc[tr&(y==0),col].median())))
    pd.DataFrame(assoc).to_csv(OUT/'train_entropy_association.csv',index=False)
    dump('extraction_runtime.json',dict(windows=len(meta),entropy_dimensions=12,median_ms=float(np.median(elapsed)),
                                       p95_ms=float(np.percentile(elapsed,95)),scope='12 entropy features only, data ready in RAM.'))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.,3.9),layout='constrained')
    test=table[table.split=='test']
    for ax,col,title in zip(axes,['macro_f1','outside_as_inside','inside_as_outside'],
                            ['Macro-F1','Outside to inside (%)','Inside to outside (%)']):
        vals=test[col].to_numpy()*(1 if col=='macro_f1' else 100)
        ax.barh(range(len(test)),vals,color=['#A0ADB8','#77ACB5','#448E9C','#287D8E','#C3CDD5']*2)
        ax.set_yticks(range(len(test)),list(test.model));ax.invert_yaxis();ax.set_title(title,fontsize=9)
        ax.set_xlim(0,1.15 if col=='macro_f1' else max(1,float(vals.max())*1.22))
        for j,v in enumerate(vals):ax.text(v,j,f' {v:.3f}' if col=='macro_f1' else f' {v:.2f}',va='center',fontsize=7)
    fig.suptitle('Fixed feature ablations; outside n=2924, inside n=96 test windows',fontsize=10)
    save(fig,'01_feature_ablation')
    fig,axes=plt.subplots(1,3,figsize=(9.,3.),layout='constrained')
    for ax,method in zip(axes,['DE','SPDEraw','SPDEdc']):
        positions=[];samples=[]
        for j,a in enumerate(prior.ANCHORS):
            for label in [0,1]:
                positions.append(j*3+label);samples.append(ent.loc[tr&(y==label),f'{method}_{a}'].to_numpy())
        bp=ax.boxplot(samples,positions=positions,widths=.65,showfliers=False,patch_artist=True)
        for j,box in enumerate(bp['boxes']):box.set_facecolor(['#A0ADB8','#287D8E'][j%2])
        ax.set_xticks(np.arange(4)*3+.5,prior.ANCHORS,rotation=30);ax.set_ylim(0,1.05);ax.set_title(method)
        ax.set_ylabel('Normalized entropy')
    fig.suptitle('Training windows only: gray outside, teal inside; outlier marks hidden, data retained',fontsize=9)
    save(fig,'02_train_entropy_distributions')
    protocol.update(completed=True,selected=selected,models=len(models),baseline_replay_passed=True,
                    windows=len(meta),feature_dimensions={k:v.shape[1] for k,v in variants.items()},
                    constant_anchor_windows=int(diag[diag.method=='DE'].constant.sum()))
    dump('protocol.json',protocol);print('COMPLETE',OUT,'SELECTED',selected,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
