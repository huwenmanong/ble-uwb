"""PyCharm: class-balanced histogram boosting, bounded GP-EI optimization."""
from pathlib import Path
import os
import sys
import json
import hashlib
import time
import subprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.utils.class_weight import compute_sample_weight
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_ble_cabin_literature_study as prior

ROOT = prior.ROOT
OUT = ROOT/'SCI-UWB/data/results/ble_cabin_boost_bayes_20261105'
OLD = ROOT/'SCI-UWB/data/results/ble_cabin_literature_20261105'


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def decode(z):
    return dict(learning_rate=float(.02*10**z[0]), max_leaf_nodes=int(round(7+24*z[1])),
                min_samples_leaf=int(round(10+50*z[2])), l2_regularization=float(11**z[3]-1),
                max_iter=int(round(60+160*z[4])))


def key(z):
    return json.dumps(decode(z), sort_keys=True)


def save_plot(fig, name):
    fig.savefig(OUT/(name+'.svg'), bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.chdir(ROOT); OUT.mkdir(parents=True, exist_ok=False)
    preflight = subprocess.run([sys.executable, 'E:/codex_skill/nature-figure/scripts/validate_figure.py',
                               __file__, '--json'], capture_output=True, encoding='utf-8',
                              env=dict(os.environ, PYTHONIOENCODING='utf-8'), check=True)
    (OUT/'figure_preflight.json').write_text(preflight.stdout, encoding='utf-8')
    assignments = prior.PRIOR/'window_assignments.csv'
    meta = pd.read_csv(assignments)
    source = ROOT/'ble_research/data/ble_static_ukf_86files_dataset.csv'
    data = pd.read_csv(source)
    lookup = {s:p.set_index('sample_index')[prior.ANCHORS] for s,p in data.groupby('source_path')}
    features = []
    for row in meta.itertuples():
        raw = lookup[row.source_path].loc[row.start:row.end].to_numpy(dtype=float)
        assert raw.shape == (16,4) and np.isfinite(raw).all()
        features.append(prior.original.window_features(raw))
    x = pd.DataFrame(features); y = meta.inside.to_numpy()
    tr = (meta.split=='train').to_numpy(); va=(meta.split=='validation').to_numpy()
    weights = compute_sample_weight('balanced', y[tr])
    protocol = dict(completed=False, interpreter=sys.executable, sklearn=sklearn.__version__, seed=20261105,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    assignments_sha256=hashlib.sha256(assignments.read_bytes()).hexdigest(),
                    dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), features=list(x.columns),
                    training='Train only, balanced sample weights, early_stopping=False, CPU threads2.',
                    objective='Validation Macro-F1; ties balanced accuracy then lower outside false-positive rate.',
                    budget='6 shared initial trials (including default), then12 GP-EI and12 random;18 each,30 unique fits.',
                    threshold=.5, split='Exact prior fixed assignments; no new grouping.',
                    space='lr log .02-.2; leaves7-31; minleaf10-60; l2 log1p0-10; iterations60-220.',
                    optimizer='GP fixed Matern nu2.5, lengthscale.4, normalized target, alpha1e-4; EI xi.001;512 candidates.',
                    limits=['Single interior position, index blocks not independent acquisitions.',
                            'Upstream full-sequence UKF dependency remains.',
                            'Historical test exposure; not blinded evaluation.',
                            'One optimizer seed, fixed validation used repeatedly.'])
    dump('protocol.json', protocol)
    trials=[]; fitted={}; observed=[]

    def evaluate(z, arm):
        params=decode(z); name=f'trial{len(trials):02d}'; begin=time.perf_counter()
        model=HistGradientBoostingClassifier(**params, early_stopping=False, random_state=20261105)
        model.fit(x.loc[tr], y[tr], sample_weight=weights)
        p=model.predict_proba(x.loc[va])[:,1]
        row=prior.measure(name, 'validation', y[va], p)
        row.update(arm=arm, parameters=params, seconds=time.perf_counter()-begin)
        trials.append(row); observed.append(np.asarray(z)); fitted[name]=model
        dump('trials.json', trials)
        print('TRIAL', name, arm, 'F1', row['macro_f1'], 'FP/FN', row['fp'],row['fn'], flush=True)
        return len(trials)-1

    rng=np.random.default_rng(20261105)
    default=np.array([np.log10(.1/.02),1.,.2,0.,.25])
    common=[evaluate(default,'shared_default')]
    for _ in range(5): common.append(evaluate(rng.uniform(size=5),'shared_initial'))
    bo=list(common)
    for _ in range(12):
        train_z=np.array([observed[i] for i in bo]); target=np.array([trials[i]['macro_f1'] for i in bo])
        gp=GaussianProcessRegressor(kernel=ConstantKernel(1.,constant_value_bounds='fixed')*
                                   Matern(length_scale=np.full(5,.4),length_scale_bounds='fixed',nu=2.5),
                                   alpha=1e-4, optimizer=None, normalize_y=True, random_state=20261105)
        gp.fit(train_z,target)
        pool=rng.uniform(size=(512,5)); mu,sd=gp.predict(pool,return_std=True)
        improvement=mu-target.max()-.001; safe=np.maximum(sd,1e-12); z=improvement/safe
        ei=improvement*ndtr(z)+safe*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        seen={key(observed[i]) for i in bo}
        for candidate in np.argsort(-ei):
            if key(pool[candidate]) not in seen:
                bo.append(evaluate(pool[candidate],'bayes_ei')); break
        else: raise RuntimeError('No unique BO candidate')
    random=list(common); random_rng=np.random.default_rng(20261106)
    for _ in range(12):
        candidate=random_rng.uniform(size=5)
        while key(candidate) in {key(observed[i]) for i in random}:
            candidate=random_rng.uniform(size=5)
        random.append(evaluate(candidate,'random'))

    def best(ids):
        return sorted(ids,key=lambda i:(-trials[i]['macro_f1'],-trials[i]['balanced_accuracy'],trials[i]['outside_as_inside'],i))[0]
    selected={'HGB_fixed':common[0], 'HGB_Bayes':best(bo), 'HGB_Random':best(random)}
    dump('selection_locked_before_test.json',dict(selected=selected,parameters={n:trials[i]['parameters'] for n,i in selected.items()}))
    test=(meta.split=='test').to_numpy(); metrics=[]
    for name,i in selected.items():
        model=fitted[trials[i]['model']]
        for split,ids in [('validation',va),('test',test)]:
            p=model.predict_proba(x.loc[ids])[:,1]
            row=prior.measure(name,split,y[ids],p);metrics.append(row)
            output=meta.loc[ids].copy();output['inside_score']=p;output['pred_inside']=(p>=.5).astype(int)
            output.to_csv(OUT/f'{name}_{split}_predictions.csv',index=False)
            print('RESULT',row,flush=True)
        joblib.dump(model,OUT/f'{name}.joblib')
    old=pd.read_csv(OLD/'metrics.csv')
    old=old[(old.split=='test')&old.model.isin(['Rich_WKNN','Balanced_SVM'])]
    combined=pd.concat([pd.DataFrame(metrics),old],ignore_index=True)
    combined.to_csv(OUT/'metrics.csv',index=False);dump('metrics.json',metrics)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':8,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,3.2),layout='constrained')
    for label,ids,color in [('Bayesian EI',bo,'#287D8E'),('Random',random,'#7D8590')]:
        ax.plot(range(1,19),np.maximum.accumulate([trials[i]['macro_f1'] for i in ids]),label=label,color=color,marker='o',markersize=3)
    ax.set(xlabel='Validation evaluations (6 shared initial)',ylabel='Best validation Macro-F1',xticks=range(2,19,2))
    ax.legend(frameon=False);save_plot(fig,'01_search_history')
    table=combined[combined.split=='test'];fig,axes=plt.subplots(1,3,figsize=(9.,3.1),layout='constrained')
    for ax,column,title in zip(axes,['macro_f1','outside_as_inside','inside_as_outside'],
                               ['Macro-F1','Outside to inside (%)','Inside to outside (%)']):
        vals=table[column].to_numpy()*(1 if column=='macro_f1' else 100)
        ax.bar(np.arange(len(table)),vals,color=['#A0ADB8','#287D8E','#C0CED4','#A0ADB8','#A0ADB8'])
        ax.set_xticks(range(len(table)),list(table.model),rotation=55,ha='right');ax.set_title(title,fontsize=9)
        ax.set_ylim(0,1.05 if column=='macro_f1' else max(1,float(vals.max())*1.25))
        for j,v in enumerate(vals):ax.text(j,v,f'{v:.3f}' if column=='macro_f1' else f'{v:.2f}',ha='center',va='bottom',fontsize=7)
    save_plot(fig,'02_test_comparison')
    protocol.update(completed=True,selected=selected,total_trials=len(trials))
    dump('protocol.json',protocol); print('COMPLETE',OUT,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2): main()
