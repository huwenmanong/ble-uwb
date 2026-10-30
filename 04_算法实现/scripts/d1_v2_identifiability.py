"""Exploratory LOSO error-detection test, including stable/consistent links.

The subset is defined using deployable raw-window std and LS residual only.
Large-error labels use truth offline; no label is in the feature matrix.
All classifier settings are fixed, without target-domain threshold selection.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_d1_loso_uncertainty_model import BASE_NUMERIC_FEATURES,CIR_NUMERIC_FEATURES,ENTROPY_NUMERIC_FEATURES

ROOT=Path(__file__).resolve().parents[2]


def scores(y,p):
    return {'n':len(y),'positive_rate':float(np.mean(y)),
            'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan'),
            'average_precision':float(average_precision_score(y,p)) if np.any(y) else float('nan')}


def main(run_dir=None):
    if run_dir is None:
        parser=argparse.ArgumentParser()
        parser.add_argument('--run-dir',type=Path,default=ROOT/'data/results/d1_v2_20261029_seed20261029')
        run_dir=parser.parse_args().run_dir
    out=Path(run_dir)
    data=pd.read_csv(out/'hidden_bias_links.csv')
    data['abs_geometry_residual']=data.geometry_residual_m.abs()
    geometry=['range_measured','range_std','sample_n','abs_geometry_residual']
    scalar=geometry+[c for c in BASE_NUMERIC_FEATURES if c not in geometry and not c.startswith('anchor_')]
    variants={'geometry':geometry,'scalar':scalar,'cir':scalar+CIR_NUMERIC_FEATURES,
              'cir_entropy':scalar+CIR_NUMERIC_FEATURES+ENTROPY_NUMERIC_FEATURES}
    start=time.perf_counter()
    results,predictions=[],[]
    for fold in sorted(data.environment_id.unique()):
        train=data[f'split_{fold}'].eq('train')
        test=data[f'split_{fold}'].eq('test')
        target=data.loc[test]
        y=target.large_error.astype(int).to_numpy()
        for variant,features in variants.items():
            med=data.loc[train,features].median().fillna(0)
            model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=40,
                learning_rate=.06,l2_regularization=1.,early_stopping=False,random_state=20261029)
            model.fit(data.loc[train,features].fillna(med),data.loc[train,'large_error'].astype(int))
            p=model.predict_proba(target[features].fillna(med))[:,1]
            for subset,mask in [('all',np.ones(len(y),bool)),('stable_consistent',target.stable_consistent.to_numpy(bool))]:
                results.append(dict(fold=fold,variant=variant,subset=subset,**scores(y[mask],p[mask])))
            frame=target[['position_id','raw_folder','channel','anchor_label','stable_consistent','large_error']].copy()
            frame['fold'],frame['variant'],frame['probability']=fold,variant,p
            predictions.append(frame)
    result=pd.DataFrame(results)
    result.to_csv(out/'identifiability_metrics.csv',index=False)
    pred=pd.concat(predictions,ignore_index=True)
    pred.to_csv(out/'identifiability_predictions.csv',index=False)
    rng=np.random.default_rng(20261029)
    uncertainty=[]
    for fold,g in pred.groupby('fold'):
        key=['position_id','channel','anchor_label','stable_consistent','large_error']
        paired=g.pivot(index=key,columns='variant',values='probability').reset_index()
        for subset in ['all','stable_consistent']:
            sample=paired if subset=='all' else paired[paired.stable_consistent]
            positions=sample.position_id.to_numpy()
            clusters=[np.flatnonzero(positions==p) for p in np.unique(positions)]
            y=sample.large_error.to_numpy(int)
            for candidate,reference in [('cir','geometry'),('cir_entropy','cir')]:
                a,b=sample[candidate].to_numpy(),sample[reference].to_numpy()
                delta=[]
                for _ in range(500):
                    ids=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
                    if len(np.unique(y[ids]))<2:
                        continue
                    delta.append(roc_auc_score(y[ids],a[ids])-roc_auc_score(y[ids],b[ids]))
                uncertainty.append(dict(fold=fold,subset=subset,candidate=candidate,reference=reference,
                    delta_auc=roc_auc_score(y,a)-roc_auc_score(y,b),ci_low=np.quantile(delta,.025),
                    ci_high=np.quantile(delta,.975),valid_resamples=len(delta)))
    uncertainty=pd.DataFrame(uncertainty)
    uncertainty.to_csv(out/'identifiability_bootstrap.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    colors={'geometry':'#65758b','scalar':'#c27c2c','cir':'#3568a8','cir_entropy':'#8963a5'}
    x=np.arange(4)
    for ax,subset,title in zip(axes,['all','stable_consistent'],['All held-out link windows','Low std and small geometric residual']):
        for k,(variant,color) in enumerate(colors.items()):
            values=result[(result.subset==subset)&(result.variant==variant)].sort_values('fold').roc_auc
            ax.bar(x+(k-1.5)*.19,values,.19,color=color,label=variant)
        ax.axhline(.5,color='black',ls='--',lw=1)
        ax.set_xticks(x,['E0','E1','E2','E3'])
        ax.set(ylabel='ROC AUC',ylim=(0,1.02),title=title,xlabel='Held-out environment')
        ax.legend(fontsize=8,ncol=2,loc='lower right')
        ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'v2_identifiability.png',dpi=180)
    plt.close(fig)
    protocol={'task':'detect absolute raw window range error >= 0.50 m','subset':'range_std <= 0.05 m and abs raw LS residual <= 0.20 m',
              'features':variants,'classifier':'HGB fixed 120 iterations,15 leaves,no early stopping',
              'training':'original source train position groups only','uncertainty':'500 paired position-cluster bootstrap replicates, exploratory, no multiplicity adjustment',
              'elapsed_s':time.perf_counter()-start,'limitation':'development on already inspected D1; no new-environment guarantee; signal increment is relative to specified four-feature geometric baseline'}
    (out/'identifiability_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result.round(4).to_string(index=False))
    print(uncertainty.round(4).to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
