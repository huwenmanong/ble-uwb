"""Controlled protocol ladder on the common 329-position D1 cohort.

P0: existing no-E2-A6, positive raw observations, old coordinates.
P1: P0 with official position labels (only label/evaluation coordinates change).
P2: finite raw observations including nonpositive outputs, official labels.
P3: official processed observations including their nonpositive outputs.
P3 is hardware-calibrated using reference distances, NOT zero-target-label data.
Only scalar diagnostics are used to isolate data-protocol changes.
"""
from __future__ import annotations
import argparse
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
import d1_v2_research as v2
import d1_v3_protocol_audit as v3

ROOT=v2.ROOT
PRIOR=ROOT/'data/results/d1_v3_protocol_20261029_repeat01'
FEATURES=list(v2.BASE_NUMERIC_FEATURES)+['range_std','sample_n']
STAGES=['P0_old_labels_positive','P1_fixed_labels_positive','P2_fixed_labels_finite','P3_official_calibrated']
KEYS=v2.KEYS
SEED=20261029


def relabel(win, official):
    win=win.copy()
    win['canonical_position']=win.raw_folder.map(lambda x:v3.canonical(*x.split('_')))
    keys=['environment_id','canonical_position','anchor_label','channel']
    match=win.merge(official[keys+['processed_x','processed_y','processed_z']],on=keys,
                    how='left',validate='one_to_one')
    assert match[['processed_x','processed_y','processed_z']].notna().all().all()
    for axis in 'xyz':
        match['tag_'+axis]=match.pop('processed_'+axis)
    match['range_true_offline']=np.sqrt(sum((match['tag_'+axis]-match['anchor_'+axis])**2 for axis in 'xyz'))
    match['error_label']=match.range_measured-match.range_true_offline
    assert np.allclose(match.tag_z,match.environment_id.map(v2.HEIGHT))
    return match.drop(columns='canonical_position')


def raw_windows(raw):
    specs={c:(c,'mean') for c in v2.BASE_NUMERIC_FEATURES}
    specs.update({c:(c,'first') for c in ['tag_x','tag_y','tag_z']+[f'split_{f}' for f in v2.FOLDS]})
    specs.update(sample_n=('range_measured','size'),range_std=('range_measured','std'))
    w=raw.groupby(KEYS,sort=True).agg(**specs).reset_index()
    w['range_std']=w.range_std.fillna(0.)
    w['position_id']=w.environment_id+'/'+w.raw_folder
    return w


def official_windows(raw, cohort, out):
    metadata=cohort[['environment_id','raw_folder','position_id']+[f'split_{f}' for f in v2.FOLDS]].drop_duplicates()
    metadata['canonical_position']=metadata.raw_folder.map(lambda s:v3.canonical(*s.split('_')))
    assert not metadata.duplicated(['environment_id','canonical_position']).any()
    metadata=metadata.set_index(['environment_id','canonical_position'])
    records, audits = [], {}
    fieldmap={'range_measured':'range','anchor_x':'x_anchor','anchor_y':'y_anchor','anchor_z':'z_anchor'}
    for env in v2.FOLDS:
        print('Read official scalar records:',env,flush=True)
        labels=raw[raw.environment_id.eq(env)][['anchor_id','anchor_label']].drop_duplicates()
        assert labels.anchor_id.is_unique
        labels=dict(zip(labels.anchor_id,labels.anchor_label))
        anchor_table=pd.read_csv(ROOT/'数据集论文/data_set'/env/'anchors.csv').set_index('Anchor')
        groups, info, excluded, nonpositive = {}, {}, 0, 0
        for r in v3.scalar_json_records(ROOT/'数据集论文/data_set'/env/'data.json',info):
            pos=r['_container_position']
            assert v3.canonical(r['x_tag'],r['y_tag'],r['z_tag'])==pos
            if (env,pos) not in metadata.index:
                excluded+=1
                continue
            anchor=labels[r['anchor_id']]
            assert not (env=='environment2' and anchor=='A6')
            channel='ch'+str(r['channel_number'])
            vals=np.array([r[fieldmap.get(c,c)] for c in v2.BASE_NUMERIC_FEATURES],dtype=float)
            assert np.isfinite(vals).all()
            nonpositive+=int(r['range']<=0)
            key=(pos,anchor,channel)
            if key not in groups:
                assert np.allclose([r[f'{ax}_anchor'] for ax in 'xyz'],anchor_table.loc[anchor,['x','y','z']])
                groups[key]=dict(n=0,sums=np.zeros(len(vals)),mean=0.,m2=0.,tag=[r[f'{ax}_tag'] for ax in 'xyz'])
            g=groups[key]
            assert np.allclose(g['tag'],[r[f'{ax}_tag'] for ax in 'xyz'])
            g['n']+=1
            g['sums']+=vals
            delta=r['range']-g['mean']
            g['mean']+=delta/g['n']
            g['m2']+=delta*(r['range']-g['mean'])
        info.update(cohort_excluded_rows=excluded,cohort_nonpositive_rows=nonpositive)
        audits[env]=info
        for (pos,anchor,channel),g in groups.items():
            row=dict(zip(v2.BASE_NUMERIC_FEATURES,g['sums']/g['n']))
            row.update(metadata.loc[(env,pos)].to_dict())
            row.update(environment_id=env,anchor_label=anchor,channel=channel,sample_n=g['n'],
                       range_std=np.sqrt(max(g['m2'],0)/(g['n']-1)) if g['n']>1 else 0.)
            row.update(dict(zip(['tag_x','tag_y','tag_z'],g['tag'])))
            row['range_true_offline']=sum((row['tag_'+ax]-row['anchor_'+ax])**2 for ax in 'xyz')**.5
            row['error_label']=row['range_measured']-row['range_true_offline']
            records.append(row)
    old=json.loads((PRIOR/'source_audit.json').read_text())['official_files']
    for env in v2.FOLDS:
        assert audits[env]['sha256']==old[env]['sha256']
        assert audits[env]['records']==old[env]['records']
    (out/'official_sources.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    return pd.DataFrame(records).sort_values(KEYS).reset_index(drop=True)


def run_stage(win,out):
    out.mkdir()
    win.to_csv(out/'window_features.csv',index=False)
    roles=[]
    original=pd.read_csv(PRIOR/'position_roles.csv').set_index(['fold','position_id']).role
    for fold in v2.FOLDS:
        g=pd.DataFrame(dict(fold=fold,position_id=win.position_id,role=v2.assign_roles(win,fold))).drop_duplicates()
        for r in g.itertuples():
            assert r.role==original.loc[(r.fold,r.position_id)]
        roles.append(g)
    pd.concat(roles).to_csv(out/'position_roles.csv',index=False)
    locs,intervals=[],[]
    for fold in v2.FOLDS:
        print(out.name,fold,flush=True)
        target=win[win.environment_id.eq(fold)].copy()
        target['std_mean_scale']=np.maximum(.03,target.range_std/np.sqrt(target.sample_n))
        locs.extend(v2.locate(target,fold,'raw',[
            ('raw_LS','range_measured',None,False),('raw_Huber','range_measured',None,True),
            ('raw_std_WLS','range_measured','std_mean_scale',False)]))
        rows,ints=v2.run_fold(win,fold,'scalar',FEATURES,SEED,out)
        locs.extend(rows)
        intervals.extend(ints)
    loc=pd.DataFrame(locs)
    assert np.isfinite(loc.error_m).all()
    loc.to_csv(out/'localization.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'intervals.csv',index=False)
    summary=pd.DataFrame([dict(fold=f,variant=v,method=m,success_rate=g.success.mean(),**v2.metrics(g.error_m))
        for (f,v,m),g in loc.groupby(['fold','variant','method'])])
    summary.to_csv(out/'summary.csv',index=False)
    v2.bootstrap(loc,SEED).to_csv(out/'paired_bootstrap.csv',index=False)
    return loc,summary


def paired_protocol_bootstrap(loc,out):
    rng=np.random.default_rng(SEED)
    rows=[]
    for candidate,reference in zip(STAGES[1:],STAGES[:-1]):
        for fold in v2.FOLDS:
            for method in ['raw_Huber','corrected_Huber','risk_Huber']:
                g=loc[(loc.fold==fold)&(loc.method==method)&loc.stage.isin([candidate,reference])]
                p=g.pivot(index=['raw_folder','channel'],columns='stage',values='error_m')
                assert p.notna().all().all()
                a,b=p[candidate].to_numpy(),p[reference].to_numpy()
                pos=p.index.get_level_values('raw_folder').to_numpy()
                clusters=[np.flatnonzero(pos==q) for q in np.unique(pos)]
                deltas=[]
                for _ in range(500):
                    ids=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
                    deltas.append(np.quantile(a[ids],.9)-np.quantile(b[ids],.9))
                rows.append(dict(candidate=candidate,reference=reference,fold=fold,method=method,
                    delta_p90_m=np.quantile(a,.9)-np.quantile(b,.9),ci_low_m=np.quantile(deltas,.025),
                    ci_high_m=np.quantile(deltas,.975),resamples=500))
    pd.DataFrame(rows).to_csv(out/'paired_protocol_differences.csv',index=False)


def figures(summary,out):
    # Quantitative grid: how protocol changes affect the same fixed methods.
    # Common cohort; descriptive P90 only, not uncertainty bands or model gains.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
    colors=['#8d8d8d','#9b719f','#bd7c47','#3b759c']
    labels=['P0: old labels, positive','P1: fixed labels, positive','P2: fixed labels, finite','P3: official calibrated']
    ymax=summary[summary.method.isin(['corrected_Huber','risk_Huber'])].p90_m.max()*1.15
    for ax,method,title in zip(axes,['corrected_Huber','risk_Huber'],['a  Bias correction + Huber','b  Bias correction + risk Huber']):
        for stage,label,color,marker in zip(STAGES,labels,colors,['o','s','^','D']):
            g=summary[(summary.stage==stage)&(summary.method==method)].set_index('fold').reindex(v2.FOLDS)
            ax.plot(range(4),g.p90_m,marker=marker,color=color,label=label,lw=1.2)
        ax.set_xticks(range(4),['E0','E1','E2','E3'])
        ax.set(xlabel='Held-out environment',ylabel='Localization P90 (m)',title=title,ylim=(0,ymax))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=6.5)
    fig.savefig(out/'protocol_ladder.png',dpi=600)
    fig.savefig(out/'protocol_ladder.pdf')
    fig.savefig(out/'protocol_ladder.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures-only',type=Path)
    args=parser.parse_args()
    if args.figures_only:
        figures(pd.read_csv(args.figures_only/'all_summary.csv'),args.figures_only)
        return
    started=time.perf_counter()
    out=v3.unique_dir(ROOT/'data/results/d1_v4_protocol_ladder_20261029')
    print('Output:',out,flush=True)
    for src,name in [(Path(__file__),'experiment_source.py'),(Path(v2.__file__),'v2_dependency_source.py'),(Path(v3.__file__),'v3_dependency_source.py')]:
        shutil.copyfile(src,out/name)
    v2.self_test()
    prior=pd.read_csv(PRIOR/'window_features.csv',float_precision='round_trip')
    official_meta=pd.read_csv(PRIOR/'official_json_window_audit.csv',float_precision='round_trip')
    p1=relabel(prior,official_meta)
    assert np.array_equal(prior[FEATURES].to_numpy(),p1[FEATURES].to_numpy(),equal_nan=True)
    raw=pd.read_csv(ROOT/'data/interim/d1_stage1_scalar_features.csv')
    raw=raw[~(raw.environment_id.eq('environment2')&raw.anchor_label.eq('A6'))].copy()
    assert np.isfinite(raw.range_measured).all()
    p2=relabel(raw_windows(raw),official_meta)
    p3=official_windows(raw,prior,out)
    # Both cohorts have identical positions/channels; P2/P3 also identical links.
    casekeys=['environment_id','raw_folder','channel']
    for frame in [p1,p2,p3]:
        assert set(map(tuple,frame[casekeys].drop_duplicates().to_numpy()))==set(map(tuple,prior[casekeys].drop_duplicates().to_numpy()))
        assert frame.position_id.nunique()==329
    assert set(map(tuple,p2[KEYS].to_numpy()))==set(map(tuple,p3[KEYS].to_numpy()))
    assert len(p2)==len(p3)==15288
    loc0=pd.read_csv(PRIOR/'localization.csv',float_precision='round_trip')
    loc0=loc0[loc0.variant.isin(['raw','scalar'])].copy()
    sum0=pd.read_csv(PRIOR/'summary.csv',float_precision='round_trip')
    sum0=sum0[sum0.variant.isin(['raw','scalar'])].copy()
    locs=[loc0.assign(stage=STAGES[0])]
    summaries=[sum0.assign(stage=STAGES[0])]
    counts=[dict(stage=STAGES[0],rows=int(prior.sample_n.sum()),windows=len(prior),positions=329)]
    for stage,win in zip(STAGES[1:],[p1,p2,p3]):
        loc,summary=run_stage(win,out/stage)
        locs.append(loc.assign(stage=stage))
        summaries.append(summary.assign(stage=stage))
        counts.append(dict(stage=stage,rows=int(win.sample_n.sum()),windows=len(win),positions=329))
    loc=pd.concat(locs,ignore_index=True)
    summary=pd.concat(summaries,ignore_index=True)
    loc.to_csv(out/'all_localization.csv',index=False)
    summary.to_csv(out/'all_summary.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'cohort_counts.csv',index=False)
    paired_protocol_bootstrap(loc,out)
    figures(summary,out)
    protocol=dict(version='v4-protocol-ladder',completed=True,seed=SEED,stages=dict(zip(STAGES,[
        'existing v3 no A6, raw positive only, original metadata labels',
        'same exact deployment feature matrix as P0; official position labels',
        'finite raw measurements retained, including nonpositive estimates; official labels',
        'official hardware-calibrated and cleaned measurements, including nonpositive estimates'])),
        cohort='common frozen 329 positions; E2-A6 excluded; previously excluded E3 position remains outside all stages',
        scope='scalar diagnostics only; all models and source position roles unchanged; full static-window planar localization',
        important='P3 is NOT zero-target-label deployment: author preprocessing used reference distances. P2->P3 combines hardware offset compensation and author outlier cleaning.',
        negative_policy='finite estimated ranges retained in P2/P3 as observations; no clipping or truth-error threshold deletion; positivity evaluated separately in P1',
        leakage_boundary='truth coordinates/range are training labels or evaluation only; no NLOS flag is a feature',
        hashes={'script':v2.digest(__file__),'v2':v2.digest(v2.__file__),'v3':v2.digest(v3.__file__),
                'raw_scalar':v2.digest(ROOT/'data/interim/d1_stage1_scalar_features.csv')},
        uncertainty='500 paired position-cluster bootstrap samples, exploratory, no multiplicity correction',
        python=platform.python_version(),elapsed_s=time.perf_counter()-started)
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary[summary.method.isin(['raw_Huber','corrected_Huber','risk_Huber'])][['stage','fold','method','p90_m','max_m']].round(4).to_string(index=False))
    print('Completed:',out,'seconds:',round(protocol['elapsed_s'],1),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
