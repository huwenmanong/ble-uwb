"""D1 source-protocol audit and controlled A6-removal / weight sensitivity.

No raw data or previous results are modified. Official JSON is streamed without
loading its CIR arrays. JSON is audited, NOT used to fit zero-shot models.
The no-A6 experiment deliberately retains v2's range>0 filter to isolate the
documented hardware-failure exclusion; this is not yet a definitive benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import d1_v2_research as v2

ROOT = v2.ROOT
PRIOR = ROOT / 'data/results/d1_v2_20261029_seed20261029'
PREFIX = re.compile(r'\{"tag_id"\s*:[^{}]*?"cir"\s*:\s*\[')
POSITION_KEY = re.compile(r'"(-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?)"\s*:\s*\{')
SEED = 20261029


def unique_dir(stem):
    out, i = stem, 0
    while out.exists():
        i += 1
        out = stem.with_name(stem.name + f'_repeat{i:02d}')
    out.mkdir(parents=True)
    return out


def canonical(x, y, z):
    return f'{float(x):.2f}_{float(y):.2f}_{float(z):.2f}'


def scalar_json_records(path, audit):
    """Schema-specific streaming extraction; CIR strings contain no objects.

    Cross-check extracted count against independent tag_id byte-token count.
    Prefixes only are decoded: memory bounded by 4 MiB plus one CIR record.
    """
    buf, token_tail, count, tokens = '', b'', 0, 0
    position = None
    sha = hashlib.sha256()
    token = b'"tag_id"'
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            sha.update(chunk)
            tokens += (token_tail + chunk).count(token)
            token_tail = (token_tail + chunk)[-(len(token)-1):]
            buf += chunk.decode('utf-8')  # dataset JSON is ASCII numeric/text
            end = 0
            for match in PREFIX.finditer(buf):
                headers = POSITION_KEY.findall(buf[end:match.start()])
                if headers:
                    position = canonical(*headers[-1].split('_'))
                assert position is not None, 'Missing container position'
                prefix = match.group(0)
                record = json.loads(prefix[:prefix.rfind(',')] + '}')
                record['_container_position'] = position
                count += 1
                yield record
                end = match.end()
            if end:
                buf = buf[end:]
            assert len(buf) < 1024 * 1024, 'Unexpected JSON schema / oversized record'
    assert count == tokens, (count, tokens)
    audit.update(records=count, independently_counted_tag_ids=tokens, sha256=sha.hexdigest())


def source_audit(out):
    cols = ['environment_id','raw_folder','channel','anchor_label','anchor_id',
            'range_measured','range_true_offline','range_valid','tag_x','tag_y','tag_z']
    raw = pd.read_csv(ROOT/'data/interim/d1_stage1_scalar_features.csv', usecols=cols)
    assert np.isfinite(raw.range_measured).all()
    raw['canonical_position'] = raw.raw_folder.map(lambda s: canonical(*s.split('_')))
    raw['nonpositive'] = raw.range_measured.le(0)
    raw['raw_error_all_finite'] = raw.range_measured - raw.range_true_offline
    counts = raw.groupby(['environment_id','anchor_label']).agg(
        raw_rows=('range_measured','size'), nonpositive_rows=('nonpositive','sum'),
        min_raw_range_m=('range_measured','min')).reset_index()
    counts.to_csv(out/'raw_nonpositive_by_anchor.csv',index=False)
    rawwin = raw.groupby(['environment_id','canonical_position','anchor_label','channel']).agg(
        raw_n=('range_measured','size'), raw_mean_m=('range_measured','mean'),
        raw_true_m=('range_true_offline','first'),raw_x=('tag_x','first'),raw_y=('tag_y','first'),raw_z=('tag_z','first')).reset_index()
    rawwin.to_csv(out/'raw_all_finite_window_audit.csv',index=False)
    audits, windows = {}, []
    for env in v2.FOLDS:
        print(f'Audit official JSON stream: {env}',flush=True)
        mapping = raw[raw.environment_id.eq(env)][['anchor_id','anchor_label']].drop_duplicates()
        assert mapping.anchor_id.is_unique
        labels = dict(zip(mapping.anchor_id, mapping.anchor_label))
        groups, info = {}, {}
        path = ROOT/'数据集论文/data_set'/env/'data.json'
        for r in scalar_json_records(path, info):
            a = labels[r['anchor_id']]
            pos = r['_container_position']
            key = (pos,a,'ch'+str(r['channel_number']))
            distance = float(r['range'])
            truth = sum((float(r[f'{axis}_tag'])-float(r[f'{axis}_anchor']))**2 for axis in 'xyz')**.5
            if key not in groups:
                groups[key] = [0,0.,0.,float('inf'),-float('inf'),truth,0,r['x_tag'],r['y_tag'],r['z_tag']]
            g = groups[key]
            assert np.isclose(truth,g[5])
            g[0] += 1
            g[1] += distance
            g[2] += distance**2
            g[3],g[4] = min(g[3],distance),max(g[4],distance)
            g[6] += int(distance <= 0)
        audits[env] = info
        for (pos,a,ch),g in groups.items():
            windows.append(dict(environment_id=env,canonical_position=pos,anchor_label=a,channel=ch,
                processed_n=g[0],processed_mean_m=g[1]/g[0],processed_min_m=g[3],processed_max_m=g[4],
                processed_true_m=g[5],processed_nonpositive_rows=g[6],processed_x=g[7],processed_y=g[8],processed_z=g[9]))
        print(f'{env}: {info["records"]} official measurements, {len(groups)} windows',flush=True)
    official = pd.DataFrame(windows)
    assert not ((official.environment_id=='environment2') & (official.anchor_label=='A6')).any()
    official.to_csv(out/'official_json_window_audit.csv',index=False)
    keys = ['environment_id','canonical_position','anchor_label','channel']
    pairs = rawwin.merge(official,on=keys,how='outer',validate='one_to_one',indicator=True)
    pairs['processed_minus_raw_m'] = pairs.processed_mean_m-pairs.raw_mean_m
    pairs['coordinate_shift_m'] = np.sqrt(sum((pairs['processed_'+axis]-pairs['raw_'+axis])**2 for axis in 'xyz'))
    pairs.to_csv(out/'raw_vs_official_windows.csv',index=False)
    matched = pairs[pairs._merge=='both']
    shifts = matched.groupby(['environment_id','anchor_label','channel']).agg(
        matched_windows=('processed_minus_raw_m','size'),
        median_shift_m=('processed_minus_raw_m','median'),
        min_shift_m=('processed_minus_raw_m','min'),max_shift_m=('processed_minus_raw_m','max')).reset_index()
    shifts.to_csv(out/'official_range_shifts.csv',index=False)
    (out/'source_audit.json').write_text(json.dumps(dict(
        official_files=audits,raw_input_rows=len(raw),raw_nonpositive_rows=int(raw.nonpositive.sum()),
        raw_E2_A6_rows=int(((raw.environment_id=='environment2')&(raw.anchor_label=='A6')).sum()),
        join_counts={str(k):int(v) for k,v in pairs._merge.value_counts().items()},
        caveat='Raw table already excludes the previously frozen E3 coordinate-conflict folder. Official JSON is not used as zero-target-label training data.'),indent=2),encoding='utf-8')
    return counts, shifts


def controlled_experiment(out):
    # Default pandas float parsing changes some last bits, which can change HGB
    # histogram splits. Round-trip parsing exactly reproduces v2.load_windows().
    before = pd.read_csv(PRIOR/'window_features.csv',float_precision='round_trip')
    excluded = before.environment_id.eq('environment2') & before.anchor_label.eq('A6')
    before.loc[excluded].to_csv(out/'excluded_hardware_failure_windows.csv',index=False)
    win = before.loc[~excluded].reset_index(drop=True)
    win.to_csv(out/'window_features.csv',index=False)
    # Check frozen position roles remain exactly the same.
    prior_roles = pd.read_csv(PRIOR/'position_roles.csv').set_index(['fold','position_id']).role
    roles = []
    for fold in v2.FOLDS:
        role = v2.assign_roles(win,fold)
        g = pd.DataFrame({'fold':fold,'position_id':win.position_id,'role':role}).drop_duplicates()
        for r in g.itertuples():
            assert r.role == prior_roles.loc[(r.fold,r.position_id)]
        roles.append(g)
    pd.concat(roles).to_csv(out/'position_roles.csv',index=False)
    variants = {'scalar':list(v2.BASE_NUMERIC_FEATURES)+['range_std','sample_n'],
                'cir':list(v2.BASE_NUMERIC_FEATURES)+['range_std','sample_n']+v2.CIR_NUMERIC_FEATURES}
    locs, intervals = [], []
    for fold in v2.FOLDS:
        target = win[win.environment_id.eq(fold)]
        locs.extend(v2.locate(target,fold,'raw',[
            ('raw_LS','range_measured',None,False),('raw_Huber','range_measured',None,True)]))
        for variant,features in variants.items():
            print(f'No-A6 retraining: {fold} / {variant}',flush=True)
            ls,rs = v2.run_fold(win,fold,variant,features,SEED,out)
            locs.extend(ls)
            intervals.extend(rs)
    loc = pd.DataFrame(locs)
    assert np.isfinite(loc[['x','y','error_m']]).all().all() and loc.success.all()
    loc.to_csv(out/'localization.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'intervals.csv',index=False)
    summary = pd.DataFrame([dict(fold=f,variant=v,method=m,**v2.metrics(g.error_m))
        for (f,v,m),g in loc.groupby(['fold','variant','method'])])
    summary.to_csv(out/'summary.csv',index=False)
    v2.bootstrap(loc,SEED).to_csv(out/'paired_bootstrap.csv',index=False)
    old = pd.read_csv(PRIOR/'summary.csv')
    comparison = old.merge(summary,on=['fold','variant','method'],suffixes=('_with_A6','_without_A6'),validate='one_to_one')
    comparison.to_csv(out/'controlled_A6_comparison.csv',index=False)
    # E2 is held out: its bias/scale models are trained on unchanged data.
    oldlinks = pd.read_csv(PRIOR/'links_cir_environment2.csv',float_precision='round_trip')
    newlinks = pd.read_csv(out/'links_cir_environment2.csv',float_precision='round_trip')
    chk = oldlinks.merge(newlinks,on=v2.KEYS,suffixes=('_old','_new'),validate='one_to_one')
    for c in ['corrected_range','risk_scale']:
        assert np.allclose(chk[c+'_old'],chk[c+'_new'],atol=1e-12,rtol=0)
    return before,win,loc,summary,comparison


def weight_sensitivity(win, loc, out):
    """Fixed predictions and physical Huber threshold; not test-tuned selection."""
    result, geometry = [], []
    for fold in v2.FOLDS:
        links = pd.read_csv(out/f'links_cir_{fold}.csv',float_precision='round_trip')
        meta = win[v2.KEYS+['anchor_x','anchor_y','anchor_z','tag_x','tag_y']]
        links = links.merge(meta,on=v2.KEYS,validate='one_to_one')
        estimates = loc[(loc.fold==fold)&(loc.variant=='cir')].set_index(['raw_folder','channel','method'])
        for (folder,ch),g in links.groupby(['raw_folder','channel']):
            a = g[['anchor_x','anchor_y','anchor_z']].to_numpy()
            target = g[['tag_x','tag_y']].iloc[0].to_numpy()
            w = 1/g.risk_scale.to_numpy()**2
            w /= w.mean()
            baseline = estimates.loc[(folder,ch,'corrected_Huber')]
            x0 = baseline[['x','y']].to_numpy(float)
            dxy = x0-a[:,:2]
            d = np.sqrt((dxy**2).sum(axis=1)+(v2.HEIGHT[fold]-a[:,2])**2)
            j = dxy/np.maximum(d[:,None],1e-12)
            eig0 = np.linalg.eigvalsh(j.T@j)
            eig1 = np.linalg.eigvalsh(j.T@(w[:,None]*j))
            weighted = estimates.loc[(folder,ch,'risk_Huber')]
            err = abs(g.corrected_error_label.to_numpy())
            geometry.append(dict(fold=fold,raw_folder=folder,channel=ch,
                anchors=len(g),effective_anchors=float(w.sum()**2/(w*w).sum()),
                max_weight_share=float(w.max()/w.sum()),
                condition_equal=float(eig0[-1]/max(eig0[0],1e-12)),
                condition_risk=float(eig1[-1]/max(eig1[0],1e-12)),
                minimum_eigenvalue_ratio=float(eig1[0]/max(eig0[0],1e-12)),
                uniform_absolute_range_error_m=float(err.mean()),
                weighted_absolute_range_error_m=float(np.average(err,weights=w)),
                baseline_error_m=baseline.error_m,risk_error_m=weighted.error_m,
                error_delta_m=weighted.error_m-baseline.error_m))
            for alpha in [0,.25,.5,.75,1]:
                if alpha in [0,1]:
                    row = baseline if alpha==0 else weighted
                    error,success = float(row.error_m),bool(row.success)
                else:
                    xy,success,_,_ = v2.solve(a,g.corrected_range,v2.HEIGHT[fold],(1-alpha)+alpha*w,True)
                    error = float(np.linalg.norm(xy-target))
                result.append(dict(fold=fold,raw_folder=folder,channel=ch,alpha=alpha,error_m=error,success=success))
    results,geom = pd.DataFrame(result),pd.DataFrame(geometry)
    assert results.success.all() and np.isfinite(results.error_m).all()
    results.to_csv(out/'weight_sensitivity_cases.csv',index=False)
    geom.to_csv(out/'geometry_diagnostics.csv',index=False)
    agg = pd.DataFrame([dict(fold=f,alpha=a,**v2.metrics(g.error_m)) for (f,a),g in results.groupby(['fold','alpha'])])
    agg.to_csv(out/'weight_sensitivity_summary.csv',index=False)
    return agg,geom


def figures(before,summary,comparison,sweep,geom,out):
    # Figure contract: quantify the documented fault confound, then show that
    # fixed risk-weight strength is not uniformly beneficial. All D1 folds shown.
    # 329 position clusters, 1974 position/channel cases; descriptive curves,
    # not confidence intervals. Source CSVs are saved beside every figure.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    fig,axes = plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
    e2 = before[before.environment_id.eq('environment2')]
    for label,g,color in [('A6 (documented fault)',e2[e2.anchor_label.eq('A6')],'#ad6a3b'),
                         ('Other anchors',e2[e2.anchor_label.ne('A6')],'#3c7199')]:
        values = np.sort(g.error_label.to_numpy())
        axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=f'{label}, n={len(g)}',color=color)
    axes[0].set(xlabel='Signed raw window error (m)',ylabel='Empirical CDF',title='a  E2: hardware failure is not NLOS')
    axes[0].legend(fontsize=7,loc='upper left')
    methods = [('raw','raw_LS'),('raw','raw_Huber'),('cir','corrected_Huber'),('cir','risk_Huber')]
    c = comparison[comparison.fold.eq('environment2')].set_index(['variant','method'])
    x=np.arange(len(methods))
    for shift,col,label,color in [(-.18,'p90_m_with_A6','With A6','#ad6a3b'),(.18,'p90_m_without_A6','Without A6','#3c7199')]:
        vals=[c.loc[m,col] for m in methods]
        axes[1].bar(x+shift,vals,.36,label=label,color=color)
    axes[1].set_xticks(x,['Raw LS','Raw Huber','Corrected\nHuber','Risk\nHuber'],fontsize=7)
    axes[1].set(ylabel='Localization P90 (m)',title='b  One-factor A6 exclusion: E2')
    axes[1].legend(fontsize=7)
    fig.savefig(out/'v3_A6_protocol_audit.png',dpi=600)
    fig.savefig(out/'v3_A6_protocol_audit.pdf')
    fig.savefig(out/'v3_A6_protocol_audit.svg')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
    for ax,fold in zip(axes.flat,v2.FOLDS):
        g=sweep[sweep.fold.eq(fold)]
        ax.plot(g.alpha,g.p90_m,'o-',color='#3c7199',label='P90')
        ax.plot(g.alpha,g.p95_m,'s--',color='#ad6a3b',label='P95')
        ax.set(xlabel='Risk weight strength alpha',ylabel='Localization error (m)',
               title=fold.replace('environment','E')+' | fixed predictions',ylim=(0,1.5))
        ax.set_xticks([0,.25,.5,.75,1])
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle('Only E2-A6 excluded; descriptive sensitivity, not parameter selection',fontsize=8)
    fig.savefig(out/'v3_weight_sensitivity.png',dpi=600)
    fig.savefig(out/'v3_weight_sensitivity.pdf')
    fig.savefig(out/'v3_weight_sensitivity.svg')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--skip-json-audit',action='store_true',help='Quick rerun only; omits independent official-source audit')
    parser.add_argument('--figures-only',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.figures_only:
        if args.run_dir is None:
            parser.error('--figures-only requires --run-dir')
        out=args.run_dir
        assert json.loads((out/'protocol.json').read_text())['completed']
        figures(pd.read_csv(PRIOR/'window_features.csv'),pd.read_csv(out/'summary.csv'),
                pd.read_csv(out/'controlled_A6_comparison.csv'),pd.read_csv(out/'weight_sensitivity_summary.csv'),
                pd.read_csv(out/'geometry_diagnostics.csv'),out)
        shutil.copyfile(__file__,out/'figure_source.py')
        print('Figures regenerated without refitting:',out)
        return
    out=unique_dir(ROOT/'data/results/d1_v3_protocol_20261029')
    started=time.perf_counter()
    shutil.copyfile(__file__,out/'audit_source.py')
    shutil.copyfile(Path(v2.__file__),out/'v2_dependency_source.py')
    print('Output:',out,flush=True)
    v2.self_test()
    if not args.skip_json_audit:
        source_audit(out)
    before,win,loc,summary,comparison=controlled_experiment(out)
    sweep,geom=weight_sensitivity(win,loc,out)
    figures(before,summary,comparison,sweep,geom,out)
    protocol=dict(version='v3-protocol-diagnostic-20261029',seed=SEED,
        source_evidence='Scientific Data 2023, DOI 10.1038/s41597-023-02639-5, pp.8-9,15',
        intervention='Exclude all environment2/A6 windows from all source-fit, calibration and held-out roles; retain all other v2 settings',
        retained_caveats=['v2 range>0 filter on uncalibrated raw data remains for this controlled diagnostic',
                         'known E3 coordinate-conflict folder remains excluded',
                         'no official target-domain offset used in models',
                         'no entropy variant rerun; not a final feature verdict',
                         'all D1 environments already inspected; external validation pending'],
        alpha_grid=[0,.25,.5,.75,1],alpha_formula='w(alpha)=(1-alpha)+alpha*(risk^-2 / mean(risk^-2)); Huber delta unchanged at 0.30 m',
        alpha_selection='descriptive sensitivity only; no target-environment hyperparameter selection',
        geometry='J computed at corrected-Huber estimate, no truth input; JTWJ is a local diagnostic, not a certified covariance or full robust Hessian',
        raw_windows_before=len(before),raw_windows_after=len(win),positions=win.position_id.nunique(),
        baseline_localization_rows=len(loc),weight_sensitivity_rows=len(pd.read_csv(out/'weight_sensitivity_cases.csv')),
        source_sha256={'v2_windows':v2.digest(PRIOR/'window_features.csv'),'audit_script':v2.digest(__file__),
                       'v2_dependency':v2.digest(v2.__file__)},elapsed_s=time.perf_counter()-started,completed=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(summary[['fold','variant','method','p90_m','max_m']].round(4).to_string(index=False))
    print('Weight sensitivity:\n',sweep[['fold','alpha','p90_m','p95_m','max_m']].round(4).to_string(index=False))
    print(f'Completed in {protocol["elapsed_s"]:.1f}s: {out}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
