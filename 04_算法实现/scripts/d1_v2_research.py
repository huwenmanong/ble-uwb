"""Independent, window-level D1 audit and LOSO experiment. Run in bluekey.

This is an offline static-window experiment, NOT causal per-frame tracking.
Ground truth is used as training labels/evaluation only. Height is a declared
known installation parameter. Existing v1 artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from run_d1_frozen_loso_baselines import solve_position as legacy_solver
from train_d1_loso_uncertainty_model import BASE_NUMERIC_FEATURES, CIR_NUMERIC_FEATURES, ENTROPY_NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[2]
KEYS = ['environment_id', 'raw_folder', 'channel', 'anchor_label']
FOLDS = [f'environment{i}' for i in range(4)]
HEIGHT = dict(zip(FOLDS, [1.20, 0.95, 1.32, 1.20]))
PROTOCOL = {
    'version': 'd1-window-v2-20261029',
    'unit': 'complete static link window, up to 31 valid measurements',
    'height_m': HEIGHT,
    'height_assumption': 'known installation height, never inferred from test labels',
    'target': 'signed mean measured range minus geometric true range',
    'split': 'original LOSO; original calibration positions split into scale_fit / interval_calibration',
    'bias_model': 'HGB squared_error, 120 iterations, 15 leaves, early_stopping=False',
    'scale_model': 'HGB absolute_error on absolute window residual, 80 iterations, 7 leaves',
    'scale_definition': 'conditional absolute-error magnitude proxy, NOT a standard deviation',
    'scale_floor_m': 0.03,
    'huber_delta_m': 0.30,
    'solver': 'scipy TRF, analytic Jacobian, centroid and algebraic starts, lowest objective',
    'intervals': 'global / scaled pooled-link quantiles / scaled position-max quantiles',
    'guarantee': 'empirical cross-domain evaluation only; no distribution-free target-domain guarantee',
    'feature_variants': ['scalar', 'cir', 'entropy', 'cir_entropy'],
    'limitation': 'D1 folds have already been inspected; results are development evidence, external final validation pending',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def quantile(values, p):
    a = np.sort(np.asarray(values, dtype=float))
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Invalid calibration scores')
    k = math.ceil((len(a) + 1) * p)
    return float(a[k - 1]) if k <= len(a) else math.inf


def solve(anchors, ranges, height, weights=None, robust=False):
    """Weights are relative heuristics; robust threshold stays in meters."""
    anchors, ranges = np.asarray(anchors, float), np.asarray(ranges, float)
    if weights is None:
        weights = np.ones(len(ranges))
    weights = np.asarray(weights, float)
    if len(ranges) < 3 or not np.isfinite(ranges).all() or np.linalg.matrix_rank(anchors[1:, :2] - anchors[0, :2]) < 2:
        return np.full(2, np.nan), False, math.inf, 0
    weights = weights / weights.mean()
    rootw = np.sqrt(weights)
    dz = height - anchors[:, 2]

    def residual(x):
        return (np.sqrt(np.sum((x - anchors[:, :2]) ** 2, axis=1) + dz ** 2) - ranges) * rootw

    def jacobian(x):
        diff = x - anchors[:, :2]
        dist = np.sqrt(np.sum(diff ** 2, axis=1) + dz ** 2)
        return diff / np.maximum(dist[:, None], 1e-12) * rootw[:, None]

    def weighted_huber(z):
        # z is weighted residual squared. Convert back to physical residual
        # squared so adding weights does not change the 0.30 m threshold.
        u = z / weights
        delta = PROTOCOL['huber_delta_m']
        tail = u > delta ** 2
        rho = np.vstack([u.copy(), np.ones_like(u), np.zeros_like(u)])
        rho[0, tail] = 2 * delta * np.sqrt(u[tail]) - delta ** 2
        rho[1, tail] = delta / np.sqrt(u[tail])
        rho[2, tail] = -0.5 * delta / (u[tail] ** 1.5)
        rho[0] *= weights
        rho[2] /= weights
        return rho

    adjusted = ranges ** 2 - dz ** 2
    a = 2 * (anchors[1:, :2] - anchors[0, :2])
    b = np.sum(anchors[1:, :2] ** 2, axis=1) - np.sum(anchors[0, :2] ** 2) - adjusted[1:] + adjusted[0]
    starts = [anchors[:, :2].mean(axis=0), np.linalg.lstsq(a, b, rcond=None)[0]]
    candidates = []
    for start in starts:
        result = least_squares(residual, start, jac=jacobian, method='trf',
                               loss=weighted_huber if robust else 'linear', max_nfev=300,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        candidates.append(result)
    best = min(candidates, key=lambda x: x.cost)
    return best.x, bool(best.success), float(best.cost), sum(r.nfev for r in candidates)


def self_test():
    anchors = np.array([[0, 0, 1], [8, 0, 2], [8, 8, 1.5], [0, 8, .5]])
    target, h = np.array([2.1, 3.4]), 1.2
    ranges = np.sqrt(np.sum((anchors[:, :2] - target) ** 2, axis=1) + (anchors[:, 2] - h) ** 2)
    for robust in (False, True):
        x, ok, _, _ = solve(anchors, ranges, h, [1, 2, 3, 4], robust)
        assert ok and np.linalg.norm(x - target) < 1e-6
    noisy = ranges + np.array([0.1, -0.15, 1.5, -.05])
    x, ok, cost, _ = solve(anchors, noisy, h, [1, 2, 3, 4], True)
    r = np.sqrt(np.sum((anchors[:, :2] - x) ** 2, axis=1) + (anchors[:, 2] - h) ** 2) - noisy
    w = np.array([1, 2, 3, 4]) / 2.5
    expected = .5 * np.sum(w * np.where(abs(r) <= .3, r ** 2, .6 * abs(r) - .09))
    assert ok and np.isclose(cost, expected)
    assert quantile(np.arange(1, 10), .9) == 9
    assert math.isinf(quantile([1, 2], .95))
    print('self-tests passed: known-position recovery, physical weighted-Huber objective, finite-sample quantiles', flush=True)


def load_windows():
    scalar_path = ROOT / 'data/interim/d1_stage1_scalar_features.csv'
    row = pd.read_csv(scalar_path)
    valid = row.loc[row.range_valid.eq(1)].copy()
    specs = {c: (c, 'mean') for c in BASE_NUMERIC_FEATURES}
    specs.update({c: (c, 'first') for c in ['tag_x', 'tag_y', 'tag_z', 'range_true_offline'] + [f'split_{f}' for f in FOLDS]})
    specs.update(sample_n=('range_measured', 'size'), range_std=('range_measured', 'std'))
    win = valid.groupby(KEYS, sort=True).agg(**specs).reset_index()
    win['range_std'] = win.range_std.fillna(0.0)
    win['error_label'] = win.range_measured - win.range_true_offline
    win['position_id'] = win.environment_id + '/' + win.raw_folder
    assert np.allclose(win.tag_z, win.environment_id.map(HEIGHT))
    paths = [scalar_path]
    cir_path = ROOT / 'data/interim/d1_stage1_cir_features.csv'
    cir = pd.read_csv(cir_path)
    rowkeys = KEYS + ['measurement_index']
    cir = valid[rowkeys].merge(cir[rowkeys + CIR_NUMERIC_FEATURES], on=rowkeys, how='left', validate='one_to_one')
    assert cir[CIR_NUMERIC_FEATURES].notna().all().all(), 'Missing CIR feature rows'
    cir = cir.groupby(KEYS, sort=True)[CIR_NUMERIC_FEATURES].mean().reset_index()
    win = win.merge(cir, on=KEYS, how='left', validate='one_to_one')
    paths.append(cir_path)
    entropy_path = ROOT / 'data/interim/d1_stage1_entropy_link_features.csv'
    entropy = pd.read_csv(entropy_path)
    win = win.merge(entropy[KEYS + ENTROPY_NUMERIC_FEATURES], on=KEYS, how='left', validate='one_to_one')
    assert win[ENTROPY_NUMERIC_FEATURES].notna().all().all(), 'Missing entropy windows'
    paths.append(entropy_path)
    info = {'original_rows': len(row), 'valid_rows': len(valid), 'windows': len(win),
            'positions': win.position_id.nunique(), 'source_sha256': {str(p): digest(p) for p in paths}}
    return win, info


def assign_roles(win, fold):
    role = win[f'split_{fold}'].replace({'train': 'bias_fit', 'calibration': 'pending'}).copy()
    for env, g in win.loc[role.eq('pending')].groupby('environment_id'):
        positions = sorted(g.position_id.unique(), key=lambda p: hashlib.sha256(('v2-scale-cal/' + p).encode()).hexdigest())
        for i, pos in enumerate(positions):
            role.loc[win.position_id.eq(pos)] = 'scale_fit' if i % 2 == 0 else 'interval_calibration'
    assert not role.eq('pending').any()
    assert pd.DataFrame({'position': win.position_id, 'role': role}).groupby('position').role.nunique().max() == 1
    return role


def metrics(values):
    a = np.asarray(values, float)
    finite = np.isfinite(a)
    v = a[finite]
    return {'n_total': len(a), 'n_finite': int(finite.sum()), 'nonfinite_rate': float((~finite).mean()),
            'mean_m': float(v.mean()), 'median_m': float(np.median(v)), 'rmse_m': float(np.sqrt(np.mean(v ** 2))),
            'p90_m': float(np.quantile(v, .9)), 'p95_m': float(np.quantile(v, .95)), 'max_m': float(v.max()),
            'fail_0_5m': float(np.mean((a > .5) | ~finite)), 'fail_1m': float(np.mean((a > 1) | ~finite))}


def locate(frame, fold, variant, specs):
    rows = []
    for (_, folder, channel), g in frame.groupby(['environment_id', 'raw_folder', 'channel'], sort=True):
        anchors = g[['anchor_x', 'anchor_y', 'anchor_z']].to_numpy(float)
        target = g[['tag_x', 'tag_y']].iloc[0].to_numpy(float)
        for method, range_col, weight_col, robust in specs:
            scale = g[weight_col].to_numpy(float) if weight_col else np.ones(len(g))
            weights = 1 / np.maximum(scale, .03) ** 2
            start = time.perf_counter()
            x, success, cost, nfev = solve(anchors, g[range_col], HEIGHT[fold], weights, robust)
            rows.append(dict(fold=fold, variant=variant, raw_folder=folder, channel=channel, method=method,
                             x=float(x[0]), y=float(x[1]), error_m=float(np.linalg.norm(x - target)), success=success,
                             cost=cost, nfev=nfev, solve_ms=1000 * (time.perf_counter() - start), anchors=len(g)))
    return rows


def model(seed, scale=False):
    return HistGradientBoostingRegressor(loss='absolute_error' if scale else 'squared_error',
        max_iter=80 if scale else 120, max_leaf_nodes=7 if scale else 15,
        min_samples_leaf=40, learning_rate=.06, l2_regularization=1.0,
        early_stopping=False, random_state=seed)


def run_fold(win, fold, variant, features, seed, out):
    role = assign_roles(win, fold)
    base = win.loc[role.eq('bias_fit'), features]
    median = base.median().fillna(0)
    numeric = win[features].fillna(median)
    # Known radio channel is a deployable setting; anchor identity is omitted
    # to reduce layout memorization. Coordinates are included explicitly.
    channel = pd.get_dummies(win.channel, prefix='channel', dtype=float)
    x = pd.concat([numeric, channel], axis=1)
    train, scale_fit, cal, test = [role.eq(v).to_numpy() for v in ['bias_fit', 'scale_fit', 'interval_calibration', 'test']]
    bias = model(seed).fit(x.loc[train], win.loc[train, 'error_label'])
    prediction = bias.predict(x)
    residual = win.error_label.to_numpy() - prediction
    scale_model = model(seed, True).fit(x.loc[scale_fit], np.abs(residual[scale_fit]))
    scale = np.maximum(.03, scale_model.predict(x))
    cal_scores = abs(residual[cal]) / scale[cal]
    global_q, scaled_q = quantile(abs(residual[cal]), .90), quantile(cal_scores, .90)
    group_scores = pd.DataFrame({'position': win.loc[cal, 'position_id'].to_numpy(), 'score': cal_scores}).groupby('position').score.max()
    group_q = quantile(group_scores, .90)
    work = win.loc[test].copy()
    work['corrected_range'] = work.range_measured - prediction[test]
    work['corrected_error_label'] = residual[test]
    work['risk_scale'] = scale[test]
    work['global_halfwidth90'] = global_q
    work['scaled_halfwidth90'] = scale[test] * scaled_q
    work['position_max_halfwidth90'] = scale[test] * group_q
    riskrows = []
    for method in ['global', 'scaled', 'position_max']:
        width = work[f'{method}_halfwidth90'].to_numpy()
        error = abs(work.corrected_error_label.to_numpy())
        covered = error <= width
        bypos = pd.DataFrame({'position': work.position_id, 'covered': covered}).groupby('position').covered.all()
        riskrows.append(dict(fold=fold, variant=variant, interval=method, nominal=.90, n_links=len(work),
            calibration_positions=int(win.loc[cal].position_id.nunique()),
            link_coverage=float(covered.mean()), whole_position_coverage=float(bypos.mean()),
            mean_full_width_m=float((2 * width).mean()),
            interval_score=float(np.mean(2 * width + 20 * np.maximum(error - width, 0))),
            risk_spearman=float(pd.Series(scale[test]).corr(pd.Series(error), method='spearman')),
            corrected_mae_m=float(error.mean()), raw_mae_m=float(abs(work.error_label).mean()),
            in_sample_scale_q90=quantile(abs(residual[scale_fit]) / scale[scale_fit], .90),
            independent_scaled_q90=scaled_q))
    work[KEYS + ['position_id', 'sample_n', 'range_measured', 'corrected_range', 'error_label', 'corrected_error_label',
                  'risk_scale', 'global_halfwidth90', 'scaled_halfwidth90', 'position_max_halfwidth90']].to_csv(out / f'links_{variant}_{fold}.csv', index=False)
    loc = locate(work, fold, variant, [('corrected_LS', 'corrected_range', None, False),
        ('corrected_Huber', 'corrected_range', None, True),
        ('risk_WLS', 'corrected_range', 'risk_scale', False),
        ('risk_Huber', 'corrected_range', 'risk_scale', True)])
    return loc, riskrows


def solver_audit(win, out):
    records = []
    for (env, folder, channel), g in win.groupby(['environment_id', 'raw_folder', 'channel'], sort=True):
        a = g[['anchor_x', 'anchor_y', 'anchor_z']].to_numpy(float)
        ranges = g.range_measured.to_numpy(float)
        height = HEIGHT[env]
        source = [dict(r, range_mean=r['range_measured'], quality_weight=1.) for r in g.to_dict('records')]
        for method, mode in [('raw_LS', 'equal'), ('raw_std_WLS', 'std')]:
            scales = np.maximum(.03, g.range_std.to_numpy() / np.sqrt(g.sample_n.to_numpy())) if mode == 'std' else np.ones(len(g))
            w = 1 / scales ** 2
            w /= w.mean()
            ox, oy, old_ok, _ = legacy_solver(source, mode, False)
            newxy, new_ok, new_cost, _ = solve(a, ranges, height, w, False)
            oldres = np.sqrt(np.sum((a[:, :2] - [ox, oy]) ** 2, axis=1) + (a[:, 2] - height) ** 2) - ranges
            records.append(dict(fold=env, raw_folder=folder, channel=channel, method=method,
                old_success=old_ok, new_success=new_ok, old_objective=float(.5 * np.sum(w * oldres ** 2)),
                new_objective=new_cost, shift_m=float(np.linalg.norm(newxy - [ox, oy]))))
    audit = pd.DataFrame(records)
    audit.to_csv(out / 'solver_audit.csv', index=False)
    assert (audit.new_objective <= audit.old_objective + 1e-5).all(), 'Stable solver increased same objective'
    return audit


def bootstrap(loc, seed):
    """Exploratory paired position-cluster bootstrap, all channels retained."""
    rng, rows = np.random.default_rng(seed), []
    for (fold, variant), group in loc[loc.variant.ne('raw')].groupby(['fold', 'variant']):
        pivot = group.pivot(index=['raw_folder', 'channel'], columns='method', values='error_m')
        positions = pivot.index.get_level_values(0).unique()
        clusters = [np.flatnonzero(pivot.index.get_level_values(0) == p) for p in positions]
        for method in ['risk_WLS', 'risk_Huber']:
            baseline = 'corrected_Huber' if method == 'risk_Huber' else 'corrected_LS'
            a, b = pivot[method].to_numpy(), pivot[baseline].to_numpy()
            assert np.isfinite(a).all() and np.isfinite(b).all()
            deltas = []
            for _ in range(500):
                ids = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
                deltas.append(np.quantile(a[ids], .9) - np.quantile(b[ids], .9))
            rows.append(dict(fold=fold, variant=variant, method=method, baseline=baseline,
                positions=len(positions), delta_p90_m=float(np.quantile(a, .9) - np.quantile(b, .9)),
                ci_low_m=float(np.quantile(deltas, .025)), ci_high_m=float(np.quantile(deltas, .975)),
                resamples=500, caveat='exploratory within-D1 positions; no correction for multiple comparisons'))
    return pd.DataFrame(rows)


def figures(summary, risk, audit, out):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(4)
    for method, color, marker in [('raw_LS', '#65758b', 'o'), ('raw_Huber', '#c27c2c', 's')]:
        g = summary[(summary.variant == 'raw') & (summary.method == method)].set_index('fold').reindex(FOLDS)
        ax[0].plot(x, g.p90_m, marker=marker, color=color, label=method)
    for method, color, marker in [('corrected_Huber', '#3568a8', '^'), ('risk_Huber', '#8963a5', 'D')]:
        g = summary[(summary.variant == 'cir') & (summary.method == method)].set_index('fold').reindex(FOLDS)
        ax[0].plot(x, g.p90_m, marker=marker, color=color, label='CIR ' + method)
    ax[0].set(ylabel='Localization P90 (m)', title='Stable-solver comparison', ylim=(0, None))
    ax[0].legend(fontsize=8)
    for variant, color, marker in [('scalar', '#65758b', 'o'), ('cir', '#3568a8', 's'), ('entropy', '#c27c2c', '^'), ('cir_entropy', '#8963a5', 'D')]:
        g = summary[(summary.variant == variant) & (summary.method == 'corrected_Huber')].set_index('fold').reindex(FOLDS)
        ax[1].plot(x, g.p90_m, marker=marker, color=color, label=variant)
    ax[1].set(ylabel='Localization P90 (m)', title='Feature ablation: corrected Huber', ylim=(0, None))
    ax[1].legend(fontsize=8)
    for method, color, marker in [('global', '#65758b', 'o'), ('scaled', '#3568a8', 's'), ('position_max', '#c27c2c', '^')]:
        g = risk[(risk.variant == 'cir') & (risk.interval == method)].set_index('fold').reindex(FOLDS)
        ax[2].plot(x, 100 * g.link_coverage, marker=marker, color=color, label=method)
    ax[2].axhline(90, color='black', linestyle='--', lw=1)
    ax[2].set(ylabel='Held-out link coverage (%)', title='CIR model: independent calibration', ylim=(0, 103))
    ax[2].legend(fontsize=8)
    for a in ax:
        a.set_xticks(x, ['E0', 'E1', 'E2', 'E3'])
        a.set_xlabel('Held-out environment')
        a.grid(alpha=.2)
    fig.savefig(out / 'v2_overview.png', dpi=180)
    fig.savefig(out / 'v2_overview.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    rates = audit.groupby('fold')[['old_success', 'new_success']].mean().reindex(FOLDS)
    axes[0].bar(x - .17, 100 * rates.old_success, .34, label='Legacy GN', color='#65758b')
    axes[0].bar(x + .17, 100 * rates.new_success, .34, label='TRF, two starts', color='#3568a8')
    axes[0].set(ylabel='Solver success (%)', ylim=(0, 103), title='Same-objective numerical audit')
    axes[0].set_xticks(x, ['E0', 'E1', 'E2', 'E3'])
    axes[0].legend()
    for interval, marker in [('global', 'o'), ('scaled', 's'), ('position_max', '^')]:
        g = risk[(risk.variant == 'cir') & (risk.interval == interval)]
        axes[1].scatter(g.mean_full_width_m, 100 * g.link_coverage, marker=marker, s=50, label=interval)
        offsets = {'environment0': (-15, -14), 'environment1': (5, 6), 'environment2': (5, -13), 'environment3': (-10, 7)}
        for r in g.itertuples():
            axes[1].annotate(r.fold.replace('environment', 'E'), (r.mean_full_width_m, 100*r.link_coverage), xytext=offsets[r.fold], textcoords='offset points', fontsize=8)
    axes[1].axhline(90, linestyle='--', color='black', lw=1)
    axes[1].set(xlabel='Mean full interval width (m)', ylabel='Held-out link coverage (%)', title='Coverage must be read with width', ylim=(0,103))
    axes[1].legend(fontsize=8)
    fig.savefig(out / 'v2_solver_and_interval_width.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    stem = ROOT / 'data/results' / f'd1_v2_20261029_seed{args.seed}'
    out, repeat = stem, 0
    while out.exists():
        repeat += 1
        out = stem.with_name(stem.name + f'_repeat{repeat:02d}')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / 'experiment_source.py')
    started = time.perf_counter()
    (out / 'protocol.json').write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2), encoding='utf-8')
    win, info = load_windows()
    print(json.dumps(info, ensure_ascii=False), flush=True)
    win.to_csv(out / 'window_features.csv', index=False)
    manifests = []
    for fold in FOLDS:
        role = assign_roles(win, fold)
        manifests.append(pd.DataFrame({'fold': fold, 'position_id': win.position_id, 'role': role}).drop_duplicates())
    manifest = pd.concat(manifests, ignore_index=True)
    manifest.to_csv(out / 'position_roles.csv', index=False)
    print('Auditing legacy and stable solvers on identical raw objectives ...', flush=True)
    audit = solver_audit(win, out)
    print('Same-objective solver audit completed', flush=True)
    base = list(BASE_NUMERIC_FEATURES) + ['range_std', 'sample_n']
    variants = {'scalar': base, 'cir': base + CIR_NUMERIC_FEATURES,
                'entropy': base + ENTROPY_NUMERIC_FEATURES,
                'cir_entropy': base + CIR_NUMERIC_FEATURES + ENTROPY_NUMERIC_FEATURES}
    loc_rows, risk_rows = [], []
    for fold in FOLDS:
        target = win.loc[win.environment_id.eq(fold)].copy()
        target['std_mean_scale'] = np.maximum(.03, target.range_std / np.sqrt(target.sample_n))
        loc_rows.extend(locate(target, fold, 'raw', [('raw_LS','range_measured',None,False),
            ('raw_Huber','range_measured',None,True), ('raw_std_WLS','range_measured','std_mean_scale',False)]))
        for variant, features in variants.items():
            print(f'{fold} / {variant}: fit independent models and solve positions', flush=True)
            loc, risk = run_fold(win, fold, variant, features, args.seed, out)
            loc_rows.extend(loc)
            risk_rows.extend(risk)
            pd.DataFrame(loc_rows).to_csv(out / 'localization.csv', index=False)
            pd.DataFrame(risk_rows).to_csv(out / 'intervals.csv', index=False)
    loc, risk = pd.DataFrame(loc_rows), pd.DataFrame(risk_rows)
    summary = pd.DataFrame([dict(fold=fold, variant=variant, method=method,
        success_rate=float(g.success.mean()), mean_solve_ms=float(g.solve_ms.mean()), **metrics(g.error_m))
        for (fold, variant, method), g in loc.groupby(['fold','variant','method'])])
    summary.to_csv(out / 'summary.csv', index=False)
    bootstrap(loc, args.seed).to_csv(out / 'paired_bootstrap.csv', index=False)
    figures(summary, risk, audit, out)
    info.update(elapsed_s=time.perf_counter()-started, python=platform.python_version(), scipy=scipy.__version__,
                sklearn=sklearn.__version__, seed=args.seed, script_sha256=digest(__file__), completed=True)
    (out / 'run_manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed in {info["elapsed_s"]:.1f} s: {out}', flush=True)
    return out


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
