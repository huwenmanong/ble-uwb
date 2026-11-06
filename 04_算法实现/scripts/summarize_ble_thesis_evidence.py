"""Evidence audit, fixed inside/outside case study and scientific figures.

Figure contract: quantify the effect of validation repair, show existing regional
errors, label the single-inside-source case study, and separate acquisition from
inference latency. Quantitative grids; Python only; no inferential p values.
All original observations are retained. No synthetic observations are created.
"""
from pathlib import Path
import json
import platform
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
ANCHORS = ['3DD5', '208A', '230D', 'B5D7']
mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def savefig(fig, path):
    fig.savefig(str(path) + '.svg', bbox_inches='tight')
    fig.savefig(str(path) + '.pdf', bbox_inches='tight')
    fig.savefig(str(path) + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def score_predictions(frame, truth, prediction):
    rows = []
    for (model, task, fold), part in frame.groupby(['model', 'task', 'fold']):
        rows.append(dict(model=model, task=task, fold=int(fold), n=len(part),
                         accuracy=accuracy_score(part[truth], part[prediction]),
                         macro_f1=f1_score(part[truth], part[prediction], average='macro', zero_division=0)))
    return pd.DataFrame(rows)


def audit_dynamic(out):
    previous = ROOT / 'ble_research/results/dynamic_learning_trend_bilstm_attn_focal_pso'
    original = pd.read_csv(previous / 'predictions.csv')
    updated = pd.read_csv(out / 'dynamic_nested/predictions.csv')
    checks = []
    for name, frame in [('historical_test_selected', original), ('corrected_inner_validation', updated)]:
        for task, sub in frame.groupby('task'):
            assert sub['window_id'].nunique() == len(sub) == 2507
            assert sub.groupby('source_id')['fold'].nunique().max() == 1
            checks.append(dict(result=name, task=task, windows=len(sub), sources=sub.source_id.nunique(),
                               source_in_single_test_fold=True))
    keys = ['window_id', 'task', 'fold', 'source_id', 'y_true']
    assert set(map(tuple, original[keys].to_numpy())) == set(map(tuple, updated[keys].to_numpy()))
    old = score_predictions(original, 'y_true', 'y_pred'); old['result'] = 'historical_test_selected'
    new = score_predictions(updated, 'y_true', 'y_pred'); new['result'] = 'corrected_inner_validation'
    all_scores = pd.concat([old, new], ignore_index=True)
    all_scores.to_csv(out / 'dynamic_recomputed_fold_metrics.csv', index=False)
    pd.DataFrame(checks).to_csv(out / 'dynamic_source_checks.csv', index=False)
    summary = all_scores.groupby(['result', 'task']).agg(macro_f1=('macro_f1', 'mean'),
                    fold_std=('macro_f1', 'std'), accuracy=('accuracy', 'mean')).reset_index()
    summary.to_csv(out / 'dynamic_protocol_comparison.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, task in zip(axes, ['action', 'direction']):
        for fold in range(1, 6):
            a = old[(old.task == task) & (old.fold == fold)].macro_f1.iloc[0]
            b = new[(new.task == task) & (new.fold == fold)].macro_f1.iloc[0]
            ax.plot([0, 1], [a, b], 'o-', color='#517b91', alpha=.7, linewidth=1, markersize=3)
        ax.set(xticks=[0, 1], xticklabels=['Historical\ntest-selected', 'Corrected\ninner validation'],
               ylabel='Macro-F1', title=f'{task.capitalize()} (five fixed outer folds)', ylim=(0, 1.02))
        ax.grid(axis='y', alpha=.15)
    savefig(fig, out / '01_dynamic_protocol_comparison')
    return summary


def regional_evaluation(out):
    path = ROOT / 'ble_research/results/static_learning_baselines/static_learning_predictions.csv'
    frame = pd.read_csv(path)
    rf = frame[(frame.model == 'rf') & (frame.feature_protocol == 'direct')]
    rows = []
    for task in ['direction', 'distance_bin']:
        sub = rf[rf.task == task].copy()
        assert len(sub) == 12023
        assert sub.groupby('source_path')['fold'].nunique().max() == 1
        for (direction, distance), part in sub.groupby(['direction', 'distance_m']):
            rows.append(dict(task=task, direction=direction, distance_m=distance, windows=len(part),
                             accuracy=float((part.true_label == part.pred_label).mean())))
    pd.DataFrame(rows).to_csv(out / 'static_region_by_direction_distance.csv', index=False)
    distance = rf[rf.task == 'distance_bin']
    labels = ['near_1_5m', 'mid_6_10m', 'far_12m_plus']
    counts = confusion_matrix(distance.true_label, distance.pred_label, labels=labels)
    rates = counts / counts.sum(axis=1, keepdims=True)
    pd.DataFrame(counts, index=labels, columns=labels).to_csv(out / 'static_range_confusion_counts.csv')
    pd.DataFrame(rates, index=labels, columns=labels).to_csv(out / 'static_range_confusion_rates.csv')
    direction = rf[rf.task == 'direction']
    recalls = direction.groupby('true_label').apply(lambda d: (d.true_label == d.pred_label).mean())
    recalls.rename('recall').to_csv(out / 'static_direction_recall.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    axes[0].bar(recalls.index, recalls.values, color='#6d91a3')
    axes[0].set(ylabel='Recall', ylim=(0, 1.03), title='Vehicle-relative direction (RF)')
    axes[0].tick_params(axis='x', labelrotation=30)
    axes[1].imshow(rates, vmin=0, vmax=1, cmap='Blues')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{100*rates[i,j]:.1f}%', ha='center', va='center',
                         color='white' if rates[i,j] > .6 else 'black')
    axes[1].set(xticks=range(3), yticks=range(3), xticklabels=['Near', 'Mid', 'Far'],
                yticklabels=['Near', 'Mid', 'Far'], xlabel='Predicted', ylabel='True',
                title='Existing coarse range bins (RF)')
    savefig(fig, out / '02_vehicle_region_metrics')
    return dict(windows_per_task=12023, sources=85, model='RF', protocol='existing GroupKFold by source_path',
                range_macro_f1=f1_score(distance.true_label, distance.pred_label, average='macro'),
                direction_macro_f1=f1_score(direction.true_label, direction.pred_label, average='macro'))


def inside_case_study(out):
    data = pd.read_csv(ROOT / 'ble_research/data/ble_static_ukf_86files_dataset.csv')
    rows, split_rows = [], []
    window = 16
    for source, part in data.groupby('source_path', sort=True):
        part = part.sort_values('sample_index')
        values = part[ANCHORS].to_numpy(float)
        assert np.isfinite(values).all()
        cutoff = int(len(part) * .7)
        for split, lo, hi in [('train', 0, cutoff), ('test', cutoff + window, len(part))]:
            count = 0
            for start in range(lo, hi - window + 1, window):
                v = values[start:start + window]
                means = v.mean(axis=0)
                feature = np.r_[means, v.std(axis=0), [means[a] - means[b] for a in range(4) for b in range(a+1, 4)]]
                row = dict(source_path=source, split=split, start=start, end=start+window-1,
                           inside=int(part.direction.iloc[0] == 'I'), direction=part.direction.iloc[0])
                row.update({f'f{i}': float(x) for i, x in enumerate(feature)})
                rows.append(row); count += 1
            split_rows.append(dict(source_path=source, split=split, original_rows=len(part),
                                   start=lo, stop=hi, windows=count, cutoff=cutoff,
                                   exclusion='16-sample temporal gap plus incomplete final windows'))
    windows = pd.DataFrame(rows)
    train = windows[windows.split == 'train']; test = windows[windows.split == 'test'].copy()
    features = [f'f{i}' for i in range(14)]
    class_sources = train.groupby('inside').source_path.nunique().to_dict()
    group_sizes = train.groupby('source_path').size().to_dict()
    weights = np.array([1/(class_sources[y] * group_sizes[s]) for y, s in zip(train.inside, train.source_path)])
    weights /= weights.mean()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42))
    model.fit(train[features], train.inside, logisticregression__sample_weight=weights)
    test['probability_inside'] = model.predict_proba(test[features])[:, 1]
    test['pred_inside'] = (test.probability_inside >= .5).astype(int)
    counts = confusion_matrix(test.inside, test.pred_inside, labels=[0, 1])
    rates = counts / counts.sum(axis=1, keepdims=True)
    test.to_csv(out / 'inside_timeblock_test_predictions.csv', index=False)
    pd.DataFrame(split_rows).to_csv(out / 'inside_timeblock_split_manifest.csv', index=False)
    metrics = dict(model='fixed logistic regression, C=1, probability threshold=0.5',
                   window=window, stride=window, train_fraction=.7, temporal_gap_samples=window,
                   train_windows=len(train), test_windows=len(test),
                   inside_train_windows=int(train.inside.sum()), inside_test_windows=int(test.inside.sum()),
                   inside_source_groups=class_sources[1], outside_source_groups=class_sources[0],
                   accuracy=accuracy_score(test.inside, test.pred_inside),
                   macro_f1=f1_score(test.inside, test.pred_inside, average='macro'),
                   outside_predicted_inside_rate=float(rates[0, 1]),
                   inside_predicted_outside_rate=float(rates[1, 0]),
                   validation='within-source chronological case study, NOT independent-source generalization',
                   limitations='one inside source; unknown internal positions/sessions; upstream historical UKF not independently audited; not an unlocking/security certification')
    (out / 'inside_case_study.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)
    ax.imshow(rates, vmin=0, vmax=1, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{counts[i,j]}\n({rates[i,j]*100:.1f}%)', ha='center', va='center',
                    color='white' if rates[i,j] > .6 else 'black')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Outside', 'Inside'], yticklabels=['Outside', 'Inside'],
           xlabel='Predicted', ylabel='True', title='Within-source time-block case study\nOne inside source; not independent validation')
    savefig(fig, out / '03_inside_outside_case_study')
    return metrics


def latency_analysis(out):
    samples = pd.read_csv(ROOT / 'ble_research/data/dynamic_kalman/ble_dynamic_kalman_samples.csv')
    windows = pd.read_csv(ROOT / 'ble_research/data/dynamic_kalman/ble_dynamic_kalman_window_manifest.csv')
    assert not samples.duplicated(['source_id', 'sample_index']).any()
    lookup = samples.set_index(['source_id', 'sample_index']).time_seconds
    for name, index_col in [('start', 'window_start_index'), ('end', 'window_end_index'), ('center', 'window_center_index')]:
        keys = pd.MultiIndex.from_arrays([windows.source_id, windows[index_col]])
        windows[f'{name}_time_s'] = lookup.reindex(keys).to_numpy()
    windows['acquisition_span_s'] = windows.end_time_s - windows.start_time_s
    windows['center_label_availability_lag_s'] = windows.end_time_s - windows.center_time_s
    ordered = windows.sort_values(['source_id', 'window_start_index'])
    ordered['update_interval_s'] = ordered.groupby('source_id').end_time_s.diff()
    ordered.to_csv(out / 'dynamic_window_timing.csv', index=False)
    diffs = samples.sort_values(['source_id', 'sample_index']).groupby('source_id').time_seconds.diff().dropna()
    timing = dict(dynamic_window=20, dynamic_stride=5, windows=len(windows), sources=windows.source_id.nunique(),
                  observed_sample_interval_median_s=float(diffs.median()), nonpositive_intervals=int((diffs <= 0).sum()),
                  acquisition_span_median_s=float(windows.acquisition_span_s.median()),
                  acquisition_span_p95_s=float(windows.acquisition_span_s.quantile(.95)),
                  update_interval_median_s=float(ordered.update_interval_s.median()),
                  center_label_availability_lag_median_s=float(windows.center_label_availability_lag_s.median()),
                  static_ble_window=16, static_ble_stride=4, static_ble_sampling_rate='not verified',
                  uwb_window=30, uwb_stride=10, uwb_sampling_rate='not verified',
                  note='Observed gaps include acquisition/logging variation, not only radio sample periods. End-of-window output avoids claiming center-time availability. No real trajectory position accuracy implied.')
    (out / 'latency_accounting.json').write_text(json.dumps(timing, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    axes[0].hist(windows.acquisition_span_s, bins=30, color='#6d91a3')
    axes[0].set(xlabel='Observed window span (s)', ylabel='Window count', title='Acquisition of 20 samples')
    axes[1].hist(ordered.update_interval_s.dropna(), bins=30, color='#bd9a64')
    axes[1].set(xlabel='Observed update interval (s)', ylabel='Update count', title='Five-sample stride')
    savefig(fig, out / '04_window_latency')
    return timing


def main(out=None):
    out = Path(out) if out is not None else ROOT / 'SCI-UWB/data/results/ble_evidence_closure_20261105'
    summary = audit_dynamic(out)
    regional = regional_evaluation(out)
    inside = inside_case_study(out)
    latency = latency_analysis(out)
    final = dict(dynamic=summary.to_dict('records'), regional=regional, inside_case_study=inside,
                 latency=latency, platform=platform.platform(), findings=dict(
                     historical_dynamic_test_used_for_early_stopping=True,
                     static_pso_requires_true_direction=True,
                     direction_and_action_are_collection_category_labels=True,
                     distance_stage_is_time_interpolated_weak_label=True))
    (out / 'evidence_summary.json').write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'figure_qa.md').write_text('# Figure QA\n\nPython/matplotlib quantitative grids; SVG/PDF editable text and 300dpi PNG.\n'
        '01: all five paired folds; historical test-selected results explicitly labeled; no significance tests.\n'
        '02: all 12023 saved RF outer predictions per task; sample-weighted recall, not source-averaged accuracy.\n'
        '03: fixed chronological case study; all available source groups, one inside group; 16-sample gap and incomplete windows documented in split manifest. No threshold search.\n'
        '04: all 2507 dynamic windows and all within-source update intervals; first interval per source undefined by design, not a discarded observation.\n'
        'Raw captures and historical predictions unchanged. No simulated points, no independence-based confidence intervals.\n', encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
