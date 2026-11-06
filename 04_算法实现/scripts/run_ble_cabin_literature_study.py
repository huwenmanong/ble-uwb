"""PyCharm entry: literature-inspired cabin discrimination, one frozen protocol."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import subprocess

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ble_research'))
import run_ble_static_learning_baselines as original
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRIOR = ROOT / 'SCI-UWB/data/results/ble_inside_outside_fixed_20261105'
OUT = ROOT / 'SCI-UWB/data/results/ble_cabin_literature_20261105'
ANCHORS = ['3DD5', '208A', '230D', 'B5D7']
PAIRS = [(a, b) for a in range(4) for b in range(a+1, 4)]
PRIMARY = 'RelativeConfidence_WKNN'


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class ConfidenceWKNN:
    """Train-scaled distances; optional relative RSSI and pair-specific variability."""
    def __init__(self, relative=False, confidence=False):
        self.relative = relative
        self.confidence = confidence

    def fit(self, values, variability, labels):
        self.dim = 10 if self.relative else 4
        self.scaler = StandardScaler().fit(values[:, :self.dim])
        self.x = self.scaler.transform(values[:, :self.dim])
        self.v = (variability[:, :self.dim] / self.scaler.scale_) ** 2
        self.y = np.asarray(labels)
        self.blocks = [slice(0, 4), slice(4, 10)] if self.relative else [slice(0, 4)]
        return self

    def predict_proba(self, values, variability):
        x = self.scaler.transform(values[:, :self.dim])
        v = (variability[:, :self.dim] / self.scaler.scale_) ** 2
        result = []
        for start in range(0, len(x), 32):
            z = x[start:start+32]
            delta = (z[:, None, :] - self.x[None, :, :]) ** 2
            distance = np.zeros(delta.shape[:2])
            for block in self.blocks:
                if self.confidence:
                    w = np.clip(1 / (1 + v[start:start+32, None, block] + self.v[None, :, block]), .1, 1.)
                    distance += np.sum(w * delta[:, :, block], axis=2) / np.sum(w, axis=2)
                else:
                    distance += np.mean(delta[:, :, block], axis=2)
            distance /= len(self.blocks)
            idx = np.argpartition(distance, 4, axis=1)[:, :5]
            near = np.sqrt(np.take_along_axis(distance, idx, axis=1))
            vote = 1 / np.maximum(near, 1e-9)
            p = np.sum(vote * self.y[idx], axis=1) / np.sum(vote, axis=1)
            result.extend(p)
        p = np.asarray(result)
        return np.column_stack([1-p, p])


def measure(name, split, y, p):
    prediction = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(model=name, split=split, n=len(y), accuracy=float(accuracy_score(y, prediction)),
                balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                macro_f1=float(f1_score(y, prediction, average='macro')),
                outside_as_inside=float(fp/(tn+fp)), inside_as_outside=float(fn/(tp+fn)),
                inside_recall=float(tp/(tp+fn)), inside_precision=float(tp/max(tp+fp, 1)),
                roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def export_figure(fig, name):
    fig.savefig(OUT / (name+'.svg'), bbox_inches='tight')
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name+'.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def figures(metrics, selected, meta, probabilities):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    # Quantitative grid: class-balanced performance versus two directional errors.
    table = pd.DataFrame(metrics)
    test = table[table.split == 'test'].set_index('model')
    names = list(test.index)
    colors = ['#287D8E' if n == PRIMARY else '#A9B3BD' for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.8), sharey=True, layout='constrained')
    for ax, column, title in zip(axes, ['balanced_accuracy', 'outside_as_inside', 'inside_as_outside'],
                                 ['Balanced accuracy (%)', 'Outside to inside (%)', 'Inside to outside (%)']):
        vals = 100*test[column].to_numpy()
        ax.barh(np.arange(len(names)), vals, color=colors)
        ax.set_xlabel(title)
        ax.set_xlim(0, max(100 if column == 'balanced_accuracy' else 1, vals.max()*1.3))
        for i, value in enumerate(vals):
            ax.text(value, i, f' {value:.2f}', va='center', fontsize=7)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    fig.suptitle('Fixed protocol; outside n=2924, inside n=96 test windows', fontsize=10)
    export_figure(fig, '01_model_comparison')
    compare = list(dict.fromkeys(['Rich_WKNN', PRIMARY, selected]))
    fig, axes = plt.subplots(1, len(compare), figsize=(3.3*len(compare), 3.3), layout='constrained', squeeze=False)
    for ax, name in zip(axes[0], compare):
        r = test.loc[name]
        counts = np.array([[r.tn, r.fp], [r.fn, r.tp]], dtype=int)
        norm = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(norm, vmin=0, vmax=1, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{counts[i,j]}\n{100*norm[i,j]:.1f}%', ha='center', va='center',
                        color='white' if norm[i,j] > .5 else 'black')
        ax.set_xticks([0, 1], ['Outside', 'Inside'])
        ax.set_yticks([0, 1], ['Outside', 'Inside'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(name, fontsize=9)
    export_figure(fig, '02_confusion_matrices')
    subset = meta[meta.split == 'test'].copy()
    for name in ['Rich_WKNN', PRIMARY, selected]:
        subset[name] = (probabilities[name]['test'] >= .5).astype(int)
    rows = []
    for direction, part in subset.groupby('direction'):
        for name in dict.fromkeys(['Rich_WKNN', PRIMARY, selected]):
            rows.append(dict(direction=direction, model=name, n=len(part),
                             error=float(np.mean(part[name] != part.inside))))
    by_direction = pd.DataFrame(rows)
    by_direction.to_csv(OUT / 'test_error_by_direction.csv', index=False)
    pivot = by_direction.pivot(index='direction', columns='model', values='error') * 100
    fig, ax = plt.subplots(figsize=(6.8, 3.4), layout='constrained')
    ax.imshow(pivot.to_numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(pivot.columns)), list(pivot.columns), rotation=15, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)), list(pivot.index))
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f'{value:.2f}%', ha='center', va='center', color='white' if value>50 else 'black')
    ax.set_title('Test window error by acquisition direction (I: single interior position)')
    export_figure(fig, '03_direction_errors')


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=False)
    # Persist the Python workflow preference inside the research workspace.
    env = dict(os.environ, NATURE_FIGURE_CONFIG=str(ROOT/'SCI-UWB/data/nature_figure_backend.json'))
    subprocess.run([sys.executable, 'E:/codex_skill/nature-figure/scripts/nature_figure_backend.py', 'set', 'python'],
                   check=True, env=env)
    subprocess.run([sys.executable, 'E:/codex_skill/nature-figure/scripts/validate_figure.py', __file__], check=True)
    assignment_path = PRIOR/'window_assignments.csv'
    meta = pd.read_csv(assignment_path)
    data_path = ROOT/'ble_research/data/ble_static_ukf_86files_dataset.csv'
    data = pd.read_csv(data_path)
    lookup = {source: part.set_index('sample_index')[ANCHORS] for source, part in data.groupby('source_path')}
    rich, values, variability = [], [], []
    for row in meta.itertuples():
        raw = lookup[row.source_path].loc[row.start:row.end].to_numpy(dtype=float)
        assert raw.shape == (16, 4) and np.isfinite(raw).all()
        rich.append(original.window_features(raw))
        pairs = np.column_stack([raw[:, a]-raw[:, b] for a,b in PAIRS])
        combined = np.column_stack([raw, pairs])
        center = np.median(combined, axis=0)
        values.append(center)
        variability.append(1.4826*np.median(np.abs(combined-center), axis=0))
    rich = pd.DataFrame(rich).to_numpy(); values=np.asarray(values); variability=np.asarray(variability)
    train = (meta.split == 'train').to_numpy()
    validation = (meta.split == 'validation').to_numpy()
    y = meta.inside.to_numpy()
    models = {
        'Median_WKNN': ('median', make_pipeline(StandardScaler(), KNeighborsClassifier(5, weights='distance'))),
        'Rich_WKNN': ('rich', make_pipeline(StandardScaler(), KNeighborsClassifier(5, weights='distance'))),
        'PCA95_WKNN': ('rich', make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver='full'),
                                           KNeighborsClassifier(5, weights='distance'))),
        'Balanced_Logistic': ('rich', make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced',
                                                                                       max_iter=2000, random_state=20261105))),
        'Balanced_SVM': ('rich', make_pipeline(StandardScaler(), SVC(C=1., gamma='scale', class_weight='balanced'))),
        'Balanced_RF': ('rich', RandomForestClassifier(n_estimators=400, class_weight='balanced',
                                                      random_state=20261105, n_jobs=-1)),
        'Block_WKNN': ('custom', ConfidenceWKNN()),
        'Confidence_WKNN': ('custom', ConfidenceWKNN(confidence=True)),
        'Relative_WKNN': ('custom', ConfidenceWKNN(relative=True)),
        PRIMARY: ('custom', ConfidenceWKNN(relative=True, confidence=True)),
    }
    protocol = dict(completed=False, interpreter=sys.executable, seed=20261105,
                    assignment_sha256=hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
                    dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    window=16, stride=4, split='Exact window assignments from prior fixed protocol; no new split.',
                    counts=meta.groupby(['split','inside']).size().reset_index(name='windows').to_dict('records'),
                    primary=PRIMARY, threshold='.5 votes/probability; SVM decision threshold0.',
                    selection='Highest validation balanced accuracy, then Macro-F1, then lower outside_as_inside; no tuning.',
                    limitations=['One fixed interior source split into separated index blocks.',
                                 'Upstream UKF preprocessing precedes split; dependence not eliminated.',
                                 'Previously explored data, not new blinded acquisition.',
                                 'PCA projection is an adaptation, not literal P061 eigenvalue-to-anchor weights.',
                                 'Window MAD reliability is an adaptation, not literal P061 weighting.'],
                    models={n:repr(model) for n, (_, model) in models.items()})
    dump('protocol.json', protocol)
    meta.to_csv(OUT/'window_assignments.csv', index=False)
    feature_names = [f'median_{a}' for a in ANCHORS]+[f'median_diff_{ANCHORS[a]}_{ANCHORS[b]}' for a,b in PAIRS]
    pd.DataFrame(values, columns=feature_names).to_csv(OUT/'window_medians.csv', index=False)
    probabilities = {}; metrics=[]; runtimes=[]

    def predict(kind, model, ids):
        x = rich[ids] if kind == 'rich' else values[ids, :4]
        if kind == 'custom': return model.predict_proba(values[ids], variability[ids])[:, 1]
        if isinstance(model, type(models['Balanced_SVM'][1])) and kind == 'rich' and isinstance(model.steps[-1][1], SVC):
            # Monotonic score mapping only; not a calibrated probability.
            decision = model.decision_function(x)
            return 1 / (1 + np.exp(-np.clip(decision, -50, 50)))
        return model.predict_proba(x)[:, 1]

    for name, (kind, model) in models.items():
        started = time.perf_counter()
        if kind == 'custom': model.fit(values[train], variability[train], y[train])
        else: model.fit(rich[train] if kind == 'rich' else values[train, :4], y[train])
        fit_s = time.perf_counter()-started
        p = predict(kind, model, validation)
        probabilities[name] = {'validation': p}
        row = measure(name, 'validation', y[validation], p)
        metrics.append(row); runtimes.append(dict(model=name, fit_seconds=fit_s))
        print('VALIDATION', row, flush=True)
    ranking = sorted(metrics, key=lambda r:(-r['balanced_accuracy'], -r['macro_f1'], r['outside_as_inside']))
    selected = ranking[0]['model']
    dump('selection_locked_before_test.json', dict(selected=selected, primary=PRIMARY, validation=metrics))
    # Evaluate all prespecified comparisons only after the validation selection is recorded.
    test = (meta.split == 'test').to_numpy()
    for name, (kind, model) in models.items():
        started = time.perf_counter(); p = predict(kind, model, test)
        runtimes[list(models).index(name)]['test_batch_predict_seconds'] = time.perf_counter()-started
        probabilities[name]['test'] = p
        row = measure(name, 'test', y[test], p); metrics.append(row)
        print('TEST', row, flush=True)
    for name in models:
        for split, p in probabilities[name].items():
            output = meta[meta.split == split].copy()
            output['inside_score'] = p; output['pred_inside'] = (p >= .5).astype(int)
            output.to_csv(OUT/f'{name}_{split}_predictions.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv', index=False); dump('metrics.json', metrics)
    dump('runtime.json', runtimes)
    # Replay controls verify that adding candidates did not change baseline data processing.
    previous = pd.read_csv(PRIOR/'metrics.csv')
    for old_name, new_name in [('WKNN_k5', 'Rich_WKNN'), ('RF_balanced','Balanced_RF')]:
        for split in ['validation', 'test']:
            old = previous[(previous.model==old_name)&(previous.split==split)].iloc[0]
            new = next(r for r in metrics if r['model']==new_name and r['split']==split)
            for key in ['tn','fp','fn','tp']: assert int(old[key]) == new[key], (old_name, split, key)
    figures(metrics, selected, meta, probabilities)
    protocol.update(completed=True, validation_selected=selected, prior_baselines_exactly_replayed=True)
    dump('protocol.json', protocol)
    print('COMPLETE', OUT, 'VALIDATION_SELECTED', selected, flush=True)


if __name__ == '__main__':
    main()
