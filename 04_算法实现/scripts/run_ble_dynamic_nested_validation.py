"""PyCharm entry: preserve historical outer folds, select epochs on inner groups only."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ble_research'))
import run_ble_dynamic_learning_baselines as m
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

OUTPUT = ROOT / 'SCI-UWB/data/results/ble_dynamic_nested_20261105'


def predict_checkpoint(model_name, x, classes, args, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = m.SequenceClassifier(model_name=model_name, input_size=x.shape[2],
                                hidden_size=args.hidden_size, num_classes=classes,
                                dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), args.batch_size):
            batch = torch.tensor(x[start:start+args.batch_size], dtype=torch.float32, device=device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def main():
    os.chdir(ROOT)
    sys.argv = ['nested-validation']
    args = m.parse_args()
    config_path = ROOT / 'ble_research/results/dynamic_learning_trend_bilstm_attn_focal_pso/config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.input_dir = ROOT / 'ble_research/data/dynamic_kalman'
    args.pso_parameters = ROOT / config['pso_parameters']
    args.no_epoch_log = True
    OUTPUT.mkdir(parents=True, exist_ok=True)
    arrays, manifest, _ = m.load_dynamic_dataset(args.input_dir)
    x = m.build_sequence_features(arrays, args.sequence_feature_mode)
    groups = arrays['source_id']
    outer = m.make_group_splits(groups, args.n_splits)
    plan = []
    for fold, (pool, test) in enumerate(outer, 1):
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                      random_state=args.seed+fold).split(pool, groups=groups[pool]))
        train, validation = pool[a], pool[b]
        group_sets = [set(groups[ids].tolist()) for ids in (train, validation, test)]
        assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
        for task in ('direction', 'action'):
            y = arrays[m.TASKS[task]['target_key']]
            for ids in (train, validation, test):
                assert set(y[ids].tolist()) == set(range(len(m.TASKS[task]['labels']))), (fold, task)
        plan.append(dict(fold=fold, train=train.tolist(), validation=validation.tolist(), test=test.tolist(),
                         train_groups=sorted(map(str, group_sets[0])),
                         validation_groups=sorted(map(str, group_sets[1])),
                         test_groups=sorted(map(str, group_sets[2]))))
    protocol = dict(completed=False, interpreter=sys.executable, torch=torch.__version__,
                    device='cuda' if torch.cuda.is_available() else 'cpu', config=config,
                    historical_source_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
                    outer='Historical make_group_splits unchanged', inner='GroupShuffleSplit .2 seed42+fold',
                    scaler='Fit inner-training only; validation/test transformed with same parameters',
                    epoch_selection='Inner-validation Macro-F1 only',
                    test_prediction='Checkpoint inference receives X only; test labels used by metrics afterwards',
                    pso_boundary='Prior static calibration; true direction used for TRAINING auxiliary labels only',
                    limitations='Previously explored data, not new blind data; center-aligned windows; one training seed',
                    splits=plan)
    protocol_path = OUTPUT / 'protocol.json'
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text(encoding='utf-8'))
        if prior['historical_source_sha256'] != protocol['historical_source_sha256'] or prior['splits'] != plan:
            raise RuntimeError('Protocol/source changed; use a new output directory.')
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verified all 5 outer folds: train/validation/test source groups disjoint.', flush=True)
    print('Interpreter:', sys.executable, 'Device:', protocol['device'], flush=True)
    guidance = m.build_pso_action_trend_guidance(arrays, manifest, args)
    variants = [('lstm_ce', 'lstm', 'ce', False),
                ('trend_focal', 'trend_bilstm_attn', 'focal', False),
                ('trend_focal_pso', 'trend_bilstm_attn', 'focal', True)]
    metrics = []
    for variant, model_name, loss, with_pso in variants:
        args.sequence_loss = loss
        for task in ('direction', 'action'):
            y = arrays[m.TASKS[task]['target_key']].astype(int)
            classes = len(m.TASKS[task]['labels'])
            for split in plan:
                fold = split['fold']
                job = OUTPUT / variant / task / f'fold{fold}'
                job.mkdir(parents=True, exist_ok=True)
                done = job / 'completed_metrics.json'
                if done.exists():
                    metrics.append(json.loads(done.read_text(encoding='utf-8')))
                    continue
                train, val, test = (np.asarray(split[key]) for key in ('train', 'validation', 'test'))
                heldout = np.concatenate([val, test])
                x_train, x_heldout = m.normalize_sequence_by_train(x, train, heldout)
                x_val, x_test = x_heldout[:len(val)], x_heldout[len(val):]
                pseudo = guidance.iloc[train]['pso_pseudo_action_id'].to_numpy(dtype=np.int64) if with_pso and task == 'action' else None
                confidence = guidance.iloc[train]['pso_trend_confidence'].to_numpy(dtype=np.float32) if pseudo is not None else None
                args.pso_guided_loss = 'action_trend' if pseudo is not None else 'none'
                m.set_seed(args.seed+fold)
                print('TRAIN', variant, task, fold, flush=True)
                _, curve = m.train_sequence_model(model_name, task, fold, x_train, y[train],
                                                   x_val, y[val], classes, args, job, pseudo, confidence)
                checkpoint = job / f'model_{task}_{model_name}_fold{fold}.pt'
                predicted = predict_checkpoint(model_name, x_test, classes, args, checkpoint)
                result = asdict(m.collect_metrics(y[test], predicted, variant, task, fold, len(train), len(test)))
                result.update(validation_size=len(val), best_epoch=int(curve['best_epoch'].iloc[0]))
                pd.DataFrame(dict(window_id=manifest.iloc[test]['window_id'].to_numpy(),
                                  source_id=groups[test], y_true=y[test], y_pred=predicted)).to_csv(job/'test_predictions.csv', index=False)
                done.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                metrics.append(result)
                pd.DataFrame(metrics).to_csv(OUTPUT/'fold_metrics.csv', index=False)
                print('TEST', result, flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUTPUT/'fold_metrics.csv', index=False)
    summary = m.summarize_metrics(frame)
    summary.to_csv(OUTPUT/'summary_metrics.csv', index=False)
    protocol['completed'] = True
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(summary.to_string(index=False), flush=True)
    print('COMPLETE', OUTPUT, flush=True)


if __name__ == '__main__':
    main()
