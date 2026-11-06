"""PyCharm entry: one frozen dynamic protocol repair, then evidence summaries.

The original training module and all historical results remain unchanged.
No partition search or architecture/hyperparameter search is performed.
"""
from pathlib import Path
import copy
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'ble_research'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ble_dynamic_learning_baselines as m
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

OUT = ROOT / 'SCI-UWB/data/results/ble_evidence_closure_20261105'
STATE = {}
SPLITS = {}
PROTOCOL = []
TIMINGS = []


def prepare_normalization(x, train_idx, test_idx):
    key = tuple(train_idx.tolist())
    assert key in SPLITS
    fold = SPLITS[key]
    groups = STATE['groups'][train_idx]
    inner, validation = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                              random_state=42 + fold).split(train_idx, groups=groups))
    assert not set(groups[inner]) & set(groups[validation])
    mean = x[train_idx[inner]].mean(axis=(0, 1), keepdims=True)
    std = x[train_idx[inner]].std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1
    STATE.update(fold=fold, inner=inner, validation=validation, outer_train=train_idx,
                 outer_test=test_idx, mean=mean, std=std)
    np.savez(OUT / 'dynamic_nested' / f'normalization_fold{fold}.npz', mean=mean, std=std)
    return ((x[train_idx] - mean) / std).astype(np.float32), ((x[test_idx] - mean) / std).astype(np.float32)


def repaired_training(model_name, task, fold, x_train, y_train, x_test, y_test,
                      num_classes, args, output_dir, pso_pseudo_train=None, pso_conf_train=None):
    assert fold == STATE['fold']
    m.set_seed(args.seed + fold)
    inner, validation = STATE['inner'], STATE['validation']
    assert len(np.unique(y_train[inner])) == num_classes
    assert len(np.unique(y_train[validation])) == num_classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = m.SequenceClassifier(model_name=model_name, input_size=x_train.shape[2],
                                 hidden_size=args.hidden_size, num_classes=num_classes,
                                 dropout=args.dropout).to(device)
    tensors = [torch.tensor(x_train[inner]), torch.tensor(y_train[inner], dtype=torch.long)]
    if pso_pseudo_train is not None:
        tensors.extend([torch.tensor(pso_pseudo_train[inner], dtype=torch.long),
                        torch.tensor(pso_conf_train[inner], dtype=torch.float32)])
    generator = torch.Generator().manual_seed(args.seed + fold)
    train_loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size,
                              shuffle=True, generator=generator)
    validation_loader = DataLoader(TensorDataset(torch.tensor(x_train[validation]),
                                  torch.tensor(y_train[validation], dtype=torch.long)),
                                  batch_size=args.batch_size, shuffle=False)
    criterion = m.make_sequence_criterion(y_train[inner], num_classes, device, args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score, best_epoch, best_state, stale, rows = -np.inf, 0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            xb, yb = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if len(batch) == 4 and args.pso_guided_weight > 0:
                pseudo, confidence = batch[2].to(device), batch[3].to(device)
                valid = (pseudo >= 0) & (confidence > 0)
                if args.pso_guided_agree_only:
                    valid &= pseudo == yb
                if valid.any():
                    auxiliary = torch.nn.functional.cross_entropy(logits[valid], pseudo[valid], reduction='none')
                    loss += args.pso_guided_weight * (auxiliary * confidence[valid]).sum() / confidence[valid].sum().clamp_min(1e-6)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_loss, true_val, pred_val = m.evaluate_torch_model(model, validation_loader, device, criterion)
        score = f1_score(true_val, pred_val, average='macro', zero_division=0)
        rows.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_loss=val_loss,
                         val_accuracy=accuracy_score(true_val, pred_val), val_macro_f1=score))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        (OUT / 'progress.json').write_text(json.dumps(dict(task=task, fold=fold, epoch=epoch,
             validation_macro_f1=score, test_evaluated=False)), encoding='utf-8')
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    # Outer test labels are first evaluated here, after checkpoint selection ends.
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long)),
                            batch_size=args.batch_size, shuffle=False)
    _, _, pred = m.evaluate_torch_model(model, test_loader, device, criterion)
    curve = pd.DataFrame(rows)
    curve['best_epoch'] = best_epoch
    curve.to_csv(output_dir / f'curve_{task}_{model_name}_fold{fold}.csv', index=False)
    torch.save(best_state, output_dir / f'model_{task}_{model_name}_fold{fold}.pt')
    full_groups = STATE['groups']
    train_sources = sorted(map(int, set(full_groups[STATE['outer_train'][inner]])))
    val_sources = sorted(map(int, set(full_groups[STATE['outer_train'][validation]])))
    test_sources = sorted(map(int, set(full_groups[STATE['outer_test']])))
    assert not set(train_sources) & set(test_sources)
    assert not set(val_sources) & set(test_sources)
    PROTOCOL.append(dict(task=task, fold=fold, inner_training_sources=train_sources,
                         validation_sources=val_sources, test_sources=test_sources,
                         best_epoch=best_epoch, best_validation_macro_f1=best_score,
                         normalization_fit='inner_training_only', test_evaluation_count=1))
    (OUT / 'split_protocol.json').write_text(json.dumps(PROTOCOL, indent=2), encoding='utf-8')
    if fold == 1:
        cpu_model = copy.deepcopy(model).cpu().eval()
        example = torch.tensor(x_test[:1])
        durations = []
        with torch.inference_mode():
            for _ in range(20):
                cpu_model(example)
            for _ in range(100):
                tick = time.perf_counter()
                cpu_model(example)
                durations.append((time.perf_counter() - tick) * 1000)
        TIMINGS.append(dict(task=task, cpu_median_ms=float(np.median(durations)),
                            cpu_p95_ms=float(np.percentile(durations, 95)), batch_size=1,
                            repeats=100, warmup=20, threads=torch.get_num_threads(),
                            parameters=sum(p.numel() for p in cpu_model.parameters()),
                            scope='forward only; excludes acquisition, preprocessing, communication'))
        pd.DataFrame(TIMINGS).to_csv(OUT / 'dynamic_forward_timings.csv', index=False)
    print(f'CORRECTED {task} fold={fold} best_epoch={best_epoch} outer_F1={f1_score(y_test,pred,average="macro"):.6f}', flush=True)
    return pred, curve


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'dynamic_nested').mkdir()
    torch.set_num_threads(1)
    print('Interpreter:', sys.executable, 'Torch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), flush=True)
    arrays, _, _ = m.load_dynamic_dataset(ROOT / 'ble_research/data/dynamic_kalman')
    STATE['groups'] = arrays['source_id']
    for fold, (training, _) in enumerate(GroupKFold(5).split(np.zeros(len(STATE['groups'])), groups=STATE['groups']), 1):
        SPLITS[tuple(training.tolist())] = fold
    m.normalize_sequence_by_train = prepare_normalization
    m.train_sequence_model = repaired_training
    sys.argv = ['dynamic_protocol_repair', '--output-dir', str(OUT / 'dynamic_nested'),
                '--models', 'trend_bilstm_attn', '--tasks', 'direction,action', '--n-splits', '5',
                '--seed', '42', '--epochs', '30', '--batch-size', '128', '--hidden-size', '64',
                '--dropout', '.25', '--lr', '.001', '--patience', '8', '--sequence-loss', 'focal',
                '--focal-gamma', '1.5', '--pso-guided-loss', 'action_trend', '--pso-guided-weight', '.2',
                '--sequence-feature-mode', 'rssi_delta', '--no-epoch-log']
    m.main()
    from summarize_ble_thesis_evidence import main as summarize
    summarize(OUT)
    (OUT / 'completed.json').write_text(json.dumps(dict(completed=True, runs=10,
         scope='same frozen five outer folds; one architecture and seed; one grouped inner holdout',
         historical_test_selection=True, independent_external_validation=False,
         interpreter=sys.executable), indent=2), encoding='utf-8')
    print('ALL FOUR EVIDENCE TASKS FINISHED:', OUT, flush=True)


if __name__ == '__main__':
    main()
