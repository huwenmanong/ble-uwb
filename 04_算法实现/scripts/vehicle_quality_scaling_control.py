"""CPU-only paired scaling experiment; run directly in bluekey/PyCharm."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from pathlib import Path
import sys, json, hashlib, inspect
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import run_uwb_physics_quality_guided_cnn as m

def main(experiment='scaling'):
    os.chdir(ROOT)
    assert experiment in ['scaling', 'input_ablation']
    directory = 'vehicle_quality_scaling_control_20261030' if experiment == 'scaling' else 'vehicle_deterministic_input_ablation_20261030'
    out = ROOT/'SCI-UWB/data/results'/directory
    if out.exists():
        raise FileExistsError(f'Preserving prior results: {out}')
    out.mkdir(parents=True)
    cfg = json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    sys.argv = ['quality_control']; args = m.parse_args()
    for key in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:
        setattr(args,key,cfg[key])
    source = ROOT/cfg['input']; wide = pd.read_csv(source)
    seq, meta = m.build_sequence_windows(wide,args.window,args.step)
    aux,q,r = m.build_window_auxiliary(wide,args.window,args.step)
    assert meta[['source_file','window_start','window_end']].equals(aux[['source_file','window_start','window_end']])
    wls = m.compute_wls_features(r,q); groups = meta.source_file.to_numpy()
    state = {}; logs = []; predictions = []; metrics = []
    def split(ids, seed, val_fraction=.2):
        a,b = next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]))
        state['fold'] = seed-cfg['seed']
        return ids[a],ids[b]
    m.split_train_val = split
    cls = m.DirectQualityPhysicsCNN
    def factory(*a,**kw):
        m.set_seed(state['seed']+state['fold'])
        return cls(*a,**kw)
    m.DirectQualityPhysicsCNN = factory
    def scale(fit,*arrays):
        scaler = (RobustScaler() if state['mode']=='robust' else StandardScaler()).fit(fit)
        result = [scaler.transform(a).astype(np.float32) for a in arrays]
        if state['mode']=='standard_clip5':
            result = [np.clip(a,-5,5) for a in result]
        return result
    m.standardize_matrix = scale
    code = inspect.getsource(m.train_model)
    assert 'for _ in range(args.epochs):' in code
    code = code.replace('for _ in range(args.epochs):','for epoch in range(args.epochs):').replace('        if val_loss < best_val - 1e-6:', '        record_epoch(epoch+1,val_loss)\n        if val_loss < best_val - 1e-6:')
    ns = dict(m.__dict__)
    ns['record_epoch'] = lambda epoch,loss: logs.append(dict(**state,epoch=epoch,val_loss=loss))
    exec(compile(code,'instrumented_training','exec'),ns)
    m.train_model = ns['train_model']; torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    modes = ['standard','standard_clip5','robust'] if experiment == 'scaling' else ['full','no_quality','sequence_only']
    for seed in [20261030,20261040,20261050]:
        for mode in modes:
            state.update(seed=seed,mode=mode)
            print(seed,mode,flush=True)
            q_input = np.zeros_like(q) if mode in ['no_quality','sequence_only'] else q
            wls_input = np.zeros_like(wls) if mode == 'sequence_only' else wls
            result,p = m.evaluate_model(mode,seq,meta,q_input,wls_input,args,'direct')
            result.update(seed=seed,mode=mode); metrics.append(result)
            p['seed']=seed; predictions.append(p)
            pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'epoch_logs.csv',index=False)
    protocol = dict(completed=True,experiment=experiment,modes=modes,device='cpu',torch=torch.__version__,seeds=[20261030,20261040,20261050],fold_seed='seed+fold',split_seed=cfg['seed'],clip_threshold=5 if experiment=='scaling' else None,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),limitations='Exploratory repeated analysis of existing processed data, not independent confirmation; scaling fit only on inner training groups. Input ablation keeps architecture and constant branch biases; no_quality retains quality-derived WLS inputs.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False)); print(out)

if __name__=='__main__':
    main()

