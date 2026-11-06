"""PyCharm: audit recorded timing and benchmark existing CPU checkpoints; no training."""
from pathlib import Path
import json
import os
import platform
import sys
import time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ble_research'))
import run_ble_dynamic_learning_baselines as m
import numpy as np
import pandas as pd
import torch

OUT=ROOT/'SCI-UWB/data/results/window_timing_20261105'

def stats(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    return dict(n=len(x),minimum=float(np.min(x)),median=float(np.median(x)),
                p95=float(np.percentile(x,95)),maximum=float(np.max(x))) if len(x) else dict(n=0)

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=False)
    samples=pd.read_csv(ROOT/'ble_research/data/dynamic_kalman/ble_dynamic_kalman_samples.csv')
    windows=pd.read_csv(ROOT/'ble_research/data/dynamic_kalman/ble_dynamic_kalman_window_manifest.csv')
    assert not samples.duplicated(['source_id','sample_index']).any()
    lookup=samples.set_index(['source_id','sample_index']).time_seconds
    gaps=samples.sort_values(['source_id','sample_index']).groupby('source_id').time_seconds.diff()
    source_checks=[]
    for sid,part in samples.groupby('source_id'):
        d=part.sort_values('sample_index').time_seconds.diff().dropna()
        source_checks.append(dict(source_id=int(sid),samples=len(part),nonpositive=int((d<=0).sum()),
                                  missing_times=int(part.time_seconds.isna().sum()),**stats(d)))
    valid_sources={r['source_id'] for r in source_checks if r['nonpositive']==0 and r['missing_times']==0}
    for label,col in [('start','window_start_index'),('end','window_end_index'),('center','window_center_index')]:
        keys=pd.MultiIndex.from_arrays([windows.source_id,windows[col]])
        windows[label+'_time_s']=lookup.reindex(keys).to_numpy()
    windows['span_s']=windows.end_time_s-windows.start_time_s
    windows['center_lag_s']=windows.end_time_s-windows.center_time_s
    windows=windows.sort_values(['source_id','window_start_index'])
    windows['update_s']=windows.groupby('source_id').end_time_s.diff()
    windows['source_monotonic']=windows.source_id.isin(valid_sources)
    windows.to_csv(OUT/'window_times.csv',index=False)
    pd.DataFrame(source_checks).to_csv(OUT/'source_time_checks.csv',index=False)
    valid=windows[windows.source_monotonic]
    audit=dict(sources=len(source_checks),valid_sources=len(valid_sources),windows=len(windows),valid_windows=len(valid),
               nonpositive_intervals=int((gaps<=0).sum()),missing_time_values=int(samples.time_seconds.isna().sum()),
               positive_recorded_intervals_s=stats(gaps[gaps>0]),window_span_s=stats(valid.span_s),
               center_availability_lag_s=stats(valid.center_lag_s),update_interval_s=stats(valid.update_s),
               source_checks=source_checks,
               note='Recorded/logged timestamps, not verified device sampling rate; no interpolation or rollover repair.',
               static_ble='No verified timestamps in existing UKF total table; W16/S4: span15/f, updates4/f.',
               uwb='Sampling rate not verified this run; W30/S10: span29/f, updates10/f.')
    (OUT/'timing_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    torch.set_num_threads(1)
    arrays,manifest,_=m.load_dynamic_dataset(ROOT/'ble_research/data/dynamic_kalman')
    x=m.build_sequence_features(arrays,'rssi_delta')
    prior=ROOT/'SCI-UWB/data/results/ble_dynamic_nested_20261105'
    protocol=json.loads((prior/'protocol.json').read_text(encoding='utf-8'))
    split=protocol['splits'][0]
    _,test=m.normalize_sequence_by_train(x,np.array(split['train']),np.array(split['test']))
    # Representative first held-out window, already normalized; transfer/file loading excluded.
    example=torch.tensor(test[:1],dtype=torch.float32)
    timings=[]
    for variant,model_name in [('lstm_ce','lstm'),('trend_focal','trend_bilstm_attn'),('trend_focal_pso','trend_bilstm_attn')]:
        for task,classes in [('direction',3),('action',2)]:
            model=m.SequenceClassifier(model_name=model_name,input_size=test.shape[2],hidden_size=64,
                                       num_classes=classes,dropout=.25).cpu().eval()
            checkpoint=prior/variant/task/'fold1'/f'model_{task}_{model_name}_fold1.pt'
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            durations=[]
            with torch.inference_mode():
                for _ in range(20):model(example)
                for _ in range(200):
                    start=time.perf_counter_ns();model(example)
                    durations.append((time.perf_counter_ns()-start)/1e6)
            timings.append(dict(variant=variant,task=task,parameters=sum(p.numel() for p in model.parameters()),
                                batch=1,warmup=20,repeats=200,threads=1,latency_ms=stats(durations),
                                checkpoint=str(checkpoint.relative_to(ROOT))))
    result=dict(completed=True,interpreter=sys.executable,torch=torch.__version__,platform=platform.platform(),
                processor=platform.processor(),scope='CPU float32 forward only; input ready in RAM; no training.',
                excluded=['acquisition','radio communication','window assembly','feature building','normalization',
                          'filtering','device transfer','vehicle command execution'],timings=timings)
    (OUT/'cpu_forward_benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='source_checks'},indent=2),flush=True)
    print(json.dumps(result,indent=2),flush=True)
    print('COMPLETE',OUT,flush=True)

if __name__=='__main__':main()
