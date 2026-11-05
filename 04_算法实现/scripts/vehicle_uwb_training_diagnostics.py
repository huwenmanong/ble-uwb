"""Instrument unchanged loss/optimizer, fixed per-fold seeds, CPU repeat check."""
from pathlib import Path
import sys,os,json,inspect,hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import run_uwb_physics_quality_guided_cnn as m

def main():
    os.chdir(ROOT);out=ROOT/'SCI-UWB/data/results/vehicle_training_diagnostics_20261030_cpu';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    sys.argv=['diagnostic'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:setattr(args,k,cfg[k])
    source=ROOT/cfg['input'];wide=pd.read_csv(source)
    seq,meta=m.build_sequence_windows(wide,args.window,args.step);aux,q,r=m.build_window_auxiliary(wide,args.window,args.step);wls=m.compute_wls_features(r,q)
    groups=meta.source_file.to_numpy();logs=[];scales=[];state={'repeat':0,'fold':0}
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]))
        state['fold']=seed-cfg['seed'];return ids[a],ids[b]
    m.split_train_val=split
    original_class=m.DirectQualityPhysicsCNN
    def factory(*a,**kw):
        m.set_seed(20261030+state['fold']);return original_class(*a,**kw)
    m.DirectQualityPhysicsCNN=factory
    text=inspect.getsource(m.train_model).replace('for _ in range(args.epochs):','for epoch in range(args.epochs):').replace('        if val_loss < best_val - 1e-6:',"        record_epoch(epoch + 1, val_loss)\n        if val_loss < best_val - 1e-6:")
    namespace=dict(m.__dict__)
    namespace['record_epoch']=lambda epoch,loss:logs.append(dict(**state,epoch=epoch,val_loss=loss))
    exec(compile(text,'instrumented_training','exec'),namespace)
    original=namespace['train_model']
    def train(*a,**kw):
        for name,tr,va in [('sequence',a[1],a[5]),('quality',a[2],a[6]),('physical',a[3],a[7])]:
            scales.append(dict(**state,feature=name,train_abs_max=float(np.max(np.abs(tr))),validation_abs_max=float(np.max(np.abs(va))),validation_abs_over_5_fraction=float(np.mean(np.abs(va)>5))))
        return original(*a,**kw)
    # Actual positional order: model,x,q,wls,y,weights,xval,qval,wlsval,yval,args.
    def train_checked(*a,**kw):
        for name,tr,va in [('sequence',a[1],a[6]),('quality',a[2],a[7]),('physical',a[3],a[8])]:
            scales.append(dict(**state,feature=name,train_abs_max=float(np.max(np.abs(tr))),validation_abs_max=float(np.max(np.abs(va))),validation_abs_over_5_fraction=float(np.mean(np.abs(va)>5))))
        return original(*a,**kw)
    m.train_model=train_checked
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():raise RuntimeError('This audit requires CPU-only execution')
    predictions=[]
    for repeat in [1,2]:
        state['repeat']=repeat;print('Repeat',repeat,flush=True)
        result,p=m.evaluate_model('full',seq,meta,q,wls,args,'direct');p['repeat']=repeat;predictions.append(p)
    a,b=predictions;difference=float(np.max(np.abs(a[['pred_x_cm','pred_y_cm']].to_numpy()-b[['pred_x_cm','pred_y_cm']].to_numpy())))
    pd.concat(predictions).to_csv(out/'repeat_predictions.csv',index=False);pd.DataFrame(logs).to_csv(out/'epoch_logs.csv',index=False);pd.DataFrame(scales).to_csv(out/'feature_scales.csv',index=False)
    old=pd.read_csv(ROOT/'SCI-UWB/data/results/vehicle_uwb_stability_20261030/predictions.csv')
    by=old.groupby(['seed','source_file','model']).distance_error_cm.mean().unstack('model');by['full_minus_sequence_cm']=by.full-by.sequence_only_information;by.to_csv(out/'per_source_seed_differences.csv')
    summary=by.groupby('source_file').full_minus_sequence_cm.agg(['mean','min','max']);summary.to_csv(out/'source_consistency.csv')
    report={'completed':True,'repeat_max_coordinate_difference_cm':difference,'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'torch':torch.__version__,'original_model_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'per_fold_seed':'20261030+fold','limitations':'Deterministic repeats change seed protocol; do not establish cause of earlier variation. Fixed processed dataset.'}
    (out/'protocol.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report);print(summary.to_string());print(pd.DataFrame(scales).groupby('feature').validation_abs_max.max());print(out)

if __name__=='__main__':main()

