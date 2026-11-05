"""Run in PyCharm with the configured project interpreter."""
from vehicle_ppt_gated_filter_control import *
import platform
def run(training_seed=20261030):
    os.chdir(ROOT)
    directory='pycharm_gated_mechanisms_20261105' if training_seed==20261030 else f'pycharm_gated_mechanisms_20261105_seed{training_seed}'
    out=ROOT/'SCI-UWB/data/results'/directory;out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_pqg_cw_cnn_robust_kalman_tuning_results/best_tuned_config.json').read_text())
    sys.argv=['mechanisms'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','lr','weight_decay','seed']:setattr(args,k,cfg[k])
    wide=pd.read_csv(args.input);seq,meta=m.build_sequence_windows(wide,args.window,args.step);aux,q,r=m.build_window_auxiliary(wide,args.window,args.step);groups=meta.source_file.to_numpy();state={};cls=m.PhysicsQualityGuidedCNN
    def split(ids,seed,val_fraction=.2):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed).split(ids,groups=groups[ids]));state['fold']=seed-args.seed;return ids[a],ids[b]
    def factory(*a,**kw):
        m.set_seed(training_seed+state['fold']);model=cls(*a,**kw)
        if state['mode']=='identity_anchor_gate':model.quality_gate.register_forward_hook(lambda module,inputs,output:(torch.ones_like(output[0]),output[1]))
        if state['mode']=='direct_head_only':model.fusion_gate.register_forward_hook(lambda module,inputs,output:torch.ones_like(output))
        return model
    m.split_train_val=split;m.PhysicsQualityGuidedCNN=factory;torch.use_deterministic_algorithms(True)
    print('Interpreter:',sys.executable,'Torch:',torch.__version__,flush=True)
    rows=[]
    for mode in ['full','identity_anchor_gate','direct_head_only']:
        state['mode']=mode;print('Running',mode,flush=True)
        _,raw=m.evaluate_model(mode,seq,meta,q,m.compute_wls_features(r,q),args,'gated');raw.to_csv(out/f'{mode}_raw.csv',index=False)
        filtered=apply_filter(raw,**{k:cfg[k] for k in ['process_var','measurement_var','init_var','robust_c']});filtered.to_csv(out/f'{mode}_filtered.csv',index=False)
        for stage,p in [('raw',raw),('filtered',filtered)]:rows.append(metrics_frame(mode+'_'+stage,'grouped',p))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,interpreter=sys.executable,torch=torch.__version__,training_seed=training_seed,fold_seed='training_seed+fold',config=cfg,limitations='One seed, original grouped split; historical fixed filter. Hooks retain initialization and parameter layout; disabled heads remain allocated.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':
    import importlib
    for training_seed in [20261040,20261050]:
        importlib.reload(m)
        print('BASE SEED',training_seed,flush=True)
        run(training_seed)
