"""Frozen PPT gated model evaluated on saved balanced source partitions."""
from vehicle_ppt_gated_filter_control import *
import numpy as np
def run():
    os.chdir(ROOT);partition=int(os.environ['PPT_PARTITION']);root=ROOT/'SCI-UWB/data/results'
    out=root/f'ppt_partition_control_20261105_{partition}';out.mkdir(exist_ok=False)
    saved=json.loads((root/'vehicle_split_sensitivity_20261102/protocol.json').read_text(encoding='utf-8'))
    plans=[x for x in saved['plans'] if x['partition']==partition];assert len(plans)==5
    cfg=json.loads((ROOT/'UWB测距数据/uwb_pqg_cw_cnn_robust_kalman_tuning_results/best_tuned_config.json').read_text())
    sys.argv=['partition'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','lr','weight_decay','seed']:setattr(args,k,cfg[k])
    wide=pd.read_csv(args.input);seq,meta=m.build_sequence_windows(wide,args.window,args.step);aux,q,r=m.build_window_auxiliary(wide,args.window,args.step);groups=meta.source_file.to_numpy();state={}
    class FixedCV:
        def __init__(self,*a,**kw):pass
        def split(self,*a,**kw):
            for plan in plans:
                test=np.flatnonzero(np.isin(groups,plan['test']));train=np.flatnonzero(~np.isin(groups,plan['test']));yield train,test
    def split(ids,seed,val_fraction=.2):
        fold=seed-args.seed;state['fold']=fold;plan=plans[fold-1]
        core=np.flatnonzero(np.isin(groups,plan['train']));val=np.flatnonzero(np.isin(groups,plan['validation']))
        assert set(core)|set(val)==set(ids)
        return core,val
    cls=m.PhysicsQualityGuidedCNN
    def factory(*a,**kw):
        m.set_seed(20261030+state['fold']);return cls(*a,**kw)
    m.GroupKFold=FixedCV;m.split_train_val=split;m.PhysicsQualityGuidedCNN=factory;torch.use_deterministic_algorithms(True)
    _,raw=m.evaluate_model('gated',seq,meta,q,m.compute_wls_features(r,q),args,'gated')
    rb=pd.read_csv(root/'vehicle_split_sensitivity_20261102/predictions.csv');rb=rb[(rb.partition==partition)&(rb.model=='rbf_mean3')].copy()
    kw={k:cfg[k] for k in ['process_var','measurement_var','init_var','robust_c']};frames=[];rows=[]
    for name,p in [('gated_raw',raw),('gated_filtered',apply_filter(raw,**kw)),('rbf_raw',rb),('rbf_filtered',apply_filter(rb,**kw))]:
        rows.append(metrics_frame(name,'saved_balanced_partition',p));p['comparison_method']=name;frames.append(p)
    pd.concat(frames).to_csv(out/'predictions.csv',index=False);pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,partition=partition,plans=plans,training_seed='20261030+fold',config=cfg,limitations='One CNN seed; repeated exploratory existing data; historical filter selected on gated model.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':run()
