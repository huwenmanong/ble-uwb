"""Three predeclared group partitions; no test-based selection."""
from vehicle_rbf_control import *
def run(training_seeds=(20261030,)):
    os.chdir(ROOT);out=ROOT/'SCI-UWB/data/results'/('vehicle_split_multiseed_20261102' if len(training_seeds)>1 else 'vehicle_split_sensitivity_20261102');out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    sys.argv=['sensitivity'];args=m.parse_args()
    for k in ['window','step','epochs','patience','batch_size','aux_lambda','seed']:setattr(args,k,cfg[k])
    wide=pd.read_csv(ROOT/cfg['input']);seq,meta=m.build_sequence_windows(wide,args.window,args.step);aux,q,r=m.build_window_auxiliary(wide,args.window,args.step)
    assert meta[['source_file','window_start','window_end']].equals(aux[['source_file','window_start','window_end']])
    groups=meta.source_file.to_numpy();y=meta[['x_cm','y_cm']].to_numpy();wls=m.compute_wls_features(r,q);weights=m.sample_weights(meta)
    frames=[];logs=[];plans=[];state={};cls=m.DirectQualityPhysicsCNN
    for partition in [20261102,20261103,20261104]:
        names=np.random.default_rng(partition).permutation(np.unique(groups));folds=[]
        for fold,testnames in enumerate(np.array_split(names,5),1):
            test=np.flatnonzero(np.isin(groups,testnames));train=np.flatnonzero(~np.isin(groups,testnames))
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=partition+fold).split(train,groups=groups[train]));core,val=train[a],train[b]
            folds.append((train,test,core,val));plans.append(dict(partition=partition,fold=fold,train=sorted(set(groups[core])),validation=sorted(set(groups[val])),test=sorted(set(groups[test]))))
        class FixedCV:
            def __init__(self,*a,**kw):pass
            def split(self,*a,**kw):
                for train,test,core,val in folds:yield train,test
        def split(ids,seed,val_fraction=.2):
            fold=seed-args.seed;state['fold']=fold;return folds[fold-1][2:]
        def factory(*a,**kw):
            m.set_seed(state['training_seed']+state['fold']);return cls(*a,**kw)
        m.GroupKFold=FixedCV;m.split_train_val=split;m.DirectQualityPhysicsCNN=factory
        print('CNN partition',partition,flush=True)
        for training_seed in training_seeds:
            state['training_seed']=training_seed
            print('training seed',training_seed,flush=True)
            _,p=m.evaluate_model('full',seq,meta,q,wls,args,'direct');p['partition']=partition;p['training_seed']=training_seed;frames.append(p)
        pred=np.zeros_like(y)
        for fold,(_,test,core,val) in enumerate(folds,1):
            sx=StandardScaler().fit(r[core]);sy=StandardScaler().fit(y[core]);best=None
            for alpha in [.001,.01,.1,1,10]:
                for factor in [.01,.1,1,10]:
                    model=KernelRidge(alpha=alpha,kernel='rbf',gamma=factor/3).fit(sx.transform(r[core]),sy.transform(y[core]),sample_weight=weights[core])
                    loss=float(np.mean((model.predict(sx.transform(r[val]))-sy.transform(y[val]))**2));logs.append(dict(partition=partition,fold=fold,alpha=alpha,gamma=factor/3,val_mse=loss))
                    if best is None or loss<best[0]:best=(loss,model)
            pred[test]=sy.inverse_transform(best[1].predict(sx.transform(r[test])))
        p=meta.copy();p['model']='rbf_mean3';p['partition']=partition;p['training_seed']=0;p['pred_x_cm']=pred[:,0];p['pred_y_cm']=pred[:,1];p['distance_error_cm']=np.linalg.norm(pred-y,axis=1);frames.append(p)
    p=pd.concat(frames);p.to_csv(out/'predictions.csv',index=False)
    metrics=p.groupby(['partition','model','training_seed']).distance_error_cm.agg(mean_cm='mean',p90_cm=lambda s:s.quantile(.9),over100_pct=lambda s:100*(s>100).mean());metrics.to_csv(out/'metrics.csv')
    p.groupby(['partition','model','source_file']).distance_error_cm.mean().to_csv(out/'source_means.csv')
    pd.DataFrame(logs).to_csv(out/'tuning.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,plans=plans,training_seeds=training_seeds,fold_seed='training_seed+fold',limitations='Outer and inner partitions changed together. Descriptive repeated-data sensitivity, not independent confirmation or causal isolation.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(metrics.to_string())
if __name__=='__main__':
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):run()

