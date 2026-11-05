"""Deterministic RBF controls; inner-group tuning only."""
from vehicle_model_complexity_control import *
from sklearn.kernel_ridge import KernelRidge
from threadpoolctl import threadpool_limits

def run():
    os.chdir(ROOT);out=ROOT/'SCI-UWB/data/results/vehicle_rbf_control_20261102';out.mkdir(exist_ok=False)
    cfg=json.loads((ROOT/'UWB测距数据/uwb_physics_quality_guided_cnn_results/physics_quality_guided_cnn_config.json').read_text(encoding='utf-8'))
    source=ROOT/cfg['input'];wide=pd.read_csv(source)
    meta,q,r=m.build_window_auxiliary(wide,cfg['window'],cfg['step'])
    y=meta[['x_cm','y_cm']].to_numpy();groups=meta.source_file.to_numpy();weights=m.sample_weights(meta)
    logs=[];frames=[];metrics=[]
    for name,x in [('rbf_mean3',r),('rbf_stats24',q)]:
        pred=np.zeros_like(y)
        for fold,(train,test) in enumerate(GroupKFold(5).split(x,y,groups),1):
            a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=cfg['seed']+fold).split(train,groups=groups[train]));core,val=train[a],train[b]
            sx=StandardScaler().fit(x[core]);sy=StandardScaler().fit(y[core]);best=None
            for alpha in [.001,.01,.1,1,10]:
                for factor in [.01,.1,1,10]:
                    gamma=factor/x.shape[1]
                    model=KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma).fit(sx.transform(x[core]),sy.transform(y[core]),sample_weight=weights[core])
                    loss=float(np.mean((model.predict(sx.transform(x[val]))-sy.transform(y[val]))**2))
                    logs.append(dict(model=name,fold=fold,alpha=alpha,gamma=gamma,val_mse=loss))
                    if best is None or loss<best[0]:best=(loss,model)
            pred[test]=sy.inverse_transform(best[1].predict(sx.transform(x[test])))
            print(name,fold,flush=True)
        p=meta.copy();p['model']=name;p['pred_x_cm']=pred[:,0];p['pred_y_cm']=pred[:,1];e=np.linalg.norm(pred-y,axis=1);p['distance_error_cm']=e;frames.append(p)
        metrics.append(dict(model=name,mean_cm=e.mean(),rmse_cm=np.sqrt(np.mean(e**2)),p90_cm=np.quantile(e,.9),p95_cm=np.quantile(e,.95),over100_pct=100*np.mean(e>100)))
    pd.concat(frames).to_csv(out/'predictions.csv',index=False);pd.DataFrame(logs).to_csv(out/'tuning.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(completed=True,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),split_seed=cfg['seed'],refit=False,grid_alpha=[.001,.01,.1,1,10],grid_gamma_factor=[.01,.1,1,10],limitations='Exploratory existing processed data; deterministic single run; train-core scaling and weights; normalized validation MSE selection; no independent confirmation.'),indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()

