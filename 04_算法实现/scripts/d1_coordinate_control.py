"""Isolate authoritative coordinate-label correction, without changing ranges."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from threadpoolctl import threadpool_limits
import d1_v2_research as v2
from d1_v3_protocol_audit import canonical, unique_dir

def main():
    prior=v2.ROOT/'data/results/d1_v3_protocol_20261029_repeat01'
    out=unique_dir(v2.ROOT/'data/results/d1_coordinate_control_20261030')
    w=pd.read_csv(prior/'window_features.csv',float_precision='round_trip')
    meta=pd.read_csv(prior/'official_json_window_audit.csv',float_precision='round_trip')
    w['canonical_position']=w.raw_folder.map(lambda s:canonical(*s.split('_')))
    keys=['environment_id','canonical_position','anchor_label','channel']
    w=w.merge(meta[keys+['processed_x','processed_y','processed_z']],on=keys,validate='one_to_one')
    assert len(w)==15130
    old=w[['tag_x','tag_y','tag_z']].to_numpy().copy()
    for axis in 'xyz':
        w['tag_'+axis]=w['processed_'+axis]
    new=w[['tag_x','tag_y','tag_z']].to_numpy()
    w['coordinate_shift_m']=np.linalg.norm(new-old,axis=1)
    w['range_true_offline']=np.linalg.norm(new-w[['anchor_x','anchor_y','anchor_z']].to_numpy(),axis=1)
    w['error_label']=w.range_measured-w.range_true_offline
    w.to_csv(out/'window_features.csv',index=False)
    locs=[]
    features=list(v2.BASE_NUMERIC_FEATURES)+['range_std','sample_n']+v2.CIR_NUMERIC_FEATURES
    for fold in v2.FOLDS:
        print(fold,flush=True)
        rows,intervals=v2.run_fold(w,fold,'cir',features,20261029,out)
        locs.extend(rows)
    loc=pd.DataFrame(locs)
    assert np.isfinite(loc.error_m).all() and loc.success.all()
    loc.to_csv(out/'localization.csv',index=False)
    summary=pd.DataFrame([dict(fold=f,method=m,**v2.metrics(g.error_m)) for (f,m),g in loc.groupby(['fold','method'])])
    summary.to_csv(out/'summary.csv',index=False)
    v2.bootstrap(loc,20261029).to_csv(out/'paired_bootstrap.csv',index=False)
    (out/'protocol.json').write_text(json.dumps({'completed':True,'intervention':'Only authoritative tag coordinate labels corrected; ranges and deployable features unchanged','limitations':['positive-range filter retained','raw uncalibrated range retained','E3 previously excluded position not restored','development evidence only'],'changed_windows':int((w.coordinate_shift_m>1e-8).sum())},indent=2),encoding='utf-8')
    print(summary.to_string(index=False))
    print(out)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
