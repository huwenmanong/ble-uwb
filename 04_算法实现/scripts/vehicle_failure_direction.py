"""Post-hoc directional diagnostic, never a test-point correction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]/'SCI-UWB'
def main():
    out=ROOT/'data/results/vehicle_failure_direction_20261102';out.mkdir(exist_ok=False)
    path=ROOT/'data/results/vehicle_model_complexity_control_20261030'
    p=pd.read_csv(path/'predictions.csv');b=pd.read_csv(ROOT/'data/results/vehicle_rbf_control_20261102/predictions.csv');b['seed']=0
    p=pd.concat([p,b],ignore_index=True)
    target='正右侧\\-500cm，120cm.csv'
    protocol=json.loads((path/'protocol.json').read_text(encoding='utf-8'))
    fold=next(x for x in protocol['ridge_splits'] if x['model']=='ridge_mean3' and target in x['test_sources'])
    coords=p[['source_file','x_cm','y_cm']].drop_duplicates().set_index('source_file')
    coverage=coords.copy();coverage['role']=['train' if s in fold['train_sources'] else 'validation' if s in fold['val_sources'] else 'test' for s in coverage.index]
    coverage['distance_to_target_cm']=np.linalg.norm(coverage[['x_cm','y_cm']].values-coords.loc[target].values,axis=1)
    coverage.to_csv(out/'position_coverage.csv')
    rows=[]
    for (source,model,seed),g in p.groupby(['source_file','model','seed']):
        dx=g.pred_x_cm-g.x_cm;dy=g.pred_y_cm-g.y_cm
        rows.append(dict(source_file=source,model=model,seed=int(seed),bias_x_cm=dx.mean(),bias_y_cm=dy.mean(),mse_y_fraction=float((dy**2).sum()/((dx**2+dy**2).sum())),mean_cm=g.distance_error_cm.mean(),pred_y_mean=g.pred_y_cm.mean()))
    d=pd.DataFrame(rows);d.to_csv(out/'directional_errors_all_sources.csv',index=False)
    subset=d[d.source_file==target];subset.to_csv(out/'target_errors.csv',index=False)
    (out/'target_fold.json').write_text(json.dumps(fold,ensure_ascii=False,indent=2),encoding='utf-8')
    print(subset.to_string(index=False));print(coverage.sort_values('distance_to_target_cm').to_string());print('fold',fold['fold'])
if __name__=='__main__':main()

