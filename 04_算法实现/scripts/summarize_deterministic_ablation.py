"""All-seed, all-source descriptive comparison. No significance claims."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]/'SCI-UWB'
OUT=ROOT/'data/results/vehicle_deterministic_input_ablation_20261030'

def main():
    assert json.loads((OUT/'protocol.json').read_text())['completed']
    p=pd.read_csv(OUT/'predictions.csv'); d=pd.read_csv(OUT/'metrics.csv')
    keys=['seed','source_file','window_start','window_end']
    assert not p.duplicated(keys+['model']).any()
    pivot=p.pivot(index=keys,columns='model',values='distance_error_cm')
    assert pivot.notna().all().all() and len(d)==9
    # Recomputed metrics include tails and failure rate on every run.
    rows=[]
    for (seed,model),g in p.groupby(['seed','model']):
        e=g.distance_error_cm.to_numpy()
        rows.append(dict(seed=seed,model=model,mean_cm=e.mean(),rmse_cm=np.sqrt((e**2).mean()),p90_cm=np.quantile(e,.9),p95_cm=np.quantile(e,.95),over100_pct=100*np.mean(e>100)))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'extended_metrics.csv',index=False)
    summary=metrics.groupby('model').agg({c:['mean','std'] for c in ['mean_cm','rmse_cm','p90_cm','p95_cm','over100_pct']})
    summary.to_csv(OUT/'summary.csv')
    by=pivot.groupby(['seed','source_file']).mean()
    for model in ['no_quality','sequence_only']:
        by[f'full_minus_{model}']=by.full-by[model]
    by.to_csv(OUT/'source_paired_differences.csv')
    source=by.groupby('source_file')[['full_minus_no_quality','full_minus_sequence_only']].agg(['mean','min','max'])
    source.to_csv(OUT/'source_consistency.csv')
    # Check identical full-model output against the preceding scaling experiment.
    prior=pd.read_csv(ROOT/'data/results/vehicle_quality_scaling_control_20261030/predictions.csv')
    a=p[p.model=='full'].set_index(keys).sort_index()
    b=prior[prior.model=='standard'].set_index(keys).sort_index()
    assert a.index.equals(b.index)
    repeat=float(np.max(np.abs(a[['pred_x_cm','pred_y_cm']].values-b[['pred_x_cm','pred_y_cm']].values)))
    assert repeat==0, f'Full-model reproducibility changed: {repeat}'
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3))
    modes=['full','no_quality','sequence_only']
    for ax,column,label in zip(axes,['mean_cm','p90_cm'],['Mean planar error (cm)','P90 planar error (cm)']):
        for (seed,g),color,marker in zip(metrics.groupby('seed'),['#446688','#cc8844','#667766'],['o','s','^']):
            ax.plot(range(3),g.set_index('model').loc[modes,column],color=color,marker=marker,label=str(seed))
        ax.set_xticks(range(3),['Full input','No quality\ninput','Sequence\ninput only']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(title='Training seed',fontsize=7)
    fig.suptitle('Information ablation: same architecture and grouped splits')
    fig.tight_layout();fig.savefig(OUT/'paired_ablation.png',dpi=300);fig.savefig(OUT/'paired_ablation.svg');fig.savefig(OUT/'paired_ablation.pdf');plt.close(fig)
    qa=dict(full_repeat_max_coordinate_difference_cm=repeat,windows_per_run=int(len(p)/9),sources=p.source_file.nunique(),figure_contract='Quantitative two-panel grid: compare mean and tail error under paired seeds; all nine runs, no exclusions. Descriptive algorithmic repetitions, not independent data repetitions. No confidence intervals. Editable vector text plus 300dpi PNG preview; no submission TIFF required.',limitations='Fixed existing processed dataset; repeated test-source exploration. Zeroed information keeps network branches and biases. No quality still retains WLS input computed using original quality.')
    (OUT/'analysis_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(summary.to_string());print('Full repeat max difference:',repeat)
    for col in ['full_minus_no_quality','full_minus_sequence_only']:
        print(col,'always improves:',int((source[(col,'max')]<0).sum()),'always worse:',int((source[(col,'min')]>0).sum()))

if __name__=='__main__':main()

