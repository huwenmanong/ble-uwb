"""Paired seed plots: show all runs, not independent data replications."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]/'SCI-UWB'
OUT=ROOT/'data/results/vehicle_quality_scaling_control_20261030'

def main():
    assert json.loads((OUT/'protocol.json').read_text())['completed']
    d=pd.read_csv(OUT/'metrics.csv')
    assert len(d)==9 and not d.duplicated(['seed','mode']).any()
    modes=['standard','standard_clip5','robust']
    columns=['mae_distance_cm','p90_distance_error_cm']
    summary=d.groupby('mode')[columns].agg(['mean','std','min','max'])
    summary.to_csv(OUT/'seed_summary.csv')
    epochs=pd.read_csv(OUT/'epoch_logs.csv')
    best=epochs.loc[epochs.groupby(['seed','mode','fold']).val_loss.idxmin()]
    best.to_csv(OUT/'best_epochs.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.3))
    for ax,col,label in zip(axes,columns,['Mean planar error (cm)','P90 planar error (cm)']):
        for (seed,g),color,marker in zip(d.groupby('seed'),['#446688','#cc8844','#667766'],['o','s','^']):
            ax.plot(range(3),g.set_index('mode').loc[modes,col],marker=marker,color=color,label=str(seed),linewidth=1.3)
        ax.set_xticks(range(3),['Standard','Standard\n+ clip ±5','Median / IQR'])
        ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(title='Training seed',fontsize=7)
    fig.suptitle('Paired quality-scaling controls: fixed grouped splits')
    fig.tight_layout()
    fig.savefig(OUT/'paired_scaling.png',dpi=300)
    fig.savefig(OUT/'paired_scaling.svg')
    fig.savefig(OUT/'paired_scaling.pdf')
    plt.close(fig)
    qa='Quantitative two-panel comparison; all 9 runs included. Same 1351 windows and 20 source groups; outer 5-fold grouped test, inner grouped validation; three training seeds are algorithmic repeats, not independent datasets. Lines pair identical seeds. No confidence intervals or significance tests. Mean and P90 calculated separately per run. Exploratory analysis after repeated use of test sources, not a fresh confirmatory test. Python only; SVG/PDF editable text, PNG 300 dpi.\n'
    (OUT/'figure_qa.txt').write_text(qa,encoding='utf-8')
    print(summary.to_string())

if __name__=='__main__':main()

