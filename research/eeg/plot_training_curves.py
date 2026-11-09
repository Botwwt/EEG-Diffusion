"""Completed formal reference curves; CPU plotting, no new training or test access."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    config=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    data=Path(config['output_root'])/'full';assert (data/'completion.json').exists()
    records=[json.loads(line) for line in (data/'epochs.jsonl').read_text().splitlines()]
    paths=[]
    for stage,label,trainkey,valkey in [('encoder','Native CLIP regression MSE','mean_native_mse','native_mse'),
                                       ('prior','Prior epsilon MSE','mean_loss','fixed_noise_validation_loss')]:
        fig,axes=plt.subplots(2,5,figsize=(16,6),constrained_layout=True)
        for subject,ax in enumerate(axes.flat,1):
            values=[row for row in records if row.get('stage')==stage and row.get('subject')==subject]
            epochs=[row['epoch'] for row in values]
            train=[row[trainkey] for row in values]
            validation=[row['validation'][valkey] if stage=='encoder' else row[valkey] for row in values]
            ax.plot(epochs,train,label='T epoch mean',linewidth=1.)
            ax.plot(epochs,validation,label='Full V',linewidth=1.)
            proof=json.loads((data/f'sub-{subject:02d}'/f'{stage}_training.json').read_text())
            ax.axvline(proof['best_epoch'],color='gray',linestyle=':',linewidth=.8)
            ax.set_title(f'sub-{subject:02d}; selected epoch {proof["best_epoch"]}')
            ax.set_xlabel('Epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f'Formal full-data reference, seed1729; {label}\nT13230 / V1660 stimuli per subject; V is development, not final test',fontsize=11)
        path=ROOT/'audit'/f'full_reference_{stage}_curves.svg';fig.savefig(path);plt.close(fig);paths.append(str(path))
    print(json.dumps(dict(status='COMPLETED',figures=paths,final_test_accessed=False)))


if __name__=='__main__':main()
