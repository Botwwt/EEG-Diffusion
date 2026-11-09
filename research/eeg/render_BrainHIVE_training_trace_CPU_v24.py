"""Render the captured scientific trace; never interpolate missing batch losses."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent


def main():
    path=ROOT/'audit/BrainHIVE_training_trace_v24.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    raw=(ROOT/'audit/BrainHIVE_training_trace_v24.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==data['source_prefix_sha256']
    events=data['events'];x=[r['updates'] for r in events]
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,layout='constrained')
    ax=axes[0,0]
    ax.plot(x,[r['last_loss'] for r in events],'.',color='#aeb4bd',label='Sampled last batch')
    ax.plot(x,[r['exact_interval_total_mean'] for r in data['intervals']],color='#2166ac',label='All-query interval mean')
    ax.plot(x,[r['cumulative_mean_loss'] for r in events],color='#b2182b',label='All-query cumulative mean')
    ax.set(title='Training objective: different summaries',ylabel='Total loss');ax.legend(fontsize=8)
    ax=axes[0,1]
    for key,scale,label,color in [('epsilon_loss',1,'Epsilon MSE','#2166ac'),
                                ('token_loss',.1,'0.1 x token angular error','#d6604d'),
                                ('risk_auxiliary_loss',.1,'0.1 x risk MSE','#4d9221')]:
        ax.plot(x,[scale*r[key] for r in events],'.-',label=label,color=color,markersize=3)
    ax.set(title='Objective terms (sampled last batches)',ylabel='Weighted loss');ax.legend(fontsize=8)
    ax=axes[1,0]
    lo=[min(r['timesteps']) for r in events];hi=[max(r['timesteps']) for r in events]
    ax.fill_between(x,lo,hi,color='#d1e5f0',label='Batch min-max')
    ax.plot(x,[sum(r['timesteps'])/len(r['timesteps']) for r in events],color='#2166ac',label='Batch mean')
    ax.set(title='Random training timesteps (sampled batches)',ylabel='Timestep',ylim=(0,1000));ax.legend(fontsize=8)
    ax=axes[1,1]
    ax.plot(x,[r['gradient_norm'] for r in events],'.-',color='#762a83',markersize=3)
    ax.axhline(1,color='#666666',linestyle='--',label='Clip threshold')
    ax.set(title='Gradient norm before clipping (sampled)',ylabel='Norm',yscale='log');ax.legend(fontsize=8)
    ax=axes[2,0]
    ax.plot(x,[r['correction_RMS'] for r in events],color='#1b7837')
    ax.set(title='Actual raw-condition correction (sampled)',ylabel='RMS',xlabel='Optimizer updates')
    ax=axes[2,1]
    ax.plot(x,[r['mean_gate'] for r in events],color='#b35806')
    ax.set(title='Mean training gate (before Dfit calibration)',ylabel='1 - predicted risk / 2',ylim=(0,1),xlabel='Optimizer updates')
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('BrainHIVE bounded risk component: T-only runtime diagnostics\n'
        'Loss and finite gradients do not establish image success or generalization',fontsize=13)
    out=ROOT/'audit/BrainHIVE_training_trace_v24.png';assert not out.exists()
    fig.savefig(out,dpi=160);plt.close(fig)
    print(json.dumps(dict(path=str(out),bytes=out.stat().st_size,events=len(events),updates=data['observed_updates'])))


if __name__=='__main__':main()
