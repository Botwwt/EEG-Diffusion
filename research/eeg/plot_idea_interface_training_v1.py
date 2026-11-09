"""Actual loss logging diagnostics; CPU only, never a picture-success test."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'audit/matplotlib_cache_v1'))
sys.path.insert(0,'/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=['RealPaired','Conditional','Moment','Iso','Clean']);args=parser.parse_args()
    source=DATA/'runs'/f'IDEA1_interface_{args.arm}_v1/training.jsonl'
    raw=source.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    lines=raw.splitlines(keepends=True);partial=bool(lines and not lines[-1].endswith(b'\n'))
    rows=[json.loads(line) for line in (lines[:-1] if partial else lines)]
    assert rows and all(x['arm']==args.arm and x['stage']=='actual_image_denoising_training' for x in rows)
    block=[];previous_seen=0;previous_sum=0.;previous_updates=0
    for x in rows:
        updates=x['updates'];seen=x['seen_queries'];current_sum=x['mean_loss_so_far']*seen
        assert updates>previous_updates and seen>previous_seen,'Preserve and inspect repeated/replayed logs, do not silently drop records'
        assert seen==min(132300,16*updates)
        assert all(math.isfinite(x[k]) for k in ['mean_loss_so_far','last_minibatch_loss','gradient_norm_preclip','seconds'])
        average=(current_sum-previous_sum)/(seen-previous_seen)
        block.append(dict(stop_update=updates,queries=seen-previous_seen,query_weighted_mean_loss=average))
        previous_seen=seen;previous_sum=current_sum;previous_updates=updates
    reconstructed=sum(x['queries']*x['query_weighted_mean_loss'] for x in block)/previous_seen
    assert abs(reconstructed-rows[-1]['mean_loss_so_far'])<1e-10
    folder=DATA/'diagnostics'/f'IDEA1_interface_{args.arm}_training_curves_v1'/f'{rows[-1]["updates"]:06d}_{digest[:12]}'
    assert not folder.exists();folder.mkdir(parents=True)
    (folder/'source_training_log_snapshot.jsonl').write_bytes(raw)
    updates=[x['updates'] for x in rows]
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    axes[0].scatter(updates,[x['last_minibatch_loss'] for x in rows],s=12,color='#999999',alpha=.65,label='Last minibatch at each log point (sparse)')
    axes[0].plot(updates,[x['query_weighted_mean_loss'] for x in block],linewidth=1,color='#2878b5',label='Exact mean since previous log point (usually32 updates)')
    axes[0].plot(updates,[x['mean_loss_so_far'] for x in rows],linewidth=1.5,color='#268b56',label='Cumulative query-weighted mean')
    axes[0].set_ylabel('Image epsilon MSE');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].plot(updates,[x['gradient_norm_preclip'] for x in rows],linewidth=1,color='#8e4d9f')
    axes[1].set_yscale('log');axes[1].set_ylabel('Last-minibatch gradient norm\n(preclip; sparse log points)')
    axes[1].set_xlabel('Actual optimizer update');axes[1].grid(alpha=.2)
    last=rows[-1]
    fig.suptitle(f'{args.arm}, seed1729: actual image-noise training log\n{last["updates"]}/8269 updates; {last["seen_queries"]}/132300 T queries. No image-success inference.',fontsize=11)
    for name in ['training_progress.png','training_progress.svg']:fig.savefig(folder/name,dpi=150)
    plt.close(fig)
    proof=dict(status='ACTUAL_LOG_SNAPSHOT_PLOTTED_NO_MODEL_SELECTION',arm=args.arm,source=str(source),source_snapshot_sha256=digest,
        output=str(folder),updates=last['updates'],training_queries=last['seen_queries'],logged_points=len(rows),
        incomplete_final_line_excluded=partial,all_logged_numeric_values_finite=True,
        exact_interval_means_reconcile_to_cumulative_mean=True,
        current_cumulative_loss=last['mean_loss_so_far'],interval_mean_min=min(x['query_weighted_mean_loss'] for x in block),
        interval_mean_max=max(x['query_weighted_mean_loss'] for x in block),GPU_used=False,training_changed=False,
        image_success_inferred=False,checkpoint_selection_changed=False,
        limits=['Only last-minibatch points are sampled; every intervening query contributes to the reconstructed interval means.',
                'Timesteps/examples differ across minibatches; fluctuations are not a fixed-query validation curve.',
                'No convergence, capacity conclusion or actual-image success follows from these training losses.'])
    (folder/'summary.json').write_text(json.dumps(proof,indent=2)+'\n')
    alias=ROOT/'audit'/f'idea_interface_{args.arm}_actual_training_curve_latest_v1.json'
    if alias.exists():
        history=ROOT/'audit/training_curve_receipt_history';history.mkdir(exist_ok=True)
        old=alias.read_bytes();path=history/(hashlib.sha256(old).hexdigest()+'.json')
        if not path.exists():path.write_bytes(old)
    alias.write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
