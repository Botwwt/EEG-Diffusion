"""Posthoc closed-stage feature diagnostics; never adjust or fit the method."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1');sys.path.insert(0,str(R))
import brainhive_temporal_image_screen_20261028_v1 as U


def run(stage):
    start=time.monotonic();folder=U.B/f'screen{stage}';score=U.read(folder/'score/completion.json');audit=U.read(folder/'score/independent_CPU_audit.json')
    assert audit['score_completion']['sha256']==U.sha(folder/'score/completion.json')
    spec=U.read(U.check(score['spec']));feat=U.read(U.check(score['feature_completion']))
    x=np.load(U.check(feat['features']),allow_pickle=False).astype(np.float64)
    old=np.load(U.check(spec['baseline_features']),allow_pickle=False)[:stage].astype(np.float64)
    target=np.load(U.check(spec['target_features']),allow_pickle=False)[:stage].astype(np.float64)
    x/=np.linalg.norm(x,axis=-1,keepdims=True);old/=np.linalg.norm(old,axis=-1,keepdims=True);target/=np.linalg.norm(target,axis=-1,keepdims=True)
    ordinary=old[:,1];change=x-ordinary[:,None,:];truth=target-ordinary
    decomposition={};maximum=0.
    for i,arm in enumerate(U.ARMS):
        energy=(change[:,i]**2).sum(-1);twice_alignment=2*(change[:,i]*truth).sum(-1)
        exact=(np.square(x[:,i]-target).sum(-1)-np.square(ordinary-target).sum(-1))
        error=float(np.abs(exact-(energy-twice_alignment)).max());maximum=max(maximum,error);assert error<1e-12
        decomposition[arm]=dict(mean_visual_feature_move_energy=float(energy.mean()),mean_twice_target_residual_alignment=float(twice_alignment.mean()),
            mean_feature_squared_error_change=float(exact.mean()),queries_feature_error_improved=int((exact<0).sum()),
            average_target_cosine_change=float(((x[:,i]-ordinary)*target).sum(-1).mean()))
    cond_flat=np.linalg.norm(x[:,0]-x[:,1],axis=-1);cond_base=np.linalg.norm(change[:,0],axis=-1)
    correction_cos=(change[:,0]*change[:,1]).sum(-1)/(np.linalg.norm(change[:,0],axis=-1)*np.linalg.norm(change[:,1],axis=-1)).clip(1e-15)
    rows=U.read(U.check(feat['rows']));old_rows=[json.loads(s) for s in U.check(spec['baseline_manifest']).read_text().splitlines()]
    lookup={(r['query_id'],r['arm']):r for r in old_rows};same_cond_flat=same_wrong=0;identical_old={arm:0 for arm in U.ARMS}
    for i in range(stage):
        group=rows[i*3:(i+1)*3];assert [r['arm'] for r in group]==list(U.ARMS)
        same_cond_flat+=group[0]['generated_pixel_sha256']==group[1]['generated_pixel_sha256']
        same_wrong+=group[0]['generated_pixel_sha256']==group[2]['generated_pixel_sha256']
        for row in group:identical_old[row['arm']]+=row['generated_pixel_sha256']==lookup[(row['query_id'],'ordinary')]['generated_pixel_sha256']
    command=Path('/proc/self/cmdline').read_bytes().split(b'\0');index=command.index(b'-c')
    source_sha=hashlib.sha256(command[index+1]).hexdigest()
    value=dict(status='D37_CLOSED_IMAGE_FEATURE_CHANGE_DECOMPOSITION_AND_CONTROL_CLOSENESS_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage=stage,score_completion=U.ref(folder/'score/completion.json'),source_executed_via_c_sha256=source_sha,
        decomposition=decomposition,maximum_energy_alignment_identity_error=maximum,
        mean_conditional_vs_flat_feature_distance=float(cond_flat.mean()),mean_conditional_vs_ordinary_feature_distance=float(cond_base.mean()),
        aggregate_conditional_flat_distance_relative_to_conditional_base=float(cond_flat.mean()/cond_base.mean()),
        mean_conditional_flat_feature_correction_cosine=float(correction_cos.mean()),exact_RGB_conditional_equals_flat_queries=int(same_cond_flat),
        exact_RGB_conditional_equals_same_concept_wrong_queries=int(same_wrong),exact_RGB_matches_old_ordinary=identical_old,
        posthoc_not_a_new_success_gate=True,no_method_or_strength_or_policy_or_future_images_changed=True,
        no_independent_image_encoder_or_UNet_replay_claimed=True,no_EEG_or_visual_model_fit=True,new_GPU_jobs=0,seconds=time.monotonic()-start)
    output=folder/'feature_delta_analysis.json'
    if output.exists():value=U.read(output)
    else:U.write(output,value)
    print(json.dumps(value))


if __name__=='__main__':run(int(json.loads(sys.stdin.buffer.readline())['stage']))
