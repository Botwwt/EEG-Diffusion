"""T-only, exact deployed prior-sampler cost; no training or image selection."""
from pathlib import Path
import csv
import json
import time
from generate_reference_images import torch,ROOT,DiffusionPriorUNet,sample_prior,sha,tensor_sha


def main():
    started=time.monotonic();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    interface=json.loads((ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json').read_text())
    for name,expected in interface['source_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(ROOT/'audit/query_manifest_R4.csv')==interface['query_manifest_sha256']
    base=Path(cfg['output_root'])/'full'/'sub-01';proof=json.loads((base/'completion.json').read_text())
    cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
    assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
    assert cache['encoder_sha256']==proof['encoder']['checkpoint_sha256']
    queries=[]
    with (ROOT/'audit/query_manifest_R4.csv').open() as file:
        for row in csv.DictReader(file):
            if row['subject_id']=='sub-01' and row['role']=='T':
                queries.append(row)
                if len(queries)==128:break
    assert len(queries)==128 and len({q['query_id'] for q in queries})==128
    prior_root=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S1_prior_velocity_repair_v1/full/sub-01')
    prior_proof=json.loads((prior_root/'completion.json').read_text())['prior']
    assert prior_proof['prediction_type']=='v_prediction'
    assert sha(prior_proof['checkpoint'])==prior_proof['checkpoint_sha256']
    model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(prior_proof['checkpoint'],weights_only=True,map_location='cpu'),strict=True)
    assert sum(p.numel() for p in model.parameters())==9675648
    c=cache['condition'][[int(q['array_row']) for q in queries]].cuda()
    seeds=[int(q['generation_seed']) for q in queries]
    counts=dict(calls=0,sample_evaluations=0)
    def hook(module,inputs):counts['calls']+=1;counts['sample_evaluations']+=len(inputs[0])
    handle=model.register_forward_pre_hook(hook)
    with torch.inference_mode():
        warm=sample_prior(model,c[:4],seeds[:4],prediction_type='v_prediction')
        torch.cuda.synchronize();before=dict(counts);tick=time.monotonic();outputs=[]
        for start in range(0,128,4):
            outputs.append(sample_prior(model,c[start:start+4],seeds[start:start+4],prediction_type='v_prediction'))
        torch.cuda.synchronize();seconds=time.monotonic()-tick;y=torch.cat(outputs)
    handle.remove()
    measured={k:counts[k]-before[k] for k in counts}
    assert measured==dict(calls=3200,sample_evaluations=12800)
    assert torch.equal(warm,y[:4]) and torch.isfinite(y).all()
    report=dict(status='EXACT_T_NATIVE_PRIOR_CACHE_COST_MEASURED',queries=128,batch=4,
        condition_chain='REF_PRIOR_V',prior_steps=50,prior_CFG=5.,parameters=9675648,dtype='float32',
        measured_seconds=seconds,seconds_per_query=seconds/128,
        full_T_OOF_plus_three_D_teacher_queries=181800,
        extrapolated_prior_sampling_hours=seconds/128*181800/3600,
        inference_only=True,training_updates=0,images_generated=0,V_or_final_test_accessed=False,
        same_query_sampler_replay_exact=True,first4_output_sha256=tensor_sha(y[:4]),counts=measured,
        encoder_sha256=proof['encoder']['checkpoint_sha256'],prior_sha256=prior_proof['checkpoint_sha256'],
        protected_generator_source_sha256=sha(ROOT/'generate_reference_images.py'),source_sha256=sha(__file__),
        peak_private_cuda_allocated=torch.cuda.max_memory_allocated(),total_seconds=time.monotonic()-started,
        limits=['128 T queries for throughput only; no scientific conditional quality or convergence result.',
            'Extrapolation excludes OOF teacher training, per-model load, hash, cache I/O, CPU validation and safety margin.',
            'If REF_REG is selected the semantic prior cost is absent; do not charge it to that chain.'])
    path=ROOT/'audit/native_prior_cache_cost_profile.json';assert not path.exists()
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
