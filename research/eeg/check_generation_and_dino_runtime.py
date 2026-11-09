"""T-only forward checks after full-reference training, before long image generation."""
from image_evaluation import *
from generate_reference_images import sample_prior,prior_seed,query_seed
from locked_metrics import fixed_gallery_metrics
from pathlib import Path
import json
import time


def main():
    started=time.monotonic();torch.set_num_threads(2)
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text());training=Path(cfg['output_root'])/'full'
    assert (training/'completion.json').exists()
    files=sorted((Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runs/S0_semantic_evaluator')/'gt/preds').glob('*.json'))
    images=[p.parents[2]/(p.stem+'.jpg') for p in files]
    model=dino_model()
    features=dino_features(images,model,batch_size=8,real_images=True)
    ids=[p.stem for p in images]
    result=fixed_gallery_metrics(features,features,ids,ids)
    assert torch.equal(result['two_afc'],torch.ones(len(images)))
    del model;torch.cuda.empty_cache()
    from diffusion_prior import DiffusionPriorUNet
    proof=json.loads((training/'sub-01/completion.json').read_text())
    cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text());rows=protocol['image_rows']['T'][:2]
    c=cache['condition'][rows].cuda()
    model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval().requires_grad_(False)
    model.load_state_dict(torch.load(proof['prior']['checkpoint'],weights_only=True),strict=True)
    seeds=[1729,1730]
    a=sample_prior(model,c,seeds);b=sample_prior(model,c,seeds)
    assert torch.equal(a,b),'Same fixed prior stream changed under repeated identical inputs'
    perturbed=sample_prior(model,c.flip(0),seeds)
    report=dict(status='PASSED_ACTUAL_T_ONLY_RUNTIME',dino_backbone='existing frozen DINOv3 ViT-S/16',
        dino_weight_sha256=DINO_SHA256,dino_feature_shape=list(features.shape),original_self_pair_two_afc=result['two_afc'].tolist(),
        prior_same_input_same_query_stream_bitwise_equal=True,prior_shape=list(a.shape),
        prior_norms=a.norm(dim=-1).tolist(),swapped_EEG_native_condition_change_RMS=float((a-perturbed).square().mean().sqrt()),
        diagnostic_seed_namespace='fixed S0 diagnostic1729/1730; no seed selection; formal queries retain the frozen per-query hash contract',
        sampler_clip_sample=False,prior_steps=50,prior_CFG=5.,seconds=time.monotonic()-started,
        EEG_generation_quality_established=False)
    (ROOT/'audit/generation_dino_runtime_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
