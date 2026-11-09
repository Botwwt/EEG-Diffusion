"""Full V native/IP-projected point-condition diagnosis, CPU only.

Uses the actual frozen native ImageProjection weights. FP32 CPU coordinates are
a diagnostic, not a claim of bit-exact deployed BF16 UNet states or image success.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import ImageProjection
from safetensors import safe_open
from locked_metrics import fixed_gallery_metrics

ROOT=Path(__file__).resolve().parent
ASSET=Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


@torch.inference_mode()
def main():
    torch.set_num_threads(2);started=time.monotonic()
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    assert sha(ROOT/'configs/split_manifest.json')==cfg['split_sha256']
    assert sha(cfg['visual_features_path'])==cfg['visual_features_sha256']
    all_y=torch.load(cfg['visual_features_path'],map_location='cpu',weights_only=True)
    v=protocol['image_rows']['V'];y=all_y[v]
    images={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    ids=[images[i]['stimulus_id'] for i in v]
    projector=ImageProjection(image_embed_dim=1024,cross_attention_dim=2048,num_image_text_embeds=4).eval().requires_grad_(False)
    with safe_open(str(ASSET),framework='pt',device='cpu') as file:
        weights={new:file.get_tensor(old).float() for new,old in [
            ('image_embeds.weight','image_proj.proj.weight'),('image_embeds.bias','image_proj.proj.bias'),
            ('norm.weight','image_proj.norm.weight'),('norm.bias','image_proj.norm.bias')]}
    projector.load_state_dict(weights,strict=True)
    def project(x):
        return torch.cat([projector(x[i:i+128]).flatten(1) for i in range(0,len(x),128)])
    tokens=project(y);mean_native=all_y[protocol['image_rows']['T']].mean(0,keepdim=True)
    mean_tokens=project(mean_native)
    report_rows=[]
    for subject in protocol['subjects']:
        proof=json.loads((Path(cfg['output_root'])/'full'/subject/'completion.json').read_text())
        assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
        cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
        assert cache['encoder_sha256']==proof['encoder']['checkpoint_sha256']
        assert torch.equal(cache['array_rows'],torch.arange(16540))
        c=cache['condition'][v];z=project(c)
        metric=fixed_gallery_metrics(z,tokens,ids,ids)
        row=dict(subject=subject,queries=len(v),gallery_unique_images=len(v),
            raw_native_dim=1024,projected_tokens=[4,2048],raw_condition_norm_mean=float(c.norm(dim=-1).mean()),
            raw_true_norm_mean=float(y.norm(dim=-1).mean()),
            projected_condition_RMS=float(z.square().mean().sqrt()),projected_true_RMS=float(tokens.square().mean().sqrt()),
            projected_condition_true_cosine=float(F.cosine_similarity(z,tokens).mean()),
            projected_T_mean_true_cosine=float(F.cosine_similarity(mean_tokens,tokens).mean()),
            projected_MSE=float((z-tokens).square().mean()),projected_T_mean_MSE=float((mean_tokens-tokens).square().mean()),
            projected_stimulus_Top1=float(metric['top1'].float().mean()),projected_stimulus_Top5=float(metric['top5'].float().mean()),
            projected_gallery_2AFC=float(metric['two_afc'].mean()),
            projected_condition_variance_trace=float(z.var(0,unbiased=False).sum()),
            projected_true_variance_trace=float(tokens.var(0,unbiased=False).sum()),encoder_sha256=proof['encoder']['checkpoint_sha256'])
        report_rows.append(row);print(json.dumps(row),flush=True)
    report=dict(status='COMPLETED_FULL_V_CPU_PROJECTOR_DIAGNOSIS',subjects=10,queries=16600,role='V condition diagnostic only',
        coordinate_space='Raw1024 CLIP-H → actual frozen ImageProjection →4x2048 tokens',
        precision='CPU FP32 native vectors and original pretrained weights; deployed UNet uses BF16; not bit-exact deployed token replay',
        adapter_sha256=sha(ASSET),projector_parameters=sum(p.numel() for p in projector.parameters()),
        visual_feature_sha256=cfg['visual_features_sha256'],source_sha256=sha(__file__),
        per_subject=report_rows,GPU_used=False,models_trained=False,final_test_accessed=False,
        images_generated=0,seconds=time.monotonic()-started,
        limits=['Projected-space retrieval is a diagnostic; the protected image evaluator is unchanged.',
            'LayerNorm bounds amplitude but does not ensure correct condition direction or object fidelity.',
            'No stochastic prior was sampled in this diagnostic; its deployed image effects require full image evaluation.',
            'A constant T mean can have high uncentered token cosine; report it alongside the EEG prediction.'])
    output=ROOT/'audit/native_projection_full_V_diagnosis.json';assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    columns=[k for k,value in report_rows[0].items() if not isinstance(value,list)]
    with (ROOT/'audit/native_projection_full_V_summary.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(report_rows)


if __name__=='__main__':main()
