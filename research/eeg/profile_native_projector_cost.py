"""T-only cost/gradient probe of the actual native ImageProjection subset.

Updated private weights are discarded. This is not formal adapter training or
evidence of reconstruction quality/convergence. All full-data arms still need
their prerequisites, sufficient training and complete image budgets.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import zipfile
from profile_standard_generator import load_pipeline,torch,ROOT
from diffusers import DDPMScheduler
from PIL import Image
import numpy as np
import torch.nn.functional as F


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729)
    output=ROOT/'audit/native_projector_training_cost.json';assert not output.exists()
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    cfg=json.loads((ROOT/'configs/full_reference_resolved.json').read_text())
    assert sha(cfg['visual_features_path'])==cfg['visual_features_sha256']
    rows=sorted(protocol['image_rows']['T'])[:16];assert len(rows)==16
    assert not set(rows)&set(protocol['image_rows']['D']+protocol['image_rows']['V'])
    features=torch.load(cfg['visual_features_path'],weights_only=True,mmap=True)
    image_rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline()
    selected=[]
    for name,parameter in pipe.unet.named_parameters():
        if name.startswith('encoder_hid_proj.'):
            parameter.requires_grad_(True);parameter.data=parameter.data.float();selected.append((name,parameter))
    count=sum(p.numel() for _,p in selected);assert count==8400896
    assert len(selected)==4 and not any(p.requires_grad for n,p in pipe.unet.named_parameters() if not n.startswith('encoder_hid_proj.'))
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing()
    optimizer=torch.optim.AdamW([p for _,p in selected],lr=1e-4,weight_decay=.01)
    pixels=[];archives={}
    for row in rows:
        m=image_rows[row];archive=m['image_zip']
        if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
        raw=archives[archive].read(m['image_member']);assert hashlib.sha256(raw).hexdigest()==m['image_sha256']
        with Image.open(io.BytesIO(raw)) as im:
            a=np.array(im.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),copy=True)
        pixels.append(torch.from_numpy(a).permute(2,0,1).float()/127.5-1)
    for archive in archives.values():archive.close()
    with torch.inference_mode():
        latent=torch.cat([pipe.vae.encode(torch.stack(pixels[i:i+4]).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor for i in range(0,len(pixels),4)])
    # Leave inference-mode tensors behind before feeding the trainable graph.
    latent=latent.clone();del pixels
    condition=features[rows].clone().cuda()
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
    time_ids=pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=prompt.dtype,text_encoder_projection_dim=pool.shape[-1]).cuda()
    records=[]
    for batch in [4,16]:
        durations=[];torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(6):
                torch.cuda.synchronize();tick=time.monotonic()
                noise=torch.randn_like(latent[:batch]);t=torch.randint(0,1000,(batch,),device='cuda')
                noisy=scheduler.add_noise(latent[:batch],noise,t)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    prediction=pipe.unet(noisy.to(torch.bfloat16),t,encoder_hidden_states=prompt.repeat(batch,1,1),
                        added_cond_kwargs=dict(text_embeds=pool.repeat(batch,1),time_ids=time_ids.repeat(batch,1),
                            image_embeds=[condition[:batch,None,:]]),return_dict=False)[0]
                    loss=F.mse_loss(prediction.float(),noise)
                assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_([p for _,p in selected],1.,error_if_nonfinite=True)
                assert float(norm)>0
                assert all(p.grad is not None for _,p in selected)
                assert all(p.grad is None for n,p in pipe.unet.named_parameters() if not n.startswith('encoder_hid_proj.'))
                optimizer.step();torch.cuda.synchronize();durations.append(time.monotonic()-tick)
            steady=sum(durations[2:])/len(durations[2:])
            record=dict(batch=batch,status='PASSED_COST_AND_GRADIENT_PROBE',updates=6,steady_seconds_per_step=steady,
                seconds_per_training_example=steady/batch,peak_cuda_allocated=torch.cuda.max_memory_allocated(),
                scratch_loss=float(loss),gradient_norm_preclip=float(norm),trainable_parameters=count,
                trainable_names=[n for n,_ in selected],frozen_backbone_gradients_absent=True,
                one_full_132300_EEG_query_pass_hours_compute_only=steady*((132300+batch-1)//batch)/3600)
        except torch.cuda.OutOfMemoryError as exc:
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
            record=dict(batch=batch,status='OUT_OF_MEMORY',error=str(exc),trainable_parameters=count)
        records.append(record);print(json.dumps(record),flush=True)
    report=dict(status='T_ONLY_PROJECTOR_COST_PROBE_COMPLETED',formal_training=False,EEG_method_gain_measured=False,
        updated_weights_discarded=True,checkpoint_saved=False,training_seed=1729,array_rows=rows,
        coordinate='Actual1024 native image vector to frozen-IP4x2048 interface; true image conditions used for cost only',
        generator_revision=receipt['revision'],native_IP_full_branch_parameters=349188096,
        projector_trainable_parameters=count,weight_source='existing public native IP-Adapter',
        cases=records,source_sha256=sha(__file__),seconds=time.monotonic()-started,
        limits=['Six scratch updates are not sufficient training, a method result or a capacity verdict.',
            'Extrapolated compute excludes full IO/checkpointing/validation/OOF/error calibration and image evaluation.',
            'All matched Clean/Iso/Moment/Conditional/Real-Paired arms and EEG controls/final costs must pass a full budget gate.',
            'No inference step count, primary evaluator, image random stream or active model is changed by this private probe.'])
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
