"""Real T / genuine OOF pretrained VQ-VAR engineering; no efficacy selection."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,datetime,gc,hashlib,io,json,subprocess,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from EEG_generative_backend_loader_20261029_v1 import build_and_load,sha,SCALES
from eeg_multiscale_prefix_20261028_v1 import EEGPrefixDecoder
from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder

def check(ref):
    path=Path(ref['path']);assert sha(path)==ref['sha256'],str(path)
    if 'bytes' in ref:assert path.stat().st_size==ref['bytes']
    return path
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        t=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def run(spec_path):
    start=time.monotonic();spec=json.loads(spec_path.read_bytes());out=Path(spec['output']);assert not out.exists();out.mkdir();phase='PINNED_INPUTS';base_CPU=peak()
    try:
        for item in spec['source_pins']:check(item)
        memory=json.loads(check(spec['memory_completion']).read_bytes());assert memory['status']=='COMPLETE_REAL_EEG_T13230_GENUINE_OOF_AND_DCHECK400_LOCAL_MEMORY'
        assert memory['all_T_rows_exactly_once_from_concept_excluded_encoder'] and memory['T_statistics_fit_only_role_allowed_concepts']
        expected_uuid=spec['GPU_UUID'];assert expected_uuid in {'GPU-0a99d971-a46c-6e81-f157-c33243c6d57a','GPU-62a1aee8-4740-dc59-93f4-d50644db650d'}
        assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==expected_uuid
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        def budget():
            assert time.monotonic()-start<840 and peak()-base_CPU<10<<30
            assert not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()<20<<30
        identities=json.loads(check(memory['files']['identities']).read_bytes());query=identities['T'][0];assert len(identities['T'])==13230 and query['subject']=='sub-01'
        c=np.load(check(memory['files']['T_OOF_c']),mmap_mode='r',allow_pickle=False)
        h=np.load(check(memory['files']['T_OOF_memory']),mmap_mode='r',allow_pickle=False);assert c.shape==(13230,1024) and h.shape==(13230,63,1024)
        target_rows={int(row['array_row']):row for row in csv.DictReader(check(spec['image_manifest']).open()) if row['dataset_split']=='train'}
        row=target_rows[query['array_row']];assert row['stimulus_id']==query['stimulus_id'] and row['concept_id']==query['concept_id']
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        if 'T_image_payload' in spec:data=check(spec['T_image_payload']).read_bytes()
        else:
            with zipfile.ZipFile(row['image_zip']) as archive:data=archive.read(row['image_member'])
        assert hashlib.sha256(data).hexdigest()==row['image_sha256']
        image=Image.open(io.BytesIO(data)).convert('RGB')
        # Exact author validation transform, matching the fixed T token bank.
        assert image.width==image.height
        crop=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),transforms.ToTensor()])(image).mul(2).sub(1)[None]
        assert crop.shape==(1,3,256,256)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        phase='STRICT_PRETRAINED_BACKEND_LOAD'
        vae,decoder,load_proof=build_and_load(spec['author_source'],spec['vendor'],check(spec['VQ_checkpoint']),check(spec['VAR_checkpoint']))
        write(out/'strict_load.json',load_proof);budget()
        initial={k:v.detach().clone() for k,v in decoder.state_dict().items()}
        vae.cuda();decoder.cuda();decoder.rng=torch.Generator(device='cuda')
        torch.cuda.reset_peak_memory_stats();vae_before=digest(vae);decoder_before=digest(decoder)
        prefix=EEGPrefixDecoder(decoder).eval();anchored=EEGAnchoredAttentionDecoder(decoder).eval();global_eeg=torch.tensor(np.array(c[:1],copy=True),device='cuda');local_eeg=torch.tensor(np.array(h[:1],copy=True),device='cuda');anchor=global_eeg[:,None].expand_as(local_eeg)
        phase='REAL_T_VQ_ENCODING'
        with torch.no_grad():
            target_indices=vae.img_to_idxBl(crop.cuda());targets=torch.cat(target_indices,1);previous=vae.quantize.idxBl_to_var_input(target_indices)
            reconstruction=vae.idxBl_to_img(target_indices,same_shape=True,last_one=True)
        assert targets.shape==(1,680) and previous.shape==(1,679,32)
        checks={};checks['native_10_scales_680_real_T_codes']=[v.shape[1] for v in target_indices]==[s*s for s in SCALES]
        checks['real_EEG_memory_preserves_global']=torch.allclose(local_eeg.mean(1),global_eeg,atol=2e-5,rtol=2e-5)
        phase='EXACT_NATIVE_ROLLBACK_LOGITS_AND_PIXELS'
        with torch.no_grad():
            native=decoder(previous,global_eeg);off=prefix(previous,global_eeg);zero=anchored(previous,global_eeg,eeg_memory=anchor)
            checks['disabled_logits_exact_native']=torch.equal(native,off)
            checks['anchor_only_logits_exact_native']=torch.equal(native,zero)
            full=anchored(previous,global_eeg,eeg_memory=local_eeg)
            changes=float((full-native).abs().max());checks['actual_local_EEG_visible']=changes>1e-7
            native_image=decoder.autoregressive_infer_cfg(1,global_eeg,g_seed=1729,cfg=4.,top_k=900)
            disabled_image=prefix.generate(global_eeg,seed=1729,guidance=4.,top_k=900)
            zero_image=anchored.generate(global_eeg,eeg_memory=anchor,seed=1729,guidance=4.,top_k=900)
            checks['disabled_pixels_exact_native']=torch.equal(native_image,disabled_image)
            checks['anchor_only_pixels_exact_native']=torch.equal(native_image,zero_image)
        budget()
        phase='ACTUAL_NORMALIZED_QK_CACHE_REPLAY';logits=[];indices=[];cache_lengths=[];old_logits=decoder.get_logits;old_embedding=vae.quantize.embedding.forward
        def capture_logits(hidden,condition):
            value=old_logits(hidden,condition);logits.append(value[:1].detach().clone());cache_lengths.append([b.attn.cached_k.shape[2] for b in decoder.blocks]);return value
        def capture_embedding(value):indices.append(value.detach().clone());return old_embedding(value)
        decoder.get_logits=capture_logits;vae.quantize.embedding.forward=capture_embedding
        try:anchored_image=anchored.generate(global_eeg,eeg_memory=local_eeg,seed=1729,guidance=0.,top_k=900)
        finally:decoder.get_logits=old_logits;vae.quantize.embedding.forward=old_embedding
        with torch.no_grad():
            generated_history=vae.quantize.idxBl_to_var_input(indices);teacher=anchored(generated_history,global_eeg,eeg_memory=local_eeg);cached=torch.cat(logits,1)
        cache_error=float((teacher-cached).abs().max());checks['cached_vs_full_logits_same_generated_history']=torch.allclose(teacher,cached,atol=2e-3,rtol=2e-4)
        cumulative=np.cumsum([s*s for s in SCALES]).tolist();checks['native_cache_has_only_image_tokens']=cache_lengths==[[n]*16 for n in cumulative]
        checks['cache_released']=all(not b.attn.caching and b.attn.cached_k is None for b in decoder.blocks)
        phase='THREE_DISCARDED_REAL_T_UPDATES';events=[]
        for name,model,mem in [('native',decoder,None),('prefix',prefix,local_eeg),('anchored',anchored,local_eeg)]:
            decoder.load_state_dict(initial,strict=True);model.train();torch.manual_seed(1729);decoder.cond_drop_rate=0.;optimizer=torch.optim.AdamW(decoder.parameters(),lr=1e-5,weight_decay=.01)
            optimizer.zero_grad(set_to_none=True);memory_input=mem.detach().clone().requires_grad_() if mem is not None else None
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(previous,global_eeg) if name=='native' else model(previous,global_eeg,eeg_memory=memory_input)
                loss=F.cross_entropy(prediction.reshape(-1,4096),targets.reshape(-1))
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.,error_if_nonfinite=True)
            checks[name+'_existing_QKV_gradient']=bool(decoder.blocks[0].attn.mat_qkv.weight.grad is not None and decoder.blocks[0].attn.mat_qkv.weight.grad.norm()>0)
            if name!='native':checks[name+'_local_EEG_gradient']=bool(memory_input.grad is not None and torch.isfinite(memory_input.grad).all() and memory_input.grad.norm()>0)
            optimizer.step();torch.cuda.synchronize();events.append(dict(arm=name,temporary_updates=1,loss=float(loss.detach()),gradient_norm=float(norm),peak_GPU_allocated=torch.cuda.max_memory_allocated()))
            optimizer=None;decoder.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();budget()
        decoder.load_state_dict(initial,strict=True);decoder.eval();checks['all_temporary_VAR_updates_reset']=digest(decoder)==decoder_before
        checks['VQ_weights_frozen_and_unchanged']=all(p.grad is None for p in vae.parameters()) and digest(vae)==vae_before
        checks['wrappers_add_no_parameters']={id(p) for p in decoder.parameters()}=={id(p) for p in prefix.parameters()}=={id(p) for p in anchored.parameters()}
        engineering=[]
        for name,value in [('real_T_VQ_reconstruction',reconstruction.add(1).mul(.5)),('native_new_EEG_projection_untrained',native_image),('disabled_exact_native',disabled_image),('anchor_only_exact_native',zero_image),('anchored_untrained_local_memory',anchored_image)]:
            pixels=value[0].detach().float().cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy();path=out/(name+'.png');Image.fromarray(pixels).save(path);engineering.append(dict(kind=name,image=ref(path),pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),engineering_only=True,EEG_method_effect_not_evaluated=True))
        assert all(checks.values()),checks;budget()
        write(out/'completion.json',dict(status='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),load_proof=load_proof,query=query,EEG_condition_source=spec['memory_completion'],training_image_sha256=row['image_sha256'],training_crop='Author validation Resize(shortedge256,BILINEAR), square stimulus asserted, RGB[-1,1], no random augmentation',checks=checks,cache_max_logit_error=cache_error,local_memory_logit_change=changes,temporary_updates=3,temporary_update_events=events,formal_trained_models=0,new_research_images=0,engineering_images=engineering,normal_free_generation_inputs='EEG global/local memory, seed and fixed sampling arguments only',attention_l2_normalization=True,conditional_dropout_for_engineering=0.,native_image_size=256,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base_CPU,seconds=time.monotonic()-start,parameter_counts=dict(VAR=310308096,VQ=108948355,new_wrapper_parameters=0)))
        print(json.dumps(dict(status='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING',seconds=time.monotonic()-start,cache_error=cache_error)))
    except BaseException as error:
        write(out/'failure.json',dict(status='GENERATIVE_BACKEND_ENGINEERING_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
