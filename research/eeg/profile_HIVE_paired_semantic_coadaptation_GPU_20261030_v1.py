"""One bounded engineering run: EEG + original P, four-step image objective.

No production weights are saved; two temporary normalized gradient updates are
discarded. No training admission or research image score follows from a PASS.
"""
import argparse,copy,datetime,gc,hashlib,importlib.util,io,json,math,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path


def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def checked(x):assert ref(x['path'])==x;return Path(x['path'])
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def main(path):
    s=read(path);assert s['status']=='REGISTERED_ENGINEERING_PAIRED_IMAGE_SEMANTIC_EEG_P_COADAPTATION' and s['host']=='old'
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    for pin in s['source_pins']+s['payloads']:checked(pin)
    out=Path(s['output_root']);out.mkdir();start=time.monotonic();base_rss=rss();phase='IMPORT'
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent))
    import numpy as np,torch
    import torch.nn.functional as F
    from torch.func import functional_call
    from PIL import Image,ImageOps
    from brainhive_paired_semantic_effect_20261030_v1 import paired_objective,native_four_step_latent,image_feature,CLIP_pixels
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    import brainhive_native_models_v1 as N
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():
        assert time.monotonic()-start<800 and rss()-base_rss<s['maximum_extra_CPU_bytes']
        assert torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes']
        assert shutil.disk_usage(out).free>3<<30
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<s['output_cap_bytes']
    def digest(m):
        h=hashlib.sha256()
        for n,v in m.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    try:
        phase='LOAD_CRITIC_AND_NATIVE'
        spec=importlib.util.spec_from_file_location('pinned_CLIP_visual',s['CLIP_source']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        jit=torch.jit.load(s['CLIP_weights']['path'],map_location='cpu').eval()
        weights={k[len('visual.'):]:v for k,v in jit.state_dict().items() if k.startswith('visual.')}
        critic=mod.VisionTransformer(224,14,1024,24,16,768).eval().requires_grad_(False)
        critic.load_state_dict(weights,strict=True);critic.cuda().float();del jit,weights;gc.collect()
        (out/'loader').mkdir();audit=ConvertedWeightAudit(torch,out/'loader');audit.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader=audit.complete();pipe.vae.float();pipe.vae.enable_slicing()
        # Everything above is private in-memory loading; saved original weights
        # are never changed. Original FP16 VAE values are promoted to FP32.
        ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
        ordinary.load_state_dict(torch.load(s['ordinary']['path'],map_location='cpu',weights_only=True)['model'],strict=True)
        parent=torch.load(s['parent']['path'],map_location='cpu',weights_only=True)
        model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval()
        model.load_state_dict(parent['model'],strict=True)
        # Keep the real coordinate chart consumed by frozen ordinary fixed.
        for n,p in model.named_parameters():p.requires_grad_(not n.startswith('norm.'))
        # Source native implementation names the final LayerNorm "norm".
        affine=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.LayerNorm)]
        assert len(affine)==1
        affine_name,LN=affine[0]
        LN.requires_grad_(False)
        train_E=[p for p in model.parameters() if p.requires_grad]
        assert sum(p.numel() for p in train_E)==6450176
        # FP32 master copies, cast back to original FP16 for exact initial P.
        master={n:torch.nn.Parameter(p.detach().float().clone()) for n,p in P.named_parameters()}
        assert sum(p.numel() for p in master.values())==8400896
        frozen=dict(A=A,P=P,UNet=pipe.unet,VAE=pipe.vae,ordinary=ordinary,critic=critic)
        before={k:digest(v) for k,v in frozen.items()}
        bindings=read(s['input_bindings']['path'])
        raw=torch.load(s['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
        records=[];stored={};temporary_updates=0
        for pair_index in range(2):
            phase='PAIR_'+str(pair_index);budget();pair=s['engineering_rows'][2*pair_index:2*pair_index+2]
            assert len(pair)==2 and pair[0]['concept_id']==pair[1]['concept_id']
            model.load_state_dict(parent['model'],strict=True)
            with torch.no_grad():
                for n,p in P.named_parameters():master[n].copy_(p.float())
            targets=[]
            for item in pair:
                assert int(item['array_row']) in s['allowed_T_rows']
                with zipfile.ZipFile(item['image_zip']) as z:data=z.read(item['image_member'])
                assert len(data)==int(item['image_bytes']) and hashlib.sha256(data).hexdigest()==item['image_sha256']
                with Image.open(io.BytesIO(data)) as image:
                    arr=np.asarray(ImageOps.fit(image.convert('RGB'),(512,512),method=Image.Resampling.BICUBIC)).copy()
                pixels=torch.from_numpy(arr).permute(2,0,1)[None].cuda().float()/255
                with torch.no_grad():targets.append(F.normalize(critic(CLIP_pixels(pixels)).float(),dim=-1))
            target=torch.cat(targets);features=[];parities=[];seeds=[];noises=[];ticks=time.monotonic()
            seed=1729+100*pair_index
            for j,item in enumerate(pair):
                budget();idx=int(item['array_row']);eeg=raw[idx:idx+1,bindings['selected_indices'],:].cuda().reshape(1,4250)
                c=model(eeg);w=LN.weight[None];b=LN.bias[None]
                corrected=ordinary(c,w,b)[0]
                middle=A(corrected.half())
                mapped={n:p.to(dtype=torch.float16) for n,p in master.items()}
                tokens=functional_call(P,mapped,(middle,))
                with torch.no_grad():
                    native_tokens=P(A(corrected.detach().half()))
                    assert torch.equal(tokens.detach(),native_tokens)
                    native,original_noise=native_four_step_latent(pipe,prompts,native_tokens,seed,gradient=False)
                latent,actual_noise=native_four_step_latent(pipe,prompts,tokens,seed,gradient=True)
                assert original_noise==actual_noise and torch.equal(native,latent.detach())
                assert tokens.requires_grad and latent.requires_grad
                features.append(image_feature(pipe,critic,latent,gradient=True))
                parities.append(dict(tokens_bit_exact=True,all4_native_latents_bit_exact=True,noise_bit_exact=True))
                seeds.append(seed);noises.append(actual_noise)
                stored[f'pair{pair_index}_image{j}_native_latent']=native.detach().cpu().numpy()
                del native,native_tokens,latent,middle,tokens,corrected,c
            assert seeds[0]==seeds[1] and noises[0]==noises[1]
            generated=torch.cat(features);loss,terms=paired_objective(generated,target)
            budget();loss.backward();gradients={}
            for group,params in (('EEG',train_E),('P',list(master.values()))):
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                norm=torch.stack([p.grad.double().square().sum() for p in params]).sum().sqrt()
                assert norm>0
                gradients[group]=dict(parameters=sum(p.numel() for p in params),gradient_norm=float(norm),nonzero_tensors=sum(bool(p.grad.count_nonzero()) for p in params),total_tensors=len(params))
                # One discarded block-normalized update; no step-size scan,
                # no candidate head or engineering checkpoint is published.
                with torch.no_grad():
                    parameter_norm=torch.stack([p.double().square().sum() for p in params]).sum().sqrt()
                    scale=1e-6*parameter_norm/norm
                    for p in params:p.add_(p.grad,alpha=-float(scale));p.grad=None
                gradients[group]['discarded_step_relative_norm']=1e-6
            temporary_updates+=1
            assert all(p.grad is None for m in frozen.values() for p in m.parameters())
            assert LN.weight.grad is None and LN.bias.grad is None
            stored[f'pair{pair_index}_generated_features']=generated.detach().cpu().numpy()
            stored[f'pair{pair_index}_target_features']=target.detach().cpu().numpy()
            records.append(dict(pair=pair_index,rows=pair,shared_seed=seed,shared_ancestral_noise=noises[0],
                forward_parity=parities,loss=float(loss.detach()),absolute=float(terms['absolute'].detach()),
                pairing=float(terms['pairing'].detach()),effect=float(terms['effect'].detach()),gradients=gradients,
                seconds=time.monotonic()-ticks,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
            del generated,features,loss,terms,target,targets;gc.collect();torch.cuda.empty_cache()
        assert before=={k:digest(v) for k,v in frozen.items()}
        for x in s['payloads']:checked(x)
        budget();np.savez_compressed(out/'actual_pairs.npz',**stored)
        result=dict(status='COMPLETE_REAL_FULL4STEP_EEG_AND_NATIVE_P_PAIRED_IMAGE_GRADIENT_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protocol=ref(path),pairs=records,actual_EEG_and_P_parameter_count=14851072,
            temporary_joint_updates=temporary_updates,new_production_models_saved=0,new_research_images=0,formal_fit_admitted=False,
            no_new_adapter=True,ordinary_unchanged=True,frozen_numerical_weights_unchanged=True,
            shared_noise_not_an_exact_additive_noise_cancellation_theorem=True,
            CLIP_critic_shared_with_author_CLIP_metric=True,DINO_not_used_for_training=True,
            single_seed=1729,OLD_only=True,NEW_contacted=False,full_chain_OOF=False,
            existing_parent_and_ordinary_used_T=True,seconds=time.monotonic()-start,
            extra_CPU_peak=rss()-base_rss,peak_GPU_allocated=torch.cuda.max_memory_allocated(),host=host,loader=loader)
        write(out/'completion.json',result);print(json.dumps(result))
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,
            new_production_models=0,new_research_images=0,NEW_contacted=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
