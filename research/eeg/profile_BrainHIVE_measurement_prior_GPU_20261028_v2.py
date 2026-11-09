"""D39 independent native EEG-core gradients through frozen A/P/UNet.

Native branch without ordinary in the training graph. Two T zero-update parity images and eight discarded temporary updates. This
is engineering, not full candidate training or an efficacy screen.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,datetime,gc,hashlib,json,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
P=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
W=Path('/root/brainhive_reproduction_20261020_v1')

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256'],str(p)
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def run(sp):
    started=time.monotonic();spec=read(sp);assert spec['status']=='D39_NATIVE_EEG_CORE_INDEPENDENT_REAL_GRADIENT_ENGINEERING_LOCKED'
    for x in spec['source_pins']:checked(x)
    B=Path(spec['output']);assert not B.exists();B.mkdir(parents=True)
    active='INPUTS';records=[]
    try:
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(dict(runtime_manifest=spec['runtime_manifest']))
        import numpy as np
        import torch
        from torch.nn import functional as F
        from PIL import Image
        import brainhive_measurement_prior_20261028_v1 as M
        from brainhive_reversible_eeg_20261028_v1 import author_brain_class
        from brainhive_component_generator_v1 import load_native,denoise
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from verify_HIVE_second_host_native_GPU_20261028_v1 import wrapper,memory
        from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
        proof=wrapper(spec['GPU_job_name']);baseCPU=memory()['VmRSS']
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True);assert not torch.cuda.is_initialized()
        cpu=read(checked(spec['CPU_completion']));assert cpu['real_repeat_input_JVPs']==52920
        files={Path(x['path']).name:x for x in cpu['files']}
        raw=np.load(checked(files['T13230_mean4.npy']),mmap_mode='r',allow_pickle=False)
        units=np.load(checked(files['T13230_original_units.npy']),mmap_mode='r',allow_pickle=False)
        Q=np.load(checked(files['measurement_precision.npy']),allow_pickle=False);Qp=np.load(checked(files['permuted_precision.npy']),allow_pickle=False)
        cpuspec=read(checked(cpu['spec']));D30=read(checked(cpuspec['D30_spec']));D33=read(checked(cpuspec['D33_spec']))
        index=[json.loads(s) for s in checked(D33['T_index']).read_text().splitlines()]
        parent=torch.load(checked(D30['roles']['full']['checkpoint']),map_location='cpu',weights_only=True)
        target=np.load(checked(D30['T_targets']),mmap_mode='r',allow_pickle=False)
        latent_receipt=read(checked(spec['native_T_latent_receipt']))
        assert latent_receipt['array_rows']==[x['array_row'] for x in index]
        latents=np.load(checked(spec['native_T_latents']),mmap_mode='r',allow_pickle=False)
        assert latents.shape==(13230,4,64,64) and latents.dtype==np.float32
        frozen=read(checked(spec['ordinary_spec']));saved=torch.load(checked(frozen['checkpoint']),map_location='cpu',weights_only=True)
        assert saved['calibration']['selected_strength']==1
        ordinary=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False).cuda();ordinary.load_state_dict(saved['model'])
        cls=author_brain_class(checked(D30['author_models_source']),checked(D30['author_layers_source']))
        original=cls(4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False).cuda();original.load_state_dict(parent['model'],strict=True)
        student=copy.deepcopy(original)
        gamma=original.block.layers[0].norm.weight.detach();beta=original.block.layers[0].norm.bias.detach()
        assert gamma.abs().min()>1e-6
        selected=np.linspace(0,13229,32,dtype=np.int64)
        x=torch.from_numpy(np.array(raw[selected],copy=True)).cuda();u0=torch.from_numpy(np.array(units[selected],copy=True)).cuda()
        y=torch.from_numpy(np.array(target[selected],copy=True)).cuda();z=torch.from_numpy(np.array(latents[selected],copy=True)).cuda()
        with torch.no_grad():
            # Use the exact training batch shape for the frozen function;
            # FP32 GEMM rounding can depend on batch length.
            c0=torch.cat([original(x[lo:lo+16]) for lo in (0,16)])
            uGPU=(c0-beta)/gamma
        raw_forward_error=float((uGPU-u0).abs().max());assert raw_forward_error<2e-4
        matrices={'measurement':torch.from_numpy(Q).float().cuda(),'permuted_measurement':torch.from_numpy(Qp).float().cuda()}
        def digest(module):
            h=hashlib.sha256()
            for name,t in module.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        oldOrd=digest(ordinary);oldEncoder=digest(original);initial=digest(student);assert initial==oldEncoder
        stage_dir=B/'converted_weights';stage_dir.mkdir();observer=ConvertedWeightAudit(torch,stage_dir);observer.install()
        active='STRICT_NATIVE_LOAD'
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:observer.restore()
        loading=observer.complete();frozenHost={name:digest(module) for name,module in [('UNet',pipe.unet),('A',adapter),('P',projector)]}
        def budget():
            assert time.monotonic()-started<840
            assert memory()['VmHWM']-baseCPU<10<<30 and torch.cuda.max_memory_allocated()<20<<30
            assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<128<<20
        budget();active='T_ZERO_UPDATE_PIXEL_PARITY'
        parity=[];pixels=[];seed=39001729
        with torch.inference_mode():
            for name,model in [('frozen_original',original),('initial_native_core',student)]:
                cc=model(x[:1]);changed=ordinary(cc,gamma[None],beta[None],gate_mode='fixed')[0]
                tokens=projector(adapter(changed.half()))
                image,observed=sample_observed(pipe,prompts,tokens,seed,torch);pixels.append(np.asarray(image.convert('RGB')).copy())
                parity.append(dict(case=name,subject_id='sub-01',stimulus_id=index[int(selected[0])]['stimulus_id'],array_row=index[int(selected[0])]['array_row'],
                   image_seed=seed,**observed,**save_image(B/'T_parity'/f'{name}.webp',image)))
                budget()
        assert np.array_equal(pixels[0],pixels[1])
        # Cached native latents and prompt tensors suffice for training.
        # Drop unused modules without rounding or altering any retained weights.
        pipe.register_modules(vae=None,text_encoder=None,text_encoder_2=None);gc.collect();torch.cuda.empty_cache()
        pipe.unet.train();assert float(pipe.unet.config.dropout)==0;pipe.unet.enable_gradient_checkpointing()
        from diffusers import DDPMScheduler
        schedule=DDPMScheduler.from_config(pipe.scheduler.config);assert schedule.config.prediction_type=='epsilon'
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):teacher_tokens=projector(adapter(y)).float().detach()
        first_states=[];total_train_seconds=0.;eps_gradients=None
        for arm in M.ARMS:
            active='REAL_NATIVE_CORE_TEMPORARY_UPDATES_'+arm
            student.load_state_dict(parent['model'],strict=True);student.eval().requires_grad_(False)
            params=[]
            for name,p in student.named_parameters():
                if name not in ['block.layers.0.norm.weight','block.layers.0.norm.bias']:p.requires_grad_(True);params.append((name,p))
            assert len(params)==4 and sum(p.numel() for _,p in params)==6450176
            affine_before={name:parent['model'][name].clone() for name in ['block.layers.0.norm.weight','block.layers.0.norm.bias']}
            optimizer=torch.optim.AdamW([p for _,p in params],lr=1e-5,weight_decay=.01)
            rng=torch.Generator(device='cuda').manual_seed(39001730)
            for step in range(2):
                budget();ts=time.monotonic();sl=slice(step*16,(step+1)*16);optimizer.zero_grad(set_to_none=True)
                noise=torch.randn(z[sl].shape,device='cuda',generator=rng);t=torch.randint(0,1000,(16,),device='cuda',generator=rng)
                noisy=schedule.add_noise(z[sl],noise,t)
                cc=student(x[sl]);unit=(cc-beta)/gamma
                changed=cc  # independent native branch; ordinary is absent from the training graph
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    tokens=projector(adapter(changed));predicted=denoise(pipe,prompts,noisy,t,tokens)
                    leps=F.mse_loss(predicted.float(),noise)
                    ltoken=(1-F.cosine_similarity(tokens.float(),teacher_tokens[sl],dim=-1)).mean()
                # Compare the current native function with its exact frozen-GPU
                # origin. No CPU/GPU rounding offset enters the anchoring loss.
                penalty=M.anchor_penalty(unit,uGPU[sl],matrices.get(arm),arm)
                loss=leps+.1*ltoken+penalty
                if arm=='unanchored' and step==0:
                    eg=torch.autograd.grad(leps,[p for _,p in params],retain_graph=True)
                    eps_gradients={name:float(g.norm()) for (name,_),g in zip(params,eg)}
                    assert all(v>0 and math.isfinite(v) for v in eps_gradients.values())
                assert torch.isfinite(loss);loss.backward()
                norms={name:float(p.grad.norm()) for name,p in params};assert all(v>0 and math.isfinite(v) for v in norms.values())
                assert all(p.grad is None for module in [original,ordinary,adapter,projector,pipe.unet] for p in module.parameters())
                torch.nn.utils.clip_grad_norm_([p for _,p in params],1.,error_if_nonfinite=True);optimizer.step()
                for name,value in affine_before.items():assert torch.equal(student.state_dict()[name].detach().cpu(),value)
                elapsed=time.monotonic()-ts;total_train_seconds+=elapsed
                event=dict(arm=arm,step=step+1,rows=selected[sl].tolist(),t=t.tolist(),noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    epsilon_loss=float(leps),token_loss=float(ltoken),anchor_loss=float(penalty),total_loss=float(loss),all4_core_gradient_norms=norms,
                    current_unit_shift_RMS=float((unit-uGPU[sl]).square().mean().sqrt()),seconds=elapsed,encoder_state_sha256=digest(student))
                if step==0:first_states.append(event['encoder_state_sha256']);assert float(penalty)==0
                records.append(event)
                with (B/'temporary_updates.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
                print(json.dumps(dict(stage=active,**event)),flush=True)
            assert digest(student)!=oldEncoder
        assert len(set(first_states))==1
        student.load_state_dict(parent['model'],strict=True);assert digest(student)==oldEncoder
        assert digest(original)==oldEncoder and digest(ordinary)==oldOrd
        assert frozenHost=={name:digest(module) for name,module in [('UNet',pipe.unet),('A',adapter),('P',projector)]}
        budget()
        write(B/'completion.json',dict(status='D39_REAL_NATIVE_EEG_CORE_GRADIENT_AND_PARITY_ENGINEERING_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          source=ref(Path(__file__).resolve()),spec=ref(sp),CPU_completion=spec['CPU_completion'],wrapper=proof,loader=loading,host=host,
          engineering_T_queries=32,raw_CPU_vs_GPU_unit_max_error=raw_forward_error,trainable_original_core_parameters=6450176,frozen_original_affine_parameters=2048,
          ordinary_in_training_graph=False,ordinary_used_only_in_parity_and_byte_guards=True,new_adapter_or_inference_parameters=0,temporary_updates=8,temporary_query_exposures=128,temporary_states_discarded=True,
          epsilon_only_gradients_into_all4_original_core_tensors=eps_gradients,first_update_identical_across_four_arms=True,
          frozen_ordinary_SHA=sha(checked(frozen['checkpoint'])),frozen_ordinary_strength=1,original_generator_and_A_P_weights_unchanged=True,
          zero_update_T_pixel_parity_exact=True,parity_images=parity,research_images=0,full_T_training_completed=False,
          original_model_fingerprint=oldEncoder,frozen_host_fingerprints=frozenHost,training_seconds=total_train_seconds,
          seconds=time.monotonic()-started,peak_extra_CPU_bytes=memory()['VmHWM']-baseCPU,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
          native_training_throughput_is_profile_not_complete_run=True,no_novelty_or_efficacy_claim=True))
    except BaseException as error:
        write(B/'failure.json',dict(status='D39_GPU_ENGINEERING_FAILURE_PRESERVED',stage=active,error=repr(error),traceback=traceback.format_exc(),temporary_updates=len(records),seconds=time.monotonic()-started));raise

if __name__=='__main__':
    import math
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
