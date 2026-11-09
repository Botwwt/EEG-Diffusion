"""D25 native/ordinary separability, exact-off pixels, and actual noise geometry."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_noise_geometry_sampler_20261028_v1 as V

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args();assert a.max_seconds==240 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stop=[False];stage='INPUTS'
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        spec=U.read(a.spec)
        for item in spec['source_pins']:U.checked(item)
        freeze=U.read(U.checked(spec['ordinary_freeze']));U.checked(freeze['checkpoint']);U.checked(freeze['source']);assert freeze['selected_strength']==1 and freeze['change_ordinary_loss_or_weights'] is False
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_native_models_v1 import classes
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N
        D.torch=torch;cfg=C.read(a.native_registration);document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=240,maximum_output_bytes=64<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(document);budget=D.Budget(document)
        def check():assert not stop[0] and time.monotonic()-start<210;budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']));bank=np.load(U.checked(d09['check_conditions']),allow_pickle=False);rows=[json.loads(x) for x in U.checked(d09['check_manifest']).read_text().splitlines()]
        assert bank.shape==(400,3,1024) and len(rows)==1200
        for i,q in enumerate(panel['Dcheck']):
            for j,arm in enumerate(U.ARMS):
                row=rows[3*i+j];assert row['arm']==arm
                for k,v in q.items():assert row[k]==v
        # Independently replay all400 native/ordinary inputs using the unchanged
        # original full encoder and frozen original ordinary checkpoint.
        prepared=U.read(U.checked(spec['prepared_EEG']));mapping=U.read(U.checked(prepared['mapping']));assert mapping['Dcheck_queries']==panel['Dcheck']
        eeg=torch.from_numpy(np.array(np.load(U.checked(prepared['Dcheck']),mmap_mode='r'),copy=True).reshape(400,4250));contract=U.read(U.checked(spec['native_input_contract']));source=next(v for v in contract['subjects'][0]['sources'] if v['role']=='full')
        model=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(torch.load(U.checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model'],strict=True)
        component=torch.load(U.checked(freeze['checkpoint']),weights_only=False,map_location='cpu');assert component['arm']=='ordinary' and component['calibration']['selected_strength']==1
        ordinary=BoundedConditionAdapter().eval().requires_grad_(False);ordinary.load_state_dict(component['model'],strict=True)
        with torch.no_grad():
            parent=model(eeg);w=model.block.layers[0].norm.weight;b=model.block.layers[0].norm.bias;corrected=ordinary(parent,w.expand(400,-1),b.expand(400,-1))[0]
        replay={}
        for name,j,value in [('parent',0,parent),('ordinary',1,corrected)]:
            expected=torch.from_numpy(bank[:,j]);error=float(((value-expected).double().norm(dim=1)/expected.double().norm(dim=1)).max());assert error<3e-5;replay[name]=error
        del eeg,model,ordinary,component,parent,corrected;gc.collect();check();stage='NATIVE_LOADING'
        loader=a.output/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();assert host==inputs['expected_host']
        with torch.inference_mode():
            tokens=[projector(adapter(torch.from_numpy(np.array(bank[0,j:j+1],copy=True)).cuda().half())) for j in (0,1)]
            missing=projector(adapter(torch.zeros((1,1024),device='cuda',dtype=torch.float16)))
        cases=[('native_isotropic',0,'isotropic',True,False,True),('ordinary_isotropic',1,'isotropic',True,False,True),('native_response_null',0,'response_null',True,False,False),('ordinary_response_null',1,'response_null',True,False,False),('native_orthogonal_control',0,'orthogonal_control',True,False,False),('ordinary_orthogonal_control',1,'orthogonal_control',True,False,False),('ordinary_zero_response',1,'response_null',True,True,True),('ordinary_sampler_off',1,'isotropic',False,False,True)]
        proofs=[];traces={};query=panel['Dcheck'][0]
        for name,j,mode,enabled,zero,parity in cases:
            check();stage=name;tic=time.monotonic()
            with torch.inference_mode():image,proof,arrays=V.sample(pipe,prompts,tokens[j],tokens[j] if zero else missing,query['image_seed'],torch,mode,capture_arrays=True,enabled=enabled)
            saved=N.save_image(a.output/(name+'.webp'),image);baseline=rows[j]
            for k in ['initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations']:assert proof[k]==baseline[k]
            if parity:
                for k in ['generated_sha256','generated_pixel_sha256']:assert saved[k]==baseline[k]
            assert proof['UNet_NFE']==(8 if enabled else 4)
            if zero:assert all(torch.equal(v['raw_noise'],v['used_noise']) and torch.count_nonzero(v['field'])==0 for v in arrays)
            if mode=='isotropic':assert all(torch.equal(v['raw_noise'],v['used_noise']) for v in arrays)
            traces[name]=arrays;proofs.append(dict(name=name,condition_arm=U.ARMS[j],mode=mode,sampler_enabled=enabled,synthetic_zero_response=zero,exact_legacy_pixel_parity=parity,image=saved,observations=proof,seconds=time.monotonic()-tic))
        torch.save(traces,a.output/'all8_noise_traces.pt')
        for item in spec['source_pins']:U.checked(item)
        U.checked(freeze['checkpoint']);U.checked(freeze['source']);check()
        done=dict(status='D25_NATIVE_AND_FROZEN_ORDINARY_SAMPLER_SEPARABILITY_PROFILE_COMPLETE',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,host=host,loader_proof=loader_proof,ordinary_freeze=spec['ordinary_freeze'],ordinary_unchanged=True,all400_native_ordinary_condition_replay_relative_error=replay,
            cases=proofs,noise_traces=U.ref(a.output/'all8_noise_traces.pt'),new_engineering_images=8,new_research_images=0,new_parameter_updates=0,new_UNet_calls=sum(v['observations']['UNet_NFE'] for v in proofs),reference_branch_UNet_overhead_per_normal_query=4,formal_D_resumed=False,
            direct_image_efficacy_scores_read=False,seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',done);print(json.dumps({k:done[k] for k in ['status','seconds','new_UNet_calls','ordinary_unchanged','all400_native_ordinary_condition_replay_relative_error']}))
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True,ordinary_changed=False));raise

if __name__=='__main__':main()
