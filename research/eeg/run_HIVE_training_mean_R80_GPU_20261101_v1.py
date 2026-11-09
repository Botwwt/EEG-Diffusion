"""Native singleton control profile and fixed200-query generation segments."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback
import brainhive_training_mean_R80_contract_CPU_20261101_v1 as M


def run(protocol,role,mode,lo):
    started=time.monotonic();s,parent,F=M.setup(protocol);E=F.E;K=F.K
    import numpy as np
    import torch
    from PIL import Image
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
    K.require_wrapper();M.resources(s);base=K.rss()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    root=M.OUT/role;model,component,saved=M.load_control(s,role,torch);model=model.cuda()
    zero=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False).cuda()
    source,bank,records=M.load_evaluation_inputs(parent,torch)
    closure=M.read(M.OUT/'actual_source_data_closure.json');source_role=M.ROLES[role]
    source_records=[__import__('json').loads(x) for x in M.checked(closure['source_manifests'][source_role]).read_text().splitlines()]
    source_model=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
    source_model.load_state_dict(torch.load(M.checked(saved['source_component']),weights_only=True,map_location='cpu')['model'],strict=True)
    source_model=source_model.cuda()
    def inputs(index):
        sub,row=divmod(index,200);r=records[index]
        c=torch.from_numpy(bank['conditions'][sub,row:row+1].copy()).cuda()
        w=torch.from_numpy(bank['gammas'][sub:sub+1].copy()).cuda();b=torch.from_numpy(bank['betas'][sub:sub+1].copy()).cuda()
        return r,c,w,b
    if mode=='profile':
        out=root/'native_profile';assert not out.exists();out.mkdir(parents=True)
        reference=BoundedConditionAdapter(predict_risk=False).double().eval().requires_grad_(False)
        reference.load_state_dict({key:value.double() for key,value in saved['model'].items()},strict=True)
        maximum=0.;conditions=[]
        with torch.inference_mode():
            for index in range(2000):
                unused,c,w,b=inputs(index);value=model(c,w,b)[0]
                assert torch.equal(model(c,w,b,enabled=False)[0],c) and torch.equal(zero(c,w,b)[0],c)
                expected=reference(c.cpu().double(),w.cpu().double(),b.cpu().double())[0]
                maximum=max(maximum,float((value.cpu().double()-expected).abs().max()))
                assert torch.allclose(value.cpu().double(),expected,**s['numerical_guard']) and torch.isfinite(value.half()).all()
                conditions.append(value.cpu().numpy()[0])
        certified=np.asarray(conditions,np.float32);np.save(out/'certified_singleton_conditions.npy',certified,allow_pickle=False)
        indices=s['profile_indices']
    else:
        assert lo%200==0 and 0<=lo<2000
        profile=M.read(root/'native_profile/completion.json')
        assert profile['status']=='PASS_TRAIN_MEAN_ALL2000_CONDITIONS_AND10_FIVE_PATH_PIXEL_REPLAYS'
        assert profile['component']==component and profile['role']==role
        certified=np.load(M.checked(profile['certified_conditions']),allow_pickle=False)
        out=root/'generation'/f'{lo:04d}_{lo+200:04d}';assert not out.exists();out.mkdir(parents=True)
        indices=list(range(lo,lo+200))
    (out/'converted').mkdir();observer=ConvertedWeightAudit(torch,out/'converted');observer.install()
    try:pipe,A,P,prompts,host=load_native(training=False)
    finally:observer.restore()
    loader=observer.complete();modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P)
    before={key:E.module_digest(module) for key,module in modules.items()}
    def guard():
        assert time.monotonic()-started<840 and K.rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        M.resources(s)
    def replay(value,index,label,expected):
        r=records[index]
        with torch.inference_mode():image,proof=sample_observed(pipe,prompts,P(A(value.half())),r['image_seed'],torch)
        assert M.ref(expected['generated_path'])['sha256']==expected['generated_sha256']
        with Image.open(expected['generated_path']) as previous:assert np.array_equal(np.asarray(image),np.asarray(previous.convert('RGB')))
        assert proof['UNet_NFE']==4 and proof['native_IP_K_calls']==proof['native_IP_V_calls']==280
        for key in ('initial_noise_latents_sha256','ancestral_noise_observations'):assert proof[key]==expected[key]
        generated=save_image(out/f'{label}_{index:04d}.webp',image)
        M.write(out/f'{label}_{index:04d}.json',dict(index=index,path=label,expected_image_path=expected['generated_path'],
            expected_file_SHA256=expected['generated_sha256'],pixel_exact=True,**generated,**proof))
    actual=[];proofs=[]
    try:
        for index in indices:
            guard();r,c,w,b=inputs(index)
            if mode=='profile' or index==lo:
                replay(c,index,'native_replay',r)
                with torch.inference_mode():source_changed=source_model(c,w,b)[0]
                assert hashlib.sha256(source_changed.cpu().numpy().tobytes()).hexdigest()==source_records[index]['condition_sha256']
                replay(source_changed,index,'source_head_replay',source_records[index])
                if mode=='profile':
                    with torch.inference_mode():off=model(c,w,b,enabled=False)[0];zero_value=zero(c,w,b)[0]
                    assert torch.equal(off,c) and torch.equal(zero_value,c)
                    replay(off,index,'disabled_replay',r);replay(zero_value,index,'zero_head_replay',r)
            with torch.inference_mode():
                changed=model(c,w,b)[0]
                assert np.array_equal(changed.cpu().numpy().view(np.uint32),certified[index:index+1].view(np.uint32))
                tokens=P(A(changed.half()));image,proof=sample_observed(pipe,prompts,tokens,r['image_seed'],torch)
            assert proof['UNet_NFE']==4 and proof['native_IP_K_calls']==proof['native_IP_V_calls']==280
            for key in ('initial_noise_latents_sha256','ancestral_noise_observations'):assert proof[key]==source_records[index][key]
            generated=save_image(out/f'{index:04d}.webp',image)
            record=dict(query_id=r['query_id'],subject=r['subject'],stimulus_id=r['stimulus_id'],array_row=r['array_row'],
                global_query_index=index,repeat_protocol='R80',intervention='correct',EEG_source_stimulus_id=r['stimulus_id'],
                method='BrainHIVE_TRAIN_ONLY_MEAN_DIRECTION_'+source_role,arm=role,image_seed=r['image_seed'],
                original_condition_sha256=r['condition_sha256'],condition_sha256=hashlib.sha256(changed.cpu().numpy().tobytes()).hexdigest(),
                actual_A_input_fp16_sha256=hashlib.sha256(changed.half().cpu().numpy().tobytes()).hexdigest(),
                actual_tokens_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),component=component,
                source_component=saved['source_component'],derivation_protocol=M.MEAN,
                parent_checkpoint=source['reports'][index//200]['model'],strength=1.,radius=.25,source_protocol=M.ref(protocol),
                normal_inference_target_access=False,one_image_per_query=True,new_training_updates=0,source_training_cost_inherited=True,
                mean_constructed_from_training_only=True,**generated,**proof)
            M.write(out/f'{index:04d}.json',record);actual.append(changed.cpu().numpy()[0]);proofs.append(M.ref(out/f'{index:04d}.json'))
        assert {key:E.module_digest(module) for key,module in modules.items()}==before;M.checked(parent['old_ordinary']);guard()
        if mode=='profile':
            result=dict(status='PASS_TRAIN_MEAN_ALL2000_CONDITIONS_AND10_FIVE_PATH_PIXEL_REPLAYS',
                certified_conditions=M.ref(out/'certified_singleton_conditions.npy'),profile_indices=indices,
                engineering_images=50,exact_native_disabled_zero_replays=30,exact_source_replays=10,
                mean_engineering_images=10,CPU64_singleton_maxerror=maximum,numerical_guard=s['numerical_guard'])
        else:
            np.save(out/'actual_conditions.npy',np.asarray(actual,np.float32),allow_pickle=False)
            result=dict(status='COMPLETE_TRAIN_MEAN_R80_FIXED200_IMAGES',range=[lo,lo+200],images=200,
                actual_conditions=M.ref(out/'actual_conditions.npy'),native_replays=1,source_replays=1,effect_scores_read=False)
        result.update(role=role,protocol=M.ref(protocol),component=component,source_component=saved['source_component'],records=proofs,
            loader=loader,seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),maximum_extra_CPU_bytes=K.rss()-base)
        M.write(out/'completion.json',result);print(__import__('json').dumps(result))
    except BaseException as error:
        M.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--role',choices=tuple(M.ROLES),required=True)
    p.add_argument('--mode',choices=('profile','generate'),required=True);p.add_argument('--lo',type=int,default=0)
    a=p.parse_args();run(a.protocol,a.role,a.mode,a.lo)
