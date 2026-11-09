"""Final original-objective head: native singleton profile/200-query image segments."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
E=F.E;K=F.K


def run(protocol,mode,lo):
    started=time.monotonic();s=F.setup_fit(protocol);s['_protocol_pin']=K.ref(protocol)
    import numpy as np
    import torch
    from PIL import Image
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
    K.require_wrapper();K.resources(s);base=K.rss()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    torch.cuda.reset_peak_memory_stats()
    root=Path(s['output_root'])/'ordinary_full1654';model,fit,audit=F.load_final(s,torch);model=model.cuda()
    source=K.read(K.check(s['evaluation']['input_completion']));bank=np.load(K.check(source['bank']),allow_pickle=False)
    records=[__import__('json').loads(line) for line in K.check(s['evaluation']['baseline_inference']).read_text().splitlines()]
    assert len(records)==2000 and bank['conditions'].shape==(10,200,1024)
    for index,item in enumerate(source['reports']):
        state=torch.load(K.check(item['model']),weights_only=True,map_location='cpu')['model']
        assert np.array_equal(bank['gammas'][index],state['block.layers.0.norm.weight'].numpy())
        assert np.array_equal(bank['betas'][index],state['block.layers.0.norm.bias'].numpy())
    def inputs(index):
        sub,row=divmod(index,200);r=records[index]
        assert r['subject']==f'sub-{sub+1:02d}' and r['array_row']==row
        c=torch.from_numpy(bank['conditions'][sub,row:row+1].copy()).cuda()
        w=torch.from_numpy(bank['gammas'][sub:sub+1].copy()).cuda();b=torch.from_numpy(bank['betas'][sub:sub+1].copy()).cuda()
        assert hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest()==r['condition_sha256']
        return r,c,w,b
    if mode=='profile':
        out=root/'native_profile';assert not out.exists();out.mkdir()
        reference=BoundedConditionAdapter(predict_risk=False).double()
        final=torch.load(K.check(fit['component']),weights_only=True,map_location='cpu')
        reference.load_state_dict({key:value.double() for key,value in final['model'].items()},strict=True)
        maximum=0.;conditions=[]
        with torch.inference_mode():
            for index in range(2000):
                unused,c,w,b=inputs(index);value=model(c,w,b)[0]
                assert torch.equal(model(c,w,b,enabled=False)[0],c)
                expected=reference(c.cpu().double(),w.cpu().double(),b.cpu().double())[0]
                error=float((value.cpu().double()-expected).abs().max());maximum=max(maximum,error)
                assert torch.allclose(value.cpu().double(),expected,**s['numerical_guard']) and torch.isfinite(value.half()).all()
                conditions.append(value.cpu().numpy()[0])
        certified=np.asarray(conditions,np.float32);np.save(out/'certified_singleton_conditions.npy',certified,allow_pickle=False)
        indices=list(range(0,2000,200))
    else:
        assert lo%200==0 and 0<=lo<2000
        profile=K.read(root/'native_profile/completion.json')
        assert profile['status']=='PASS_ORIGINAL_ORDINARY_FULL1654_ALL2000_CONDITIONS_AND10_NATIVE_REPLAYS'
        certified=np.load(K.check(profile['certified_conditions']),allow_pickle=False)
        out=root/'generation'/f'{lo:04d}_{lo+200:04d}';assert not out.exists();out.mkdir(parents=True)
        indices=list(range(lo,lo+200))
    (out/'converted').mkdir();observer=ConvertedWeightAudit(torch,out/'converted');observer.install()
    try:pipe,A,P,prompts,host=load_native(training=False)
    finally:observer.restore()
    loader=observer.complete();modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P)
    before={key:E.module_digest(module) for key,module in modules.items()}
    def guard():
        assert time.monotonic()-started<840 and K.rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        K.resources(s)
    saved_rows=[];actual=[]
    try:
        for index in indices:
            guard();r,c,w,b=inputs(index)
            if mode=='profile' or index==lo:
                with torch.inference_mode():native,unused=sample_observed(pipe,prompts,P(A(c.half())),r['image_seed'],torch)
                assert K.ref(r['generated_path'])['sha256']==r['generated_sha256']
                with Image.open(r['generated_path']) as image:assert np.array_equal(np.asarray(native),np.asarray(image.convert('RGB')))
                save_image(out/f'native_replay_{index:04d}.webp',native)
            with torch.inference_mode():
                changed=model(c,w,b)[0]
                assert np.array_equal(changed.cpu().numpy().view(np.uint32),certified[index:index+1].view(np.uint32))
                tokens=P(A(changed.half()));image,proof=sample_observed(pipe,prompts,tokens,r['image_seed'],torch)
            generated=save_image(out/f'{index:04d}.webp',image)
            row=dict(query_id=r['query_id'],subject=r['subject'],stimulus_id=r['stimulus_id'],array_row=r['array_row'],
                global_query_index=index,repeat_protocol='R80',intervention='correct',EEG_source_stimulus_id=r['stimulus_id'],
                method='BrainHIVE_original_ordinary_FULL1654_seed1729',arm='ordinary_full1654',image_seed=r['image_seed'],
                original_condition_sha256=r['condition_sha256'],condition_sha256=hashlib.sha256(changed.cpu().numpy().tobytes()).hexdigest(),
                actual_A_input_fp16_sha256=hashlib.sha256(changed.half().cpu().numpy().tobytes()).hexdigest(),
                actual_tokens_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),component=fit['component'],
                parent_checkpoint=source['reports'][index//200]['model'],strength=1.,radius=.25,source_protocol=K.ref(protocol),
                normal_inference_target_access=False,one_image_per_query=True,**generated,**proof)
            K.write(out/f'{index:04d}.json',row);saved_rows.append(row);actual.append(changed.cpu().numpy()[0])
        assert {key:E.module_digest(module) for key,module in modules.items()}==before;K.check(s['old_ordinary']);guard()
        if mode=='profile':
            result=dict(status='PASS_ORIGINAL_ORDINARY_FULL1654_ALL2000_CONDITIONS_AND10_NATIVE_REPLAYS',
                certified_conditions=K.ref(out/'certified_singleton_conditions.npy'),native_replays=10,changed_engineering_images=10,
                CPU64_singleton_maxerror=maximum,guard=s['numerical_guard'])
        else:
            np.save(out/'actual_conditions.npy',np.asarray(actual,np.float32),allow_pickle=False)
            result=dict(status='COMPLETE_ORIGINAL_ORDINARY_FULL1654_R80_200_IMAGES',range=[lo,lo+200],images=200,
                actual_conditions=K.ref(out/'actual_conditions.npy'),records=[K.ref(out/f'{index:04d}.json') for index in indices],
                native_replays=1,effect_scores_read=False)
        result.update(protocol=K.ref(protocol),component=fit['component'],loader=loader,
            seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),maximum_extra_CPU_bytes=K.rss()-base)
        K.write(out/'completion.json',result);print(__import__('json').dumps(result))
    except BaseException as error:
        K.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--mode',choices=('profile','generate'),required=True)
    parser.add_argument('--lo',type=int,default=0);args=parser.parse_args();run(args.protocol,args.mode,args.lo)
