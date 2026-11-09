"""Native R80 final-head profile or fixed200-image segment; no query targets."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import brainhive_full1654_F_study_contract_20261101_v1 as G

def digest(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(path,arm,mode,lo):
    start=time.monotonic();s=G.setup(path);G.wrapper(s);G.resources(s);base=G.rss()
    import numpy as np,torch
    from PIL import Image
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,float(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    root=Path(s['output_root'])/arm;model,fit,audit=G.load_final(s,arm,torch);model=model.cuda()
    source=G.read(G.check(s['evaluation']['input_completion']));bank=np.load(G.check(source['bank']),allow_pickle=False)
    rows=[json.loads(x) for x in G.check(s['evaluation']['baseline_inference']).read_text().splitlines()]
    assert len(rows)==2000 and [(r['subject'],r['array_row']) for r in rows]==[(f'sub-{n:02d}',i) for n in range(1,11) for i in range(200)]
    assert bank['conditions'].shape==(10,200,1024)
    for n,report in enumerate(source['reports']):
        weights=torch.load(G.check(report['model']),weights_only=True,map_location='cpu')['model']
        assert np.array_equal(bank['gammas'][n],weights['block.layers.0.norm.weight'].numpy()) and np.array_equal(bank['betas'][n],weights['block.layers.0.norm.bias'].numpy())
    def inputs(i):
        n,j=divmod(i,200);r=rows[i];c=torch.from_numpy(bank['conditions'][n,j:j+1].copy()).cuda();w=torch.from_numpy(bank['gammas'][n:n+1].copy()).cuda();b=torch.from_numpy(bank['betas'][n:n+1].copy()).cuda()
        assert hashlib.sha256(c.cpu().numpy().tobytes()).hexdigest()==r['condition_sha256'];return r,c,w,b
    if mode=='profile':
        out=root/'native_profile';assert not out.exists();out.mkdir()
        corrected=[];reference=G.model_class(arm)().double();reference.load_state_dict({k:v.double() for k,v in torch.load(G.check(fit['component']),weights_only=True,map_location='cpu')['model'].items()},strict=True)
        maximum=0.
        with torch.inference_mode():
            for i in range(2000):
                _,c,w,b=inputs(i);v=model(c,w,b)[0].cpu();assert torch.equal(model(c,w,b,enabled=False)[0],c)
                expected=reference(c.cpu().double(),w.cpu().double(),b.cpu().double())[0];maximum=max(maximum,float((v.double()-expected).abs().max()));assert torch.allclose(v.double(),expected,**s['numerical_guard'])
                assert torch.isfinite(v.half()).all();corrected.append(v.numpy()[0])
        certified=np.asarray(corrected,np.float32);np.save(out/'certified_singleton_conditions.npy',certified,allow_pickle=False)
        indices=list(range(0,2000,200))
    else:
        assert 0<=lo<2000 and lo%200==0
        profile=G.read(root/'native_profile/completion.json');assert profile['status']=='PASS_FULL1654_FINAL_HEAD_ALL2000_CONDITIONS_AND10_NATIVE_REPLAYS'
        certified=np.load(G.check(profile['certified_conditions']),allow_pickle=False)
        out=root/'generation'/f'{lo:04d}_{lo+200:04d}';assert not out.exists();out.mkdir(parents=True)
        indices=list(range(lo,lo+200))
    (out/'converted').mkdir();observer=ConvertedWeightAudit(torch,out/'converted');observer.install()
    try:pipe,A,P,prompts,host=load_native(training=False)
    finally:observer.restore()
    loader=observer.complete();modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P);before={k:digest(v) for k,v in modules.items()}
    def budget():assert time.monotonic()-start<820 and G.rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
    records=[];actual=[]
    try:
        for i in indices:
            budget();r,c,w,b=inputs(i)
            if mode=='profile' or i==lo:
                with torch.inference_mode():image,proof=sample_observed(pipe,prompts,P(A(c.half())),r['image_seed'],torch)
                assert G.ref(r['generated_path'])['sha256']==r['generated_sha256']
                with Image.open(r['generated_path']) as old:assert np.array_equal(np.asarray(old.convert('RGB')),np.asarray(image))
                save_image(out/f'native_replay_{i:04d}.webp',image)
            with torch.inference_mode():
                changed=model(c,w,b)[0];assert np.array_equal(changed.cpu().numpy().view(np.uint32),certified[i:i+1].view(np.uint32))
                tokens=P(A(changed.half()));image,proof=sample_observed(pipe,prompts,tokens,r['image_seed'],torch)
            saved=save_image(out/f'{i:04d}.webp',image);actual.append(changed.cpu().numpy()[0])
            row=dict(query_id=r['query_id'],subject=r['subject'],stimulus_id=r['stimulus_id'],array_row=r['array_row'],global_query_index=i,repeat_protocol='R80',intervention='correct',EEG_source_stimulus_id=r['stimulus_id'],method='BrainHIVE_'+arm+'_FULL1654_seed1729',image_seed=r['image_seed'],
                arm=arm,original_condition_sha256=r['condition_sha256'],condition_sha256=hashlib.sha256(changed.cpu().numpy().tobytes()).hexdigest(),actual_A_input_fp16_sha256=hashlib.sha256(changed.half().cpu().numpy().tobytes()).hexdigest(),actual_tokens_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
                parent_checkpoint=source['reports'][i//200]['model'],component=fit['component'],strength=1.,radius=.25,source_study=G.ref(path),normal_inference_target_access=False,one_image_per_query=True,**saved,**proof)
            G.write(out/f'{i:04d}.json',row);records.append(row)
        assert {k:digest(v) for k,v in modules.items()}==before
        assert G.ref(fit['component']['path'])==fit['component'];budget()
        if mode=='profile':
            done=dict(status='PASS_FULL1654_FINAL_HEAD_ALL2000_CONDITIONS_AND10_NATIVE_REPLAYS',arm=arm,study=G.ref(path),component=fit['component'],certified_conditions=G.ref(out/'certified_singleton_conditions.npy'),CPU64_singleton_max_absolute_error=maximum,guard=s['numerical_guard'],native_replays=10,changed_engineering_images=10,loader=loader,seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base)
        else:
            np.save(out/'actual_conditions.npy',np.asarray(actual,np.float32),allow_pickle=False)
            done=dict(status='COMPLETE_FULL1654_R80_200_IMAGES',arm=arm,study=G.ref(path),range=[lo,lo+200],component=fit['component'],actual_conditions=G.ref(out/'actual_conditions.npy'),records=[G.ref(out/f'{i:04d}.json') for i in indices],native_replays=1,images=200,loader=loader,seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base,effect_scores_read=False)
        G.write(out/'completion.json',done);print(json.dumps(done))
    except BaseException as error:G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),mode=mode,lo=lo));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--arm',choices=G.ARMS,required=True);p.add_argument('--mode',choices=['profile','generate'],required=True);p.add_argument('--lo',type=int,default=0);a=p.parse_args();run(a.study,a.arm,a.mode,a.lo)
