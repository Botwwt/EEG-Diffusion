"""All 16600 V images, native host, one image/query, explicit private placement."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from brainhive_native_models_v1 import *
from brainhive_component_generator_v1 import load_native
from brainhive_bounded_component_v1 import BoundedConditionAdapter
from brainhive_lifecycle_storage_v1 import verify_remaining


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',required=True)
    parser.add_argument('--arm',choices=['parent','ordinary','risk'],required=True);args=parser.parse_args()
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    recipe_path=Path(args.recipe);recipe=json.loads(recipe_path.read_text())
    for name,value in recipe['source_sha256'].items():assert sha(ROOT/name)==value
    storage=json.loads(Path(recipe['storage_lifecycle_receipt']).read_text());assert storage['all_three_main_lifecycles_reserved']
    verify_remaining(recipe)
    segments=storage['segments'][args.arm]
    assert segments[0]['start']==0 and segments[-1]['stop']==16600
    assert all(a['stop']==b['start'] for a,b in zip(segments,segments[1:]))
    run=recipe['run_names'][args.arm];out=WORK/'images'/run;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'completion.json').exists()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pipe,adapter,projector,prompts,host=load_native(training=False)
    from PIL import Image
    import numpy as np
    component=None;calibration={};checkpoint_sha=None
    if args.arm!='parent':
        folder=WORK/'components/bounded_v1'/args.arm;done=json.loads((folder/'completion.json').read_text())
        assert done['status']=='COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_COMPONENT'
        assert sha(done['checkpoint_path'])==done['checkpoint_sha256']
        state=torch.load(done['checkpoint_path'],weights_only=True,map_location='cpu')
        assert state['recipe_sha256']==sha(recipe_path)
        component=BoundedConditionAdapter(predict_risk=args.arm=='risk').cuda().eval().requires_grad_(False)
        component.load_state_dict(state['model'],strict=True);calibration=state['calibration'];checkpoint_sha=done['checkpoint_sha256']
        if args.arm=='ordinary':component.radius=.25*calibration['selected_strength']
    configuration=dict(generator=host,arm=args.arm,component_checkpoint_sha256=checkpoint_sha,calibration=calibration,
        recipe_sha256=sha(recipe_path),source_sha256=sha(__file__),parent_role='full_source_dropout_v2',
        resolution=512,steps=4,CFG=0,images_per_query=1,training_seed=1729,normal_inference_target_access=False)
    receipt=out/'model_receipt.json'
    if receipt.exists():assert json.loads(receipt.read_text())==configuration
    else:write(receipt,configuration)
    inputs=json.loads((WORK/'data/generation_inputs/V.json').read_text());assert inputs['expected_queries']==16600
    assert not inputs['target_image_or_class_fields_present'] and inputs['repeat_protocol']=='R4'
    assert all(set(r)=={'stimulus_id','array_row'} for r in inputs['inputs'])
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    progress={};cls=type(next(p for n,p in pipe.unet.attn_processors.items() if n.endswith('attn2.processor')))
    original=cls.__call__;cross_count=sum(n.endswith('attn2.processor') for n in pipe.unet.attn_processors)
    @functools.wraps(original)
    def checked(self,*a,**kw):
        value=kw.get('ip_hidden_states');assert value is not None and value.shape==(1,4,2048)
        assert value.data_ptr()==progress['token_pointer'];progress['IP_conditioned_attention_calls']+=1
        return original(self,*a,**kw)
    cls.__call__=checked
    def unet_hook(module,args,result):
        progress['UNet_NFE']+=args[0].shape[0];assert torch.isfinite(result[0]).all()
    def vae_hook(module,args,result):
        assert torch.isfinite(result).all();progress['VAE_decoder_calls']+=1
        progress['VAE_float_min']=float(result.min());progress['VAE_float_max']=float(result.max())
    handles=[pipe.unet.register_forward_hook(unet_hook),pipe.vae.decoder.register_forward_hook(vae_hook)]
    manifest=out/'inference_manifest.jsonl';saved=[json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    done={r['query_id']:r for r in saved};assert len(done)==len(saved)
    for row in saved:assert sha(row['generated_path'])==row['generated_sha256']
    global_index=0;segment_index=0;started=time.monotonic()
    with manifest.open('a',buffering=1) as log:
        for subject in inputs['subjects']:
            folder=WORK/'runs/full_source_dropout_v2'/subject
            data=torch.load(folder/'conditions.pt',weights_only=True,map_location='cpu',mmap=True)
            checkpoint=folder/'brain_final.pt';assert data['model_sha256']==sha(checkpoint)
            brain=torch.load(checkpoint,weights_only=True,map_location='cpu')['model']
            weight=brain['block.layers.0.norm.weight'].cuda()[None];bias=brain['block.layers.0.norm.bias'].cuda()[None]
            lookup={int(row):i for i,row in enumerate(data['array_rows'])}
            for query in inputs['inputs']:
                index=global_index;global_index+=1
                while index>=segments[segment_index]['stop']:segment_index+=1
                segment=segments[segment_index];assert segment['start']<=index<segment['stop']
                q=f"{subject}/{query['stimulus_id']}/R4"
                if q in done:continue
                root=Path(segment['root']);assert root in [Path(p) for p in storage['allowed_private_roots']]
                assert shutil.disk_usage(root).free>(512<<20),'Preserve saved images and stop before private disk buffer is consumed'
                p=root/run/subject/(Path(query['stimulus_id']).stem+'.webp');p.parent.mkdir(parents=True,exist_ok=True)
                assert not p.exists() and not p.is_symlink(),'Preserve orphan file for explicit repair'
                position=lookup[query['array_row']];c=data['condition'][position:position+1]
                image_seed=int(hashlib.sha256(f"1729|{subject}|{query['stimulus_id']}|R4|{protocol['version']}".encode()).hexdigest()[:16],16)%(2**63-1)
                progress.clear();progress.update(UNet_NFE=0,IP_conditioned_attention_calls=0,VAE_decoder_calls=0)
                with torch.inference_mode():
                    raw=c.cuda();info={}
                    if component is not None:
                        cal=(calibration['slope'],calibration['intercept']) if args.arm=='risk' else (1.,0.)
                        raw,info=component(raw,weight,bias,calibration=cal)
                    native=raw.to(torch.float16);tokens=projector(adapter(native));progress['token_pointer']=tokens.data_ptr()
                    assert torch.isfinite(tokens).all() and tokens.shape==(1,4,2048)
                    image=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(image_seed),
                        prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],
                        cross_attention_kwargs={'ip_hidden_states':tokens},num_inference_steps=4,guidance_scale=0.,output_type='pil').images[0].convert('RGB')
                assert progress['UNet_NFE']==4 and progress['IP_conditioned_attention_calls']==4*cross_count and progress['VAE_decoder_calls']==1
                progress.pop('token_pointer');pixels=np.asarray(image)
                image.save(p,format='WEBP',lossless=True,quality=90,method=4)
                with Image.open(p) as decoded:assert np.array_equal(pixels,np.asarray(decoded.convert('RGB')))
                row=dict(query_id=q,subject=subject,stimulus_id=query['stimulus_id'],array_row=query['array_row'],
                    global_query_index=index,repeat_protocol='R4',intervention='correct',method='BrainHIVE_'+run,
                    EEG_source_stimulus_id=query['stimulus_id'],image_seed=image_seed,generated_path=str(p),generated_sha256=sha(p),
                    condition_sha256=hashlib.sha256(c.contiguous().numpy().tobytes()).hexdigest(),condition_norm=float(c.norm()),
                    actual_A_input_fp16_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                    component_correction_RMS=float((raw-c.cuda()).square().mean().sqrt()),
                    component_gate=float(info['gate'][0]) if info else None,
                    predicted_risk=float(info['risk'][0]) if info and info['risk'] is not None else None,
                    target_or_visual_features_read_for_condition=False,oracle_diagnostic_only=False,lossless_pixel_roundtrip=True,**progress)
                log.write(json.dumps(row)+'\n');done[q]=row
                if len(done)%100==0 or len(done)==16600:print(json.dumps(dict(stage='BRAINHIVE_COMPLETE_V_IMAGES',arm=args.arm,completed=len(done),expected=16600,seconds=time.monotonic()-started)),flush=True)
                if len(done)%500==0:verify_remaining(recipe)
    for handle in handles:handle.remove()
    cls.__call__=original;assert global_index==len(done)==16600
    write(out/'completion.json',dict(status='ALL_ACTUAL_LOSSLESS_IMAGES_SAVED_INFERENCE_ONLY',queries=16600,
        role='full',split='V',intervention='correct',seconds=time.monotonic()-started,inference_manifest_sha256=sha(manifest),
        component_arm=args.arm,images_per_query=1,selected_best_seed=False,component_gain_established=False))


if __name__=='__main__':main()
