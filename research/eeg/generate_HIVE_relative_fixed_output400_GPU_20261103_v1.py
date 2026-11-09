"""Generate frozen raw1 conditions through the unchanged HIVE native renderer."""
import argparse
import gc
import hashlib
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import brainhive_relative_fixed_output_contract_20261103_v1 as K


def main(args):
    ctx=K.setup(args.policy);s=ctx['s'];rel=ctx['relative'];r=Path(s['output_root'])
    resources=K.resources(ctx,starting=True);a=K.assets(ctx)
    sys.path.insert(0,rel['repository_root'])
    import torch
    import numpy as np
    from PIL import Image
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    started=time.monotonic();base=rss()
    attempt=r/(f'generate_{args.lo:03d}_{args.hi:03d}' if args.mode=='generate' else 'features400')
    assert not attempt.exists();attempt.mkdir()

    def budget():
        assert time.monotonic()-started<800 and rss()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30
        assert shutil.disk_usage(r).free>=resources['required_remaining_bytes']

    def bsha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    try:
        prepared=K.read(r/'CPU_preflight/completion.json')
        assert prepared['status']=='PASS_REAL_FROZEN4_RAW1_HEADS_ALL400_NUMPY64_AND_BASELINE_IDENTITIES'
        if args.mode=='features':
            import image_evaluation as E
            assert E.DINO_SHA256==rel['DINO_weight']['sha256'];K.checked(rel['DINO_weight'])
            model=E.dino_model();result={};encoded=0;reused=0
            candidates=K.read(r/'cached_candidates.json')
            for arm in K.ARMS:
                rows=[K.read(r/'records'/f'{i:03d}_{arm}.json') for i in range(400)]
                values=np.empty((400,768),np.float32);positions=[];paths=[]
                for i,row in enumerate(rows):
                    assert row['policy']==K.ref(args.policy) and row['query_id']==a['queries'][i]['query_id']
                    p=Path(row['generated_path']);assert K.ref(p)['sha256']==row['generated_sha256']
                    with Image.open(p) as im:assert bsha(np.asarray(im.convert('RGB')))==row['generated_pixel_sha256']
                    if row.get('cached_from') is not None:
                        assert i<80 and arm in candidates
                        old=np.load(K.checked(candidates[arm]['features80']),allow_pickle=False)
                        assert old.shape==(80,768);values[i]=old[i];reused+=1
                    else:positions.append(i);paths.append(p)
                for begin in range(0,len(paths),32):
                    budget();v=E.dino_features(paths[begin:begin+32],model,batch_size=32,real_images=False).float().numpy()
                    values[positions[begin:begin+32]]=v;encoded+=len(v)
                assert np.isfinite(values).all()
                p=r/f'features400_{arm}.npy'
                with p.open('xb') as f:np.save(f,values,allow_pickle=False)
                result[arm]=K.ref(p)
            budget()
            done=dict(status='COMPLETE_FROZEN_RAW1_ALL400_FEATURES',policy=K.ref(args.policy),features=result,
                newly_encoded=encoded,reused_identical_existing_features=reused,seconds=time.monotonic()-started,
                new_training_updates=0,resources=K.resources(ctx))
            K.atomic(r/'features_completion.json',done);K.atomic(attempt/'completion.json',done);return
        assert args.mode=='generate' and 0<=args.lo<args.hi<=400 and args.hi-args.lo==80
        references=K.read(r/'component_references.json');candidates=K.read(r/'cached_candidates.json')
        if not (r/'condition_completion.json').exists():
            assert args.lo==0
            from brainhive_bounded_component_v1 import BoundedConditionAdapter
            native=np.load(K.checked(prepared['native']),allow_pickle=False)
            affine=np.load(K.checked(prepared['affine']),allow_pickle=False)
            expected=np.load(K.checked(prepared['expected']),allow_pickle=False)
            c=torch.from_numpy(native).cuda();w=torch.from_numpy(affine['gamma'][None].copy()).cuda().expand_as(c)
            b=torch.from_numpy(affine['beta'][None].copy()).cuda().expand_as(c)
            conditions={};errors={};reuse={}
            for j,arm in enumerate(K.ARMS):
                state=torch.load(K.checked(references[arm]['component']),weights_only=True,map_location='cpu')
                model=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False).cuda()
                model.load_state_dict(state['model'],strict=True);model.radius=.25
                with torch.inference_mode():v=model(c,w,b,gate_mode='fixed')[0].cpu().numpy()
                error=float(np.max(abs(v.astype(np.float64)-expected[:,j])/(1+abs(expected[:,j]))));assert error<2e-6
                dest=r/f'conditions400_{arm}.npy'
                with dest.open('xb') as f:np.save(f,v,allow_pickle=False)
                conditions[arm]=K.ref(dest);errors[arm]=error
                reuse[arm]=arm in candidates and np.array_equal(v,np.load(K.checked(candidates[arm]['conditions']),allow_pickle=False))
                del model
            del c,w,b;gc.collect();torch.cuda.empty_cache()
            K.atomic(r/'condition_completion.json',dict(status='PASS_ALL1600_RAW1_GPU32_VS_INDEPENDENT_NUMPY64',
                policy=K.ref(args.policy),conditions=conditions,scaled_errors=errors,can_reuse_original80=reuse,
                old_selected_strengths_preserved={arm:v['old_selected_strength'] for arm,v in references.items()},
                all_diagnostic_strengths=1,new_training_updates=0))
        condition=K.read(r/'condition_completion.json')
        conditions={arm:np.load(K.checked(v),allow_pickle=False) for arm,v in condition['conditions'].items()}
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
        audit=ConvertedWeightAudit(torch,attempt/'converted');audit.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader=audit.complete();assert host==a['inputs']['expected_host']
        modules=dict(UNet=pipe.unet,VAE=pipe.vae,A=A,P=P)
        before={k:digest(v) for k,v in modules.items()}
        oldbank=np.load(K.checked(a['d09']['check_conditions']),allow_pickle=False)
        noise_keys=('initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations')
        for j,arm in enumerate(('parent','ordinary')):
            row=a['old_records'][j];c=torch.from_numpy(oldbank[:1,j].copy()).cuda()
            with torch.inference_mode():image,proof=sample_observed(pipe,prompts,P(A(c.half())),row['image_seed'],torch)
            with Image.open(row['generated_path']) as im:assert np.array_equal(np.asarray(image),np.asarray(im.convert('RGB')))
            assert all(proof[k]==row[k] for k in noise_keys)
            K.atomic(attempt/(arm+'_parity.json'),dict(pixel_exact=True,reference=row,observed=proof,
                **save_image(attempt/(arm+'_parity.webp'),image)))
        (r/'records').mkdir(exist_ok=True);(r/'pixels').mkdir(exist_ok=True)
        new_images=0;cached_images=0
        for i in range(args.lo,args.hi):
            v=a['queries'][i];reference=a['old_records'][3*i]
            for arm in K.ARMS:
                budget();dest=r/'records'/f'{i:03d}_{arm}.json';assert not dest.exists()
                if i<80 and condition['can_reuse_original80'][arm]:
                    old_path=Path(candidates[arm]['records_root'])/f'{i:03d}_{arm}.json';oldrow=K.read(old_path)
                    assert oldrow['query_id']==v['query_id'] and oldrow['selected_strength']==1
                    assert oldrow['condition_sha256']==bsha(conditions[arm][i:i+1])
                    assert oldrow['condition_FP16_sha256']==bsha(conditions[arm][i:i+1].astype(np.float16))
                    assert all(oldrow[k]==reference[k] for k in noise_keys)
                    assert K.ref(oldrow['generated_path'])['sha256']==oldrow['generated_sha256']
                    cached=dict(oldrow);cached.update(policy=K.ref(args.policy),fixed_strength=1,
                        original_selected_strength=references[arm]['old_selected_strength'],cached_from=K.ref(old_path))
                    K.atomic(dest,cached);cached_images+=1
                else:
                    c=torch.from_numpy(conditions[arm][i:i+1].copy()).cuda()
                    with torch.inference_mode():tokens=P(A(c.half()));image,proof=sample_observed(pipe,prompts,tokens,v['image_seed'],torch)
                    assert all(proof[k]==reference[k] for k in noise_keys)
                    saved=save_image(r/'pixels'/f'{i:03d}_{arm}.webp',image)
                    K.atomic(dest,dict(**v,query_index=i,arm=arm,policy=K.ref(args.policy),
                        fixed_strength=1,original_selected_strength=references[arm]['old_selected_strength'],
                        component=references[arm]['component'],condition_sha256=bsha(conditions[arm][i:i+1]),
                        condition_FP16_sha256=bsha(c.half().cpu().numpy()),tokens_sha256=bsha(tokens.cpu().numpy()),
                        normal_inference_target_access=False,one_image_per_query=True,cached_from=None,**saved,**proof))
                    new_images+=1
            if i%10==9:
                K.atomic(attempt/'progress.json',dict(completed_query=i+1,new_images=new_images,reused_images=cached_images,
                    seconds=time.monotonic()-started))
        assert before=={k:digest(v) for k,v in modules.items()};K.checked(rel['ordinary']);budget()
        K.atomic(attempt/'completion.json',dict(status='COMPLETE_FIXED_FROZEN_RAW1_IMAGE_BLOCK',range=[args.lo,args.hi],
            policy=K.ref(args.policy),new_images=new_images,reused_images=cached_images,loader=loader,
            all_frozen_generator_fingerprints_unchanged=True,seconds=time.monotonic()-started,
            maximum_extra_CPU=rss()-base,maximum_GPU_allocated=torch.cuda.max_memory_allocated(),resources=K.resources(ctx)))
    except BaseException as exc:
        K.atomic(attempt/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise


def rss():
    return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def digest(module):
    h=hashlib.sha256()
    for name,value in module.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--mode',choices=('generate','features'),required=True)
    ap.add_argument('--lo',type=int,default=0);ap.add_argument('--hi',type=int,default=80)
    main(ap.parse_args())
