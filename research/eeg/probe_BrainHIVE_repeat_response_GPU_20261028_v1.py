"""T-only physical EEG-repeat curvature diagnostic; does not alter any response.

Classical jackknife is a falsification tool here, not a novel algorithm. Every
candidate response is evaluated at the same actual native Q. The running UNet,
its attention processors, ordinary and the sampler return their original values.
No images, fitting, Dcheck or V scores are involved.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U

def metrics(response,correct):
    base=response[0];delta3=response[1:5].mean(0)-base;delta2=response[5:11].mean(0)-base
    minus=base-3*delta3;plus=base+3*delta3;teachers=response[11:21]
    target=teachers[correct];wrong=torch.cat([teachers[:correct],teachers[correct+1:]])
    denominator=((wrong-target)**2).mean().clamp_min(1e-10)
    out={}
    for name,value in [('base',base),('minus',minus),('plus',plus)]:
        distances=((teachers-value)**2).flatten(1).mean(1)
        positive=distances[correct];negative=torch.cat([distances[:correct],distances[correct+1:]]).mean()
        out[name+'_positive_MSE']=float(positive)
        out[name+'_J']=float((negative-positive)/denominator)
        out[name+'_within_concept_top1']=int(distances.argmin()==correct)
        out[name+'_norm']=float(value.norm())
    out['correction_relative_norm']=float((3*delta3).norm()/base.norm().clamp_min(1e-10))
    out['quadratic_scaling_residual']=float((delta2-3*delta3).norm()/(delta2.norm()+3*delta3.norm()).clamp_min(1e-10))
    return out

def main():
    p=argparse.ArgumentParser()
    for k in ['spec','native-registration','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    assert a.max_seconds==720 and not a.output.exists();a.output.mkdir(parents=True)
    start=time.monotonic();stage='INPUTS';stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        spec=U.read(a.spec)
        for item in spec['source_pins']:U.checked(item)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_native_models_v1 import classes
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        D.torch=torch;cfg=C.read(a.native_registration)
        document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=720,maximum_output_bytes=256<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(document);budget=D.Budget(document)
        def check():
            assert not stop[0] and time.monotonic()-start<675
            budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        freeze=U.read(U.checked(spec['ordinary_freeze']));component=torch.load(U.checked(freeze['checkpoint']),weights_only=False,map_location='cpu')
        assert component['arm']=='ordinary' and component['calibration']['selected_strength']==1
        ordinary=BoundedConditionAdapter().eval().requires_grad_(False).cuda();ordinary.load_state_dict(component['model'],strict=True);del component
        panel=U.read(U.checked(spec['panel']));queries=panel['rows'];assert len(queries)==80
        index=[json.loads(s) for s in U.checked(spec['index']).read_text().splitlines()];assert len(index)==13230
        by_concept={}
        for r in index:by_concept.setdefault(r['concept_id'],[]).append(r)
        for k in by_concept:by_concept[k].sort(key=lambda r:r['row']);assert len(by_concept[k])==10
        trials=np.load(U.checked(spec['trials']),mmap_mode='r');targets=np.load(U.checked(spec['targets']),mmap_mode='r')
        assert trials.shape==(13230,4,17,250) and targets.shape==(13230,1024)
        contract=U.read(U.checked(spec['native_input_contract']));sources=[s for s in contract['subjects'][0]['sources'] if s['role'] in ['oof0','oof1']]
        models={};routes={};cache={}
        for s in sources:
            m=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False).cuda();m.load_state_dict(torch.load(U.checked(dict(path=s['checkpoint_path'],sha256=s['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model'],strict=True)
            models[s['role']]=m
            bank=torch.load(U.checked(dict(path=s['condition_cache_path'],sha256=s['condition_cache_sha256'])),weights_only=True,map_location='cpu')
            # Original cache schema is checked against the route, never guessed.
            assert isinstance(bank,dict) and 'condition' in bank
            bank=bank['condition'];assert bank.ndim==2 and bank.shape[1]==1024
            cache[s['role']]=bank
            for r in s['groups']['component_train']:
                assert r['array_row'] not in routes;routes[r['array_row']]=(s['role'],r['cache_row'])
        assert len(routes)==13230
        import brainhive_repeat_response_bias_20261028_v1 as M
        means=[];conditions=[];replay=[]
        with torch.inference_mode():
            for q in queries:
                x=torch.from_numpy(np.array(trials[q['row']],copy=True)).reshape(1,4,4250).cuda()
                # Fixed order: full R4, four delete-one R3, six R2 pair means.
                subsets=M.subset_means(x)
                all_means=torch.cat([subsets.m4,subsets.m3[:,0],subsets.m2[:,0]],0)
                role,cr=routes[q['array_row']];model=models[role];c=model(all_means)
                error=float((c[0].cpu()-cache[role][cr]).double().norm()/cache[role][cr].double().norm());assert error<3e-5;replay.append(error)
                w=model.block.layers[0].norm.weight.expand(len(c),-1);b=model.block.layers[0].norm.bias.expand(len(c),-1)
                o=ordinary(c,w,b)[0];conditions.append(torch.stack([c,o]).cpu());means.append(all_means.cpu())
        conditions=torch.stack(conditions);means=torch.stack(means)
        del cache,models,ordinary,trials,all_means,x,c,o;gc.collect();torch.cuda.empty_cache();check();stage='NATIVE_LOADING'
        loader=a.output/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        loader_proof=audit.complete();check()
        tokens=[];teacher_tokens=[];corrects=[]
        with torch.inference_mode():
            for i,q in enumerate(queries):
                v=conditions[i].reshape(22,1024).cuda().half();tokens.append(projector(adapter(v)).reshape(2,11,4,2048).cpu())
                members=by_concept[q['concept_id']];f=torch.from_numpy(np.array(targets[[r['row'] for r in members]],copy=True)).cuda().half();teacher_tokens.append(projector(adapter(f)).cpu());corrects.append(next(j for j,r in enumerate(members) if r['row']==q['row']))
        tokens=torch.stack(tokens);teacher_tokens=torch.stack(teacher_tokens)
        assert tokens.shape==(80,2,11,4,2048) and teacher_tokens.shape==(80,10,4,2048)
        torch.save(dict(conditions=conditions,means=means,tokens=tokens,teacher_tokens=teacher_tokens,queries=queries,corrects=corrects),a.output/'input_bank.pt')
        del conditions,means,targets;gc.collect();check()
        names=[n for n,v in pipe.unet.attn_processors.items() if hasattr(v,'to_k_ip')];assert len(names)==70
        sampled_names=[names[i] for i in (0,23,46,69)];context=[None];rows=[];traces=[];handles=[];call_counts=[0]
        def unet_before(mod,args,kwargs):
            call_counts[0]+=1;check()
            if context[0] is not None:context[0]['step']+=1
        handles.append(pipe.unet.register_forward_pre_hook(unet_before,with_kwargs=True))
        for layer,name in enumerate(names):
            attn=pipe.unet.get_submodule(name.rsplit('.processor',1)[0]);processor=attn.processor
            def capture(mod,args,output,attn=attn,processor=processor,name=name,layer=layer):
                ctx=context[0]
                if ctx is None:return
                with torch.inference_mode():
                    positions=torch.linspace(0,output.shape[1]-1,16,device=output.device).round().long()
                    rawq=output[:,positions];h=attn.heads;d=rawq.shape[-1]//h
                    q=rawq.reshape(1,16,h,d).transpose(1,2)
                    if attn.norm_q is not None:q=attn.norm_q(q)
                    k=processor.to_k_ip(ctx['bank']).reshape(21,4,h,d).transpose(1,2)
                    v=processor.to_v_ip(ctx['bank']).reshape(21,4,h,d).transpose(1,2)
                    response=F.scaled_dot_product_attention(q.float().expand(21,-1,-1,-1),k.float(),v.float(),dropout_p=0.)
                    stats=metrics(response,ctx['correct']);item=dict(query_index=ctx['index'],concept_id=ctx['query']['concept_id'],stimulus_id=ctx['query']['stimulus_id'],arm=ctx['arm'],step=ctx['step'],layer=layer,**stats);rows.append(item)
                    if name in sampled_names and ctx['correct']==0:
                        traces.append(dict(query_index=ctx['index'],arm=ctx['arm'],step=ctx['step'],layer=layer,name=name,q=q.cpu(),k=k.cpu(),v=v.cpu(),metrics=stats,correct=ctx['correct']))
            handles.append(attn.to_q.register_forward_hook(capture))
        first_latent=None
        try:
            for i,query in enumerate(queries):
                for arm in range(2):
                    check();stage='QUERY_%03d_ARM_%d'%(i,arm)
                    bank=torch.cat([tokens[i,arm],teacher_tokens[i]],0).cuda()
                    seed=int(hashlib.sha256(('EEG-repeat-T-probe-1729:'+query['stimulus_id']).encode()).hexdigest()[:15],16)
                    context[0]=dict(index=i,query=query,arm=arm,bank=bank,correct=corrects[i],step=-1)
                    with torch.inference_mode():
                        z=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':bank[:1]},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
                    assert context[0]['step']==3
                    if i==0 and arm==0:
                        context[0]=None
                        with torch.inference_mode():z0=pipe(height=512,width=512,generator=torch.Generator(device='cuda').manual_seed(seed),prompt_embeds=prompts[0],pooled_prompt_embeds=prompts[1],cross_attention_kwargs={'ip_hidden_states':bank[:1]},num_inference_steps=4,guidance_scale=0.,output_type='latent').images
                        assert torch.equal(z,z0);first_latent=True
                U.atomic(a.output/('progress_%03d.json'%i),dict(queries_complete=i+1,rows=len(rows),seconds=time.monotonic()-start))
        finally:
            context[0]=None
            for h in handles:h.remove()
        assert len(rows)==80*2*4*70 and call_counts[0]==644 and len(traces)==8*2*4*4
        with (a.output/'all_response_metrics.jsonl').open('x') as stream:
            for r in rows:stream.write(json.dumps(r,allow_nan=False)+'\n')
        torch.save(traces,a.output/'fixed4layers_all8concept_first_stimulus_traces.pt')
        assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<256<<20
        for item in spec['source_pins']:U.checked(item)
        U.checked(freeze['checkpoint']);U.checked(freeze['source']);check()
        summary={}
        for arm,label in [(0,'native'),(1,'ordinary')]:
            values=[r for r in rows if r['arm']==arm]
            summary[label]={k:float(np.mean([r[k] for r in values])) for k in values[0] if k not in ['query_index','concept_id','stimulus_id','arm','step','layer']}
        done=dict(status='D26_TONLY_PHYSICAL_REPEAT_SAMEQ_RESPONSE_DIAGNOSTIC_COMPLETE',spec=U.ref(a.spec),wrapper=wrapper,loader_proof=loader_proof,host=host,summary=summary,query_count=80,concept_count=8,subjects=1,epochs=0,new_UNet_calls=call_counts[0],new_images=0,Dcheck_or_V_read=False,ordinary_changed=False,sampler_changed=False,attention_responses_changed=False,observer_latent_bitwise_parity=first_latent,all80_OOF_mean4_cache_max_relative_error=max(replay),metrics=U.ref(a.output/'all_response_metrics.jsonl'),traces=U.ref(a.output/'fixed4layers_all8concept_first_stimulus_traces.pt'),input_bank=U.ref(a.output/'input_bank.pt'),seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,novelty_established=False,image_gain_established=False,iid_repeat_noise_established=False,formal_D_resumed=False)
        U.atomic(a.output/'completion.json',done);print(json.dumps(dict(status=done['status'],summary=summary,seconds=done['seconds'])),flush=True)
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
