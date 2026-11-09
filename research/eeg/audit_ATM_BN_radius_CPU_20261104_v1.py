"""Independent full BN recurrence, unchanged weights, all400 native CPU outputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import atm_BN_radius_contract_20261104_v1 as B
c=B.setup(sys.argv[1]);K=c['K'];q=c['q'];out=c['out'];up=c['up'];started=time.monotonic();rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert K.ref(__file__)==q['sources']['CPU'];before=B.resources(c,starting=True)
try:
    done=K.read(out/'GPU_completion.json');assert done['status']=='COMPLETE_T_ONLY_BN_AND_RADIUS_NATIVE_PROJECTOR_DIAGNOSTIC' and done['policy']==K.ref(sys.argv[1])
    batchids=K.read(out/'BN_batch_identities.json');events=K.read(out/'epochs.json');data=np.load(out/'BN_moments_and_buffers.npz',allow_pickle=False);assert len(batchids)==830 and len(events)==10
    for epoch in range(10):
        order=np.random.default_rng(1729+epoch).permutation(1323);seen=[]
        for bi,lo in enumerate(range(0,1323,16)):
            ids=c['T'][(order[lo:lo+16,None]*10+np.arange(10)).reshape(-1)];row=batchids[epoch*83+bi];assert row==dict(epoch=epoch+1,batch=bi,queries=len(ids),array_rows_sha256=hashlib.sha256(ids.tobytes()).hexdigest());seen.extend(ids.tolist())
        assert sorted(seen)==sorted(c['T'].tolist())
        e=events[epoch];assert e['statistic_updates']==(epoch+1)*83 and e['exposures']==(epoch+1)*13230 and e['optimizer_updates']==0 and e['concept_order_sha256']==hashlib.sha256(order.tobytes()).hexdigest() and e['coverage_min']==e['coverage_max']==1
    buffer_errors={}
    for i,name in enumerate(B.BN_NAMES):
        means=data['mean_'+str(i)];vars=data['variance_'+str(i)];assert means.shape==vars.shape==(830,40) and (vars>=0).all()
        mm=data['initial_'+str(i)+'_running_mean'].astype(np.float64);vv=data['initial_'+str(i)+'_running_var'].astype(np.float64)
        for m,v in zip(means,vars):mm=(.9*mm+.1*m.astype(float)).astype(np.float32).astype(float);vv=(.9*vv+.1*v.astype(float)).astype(np.float32).astype(float)
        errors=dict(mean=float(np.max(abs(mm-data['final_'+str(i)+'_running_mean'])/(1+abs(mm)))),variance=float(np.max(abs(vv-data['final_'+str(i)+'_running_var'])/(1+abs(vv)))))
        assert max(errors.values())<5e-5 and int(data['final_'+str(i)+'_num_batches_tracked'])-int(data['initial_'+str(i)+'_num_batches_tracked'])==830;buffer_errors[name]=errors
    import torch
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);sys.path.insert(0,'/root/atm_official_20261016');from models.atms import ATMS
    original=torch.load(K.checked(up['q']['encoder']),map_location='cpu',weights_only=True,mmap=True);value=torch.load(K.checked(done['checkpoint']),map_location='cpu',weights_only=True,mmap=True);state=value['model']
    bnkeys={n+'.'+k for n in B.BN_NAMES for k in ('running_mean','running_var','num_batches_tracked')}
    assert set(state)==set(original) and all(torch.equal(v,original[k]) for k,v in state.items() if k not in bnkeys)
    model=ATMS().eval().requires_grad_(False);model.load_state_dict(state,strict=True);assert sum(p.numel() for p in model.parameters())==3200413
    raw=torch.load(K.checked(up['q']['EEG']),map_location='cpu',weights_only=True,mmap=True);conditions=np.load(K.checked(done['conditions']),allow_pickle=False);assert conditions.shape==(400,6,1024)
    errors=[]
    with torch.inference_mode():
        for lo in range(0,400,16):
            ids=[r['array_row'] for r in c['queries'][lo:lo+16]];pred=model(raw[ids],torch.ones(len(ids),dtype=torch.long)).numpy();errors.extend(np.max(abs(pred-conditions[lo:lo+len(ids),3]),axis=1).tolist())
    assert len(errors)==400 and max(errors)<5e-5
    main=K.read(up['out']/'prepared.json');old=np.load(K.checked(main['conditions']),allow_pickle=False);assert np.array_equal(conditions[:,:3],old[:,[0,2,3]])
    a=conditions[:,0].astype(float);b=conditions[:,1].astype(float);ra=np.linalg.norm(a,axis=1);rb=np.linalg.norm(b,axis=1)
    dr=b*ra[:,None]/rb[:,None];rr=a*rb[:,None]/ra[:,None]
    formula_error=max(float(abs(dr-conditions[:,4]).max()),float(abs(rr-conditions[:,5]).max()));assert formula_error<5e-6
    norm_error=max(float(abs(np.linalg.norm(conditions[:,4].astype(float),axis=1)-ra).max()),float(abs(np.linalg.norm(conditions[:,5].astype(float),axis=1)-rb).max()));assert norm_error<5e-6
    direction_error=max(float(abs(conditions[:,4].astype(float)/np.linalg.norm(conditions[:,4].astype(float),axis=1)[:,None]-b/rb[:,None]).max()),float(abs(conditions[:,5].astype(float)/np.linalg.norm(conditions[:,5].astype(float),axis=1)[:,None]-a/ra[:,None]).max()));assert direction_error<1e-7
    # Fixed8 FP32 tokens are separately recomputed with float64 affine and LN.
    weights=np.load(out/'projector_weights_FP16.npz',allow_pickle=False);w=weights['image_embeds.weight'].astype(float);bias=weights['image_embeds.bias'].astype(float);gamma=weights['norm.weight'].astype(float);beta=weights['norm.bias'].astype(float)
    selected=np.arange(0,80,10);v=conditions[selected].astype(float).reshape(-1,1024);linear=(v@w.T+bias).reshape(-1,4,2048);mean=linear.mean(-1,keepdims=True);var=((linear-mean)**2).mean(-1,keepdims=True);expected=((linear-mean)/np.sqrt(var+1e-5))*gamma+beta
    saved=np.load(out/'fixed8_tokens_FP32.npy',allow_pickle=False).reshape(-1,4,2048);token_error=float(np.max(abs(expected-saved)/(1+abs(expected))));assert token_error<5e-5
    tokens=np.load(K.checked(done['tokens']),allow_pickle=False);assert tokens.shape==(400,6,4,2048) and tokens.dtype==np.float16
    probes={}
    for j,arm in enumerate(B.ARMS):
        if not j:continue
        x=tokens[:,j].astype(float).reshape(400,-1);y=tokens[:,0].astype(float).reshape(400,-1);probes[arm]=dict(bit_equal_queries=int(np.all(x==y,1).sum()),mean_token_cosine=float(np.mean(np.sum(x*y,1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)))),mean_token_squared_error=float(np.mean((x-y)**2)))
        for k,v in probes[arm].items():assert abs(v-done['probes'][arm][k])<1e-10
    assert time.monotonic()-started<850 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0<c['parent']['maximum_extra_CPU_bytes'];K.checked(up['q']['encoder'])
    K.write(out/'CPU_audit.json',dict(status='PASS_FULL_T_BN_STATE_AND_ALL400_NATIVE_CPU_CONDITIONS_RADIUS_PROJECTOR_AUDIT',all830_batch_identifiers_and10_epoch_coverage=True,BN_running_buffer_recurrence_scaled_errors=buffer_errors,all_weights_nonBN_buffers_bit_exact=True,all400_native_CPU_condition_errors=errors,
        radius_formula_max_absolute_error=formula_error,radius_norm_max_error=norm_error,normalized_direction_max_error=direction_error,fixed48_FP64_projector_scaled_error=token_error,all2400_FP16_token_scalar_probes_recomputed=True,
        no_images_or_generator_calls=True,not_BN_only_effect_or_image_gain_evidence=True,seconds=time.monotonic()-started,extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0,resources_after=B.resources(c)))
except BaseException as exc:K.write(out/'CPU_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_evidence_retained=True));raise
