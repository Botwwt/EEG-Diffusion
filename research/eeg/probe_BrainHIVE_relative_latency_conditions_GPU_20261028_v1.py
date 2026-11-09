"""Replay original EEG encoders and frozen ordinary with changed R4 merging.

The only changed operation is the preregistered EEG temporal registration before
the original4250-to1024 projection. No weight, affine, strength or sampling change
is made; all T conditions are exported for independent CPU discrimination tests.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,gc,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U

def main():
    p=argparse.ArgumentParser()
    for name in ['spec','measurement','native-registration','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=int,required=True);a=p.parse_args()
    assert a.max_seconds==180 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stage='INPUTS';stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    try:
        spec=U.read(a.spec)
        for item in spec['source_pins']:U.checked(item)
        measure=U.read(a.measurement);assert measure['status']=='D28_COMPLETE_T_RELATIVE_LATENCY_MEASUREMENT_CPU'
        registered={r['path']:r['sha256'] for r in spec['source_pins']}
        for key in ['source','mathematics_module','input_contract','index','trials']:
            U.checked(measure[key]);assert registered.get(measure[key]['path'])==measure[key]['sha256'],key
        for key in ['trials','index','aligned_EEG','wrong_EEG','shifts','pair_lags']:U.checked(measure[key])
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_native_models_v1 import classes
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        D.torch=torch;registration=C.read(a.native_registration)
        document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=180,maximum_output_bytes=400<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=registration['resource_authorization'])
        wrapper=D.verify_wrapper(document);budget=D.Budget(document)
        def check():assert not stop[0] and time.monotonic()-start<160;budget.check()
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        freeze=U.read(U.checked(spec['ordinary_freeze']));U.checked(freeze['source'])
        payload=torch.load(U.checked(freeze['checkpoint']),weights_only=False,map_location='cpu');assert payload['arm']=='ordinary' and payload['calibration']['selected_strength']==1
        ordinary=BoundedConditionAdapter().eval().requires_grad_(False).cuda();ordinary.load_state_dict(payload['model'],strict=True);del payload
        index=[json.loads(s) for s in U.checked(measure['index']).read_text().splitlines()];assert len(index)==13230
        assert all(r['row']==i and r['partition']=='T' and r['subject_id']=='sub-01' for i,r in enumerate(index))
        raw=np.load(U.checked(measure['trials']),mmap_mode='r',allow_pickle=False);aligned=np.load(U.checked(measure['aligned_EEG']),mmap_mode='r',allow_pickle=False);wrong=np.load(U.checked(measure['wrong_EEG']),mmap_mode='r',allow_pickle=False)
        assert raw.shape==(13230,4,17,250) and aligned.shape==wrong.shape==(13230,17,250)
        contract=U.read(U.checked(spec['native_input_contract']));assert contract['subjects'][0]['subject']=='sub-01'
        sources=[s for s in contract['subjects'][0]['sources'] if s['role'] in ['oof0','oof1']];assert len(sources)==2
        byrow={r['array_row']:i for i,r in enumerate(index)};assert len(byrow)==13230
        predictions=np.lib.format.open_memmap(a.output/'predictions.npy',mode='w+',dtype=np.float32,shape=(13230,2,3,1024));seen=np.zeros(13230,dtype=np.uint8);roles=[];errors=[]
        for source in sources:
            stage=source['role'];check();model=classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False).cuda()
            cp=dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256']);model.load_state_dict(torch.load(U.checked(cp),weights_only=True,map_location='cpu')['model'],strict=True)
            cache_ref=dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256']);cache=torch.load(U.checked(cache_ref),weights_only=True,map_location='cpu',mmap=True);assert 'condition' in cache
            routes=source['groups']['component_train'];positions=np.array([byrow[r['array_row']] for r in routes],dtype=np.int64)
            assert len(positions) in [6610,6620] and np.all(seen[positions]==0)
            assert all(index[i]['fold']==int(source['role'][-1]) for i in positions)
            with torch.inference_mode():
                for lo in range(0,len(positions),256):
                    check();ids=positions[lo:lo+256];rs=routes[lo:lo+256]
                    zero=np.asarray(raw[ids]).mean(axis=1,dtype=np.float32)
                    x=np.stack([zero,np.asarray(aligned[ids]),np.asarray(wrong[ids])],axis=1).reshape(-1,4250)
                    c=model(torch.from_numpy(x.copy()).cuda()).reshape(len(ids),3,1024)
                    expected=cache['condition'][[r['cache_row'] for r in rs]]
                    error=((c[:,0].cpu()-expected).double().norm(dim=1)/expected.double().norm(dim=1)).max().item();assert error<3e-5;errors.append(error)
                    flat=c.reshape(-1,1024);w=model.block.layers[0].norm.weight.expand(len(flat),-1);b=model.block.layers[0].norm.bias.expand(len(flat),-1)
                    changed=ordinary(flat,w,b)[0].reshape(len(ids),3,1024)
                    predictions[ids,0]=c.cpu().numpy();predictions[ids,1]=changed.cpu().numpy();seen[ids]=1
            predictions.flush();roles.append(dict(role=source['role'],queries=len(positions),checkpoint=cp,condition_cache=cache_ref));del cache,model,c,changed,flat,w,b;gc.collect();torch.cuda.empty_cache()
        assert np.all(seen==1) and np.isfinite(predictions).all();predictions.flush();del predictions
        for item in spec['source_pins']:U.checked(item)
        U.checked(freeze['checkpoint']);U.checked(freeze['source']);check()
        done=dict(status='D28_ALL13230_T_NATIVE_AND_FROZEN_ORDINARY_LATENCY_CONDITIONS_EXPORTED',spec=U.ref(a.spec),CPU_measurement=U.ref(a.measurement),source=U.ref(__file__),wrapper=wrapper,
            predictions=U.ref(a.output/'predictions.npy'),index=measure['index'],targets=spec['targets'],ordinary_freeze=spec['ordinary_freeze'],native_input_contract=spec['native_input_contract'],source_models=roles,role_query_counts={r['role']:r['queries'] for r in roles},max_mean4_cache_error=max(errors),all_weights_frozen=True,axes=dict(condition_method=['native','ordinary'],EEG_frontend=['zero','aligned','wrong_assignment']),subjects=1,T_queries=13230,new_models=0,training_updates=0,new_images=0,ordinary_changed=False,all_affines_unchanged=True,selected_strength=1.,sampling_changed=False,new_UNet_calls=0,formal_D_resumed=False,seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<400<<20
        U.atomic(a.output/'completion.json',done);print(json.dumps({k:done[k] for k in ['status','seconds','max_mean4_cache_error','role_query_counts']}),flush=True)
    except BaseException as exc:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
