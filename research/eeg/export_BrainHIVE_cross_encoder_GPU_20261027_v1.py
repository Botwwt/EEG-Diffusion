"""Frozen ATMS inference only: complete all10 real OOF-T and full-D predictions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U


def main():
    p=argparse.ArgumentParser()
    for n in ('input-completion','native-registration','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();start=time.monotonic();assert a.max_seconds==180 and not a.output.exists();a.output.mkdir(parents=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));active='INPUTS';queries=0
    def log(value):
        value=value|dict(seconds=time.monotonic()-start)
        with (a.output/'export_log.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value),flush=True)
    try:
        inputs=U.read(a.input_completion);assert inputs['status']=='D15_ALL30_STRICT_MODELS_ORIGINAL_R4_INPUTS_AND_OOF_ROLES_CPU_VERIFIED'
        for item in inputs['source_pins']:U.checked(item)
        reference=torch.load(U.checked(inputs['CPU_reference']),weights_only=True,map_location='cpu')
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=1024<<20,
            maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stop[0] and time.monotonic()-start<165,'D15 finite export deadline';budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        torch.cuda.reset_peak_memory_stats();C.resources(config,training_start=True)
        sys.path[:0]=['/root/atm_official_20261016','/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages',
            '/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages']
        from models.atms import ATMS
        T=inputs['T_array_rows'];Drows=inputs['D_array_rows'];t_lookup={r:i for i,r in enumerate(T)}
        final_paths={key:a.output/(key+'.npy') for key in ('T_ATMS_OOF','D_ATMS_full')}
        temporary={key:p.with_suffix('.pending.npy') for key,p in final_paths.items()}
        for p in list(temporary.values())+list(final_paths.values()):assert not p.exists()
        output={'T_ATMS_OOF':np.lib.format.open_memmap(temporary['T_ATMS_OOF'],mode='w+',dtype=np.float32,shape=(132300,1024)),
            'D_ATMS_full':np.lib.format.open_memmap(temporary['D_ATMS_full'],mode='w+',dtype=np.float32,shape=(16500,1024))}
        coverage={key:np.zeros(len(value),np.int8) for key,value in output.items()};parities=[];affines={}
        for si,(name,subject) in enumerate(inputs['subjects'].items()):
            active='FROZEN_ATMS_EXPORT_'+name;check()
            eeg=torch.load(U.checked(subject['input']),weights_only=True,map_location='cpu',mmap=True);assert eeg.shape==(16540,63,250)
            for role,record in subject['models'].items():
                check();model=ATMS().cuda().eval().requires_grad_(False)
                state=torch.load(U.checked(record['checkpoint']),weights_only=True,map_location='cpu',mmap=True);model.load_state_dict(state,strict=True)
                norm=model.proj_eeg[2];affines[name+'/'+role]=dict(weight=norm.weight.detach().cpu(),bias=norm.bias.detach().cpu(),eps=norm.eps)
                ref=reference[name+'/'+role];parity_rows=ref['array_rows']
                with torch.inference_mode():
                    value=model(eeg[parity_rows].cuda(),torch.full((2,),si+1,dtype=torch.long,device='cuda')).float().cpu()
                relative=(value-ref['prediction']).norm(dim=-1)/ref['prediction'].norm(dim=-1)
                absolute=(value-ref['prediction']).abs().max();assert relative.max()<3e-5 and absolute<3e-4
                parities.append(dict(subject=name,role=role,array_rows=parity_rows,CPU_GPU_max_relative_L2=float(relative.max()),max_absolute=float(absolute)))
                assert torch.equal(affines[name+'/'+role]['weight'],ref['weight']) and torch.equal(affines[name+'/'+role]['bias'],ref['bias'])
                rows=record['export_array_rows'];key='D_ATMS_full' if role=='full' else 'T_ATMS_OOF'
                for lo in range(0,len(rows),128):
                    if lo%1024==0:check()
                    selected=rows[lo:lo+128];x=eeg[selected].cuda();assert torch.isfinite(x).all()
                    with torch.inference_mode():value=model(x,torch.full((len(x),),si+1,dtype=torch.long,device='cuda')).float().cpu().numpy()
                    assert value.shape==(len(selected),1024) and np.isfinite(value).all()
                    indices=si*(1650 if role=='full' else 13230)+(np.arange(lo,lo+len(selected)) if role=='full' else np.array([t_lookup[r] for r in selected]))
                    assert (coverage[key][indices]==0).all();output[key][indices]=value;coverage[key][indices]+=1;queries+=len(selected)
                log(dict(stage=active,role=role,queries=len(rows),total_predictions=queries,new_parameter_updates=0))
                del model,state
            del eeg
        assert queries==148800
        for key,value in output.items():
            assert (coverage[key]==1).all() and np.isfinite(value).all();value.flush();os.rename(temporary[key],final_paths[key])
        torch.save(affines,a.output/'original_ATMS_output_affines.pt')
        U.atomic(a.output/'CPU_GPU_parities.json',dict(status='ALL30_FROZEN_MODELS_60_INPUTS_CPU_GPU_PARITY_PASSED',rows=parities))
        for item in inputs['source_pins']:U.checked(item)
        check();result=dict(status='D15_ALL10_COMPLETE_ATMS_OOF_T_AND_FULL_D_REEXPORTED',inputs=U.ref(a.input_completion),source=U.ref(__file__),wrapper=wrapper,
            T_ATMS_OOF=U.ref(final_paths['T_ATMS_OOF']),D_ATMS_full=U.ref(final_paths['D_ATMS_full']),affines=U.ref(a.output/'original_ATMS_output_affines.pt'),
            CPU_GPU_parity=U.ref(a.output/'CPU_GPU_parities.json'),all_predictions=queries,T_queries=132300,D_queries=16500,all30_models_strict_loaded=True,
            T_each_query_from_model_excluding_its_concept=True,D_models_fit_only_T=True,original_V_checkpoint_selection_preserved=True,
            input_shape=[63,250],output_coordinates='native pooled CLIP_H_1024',new_parameter_updates=0,new_images=0,UNet_forwards=0,
            no_original_cache_bytes_equality_claim=True,original_D_resumed=False,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            seconds=time.monotonic()-start,scientific_increment_established=False)
        U.atomic(a.output/'completion.json',result);log(dict(stage='COMPLETE',total_predictions=queries));return 0
    except BaseException as error:
        U.atomic(a.output/'failure.json',dict(status='D15_EXPORT_INCOMPLETE_NO_AUTOMATIC_RETRY',active=active,queries=queries,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise


if __name__=='__main__':raise SystemExit(main())
