"""All10 complete-T fixed kernel/random/wrong fits; no image-admission claim."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V
import brainhive_encoder_kernel_innovation_20261027_v1 as K
from brainhive_bounded_component_v1 import BoundedConditionAdapter

def cpu(v):
    if torch.is_tensor(v):return v.cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    return v

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==540 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stopped=[False];stage='INPUTS'
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    try:
        s=U.read(a.spec)
        for ref in s['source_pins']:U.checked(ref)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration)
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=540,maximum_output_bytes=1536<<20,maximum_extra_CPU_bytes=10<<30,
                 maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-start<520;budget.check()
        def log(v):
            v=v|dict(seconds=time.monotonic()-start)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
            print(json.dumps(v),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        from brainhive_component_data_v1 import load_training
        train=load_training('cpu');del train['target']
        prep=U.read(U.checked(s['D21_prepared']));oldfit=U.read(U.checked(s['D21_fit']));kernels=U.read(U.checked(s['kernel_completion']))
        assert U.read(U.checked(s['kernel_CPU_completion']))['new_GPU_jobs']==0
        oldspec=U.read(U.checked(oldfit['spec']));assert train['array_rows']==oldspec['T_rows'] and train['subjects']==oldspec['subjects']
        metadata=U.read(U.checked(prep['Dfit_metadata']))['records'];concepts=[x['concept_id'] for x in metadata]
        target=torch.from_numpy(np.load(U.checked(prep['T_projected']),allow_pickle=False)).cuda()
        heldtarget=torch.from_numpy(np.load(U.checked(prep['Dfit_projected']),allow_pickle=False)).cuda()
        noise=torch.from_numpy(np.load(U.checked(oldfit['noise']),allow_pickle=False)).cuda()
        replay=U.read(U.checked(oldfit['source_replay']));perm=torch.as_tensor(replay['training_sameconcept_permutation'],device='cuda');hp=torch.as_tensor(replay['Dfit_sameconcept_permutation'],device='cuda')
        component=BoundedConditionAdapter().cuda().eval().requires_grad_(False)
        component.load_state_dict(torch.load(U.checked(oldspec['ordinary_checkpoint']),weights_only=False,map_location='cpu')['model'],strict=True)
        scores=np.full((10,820,4,5,3),np.nan);predictions=np.full((10,820,4,5,128),np.nan);covariances=np.full((10,4,5,128,128),np.nan);models={};ordinary_checks=[]
        for j,name in enumerate(train['subjects']):
            check();stage='T30_FITS_'+name;ids=train['affine_ids'][j*13230:(j+1)*13230];c0=train['condition'][j*13230:(j+1)*13230];parts=[]
            with torch.no_grad():
                for lo in range(0,13230,256):
                    ii=ids[lo:lo+256];parts.append(component(c0[lo:lo+256].cuda(),train['weight'][ii].cuda(),train['bias'][ii].cuda())[0])
            c=torch.cat(parts);folder=Path(oldfit['scores']['path']).parent
            heldc=torch.from_numpy(np.load(folder/(name+'_Dfit_ordinary_conditions.npy'),allow_pickle=False)).cuda()
            assert U.sha(folder/(name+'_Dfit_ordinary_conditions.npy'))==s['Dfit_c_refs'][name]['sha256']
            files=kernels['subjects'][name]
            def arr(key):return torch.from_numpy(np.load(U.checked(files[key]),allow_pickle=False)).cuda()
            tk=arr('T_common_kernel.npy');tr=arr('T_random_control.npy');hk=arr('Dfit_common_kernel.npy');hr=arr('Dfit_random_control.npy')
            fitted={}
            for arm,e in [('kernel',tk),('random',tr),('wrong_kernel',tk[perm])]:
                check();fitted[arm]=V.fit_joint(target,c,e);path=a.output/(name+'_'+arm+'_working_model.pt');torch.save(cpu(fitted[arm]),path);models[name+'_'+arm]=U.ref(path)
            m=fitted['kernel'];mr=fitted['random'];mw=fitted['wrong_kernel']
            base,pair,ik=V.clean_means(m,heldc,hk);br,random,ir=V.clean_means(mr,heldc,hr);bw,wrong,iw=V.clean_means(mw,heldc,hk);_,wi,_=V.clean_means(m,heldc,hk[hp])
            assert torch.allclose(base,br,atol=1e-10,rtol=0) and torch.allclose(base,bw,atol=1e-10,rtol=0)
            old=torch.load(U.checked(oldfit['models'][name+'_paired']),weights_only=False,map_location='cuda')
            assert torch.allclose(m['prior_cov'],old['prior_cov'],atol=1e-10,rtol=0) and torch.allclose(m['mean_image_from_c'],old['mean_image_from_c'],atol=1e-10,rtol=0)
            clean=[base,pair,random,wrong,wi];covs=[m['prior_cov'],m['joint_cov'],mr['joint_cov'],mw['joint_cov'],m['joint_cov']]
            truth=(heldtarget-m['image_mean'])/m['image_std']
            for ti,t in enumerate(V.TIMES):
                check();v=prep['time_values'][str(t)];alpha=v['alpha'];sigma=v['sigma'];noisy=alpha*truth+sigma*noise[ti]/m['image_std']
                for ai,(mean,cov) in enumerate(zip(clean,covs)):
                    pred,pcov=V.noisy_update(mean,cov,noisy,alpha,sigma,m['image_std']);metric=V.score(pred,truth,pcov,m['image_std'])
                    scores[j,:,ti,ai]=torch.stack(list(metric.values()),1).cpu().numpy();predictions[j,:,ti,ai]=pred.cpu().numpy();covariances[j,ti,ai]=pcov.cpu().numpy()
            log(dict(subject=name,complete_T_fits=3,fit_rows_each=13230,discarded_EEG_dimensions=1178,condition_dimensions=1024,Dfit_rows=820,
                scores=scores[j].mean((0,1)).tolist(),same_c_only_moments_as_D21=True))
            del c,parts,heldc,tk,tr,hk,hr,fitted,m,mr,mw,clean,covs,base,pair,random,wrong,wi,br,bw,ik,ir,iw,old
            torch.cuda.empty_cache();check()
        previous=np.load(U.checked(oldfit['scores']),allow_pickle=False)
        base_error=float(np.abs(scores[:,:,:,0]-previous[:,:,:,0]).max());assert base_error<1e-10
        for name,value in [('all10_Dfit_scores.npy',scores),('all10_Dfit_predictions.npy',predictions),('all10_Dfit_posterior_covariances.npy',covariances)]:U.save_array(a.output/name,value)
        result=K.readout(scores,concepts,train['subjects']);U.atomic(a.output/'readout.json',result)
        U.atomic(a.output/'completion.json',dict(status='ALL10_T30_KERNEL_RANDOM_WRONG_FITS_AND_DFIT8200_COMPLETE',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
            models=models,scores=U.ref(a.output/'all10_Dfit_scores.npy'),predictions=U.ref(a.output/'all10_Dfit_predictions.npy'),covariances=U.ref(a.output/'all10_Dfit_posterior_covariances.npy'),readout=U.ref(a.output/'readout.json'),
            all30_models_used_complete_T=True,fit_rows_per_model=13230,Dfit_queries=8200,c_only_exact_D21_score_max_error=base_error,
            no_Dcheck_V_or_official_effect_read=True,new_UNet_calls=0,new_images=0,formal_D_resumed=False,new_image_method_admitted=False,
            seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes))
    except BaseException as exc:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
