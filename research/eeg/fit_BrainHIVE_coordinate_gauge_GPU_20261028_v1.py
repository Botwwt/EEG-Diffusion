"""D24 role-private closed-form rotations, native multiweight folding, no SGD."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_native_coordinate_gauge_20261028_v1 as V

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==420 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stopped=[False];stage='INPUTS'
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    try:
        spec=U.read(a.spec)
        for item in spec['source_pins']:U.checked(item)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_native_models_v1 import classes
        from safetensors import safe_open
        D.torch=torch;cfg=C.read(a.native_registration)
        document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=420,maximum_output_bytes=1024<<20,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(document);budget=D.Budget(document)
        def check():
            assert not stopped[0] and time.monotonic()-start<390;budget.check()
            assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<(1024<<20)-(1<<20)
        def log(value):
            value=value|dict(seconds=time.monotonic()-start)
            with (a.output/'fit_log.jsonl').open('a') as f:f.write(json.dumps(value)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(value),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        with safe_open(str(U.checked(spec['visual_prior_weights'])),framework='pt',device='cpu') as f:
            vg=f.get_tensor('vision_model.block.layers.0.norm.weight').cuda();vb=f.get_tensor('vision_model.block.layers.0.norm.bias').cuda()
        assert vg.shape==vb.shape==(1024,) and (vg.abs()>1e-8).all()
        raw=torch.from_numpy(np.array(np.load(U.checked(spec['T_mean4']),mmap_mode='r'),copy=True).reshape(13230,4250)).cuda()
        target=torch.from_numpy(np.load(U.checked(spec['targets']),allow_pickle=False)).cuda();visual=V.remove_affine(target,vg,vb)
        index=[json.loads(x) for x in U.checked(spec['index']).read_text().splitlines()];assert len(index)==13230
        prepared=U.read(U.checked(spec['prepared_EEG']));mapping=U.read(U.checked(prepared['mapping']));d09=U.read(U.checked(spec['D09_completion']));panel=U.read(U.checked(U.read(U.checked(d09['inputs']))['panel']))
        assert mapping['Dcheck_queries']==panel['Dcheck']
        dev=torch.from_numpy(np.array(np.load(U.checked(prepared['Dfit']),mmap_mode='r')[0],copy=True).reshape(820,4250)).cuda();dc=torch.from_numpy(np.array(np.load(U.checked(prepared['Dcheck']),mmap_mode='r'),copy=True).reshape(400,4250)).cuda()
        contract=U.read(U.checked(spec['native_input_contract']));sub=contract['subjects'][0];assert sub['subject']=='sub-01'
        source=next(x for x in sub['sources'] if x['role']=='full');dfrows=[x['array_row'] for x in source['groups']['calibration_only_Dfit']];assert dfrows==[q['array_row'] for q in mapping['Dfit_queries'][:820]]
        bank=torch.load(U.checked(spec['target_cache']),weights_only=True,map_location='cpu',mmap=True);dftruth=bank[dfrows].cuda();del bank
        modelclass=classes()['BrainMLP'];allstates={};rotations={};proofs=[];dfmetrics={};fitseconds=0.
        oof=np.full((13230,3,1024),np.nan,np.float32);coverage=np.zeros((13230,3),np.int8);df=np.full((820,3,1024),np.nan,np.float32);checkbank=np.full((400,3,1024),np.nan,np.float32)
        def infer(model,values):
            with torch.inference_mode():return torch.cat([model(x).cpu() for x in values.split(256)])
        for role,item in spec['roles'].items():
            check();stage='FIT_'+role;old=torch.load(U.checked(item['checkpoint']),weights_only=True,map_location='cpu')['model'];native={k:v.cuda() for k,v in old.items()}
            model=modelclass(4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);model.load_state_dict(native,strict=True)
            identity=V.fold_native(native,torch.eye(1024,device='cuda',dtype=torch.float64),native[V.PREFIX+'norm.weight'],native[V.PREFIX+'norm.bias'])
            assert all(torch.equal(identity[k],native[k]) for k in native)
            with torch.inference_mode():assert torch.equal(model(raw[:32]),V.predict_from_state(raw[:32],native))
            c=infer(model,raw).cuda();u=V.remove_affine(c,native[V.PREFIX+'norm.weight'],native[V.PREFIX+'norm.bias']);ids=item['fit_positions'];held=item['held_positions'];x=u[ids];y=visual[ids]
            perm=V.same_concept_derangement([index[i]['concept_id'] for i in ids]);assert {index[i]['concept_id'] for i in ids}.isdisjoint({index[i]['concept_id'] for i in held})
            tic=time.monotonic();paired,pinfo=V.procrustes_rotation(x,y);wrong,winfo=V.procrustes_rotation(x,y[perm]);fitseconds+=time.monotonic()-tic
            rots=[paired,wrong,torch.eye(1024,device='cuda',dtype=torch.float64)];infos=[pinfo,winfo,dict(objective_before=float((x-y).square().mean()),objective_after=float((x-y).square().mean()))]
            allstates[role]={};rotations[role]={}
            for ai,arm in enumerate(V.ARMS):
                check();r=rots[ai];fold=V.fold_native(native,r,vg,vb);model.load_state_dict(fold,strict=True)
                expected=(u[:128]@r.T)*vg.double()+vb.double();observed=model(raw[:128]);error=float(((observed.double()-expected).norm(dim=1)/expected.norm(dim=1)).max());assert error<3e-5
                sample=u[:128];before=sample@sample.T;after=(sample@r.T)@(sample@r.T).T;gramerror=float((after-before).abs().max());assert gramerror<1e-7
                allstates[role][arm]={k:v.detach().cpu() for k,v in fold.items()};rotations[role][arm]=r.cpu()
                if held:
                    value=infer(model,raw[held]);oof[held,ai]=value.numpy();coverage[held,ai]+=1
                if role=='full':
                    pred=infer(model,dev);df[:,ai]=pred.numpy();checkbank[:,ai]=infer(model,dc).numpy()
                    aa=torch.nn.functional.normalize(pred.cuda(),dim=1);bb=torch.nn.functional.normalize(dftruth,dim=1);scores=torch.bmm(aa.reshape(82,10,1024),bb.reshape(82,10,1024).transpose(1,2))
                    dfmetrics[arm]=dict(MSE=float((pred.cuda()-dftruth).double().square().mean()),cosine=float((aa*bb).sum(1).mean()),within10_top1=float((scores.argmax(2)==torch.arange(10,device='cuda')).float().mean()))
                proof=dict(role=role,arm=arm,fit_stimuli=len(ids),fit_concepts=len(ids)//10,held_stimuli=len(held),native_fold_relative_error=error,sampled_unaffined_Gram_abs_error=gramerror,orthogonal_abs_error=float((r@r.T-torch.eye(1024,device='cuda')).abs().max()),mean_preservation_error=float((r.sum(1)-1).abs().max()),fit=infos[ai],all_native_parameter_count=sum(v.numel() for v in fold.values()),no_additional_inference_parameters=True)
                proofs.append(proof);log(proof)
            del c,u,x,y,paired,wrong,rots,native,model;torch.cuda.empty_cache()
        assert np.all(coverage==1) and np.isfinite(oof).all() and np.isfinite(df).all() and np.isfinite(checkbank).all()
        torch.save(dict(models=allstates,spec=U.ref(a.spec),functionally_equivalent_to_orthogonal_output_map=True),a.output/'native_models.pt');torch.save(dict(rotations=rotations,visual_weight=vg.cpu(),visual_bias=vb.cpu()),a.output/'coordinate_rotations.pt')
        for n,v in [('OOF_T_conditions.npy',oof),('Dfit_conditions.npy',df),('Dcheck400_conditions.npy',checkbank)]:U.save_array(a.output/n,v)
        U.atomic(a.output/'fit_proofs.json',dict(proofs=proofs));U.atomic(a.output/'Dfit_final_metrics.json',dfmetrics)
        modelref=U.ref(a.output/'native_models.pt');lock=dict(status='D24_ALL400_EEG_CONDITIONS_LOCKED_BEFORE_IMAGES',query_ids=[q['query_id'] for q in panel['Dcheck']],arms=V.ARMS,conditions=U.ref(a.output/'Dcheck400_conditions.npy'),model=modelref,normal_inference_other_subject_EEG=False,Dcheck_target_F_or_images_used=False,Dcheck_scores_used_for_choices=False,ordinary_applied=False)
        U.atomic(a.output/'condition_lock.json',lock);check()
        done=dict(status='D24_ALL9_NATIVE_FOLDED_COORDINATE_MODELS_FULL_ROLE_T_COMPLETE',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,model=modelref,rotations=U.ref(a.output/'coordinate_rotations.pt'),condition_lock=U.ref(a.output/'condition_lock.json'),OOF_T_conditions=U.ref(a.output/'OOF_T_conditions.npy'),Dfit_conditions=U.ref(a.output/'Dfit_conditions.npy'),Dfit_metrics=U.ref(a.output/'Dfit_final_metrics.json'),fit_proofs=U.ref(a.output/'fit_proofs.json'),models=9,closed_form_rotations_fitted=6,identity_control_models=3,SGD_updates=0,complete_fit_stimuli_all_roles=26460,paired_rotations_target_exposures=26460,matched_wrong_rotations_target_exposures=26460,parameters_per_encoder=6452224,additional_inference_parameters=0,ordinary_applied=False,no_Dfit_model_or_epoch_or_strength_selection=True,new_UNet_calls=0,new_images=0,normal_inference_individual_trials=False,formal_D_resumed=False,fit_body_seconds=fitseconds,seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',done);print(json.dumps(done))
    except BaseException as e:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
