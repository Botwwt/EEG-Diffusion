"""All10 complete T Gaussian fits, matched wrong-EEG fits, full Dfit only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V
from brainhive_bounded_component_v1 import BoundedConditionAdapter

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    return value

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','prepared','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==540 and not a.output.exists();a.output.mkdir(parents=True);started=time.monotonic();stopped=[False];stage='INPUTS'
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    try:
        spec=U.read(a.spec);assert spec['schema']==V.SCHEMA and spec['shrink']==.1 and spec['rank']==128
        for x in spec['source_pins']:U.checked(x)
        prepared=U.read(a.prepared);assert prepared['spec']==U.ref(a.spec) and prepared['no_Dcheck_V_or_official_image_pixels_read']
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration)
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=540,maximum_output_bytes=1536<<20,maximum_extra_CPU_bytes=10<<30,
                 maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-started<520;budget.check()
        def log(x):
            x=x|dict(seconds=time.monotonic()-started)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
            print(json.dumps(x),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        from brainhive_component_data_v1 import load_training
        stage='SOURCE_COMPLETE_ALL10_OOF_C';train=load_training('cpu');del train['target']
        assert train['subjects']==spec['subjects'] and train['array_rows']==spec['T_rows'] and len(train['condition'])==132300
        contract=U.read(U.checked(spec['component_input_contract']));bindings=U.read(U.checked(spec['raw_bindings']));raw=U.read(U.checked(spec['raw_T_inputs']))
        heldmeta=U.read(U.checked(prepared['Dfit_metadata']));records=heldmeta['records'];heldrows=[q['array_row'] for q in records]
        concepts=[q['concept_id'] for q in records];assert len(heldrows)==820 and len(set(concepts))==82
        target=torch.from_numpy(np.load(U.checked(prepared['T_projected']),allow_pickle=False)).cuda()
        heldtarget=torch.from_numpy(np.load(U.checked(prepared['Dfit_projected']),allow_pickle=False)).cuda()
        assert target.shape==(13230,128) and heldtarget.shape==(820,128)
        source_metadata={int(q['array_row']):q for q in csv.DictReader(U.checked(spec['image_manifest']).open()) if q['dataset_split']=='train'}
        groups={}
        for i,row in enumerate(train['array_rows']):groups.setdefault(source_metadata[row]['concept_id'],[]).append(i)
        permutation=np.arange(13230)
        for group,ix in groups.items():assert len(ix)==10;permutation[ix]=np.roll(ix,-1)
        assert np.all(permutation!=np.arange(13230)) and sorted(permutation.tolist())==list(range(13230))
        perm=torch.from_numpy(permutation).cuda();heldperm=torch.as_tensor(np.arange(820)//10*10+(np.arange(820)+1)%10,device='cuda')
        saved=torch.load(U.checked(spec['ordinary_checkpoint']),weights_only=False,map_location='cpu')
        assert saved['arm']=='ordinary' and saved['calibration']['selected_strength']==1.
        component=BoundedConditionAdapter().cuda().eval().requires_grad_(False);component.load_state_dict(saved['model'],strict=True)
        def correct(c,w,b):
            parts=[]
            with torch.no_grad():
                for lo in range(0,len(c),256):parts.append(component(c[lo:lo+256].cuda(),w[lo:lo+256].cuda(),b[lo:lo+256].cuda())[0])
            return torch.cat(parts)
        # Fixed native-PC noise, all subjects and all arms share the same draws.
        rng=np.random.default_rng(1729);noise=torch.from_numpy(rng.normal(size=(4,820,128))).cuda()
        U.save_array(a.output/'all4time_shared_PC_noise.npy',noise.cpu().numpy())
        scores=np.full((10,820,4,4,3),np.nan,np.float64);predictions=np.full((10,820,4,4,128),np.nan,np.float64)
        variance=np.full((10,4,4,128,128),np.nan,np.float64);models={};fit_seconds=0.;ordinary_checks=[]
        audit_rows=[];train_query_ids=[]
        for j,name in enumerate(spec['subjects']):
            check();stage='COMPLETE_T_FITS_'+name;t0=time.monotonic()
            c0=train['condition'][j*13230:(j+1)*13230];aids=train['affine_ids'][j*13230:(j+1)*13230]
            c=correct(c0,train['weight'][aids],train['bias'][aids]);item=raw['subjects'][name]
            eeg=torch.from_numpy(np.array(np.load(U.checked(dict(path=item['path'],sha256=item['sha256'])),mmap_mode='r'),copy=True).reshape(13230,4250)).cuda()
            assert c.shape==(13230,1024) and eeg.shape==(13230,4250)
            assert raw['source_rows']==train['array_rows']
            subject=next(x for x in contract['subjects'] if x['subject']==name);full=next(x for x in subject['sources'] if x['role']=='full')
            cache=torch.load(U.checked(dict(path=full['condition_cache_path'],sha256=full['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
            lookup={int(row):i for i,row in enumerate(cache['array_rows'])};positions=[lookup[row] for row in heldrows]
            assert cache['array_rows'][positions].tolist()==heldrows
            ck=torch.load(U.checked(dict(path=full['checkpoint_path'],sha256=full['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model']
            weight=ck['block.layers.0.norm.weight'];bias=ck['block.layers.0.norm.bias']
            heldc=correct(cache['condition'][positions].clone(),weight.expand(820,-1),bias.expand(820,-1));del cache,ck
            bind=bindings['subjects'][name];released=torch.load(U.checked(dict(path=bind['train_path'],sha256=bind['train_sha256'])),weights_only=True,map_location='cpu',mmap=True)
            helde=released[heldrows][:,bindings['selected_indices'],:].reshape(820,4250).clone().cuda()
            assert np.array_equal(released[train['array_rows'][:2]][:,bindings['selected_indices'],:].reshape(2,4250).numpy(),eeg[:2].cpu().numpy())
            del released
            if j==0:
                # CPU relative replay uses actual OOF affine groups and the
                # original component, not an invented common LayerNorm.
                first=[]
                for i in (0,13229):
                    ii=aids[i];mm=BoundedConditionAdapter().eval().requires_grad_(False);mm.load_state_dict(saved['model'],strict=True)
                    with torch.no_grad():expected=mm(c0[i:i+1],train['weight'][ii:ii+1],train['bias'][ii:ii+1])[0]
                    rel=float((expected-c[i:i+1].cpu()).norm()/expected.norm());assert rel<1e-6;first.append(rel)
                ordinary_checks=first
            fitted={}
            for arm,values in [('paired',eeg),('wrong',eeg[perm])]:
                check();fitted[arm]=V.fit_joint(target,c,values)
                path=a.output/(name+'_'+arm+'_working_model.pt');torch.save(cpu(fitted[arm]),path);models[name+'_'+arm]=U.ref(path)
            fit_seconds+=time.monotonic()-t0
            pair=fitted['paired'];wrong=fitted['wrong'];assert torch.allclose(pair['prior_cov'],wrong['prior_cov'],atol=1e-10,rtol=0)
            base,paired,innovation=V.clean_means(pair,heldc,helde)
            base2,wrongpred,_=V.clean_means(wrong,heldc,helde);assert torch.allclose(base,base2,atol=1e-10,rtol=0)
            _,wrong_infer,_=V.clean_means(pair,heldc,helde[heldperm])
            clean=[base,paired,wrongpred,wrong_infer];covs=[pair['prior_cov'],pair['joint_cov'],wrong['joint_cov'],pair['joint_cov']]
            expected_target=(heldtarget-pair['image_mean'])/pair['image_std']
            U.save_array(a.output/(name+'_Dfit_ordinary_conditions.npy'),heldc.cpu().numpy())
            for ti,t in enumerate(V.TIMES):
                check();vals=prepared['time_values'][str(t)];alpha=vals['alpha'];sigma=vals['sigma']
                noisy=alpha*expected_target+sigma*noise[ti]/pair['image_std']
                for ai,(m,cov) in enumerate(zip(clean,covs)):
                    pred,post=V.noisy_update(m,cov,noisy,alpha,sigma,pair['image_std'])
                    metric=V.score(pred,expected_target,post,pair['image_std'])
                    scores[j,:,ti,ai]=torch.stack(list(metric.values()),1).cpu().numpy()
                    predictions[j,:,ti,ai]=pred.cpu().numpy();variance[j,ti,ai]=post.cpu().numpy()
            U.save_array(a.output/(name+'_Dfit_raw_innovation.npy'),innovation.cpu().numpy())
            # First/last query per subject retained for independent source replay.
            audit_rows.extend(dict(subject=name,T_position=i,array_row=train['array_rows'][i],ordinary=c[i].cpu().tolist()) for i in (0,13229))
            log(dict(stage='ALL13230_T_PAIRED_AND_WRONG_FITS_AND820_DFIT_READOUT',subject=name,subject_index=j+1,
                     fit_rows_per_model=13230,raw_dimensions=4250,condition_dimensions=1024,latent_dimensions=128,
                     fixed_last_fit_no_checkpoint_selection=True,paired_solve_residual=pair['solve_relative_residual'],wrong_solve_residual=wrong['solve_relative_residual'],
                     Dfit_scores=scores[j].mean((0,1)).tolist(),Dcheck_read=False))
            del eeg,c,heldc,helde,fitted,pair,wrong,clean,covs,base,paired,wrongpred,wrong_infer,base2,innovation
            torch.cuda.empty_cache();check()
        assert np.isfinite(scores).all() and np.isfinite(predictions).all() and np.isfinite(variance).all()
        U.save_array(a.output/'all10_Dfit_scores.npy',scores);U.save_array(a.output/'all10_Dfit_predictions.npy',predictions);U.save_array(a.output/'all10_Dfit_posterior_covariances.npy',variance)
        U.atomic(a.output/'source_replay_rows.json',dict(T_first_last_ordinary=audit_rows,training_sameconcept_permutation=permutation.tolist(),Dfit_sameconcept_permutation=heldperm.cpu().tolist()))
        result=V.readout(scores,concepts,spec['subjects']);U.atomic(a.output/'readout.json',result);check()
        done=dict(status='ALL10_COMPLETE_T_GAUSSIAN_FITS_AND8200_DFIT_STATE_INNOVATION_DIAGNOSTICS_COMPLETE',schema=V.SCHEMA,
                  spec=U.ref(a.spec),prepared=U.ref(a.prepared),source=U.ref(__file__),wrapper=wrapper,models=models,
                  scores=U.ref(a.output/'all10_Dfit_scores.npy'),predictions=U.ref(a.output/'all10_Dfit_predictions.npy'),posterior_covariances=U.ref(a.output/'all10_Dfit_posterior_covariances.npy'),
                  readout=U.ref(a.output/'readout.json'),source_replay=U.ref(a.output/'source_replay_rows.json'),noise=U.ref(a.output/'all4time_shared_PC_noise.npy'),
                  fitted_subjects=10,T_queries_per_model=13230,fit_models=20,EEG_dimensions=4250,latent_dimensions=128,ordinary_CPU_replay_relative_errors=ordinary_checks,
                  all_T_queries_used=True,old_full_vs_OOF_shift_retained=True,all_Dfit_queries=8200,no_Dcheck_V_or_official_pixels_or_effect_scores_read=True,
                  new_UNet_calls=0,new_research_images=0,formal_D_resumed=False,fit_and_subject_export_seconds=fit_seconds,
                  seconds=time.monotonic()-started,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',done);print(json.dumps(done))
    except BaseException as exc:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started,no_automatic_retry=True));raise

if __name__=='__main__':main()
