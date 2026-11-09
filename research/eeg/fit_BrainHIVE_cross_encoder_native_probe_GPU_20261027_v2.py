"""D15 native-CLIP coordinate recheck: preserve ATMS shared output affine; all else fixed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import csv
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_cross_encoder_20261027_v1 as V

R=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    for n in ('spec','native-registration','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();start=time.monotonic();assert a.max_seconds==180 and not a.output.exists();a.output.mkdir(parents=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));active='INPUTS'
    def log(value):
        value=value|dict(seconds=time.monotonic()-start)
        with (a.output/'fit_log.jsonl').open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(value,allow_nan=False),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==V.SCHEMA and spec['ridge_fraction']==.1 and spec['ATMS_coordinate']=='original_raw_pooled_CLIP_H_output'
        for item in spec['source_pins']:U.checked(item)
        assert U.read(U.checked(spec['CPU_checks']))['status']=='D15_LINEAR_SOLVE_INCREMENT_AND_NULL_POSITIVE_GATE_TESTS_PASSED'
        exported=U.read(U.checked(spec['ATMS_export']));inputs=U.read(U.checked(exported['inputs']))
        assert exported['status']=='D15_ALL10_COMPLETE_ATMS_OOF_T_AND_FULL_D_REEXPORTED'
        import brainhive_paired_study_contract_20261025_v1 as C
        config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=256<<20,
            maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stop[0] and time.monotonic()-start<165,'D15 fixed probe deadline';budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
        from brainhive_component_data_v1 import load_training
        active='ALL10_REAL_OOF_INPUTS_AND_T_ONLY_NORMALIZERS';training=load_training('cuda');T=training['array_rows']
        assert training['subjects']==list(inputs['subjects']) and T==inputs['T_array_rows']
        c=(training['condition']-training['bias'][training['affine_ids']])/training['weight'][training['affine_ids']]
        atms=torch.from_numpy(np.array(np.load(U.checked(exported['T_ATMS_OOF']),allow_pickle=False),copy=True)).cuda()
        affines=torch.load(U.checked(exported['affines']),weights_only=True,map_location='cpu');folds=np.asarray(inputs['T_heldout_folds'])
        # Original ATMS output is already supervised in common CLIP_H coordinates.
        # Its per-model learned affine is retained. No Dfit distribution is fitted.
        # Only T learns means/RMS, separately per known subject and encoder view.
        def normalize(value):
            x=value.reshape(10,13230,1024);mean=x.mean(1,keepdim=True);scale=(x-mean).square().mean((1,2),keepdim=True).sqrt()
            assert scale.min()>.01 and torch.isfinite(scale).all()
            return ((x-mean)/scale).reshape(-1,1024),mean.squeeze(1),scale.reshape(10,1)
        c,cm,cs=normalize(c);atms,am,ass=normalize(atms)
        d12=U.read(U.checked(spec['D12_fit']));normalizer=torch.load(U.checked(d12['model']),weights_only=True,map_location='cpu',mmap=True)
        mean=normalizer['mean'].cuda();scale=normalizer['scale'].cuda();fw=normalizer['F_weight'].cuda();fb=normalizer['F_bias'].cuda()
        target=((training['target']-fb)/fw-mean)/scale;y=target.repeat(10,1)
        permutation=np.concatenate([si*13230+(U.derangement(1323)[:,None]*10+np.arange(10)[None]).reshape(-1) for si in range(10)])
        assert len(permutation)==132300;check();active='FIXED_FIVE_LINEAR_SYSTEMS_T_ONLY';fit_start=time.monotonic()
        with torch.no_grad():models=V.fit(c,atms,y,torch.as_tensor(permutation,device='cuda'))
        torch.cuda.synchronize();fit_seconds=time.monotonic()-fit_start;check()
        checkpoint=a.output/'fixed_T_linear_models.pt'
        torch.save(dict(models={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in models.items()},
            c_mean=cm.cpu(),c_scale=cs.cpu(),ATMS_mean=am.cpu(),ATMS_scale=ass.cpu(),F_mean=mean.cpu(),F_scale=scale.cpu(),F_weight=fw.cpu(),F_bias=fb.cpu(),
            T_queries=132300,unique_T_images=13230,spec_sha256=U.sha(a.spec),Dfit_seen_before_coefficients_fixed=False,ATMS_coordinate='original_raw_pooled_CLIP_H_output'),checkpoint)
        checkpoint_sha=U.sha(checkpoint);log(dict(stage='FIVE_T_LINEAR_SYSTEMS_FIXED_BEFORE_DFIT_TARGETS',fit_seconds=fit_seconds,
            training_queries=132300,models=U.ref(checkpoint),ridge_lambdas=models['regularization'],Dcheck_read=False));del training,y,target,c,atms
        active='ALL10_COMPLETE_DFIT_NO_DCHECK';contract_path=Path('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json');contract=U.read(contract_path)
        metadata={int(x['array_row']):x for x in csv.DictReader((R/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
        rawF=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        ad=np.load(U.checked(exported['D_ATMS_full']),allow_pickle=False,mmap_mode='r');dl={row:i for i,row in enumerate(inputs['D_array_rows'])}
        cx=[];ax=[];ys=[];queries=[];all_dfit_rows=[]
        for si,subject in enumerate(contract['subjects']):
            name=subject['subject'];assert name==training_subject_name(si)
            full=next(x for x in subject['sources'] if x['role']=='full');cache=torch.load(U.checked(dict(path=full['condition_cache_path'],sha256=full['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
            state=torch.load(U.checked(dict(path=full['checkpoint_path'],sha256=full['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model']
            rows=full['groups']['calibration_only_Dfit'];ids=[q['array_row'] for q in rows];ix=[q['cache_row'] for q in rows]
            assert len(rows)==820 and cache['array_rows'][ix].tolist()==ids
            cc=(cache['condition'][ix].cuda()-state['block.layers.0.norm.bias'].cuda())/state['block.layers.0.norm.weight'].cuda()
            affine=affines[name+'/full'];aa=torch.from_numpy(np.array(ad[[si*1650+dl[row] for row in ids]],copy=True)).cuda()
            cx.append((cc-cm[si])/cs[si]);ax.append((aa-am[si])/ass[si]);ys.append(((rawF[ids].cuda()-fb)/fw-mean)/scale)
            queries.extend(dict(subject_id=name,array_row=row,concept_id=metadata[row]['concept_id'],stimulus_id=metadata[row]['stimulus_id']) for row in ids)
            all_dfit_rows.append(ids)
        assert all(rows==all_dfit_rows[0] for rows in all_dfit_rows) and len(queries)==8200
        c=torch.cat(cx);atms=torch.cat(ax);y=torch.cat(ys);groups=[];perm=np.empty(8200,np.int64)
        for si in range(10):
            concept_names=sorted({q['concept_id'] for q in queries[si*820:(si+1)*820]});assert len(concept_names)==82
            local=[[si*820+j for j,q in enumerate(queries[si*820:(si+1)*820]) if q['concept_id']==concept] for concept in concept_names]
            assert all(len(g)==10 for g in local);groups.extend(local);group_permutation=U.derangement(82)
            for j,g in enumerate(local):perm[g]=local[group_permutation[j]]
        with torch.no_grad():predictions=V.predict(models,c,atms,torch.as_tensor(perm,device='cuda'))
        values={arm:V.metrics(predictions[arm],y,groups) for arm in V.ARMS};check()
        # CPU descriptive subject/concept resampling, all10x82 cells and all10 stimuli.
        result=V.readout(values,[q['subject_id'] for q in queries],[q['concept_id'] for q in queries])
        U.atomic(a.output/'readout.json',result)
        U.save_array(a.output/'Dfit_standardized_F_predictions.npy',np.stack([predictions[arm].cpu().numpy() for arm in V.ARMS],1))
        U.save_array(a.output/'Dfit_standardized_F_targets.npy',y.cpu().numpy())
        U.atomic(a.output/'Dfit_queries_and_controls.json',dict(queries=queries,shuffled_innovation_source_indices=perm.tolist(),groups=groups))
        fields=['subject_id','array_row','concept_id','stimulus_id','arm']+list(values[V.ARMS[0]])
        with (a.output/'per_query.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,q in enumerate(queries):
                for arm in V.ARMS:writer.writerow(q|dict(arm=arm)|{k:float(v[i]) for k,v in values[arm].items()})
        assert U.sha(checkpoint)==checkpoint_sha
        for item in spec['source_pins']:U.checked(item)
        check();complete=dict(status='D15_ALL10_COMPLETE_T_FIT_AND_ALL_DFIT_LINEAR_PROBE_COMPLETE',schema=V.SCHEMA,spec=U.ref(a.spec),source=U.ref(__file__),
            ATMS_export=spec['ATMS_export'],model=U.ref(checkpoint),readout=U.ref(a.output/'readout.json'),
            Dfit_predictions=U.ref(a.output/'Dfit_standardized_F_predictions.npy'),Dfit_targets=U.ref(a.output/'Dfit_standardized_F_targets.npy'),
            queries_and_controls=U.ref(a.output/'Dfit_queries_and_controls.json'),per_query=U.ref(a.output/'per_query.csv'),
            training_queries=132300,unique_T_targets=13230,Dfit_queries=8200,Dfit_concepts=82,subjects=10,linear_systems=5,
            fitted_coefficients=5*1024*1024,fit_seconds=fit_seconds,conditional_increment_supported=result['conditional_increment_supported'],
            Dcheck_read=False,new_images=0,UNet_forwards=0,old_encoder_weights_updated=False,original_D_resumed=False,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-start,
            no_hyperparameter_search_or_automatic_image_job=True,scientific_image_gain_established=False)
        U.atomic(a.output/'completion.json',complete);log(dict(stage='COMPLETE',decision=result['decision']));return 0
    except BaseException as error:
        U.atomic(a.output/'failure.json',dict(status='D15_LINEAR_PROBE_INCOMPLETE_NO_AUTOMATIC_RETRY',active=active,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise


def training_subject_name(index):return f'sub-{index+1:02d}'


if __name__=='__main__':raise SystemExit(main())
