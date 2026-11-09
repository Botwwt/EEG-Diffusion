"""D17 all10 leave-subject-out/cross-concept raw-neural soft targets; no images/F."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_crossbrain_supervision_20261027_v1 as V
import fit_BrainHIVE_raw_neural_recoverability_20261027_v1 as OLD

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();start=time.monotonic();assert a.max_seconds==240 and not a.output.exists();a.output.mkdir(parents=True)
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    spec=U.read(a.spec)
    for item in spec['source_pins']:U.checked(item)
    inputs=U.read(U.checked(spec['raw_T_inputs']));index=Path(inputs['prediction_index']);assert U.sha(index)==inputs['prediction_index_sha256']
    items=OLD.read_index(index);assert len(items)==13230
    names=sorted(inputs['subjects']);assert names==[f'sub-{i:02d}' for i in range(1,11)]
    concepts=sorted({x['concept_id'] for x in items});assert len(concepts)==1323
    groups=[[i for i,x in enumerate(items) if x['concept_id']==concept] for concept in concepts];assert all(len(x)==10 for x in groups)
    assert [items[i]['row'] for i in range(13230)]==list(range(13230))
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch=torch;document=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=512<<20,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    def check():assert not stopped[0] and time.monotonic()-start<220,'D17 finite teacher build';budget.check()
    def log(x):
        x=x|dict(seconds=time.monotonic()-start)
        with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
    table=np.empty((10,1323,10,10),np.float32);embedding=np.empty((10,13230,2,16),np.float32);coverage=np.zeros((10,1323),np.int8)
    states={};statistics=[];memberships=[]
    for fold in (0,1):
        fit=[i for i,x in enumerate(items) if x['fold']!=fold];held=[i for i,x in enumerate(items) if x['fold']==fold]
        assert {items[i]['concept_id'] for i in fit}.isdisjoint({items[i]['concept_id'] for i in held})
        fit_ix=torch.tensor(fit,device='cuda');held_ix=torch.tensor(held,device='cuda');held_position={row:i for i,row in enumerate(held)}
        group_ids=[j for j,g in enumerate(groups) if items[g[0]]['fold']==fold]
        local=torch.tensor([[held_position[i] for i in groups[j]] for j in group_ids],device='cuda')
        scores={};pcas={}
        for si,name in enumerate(names):
            check();ref=inputs['subjects'][name];source=U.checked(dict(path=ref['path'],sha256=ref['sha256']))
            raw=np.load(source,mmap_mode='r',allow_pickle=False);assert raw.shape==(13230,17,250) and raw.dtype==np.float32
            x=torch.from_numpy(np.array(raw,copy=True).reshape(13230,4250)).cuda();assert torch.isfinite(x).all()
            model=OLD.fit_subject(x[fit_ix],64);scores[si]=OLD.subject_transform(x,model).double();pcas[si]=OLD.cpu_value(model);del x
        cache={}
        def consensus(members):
            if members not in cache:
                model=OLD.fit_mcca([scores[si][fit_ix] for si in members],rank=16)
                train,_=OLD.group_transform([scores[si][fit_ix] for si in members],model)
                hold,_=OLD.group_transform([scores[si][held_ix] for si in members],model)
                cache[members]=(train,hold,model)
            return cache[members]
        folds={}
        for si in range(10):
            check();members_a,members_b=V.teacher_groups(si);aa,ah,modela=consensus(members_a);bb,bh,modelb=consensus(members_b)
            rotation=OLD.procrustes(aa,bb);ah=ah@rotation
            score=V.centered_crossbrain_scores(ah[local],bh[local]);prob=V.probabilities(score)
            table[si,group_ids]=prob.cpu().numpy();coverage[si,group_ids]+=1
            embedding[si,held,0]=ah.float().cpu().numpy();embedding[si,held,1]=bh.float().cpu().numpy()
            expected=torch.arange(10,device='cuda').expand(len(group_ids),-1);top=(score.argmax(-1)==expected).float().mean()
            correct=score.diagonal(dim1=-2,dim2=-1);wrong=[]
            for j,gid in enumerate(group_ids):wrong.append(score[j,V.within_concept_permutation(concepts[gid]).cuda(),torch.arange(10,device='cuda')])
            gap=correct-torch.stack(wrong)
            stats=dict(fold=fold,query_subject=names[si],concepts=len(group_ids),teacher_A=[names[i] for i in members_a],teacher_B=[names[i] for i in members_b],
                current_query_subject_excluded=True,all_teacher_states_fit_only_other_concepts=True,within10_top1=float(top),correct_minus_deranged_cosine=float(gap.mean()),
                diagonal_probability=float(prob.diagonal(dim1=-2,dim2=-1).mean()),entropy=float(-(prob*prob.clamp_min(1e-20).log()).sum(-1).mean()))
            statistics.append(stats);memberships.append(stats|dict(fit_concepts=sorted({items[i]['concept_id'] for i in fit}),held_concepts=[concepts[j] for j in group_ids]));log(stats)
            folds[si]=dict(teacher_A=list(members_a),teacher_B=list(members_b),A_model=OLD.cpu_value(modela),B_model=OLD.cpu_value(modelb),rotation=rotation.cpu())
        states[fold]=dict(PCA_models=pcas,teachers=folds,fit_indices=fit,held_indices=held);del scores,pcas,cache
    assert np.all(coverage==1) and np.isfinite(table).all() and np.isfinite(embedding).all()
    supported=all(x['within10_top1']>=.15 and x['correct_minus_deranged_cosine']>=.01 for x in statistics)
    U.save_array(a.output/'OOF_T_neural_probabilities.npy',table);U.save_array(a.output/'OOF_T_crossbrain_embeddings.npy',embedding)
    model_path=a.output/'crossfit_neural_teacher_states.pt';torch.save(dict(states=states,seed=1729,source_spec=U.ref(a.spec)),model_path)
    U.atomic(a.output/'mapping.json',dict(T_array_rows=[x['source_row'] for x in items],concepts=concepts,groups=groups,subjects=names,
        T_items=[{k:x[k] for k in ('row','source_row','concept_id','stimulus_id','fold')} for x in items],teacher_memberships=memberships))
    for item in spec['source_pins']:U.checked(item)
    check();complete=dict(status='D17_ALL10_LOSO_CROSSFIT_RAW_NEURAL_SOFT_TEACHERS_COMPLETE',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
        raw_T_inputs=spec['raw_T_inputs'],probabilities=U.ref(a.output/'OOF_T_neural_probabilities.npy'),embeddings=U.ref(a.output/'OOF_T_crossbrain_embeddings.npy'),
        model=U.ref(model_path),mapping=U.ref(a.output/'mapping.json'),statistics=statistics,neural_teacher_gate_passed=supported,
        gate='every subject xconcept fold: within10Top1>=.15 and correct-minus-sameconcept-row-deranged cosine>=.01',
        source_primitive='reuse original D04 PCA64/MCCA16 and ridge .1, now as leave-subject-out training-only soft targets; no new PCA/MCCA novelty claim',
        all132300_training_queries_never_use_own_EEG_in_teacher=True,all_T_teacher_transforms_exclude_query_concept=True,full_coverage=True,
        F_or_image_targets_loaded=False,Dfit_or_Dcheck_EEG_loaded=False,EEG_or_generator_parameter_updates=0,new_images=0,UNet_forwards=0,original_D_resumed=False,
        source_preprocessing_MVNN_train_only_independently_established=False,shared_stimulus_artifacts_excluded=False,
        peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-start,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'completion.json',complete);log(dict(stage='COMPLETE',neural_teacher_gate_passed=supported));return 0

if __name__=='__main__':raise SystemExit(main())
