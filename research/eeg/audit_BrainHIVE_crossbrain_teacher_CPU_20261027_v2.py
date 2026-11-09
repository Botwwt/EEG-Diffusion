"""D17 integer-count complete soft-table replay and selected raw-neural replay."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_crossbrain_supervision_20261027_v1 as V
import fit_BrainHIVE_raw_neural_recoverability_20261027_v1 as OLD

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);assert not a.output.exists() and not torch.cuda.is_initialized();a.output.mkdir(parents=True)
    done=U.read(a.completion);assert done['status']=='D17_ALL10_LOSO_CROSSFIT_RAW_NEURAL_SOFT_TEACHERS_COMPLETE'
    mapping=U.read(U.checked(done['mapping']));inputs=U.read(U.checked(done['raw_T_inputs']));groups=np.array(mapping['groups'])
    prob=np.load(U.checked(done['probabilities']),allow_pickle=False);emb=np.load(U.checked(done['embeddings']),allow_pickle=False)
    saved=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu',mmap=True)['states']
    assert prob.shape==(10,1323,10,10) and emb.shape==(10,13230,2,16)
    max_error=0.;replays=[];stats=[]
    for si in range(10):
        scores=V.centered_crossbrain_scores(torch.from_numpy(emb[si,groups,0].copy()),torch.from_numpy(emb[si,groups,1].copy()))
        actual=V.probabilities(scores).numpy();max_error=max(max_error,float(np.abs(actual-prob[si]).max()))
        assert np.allclose(actual,prob[si],atol=2e-6,rtol=2e-5)
        for fold in (0,1):
            state=saved[fold];teacher=state['teachers'][si];ma,mb=V.teacher_groups(si)
            assert teacher['teacher_A']==list(ma) and teacher['teacher_B']==list(mb) and si not in ma+mb
            assert {mapping['T_items'][i]['concept_id'] for i in state['fit_indices']}.isdisjoint({mapping['T_items'][i]['concept_id'] for i in state['held_indices']})
            selected=[state['held_indices'][0],state['held_indices'][-1]];features={}
            for sj in ma+mb:
                name=mapping['subjects'][sj];item=inputs['subjects'][name];x=np.load(item['path'],mmap_mode='r',allow_pickle=False)
                raw=torch.from_numpy(np.array(x[selected],copy=True).reshape(2,4250));features[sj]=OLD.subject_transform(raw,state['PCA_models'][sj]).double()
            aa,_=OLD.group_transform([features[sj] for sj in ma],teacher['A_model']);bb,_=OLD.group_transform([features[sj] for sj in mb],teacher['B_model']);aa=aa@teacher['rotation']
            replay=torch.stack([aa,bb],1).float().numpy();reference=emb[si,selected]
            relative=np.linalg.norm(replay-reference,axis=-1)/np.maximum(np.linalg.norm(reference,axis=-1),1e-8)
            assert relative.max()<2e-4
            replays.append(dict(subject=mapping['subjects'][si],fold=fold,selected=selected,maximum_relative_L2=float(relative.max()),own_subject_excluded=True))
            gids=[i for i,g in enumerate(groups) if mapping['T_items'][g[0]]['fold']==fold];sub=scores[gids]
            top=float((sub.argmax(-1)==torch.arange(10).expand(len(gids),-1)).float().mean())
            wrong=torch.stack([sub[j,V.within_concept_permutation(mapping['concepts'][gid]),torch.arange(10)] for j,gid in enumerate(gids)])
            gap=float((sub.diagonal(dim1=-2,dim2=-1)-wrong).mean());stats.append(dict(subject=mapping['subjects'][si],fold=fold,top1=top,cosine_gap=gap))
            original=next(x for x in done['statistics'] if x['query_subject']==mapping['subjects'][si] and x['fold']==fold)
            assert int((sub.argmax(-1)==torch.arange(10)).sum())==round(original['within10_top1']*len(gids)*10)
            assert abs(gap-original['correct_minus_deranged_cosine'])<2e-6
    supported=all(x['top1']>=.15 and x['cosine_gap']>=.01 for x in stats);assert supported==done['neural_teacher_gate_passed']
    spec=U.read(U.checked(done['spec']))
    for item in spec['source_pins']:U.checked(item)
    assert not torch.cuda.is_initialized();result=dict(status='D17_FULL_NEURAL_TABLE_STATS_EXCLUSION_AND40_RAW_INPUT_CPU_REPLAYS_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),
        all132300_soft_target_rows_recomputed=True,maximum_soft_probability_absolute_error=max_error,all20_fold_subject_memberships_exclude_own_subject=True,
        all20_fold_subject_transforms_exclude_held_concepts=True,all20_statistic_rows_recomputed=True,selected_raw_neural_replay=replays,
        raw_input_query_replays=40,all132300_raw_teacher_forward_CPU_replayed=False,neural_teacher_gate_passed=supported,
        new_images=0,new_GPU_jobs=0,new_parameter_updates=0,no_F_or_images_loaded=True,seconds=time.monotonic()-start,
        shared_artifacts_excluded=False,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps({k:result[k] for k in ('status','maximum_soft_probability_absolute_error','raw_input_query_replays','neural_teacher_gate_passed','seconds')}))

if __name__=='__main__':main()
