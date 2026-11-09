"""Bind the future single component to complete existing real OOF data.

Metadata only: no duplicated embeddings, optimizer, candidate fitting, generated
images or V/test scores. All paths come from verified parent artifacts.
"""
import csv
import json
import os
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_native_models_v1 import *


def main():
    torch.set_num_threads(2);started=time.monotonic()
    out=WORK/'data/component_input_contract_v1.json';assert not out.exists()
    proof_path=WORK/'audit/all40_parent_training_CPU_verification_v1.json'
    proof=json.loads(proof_path.read_text())
    verified={(p['role'],p['subject']):p for p in proof['models']}
    protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
    image_rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
    train=protocol['image_rows']['T'];development=protocol['image_rows']['D']
    dfit_concepts=set(protocol['d_fit_concepts']);dcheck_concepts=set(protocol['d_check_concepts'])
    dfit=[i for i in development if image_rows[i]['concept_id'] in dfit_concepts]
    dcheck=[i for i in development if image_rows[i]['concept_id'] in dcheck_concepts]
    assert (len(train),len(dfit),len(dcheck))==(13230,820,830)
    assert set(train).isdisjoint(dfit+dcheck) and set(dfit).isdisjoint(dcheck)
    assert set(train+dfit+dcheck).isdisjoint(protocol['image_rows']['V'])
    target=WORK/'data/fused_train_targets.pt'
    target_contract=json.loads((WORK/'audit/fused_target_contract.json').read_text())
    assert sha(target)==target_contract['target_sha256']
    result=[]
    for subject in sorted({p['subject'] for p in proof['models']}):
        coverage={i:0 for i in train};sources=[]
        for role in ['oof0','oof1','full']:
            directory=WORK/'runs'/(role+'_source_dropout_v2');folder=directory/subject
            registration=json.loads((directory/'registration.json').read_text())
            cache=folder/'conditions.pt';checkpoint=folder/'brain_final.pt';old=verified[role,subject]
            assert sha(cache)==old['condition_cache_sha256']
            assert sha(checkpoint)==old['checkpoint_sha256']
            data=torch.load(cache,weights_only=True,map_location='cpu',mmap=True)
            assert data['model_sha256']==old['checkpoint_sha256']
            assert data['condition'].shape==(len(data['array_rows']),1024)
            lookup={int(row):position for position,row in enumerate(data['array_rows'])}
            assert len(lookup)==len(data['array_rows'])
            if role.startswith('oof'):
                selected=list(lookup)
                fit_concepts={image_rows[i]['concept_id'] for i in registration['training_array_rows']}
                held_concepts={image_rows[i]['concept_id'] for i in selected}
                assert fit_concepts.isdisjoint(held_concepts)
                assert set(selected)<=set(train)
                for row in selected:coverage[row]+=1
                groups={'component_train':selected}
            else:
                assert set(registration['training_array_rows'])==set(train)
                groups={'calibration_only_Dfit':dfit,'calibration_check_only_Dcheck':dcheck}
            permitted=[i for ids in groups.values() for i in ids]
            assert torch.isfinite(data['condition'][[lookup[i] for i in permitted]]).all()
            state=torch.load(checkpoint,weights_only=True,map_location='cpu')['model']
            norm_keys=['block.layers.0.norm.weight','block.layers.0.norm.bias']
            assert all(state[k].shape==(1024,) and torch.isfinite(state[k]).all() for k in norm_keys)
            sources.append(dict(role=role,condition_cache_path=str(cache),condition_cache_sha256=old['condition_cache_sha256'],
                checkpoint_path=str(checkpoint),checkpoint_sha256=old['checkpoint_sha256'],
                parent_registration_path=str(directory/'registration.json'),parent_registration_sha256=sha(directory/'registration.json'),
                groups={name:[dict(array_row=i,cache_row=lookup[i]) for i in ids] for name,ids in groups.items()},
                raw_condition_contract='Unmodified BrainMLP LayerNorm output; no L2 replacement',
                actual_native_output_affine_keys=norm_keys,
                known_cache_can_contain_other_rows=role=='full',other_rows_permitted_for_component=False))
            del data,state
        assert set(coverage.values())=={1}
        result.append(dict(subject=subject,complete_real_OOF_train_queries=13230,sources=sources))
    assert len(result)==10
    write(out,dict(status='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT',
        entry_sha256=sha(__file__),parent_verification_sha256=sha(proof_path),
        split_manifest_sha256=sha(ROOT/'configs/split_manifest.json'),target_cache_path=str(target),
        target_cache_sha256=target_contract['target_sha256'],target_is_training_supervision_only=True,
        training_queries=132300,unique_T_images=13230,Dfit_queries=8200,Dcheck_queries=8300,
        subjects=result,training_seed=1729,new_tensor_cache_bytes=0,
        candidate_trained=False,candidate_hyperparameters_registered=False,
        official_R80_inputs_loaded=False,V_values_used=False,normal_inference_target_access=False,
        Dfit_policy='Calibration only, no component residual optimization',
        Dcheck_policy='Predeclared validation of calibration, no iterative fitting',
        OOF_shift_limit='Two-fold encoders saw half T; deployment full-T parent differs. Check this shift on separated D.',
        seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE_COMPONENT_INPUT_CONTRACT_READY_NOT_TRAINED',
        training_queries=132300,subjects=10,sha256=sha(out),seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
