"""Bind actual closed source models/means/manifests; no CUDA or score values."""
import argparse
from pathlib import Path
import time
import brainhive_training_mean_R80_contract_CPU_20261101_v1 as M


def run(protocol):
    started=time.monotonic();s,parent,F=M.setup(protocol,need_bindings=False)
    import numpy as np
    import torch
    assert not torch.cuda.is_initialized()
    M.resources(s,starting=True)
    mean_protocol=M.read(M.checked(M.MEAN))
    done=M.read(M.MEAN_ROOT/'completion.json');audit=M.read(M.checked(done['CPU_audit']))
    assert done['protocol']==M.MEAN and done['new_training_updates']==done['new_GPU_jobs']==0
    assert audit['status']=='PASS_BOTH_FULL_TRAIN_RAW_MEANS_AND_CPU_ONLY_INTERFACE_AUDIT'
    means=M.read(M.MEAN_ROOT/'means_completion.json')
    assert means['protocol']==M.MEAN and set(means['roles'])==set(M.ROLES.values())
    source_components={};derived_components={};mean_completions={}
    for role,source_role in M.ROLES.items():
        pin=means['roles'][source_role];value=M.read(M.checked(pin))
        assert value['status']=='COMPLETE_ALL165400_TRAIN_ONLY_RAW_FIELD_MEAN'
        assert value['role']==source_role and value['protocol']==M.MEAN and value['queries']==165400
        assert value['no_R80_inputs_or_image_score_payloads_used'] is True
        source=M.checked(value['source_component']);derived=M.checked(value['mean_component'])
        raw=torch.load(source,weights_only=True,map_location='cpu')
        control=torch.load(derived,weights_only=True,map_location='cpu')
        assert raw['arm']==source_role and raw['updates']==31014 and raw['seed']==1729
        assert control['source_component']==value['source_component'] and control['derivation_protocol']==M.MEAN
        assert control['parent_source_training_cost_inherited'] is True
        assert control['new_training_updates']==0 and control['source_training_updates']==31014
        assert control['mean_rows']==165400 and control['no_R80_inputs_or_image_score_payloads_used'] is True
        assert control['input_bank']==parent['input_bank'] and control['asset_protocol']==parent['asset_protocol']
        assert control['arm']==source_role+'_training_mean_field'
        assert control['strength']==1. and control['radius']==.25
        assert set(raw['model'])==set(control['model'])=={'trunk.0.weight','trunk.0.bias','direction.weight','direction.bias'}
        for key in ('trunk.0.weight','trunk.0.bias'):assert torch.equal(raw['model'][key],control['model'][key])
        assert torch.count_nonzero(control['model']['direction.weight'])==0
        mean=np.load(M.checked(value['mean_FP32']),allow_pickle=False)
        assert mean.dtype==np.float32 and mean.shape==(1024,) and np.isfinite(mean).all()
        assert np.array_equal(control['model']['direction.bias'].numpy(),mean)
        source_components[source_role]=value['source_component'];derived_components[role]=value['mean_component']
        mean_completions[role]=pin
    original_root=Path(parent['output_root'])/'ordinary_full1654'
    fit=M.read(original_root/'fit_completion.json');fit_audit=M.read(original_root/'fit_CPU_audit.json')
    assert fit['component']==fit_audit['component']==source_components['ordinary_full1654']
    assert fit_audit['status']=='PASS_FULL1654_ORIGINAL_ORDINARY_FINAL_STATE_AND_ALL165400_CPU_FORWARD_AUDIT'
    assert source_components['bounded_F']==mean_protocol['bounded_F_component']
    evaluation=M.read(original_root/'evaluation/evaluation_closure.json')
    assert evaluation['status']=='COMPLETE_ACTUAL_NEW_ORDINARY_ALL16_SCORE_CALLS_AND_FILES'
    assert evaluation['queries']==2000 and evaluation['all16_metrics_included'] is True
    manifests={key:parent['reference_manifests'][key] for key in ('parent','bounded_F')}
    manifests['ordinary_full1654']=evaluation['manifest']
    tables={key:parent['reference_CSVs'][key] for key in ('parent','bounded_F')}
    tables['ordinary_full1654']=evaluation['score_CSVs']
    for pin in manifests.values():M.checked(pin)
    for role_tables in tables.values():
        assert set(role_tables)=={'common','paper'}
        for pin in role_tables.values():M.checked(pin)  # Hash bytes only; do not interpret effect columns.
    source,bank,records=M.load_evaluation_inputs(parent,torch)
    rows={key:[__import__('json').loads(x) for x in M.checked(pin).read_text().splitlines()] for key,pin in manifests.items()}
    inodes=set()
    for role,items in rows.items():
        assert len(items)==2000
        for index,row in enumerate(items):
            native=records[index]
            assert all(row[key]==native[key] for key in ('query_id','subject','stimulus_id','array_row','image_seed','repeat_protocol'))
            assert row.get('normal_inference_target_access',False) is False
            path=M.checked(dict(path=row['generated_path'],bytes=Path(row['generated_path']).stat().st_size,sha256=row['generated_sha256']))
            info=path.stat();identity=info.st_dev,info.st_ino
            assert identity not in inodes;inodes.add(identity)
            if role!='parent':
                assert row['component']==source_components[role] and row['strength']==1. and row['radius']==.25
    assert len(inodes)==6000 and not torch.cuda.is_initialized()
    result=dict(status='PASS_ACTUAL_TRAIN_MEAN_R80_SOURCE_DATA_CLOSURE',protocol=M.ref(protocol),
        parent_protocol=M.PARENT,mean_protocol=M.MEAN,mean_CPU_completion=M.ref(M.MEAN_ROOT/'completion.json'),
        mean_CPU_audit=done['CPU_audit'],mean_completions=mean_completions,
        source_components=source_components,derived_components=derived_components,
        source_manifests=manifests,source_CSVs=tables,input_completion=parent['evaluation']['input_completion'],
        source_input_bank=source['bank'],source_6000_image_inodes_and_file_hashes_verified=True,
        source_R80_raw_conditions_and10_published_affines_verified=True,
        mean_construction_used_R80=False,source_score_values_read=False,CUDA_initialized=False,
        new_training_updates=0,seconds=time.monotonic()-started)
    M.write(M.OUT/'actual_source_data_closure.json',result)
    print(__import__('json').dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
