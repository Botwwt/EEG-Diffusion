"""Actual cold-process source/state/full-input/resource review, never CUDA."""
import argparse
import json
from pathlib import Path
import sys
import time

def run(proposal,engineering_protocol,receipt):
    started=time.monotonic();s=json.loads(Path(proposal).read_bytes())
    assert s['status']=='PENDING_ACTUAL_COMPLETE_ORDINARY_LIFECYCLE_REVIEW'
    sys.path.insert(0,s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(proposal).parent))
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
    import check_HIVE_full1654_ordinary_fit_state_CPU_20261101_v1 as statechecks
    import check_HIVE_full1654_recovery_exit_CPU_20261101_v1 as exitchecks
    E=F.E;K=F.K
    for pin in s['source_pins']:K.check(pin)
    assert s['method']==E.METHOD
    resources=K.resources(s,starting=True)
    state=statechecks.run(engineering_protocol)
    wrapper=exitchecks.run(Path(s['repository_root'])/'run_gpu_job_v2.py')
    inputs,latents,binding=E.verified_CPU_inputs(s)
    assert inputs['condition'].shape==(165400,1024) and latents.shape==(16540,4,64,64)
    assert not torch.cuda.is_initialized()
    value=dict(status='PASS_ACTUAL_COMPLETE_ORDINARY_SOURCE_STATE_AND_FULL_INPUT_CPU_PREFLIGHT',
        proposal=K.ref(proposal),source_pins=s['source_pins'],state_checks=state,exit_checks=wrapper,
        full_input_binding=binding,resources_before=resources,resources_after=K.resources(s),
        R80_score_or_image_payloads_opened=False,CUDA_initialized=False,
        new_model_updates=0,new_research_images=0,seconds=time.monotonic()-started)
    K.write(receipt,value)
    print(json.dumps(dict(status=value['status'],seconds=value['seconds'],CUDA_initialized=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--proposal',required=True)
    parser.add_argument('--engineering-protocol',required=True);parser.add_argument('--receipt',required=True)
    args=parser.parse_args();run(args.proposal,args.engineering_protocol,args.receipt)
