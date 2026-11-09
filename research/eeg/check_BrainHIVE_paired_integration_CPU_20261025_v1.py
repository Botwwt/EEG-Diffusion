"""Actual full identity plan plus native-processor/checkpoint gradient tests."""
import argparse
import copy
import csv
import json
from pathlib import Path
import torch
from brainhive_paired_effect_integration_20261025_v1 import build_pair_plan,response_objectives,PairedEffectCapture

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contract',type=Path,
        default=ROOT/'audit/decisive_T_analysis_20261025_v1/component_input_contract_snapshot.json')
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(1729)
    contract=json.loads(args.contract.read_bytes());protocol=json.loads((ROOT/'configs/split_manifest.json').read_bytes())
    with (ROOT/'audit/image_manifest.csv').open(encoding='utf-8',newline='') as stream:
        metadata={int(r['array_row']):r for r in csv.DictReader(stream) if r['dataset_split']=='train'}
    plan=build_pair_plan(contract,protocol,metadata)
    assert plan['queries']==132300 and len(plan['groups'])==20
    assert plan==build_pair_plan(contract,protocol,metadata)
    bad=copy.deepcopy(contract);bad['subjects'][0]['sources'][1]['role']='full'
    try:build_pair_plan(bad,protocol,metadata)
    except AssertionError:pass
    else:raise AssertionError('OOF role corruption accepted')
    si=torch.randn(3,2,4,8,requires_grad=True);sj=torch.randn_like(si,requires_grad=True)
    teachers=[torch.randn_like(si,requires_grad=True) for _ in range(4)]
    terms,info=response_objectives(si,sj,*teachers)
    assert torch.allclose(terms['A'],info['common_component']+terms['D'])
    shift=torch.randn_like(si)
    shifted,_=response_objectives(si+shift,sj+shift,*teachers)
    assert torch.allclose(terms['D'],shifted['D'],atol=1e-6)
    assert not torch.allclose(terms['A'],shifted['A'])
    terms['D'].backward();assert si.grad.norm()>0 and sj.grad.norm()>0 and all(t.grad is None for t in teachers)
    # Exercise the REAL pinned native processors and real checkpoint replay,
    # with two blocks so the second Q depends on earlier conditioning.
    from check_BrainHIVE_functional_parentQ_CPU_v2 import TwoBlockNativeTinyUNet,snapshot_queries,assert_queries_unchanged
    from check_BrainHIVE_response_rank_CPU_v1 import native_processor_class
    cls,_=native_processor_class();unet=TwoBlockNativeTinyUNet(cls).eval().requires_grad_(False)
    capture=PairedEffectCapture(unet,expected_layers=2)
    hidden=torch.randn(2,19,32);text=torch.randn(2,3,8);base=torch.randn(2,4,8)
    left=torch.randn_like(base,requires_grad=True);right=torch.randn_like(base,requires_grad=True)
    teachers=[torch.randn_like(base) for _ in range(4)]
    try:
        with torch.no_grad(),capture.capture():unet(hidden,text,base)
        snapshot=snapshot_queries(capture)
        predicted=unet(hidden,text,left)
        auxiliary,terms,info=capture.auxiliary(left,right,*teachers,arm='D')
        assert not capture.queries
        assert_queries_unchanged(capture,snapshot,consumed=True)
        (predicted.square().mean()+auxiliary).backward()
        assert left.grad.norm()>0 and right.grad.norm()>0
        assert all(p.grad is None for p in unet.parameters())
        assert not capture.active and not capture.queries and unet.generator_forwards==2
        assert unet.block_calls['first']>2 and unet.block_calls['second']>2
        assert_queries_unchanged(capture,snapshot,consumed=True)
    finally:capture.close()
    assert not torch.cuda.is_initialized()
    print('FULL132300_PAIR_PLAN_NATIVE_PROCESSOR_CHECKPOINT_AND_TWO_BRANCH_GRADIENT_CPU_CHECKS_PASSED')


if __name__=='__main__':main()
