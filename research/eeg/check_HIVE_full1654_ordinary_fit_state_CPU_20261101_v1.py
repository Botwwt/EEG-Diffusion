"""Future-fit state transactions and exact coverage; no training admission/GPU."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
E=F.E;K=F.K


def run(engineering_protocol=None):
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    checks=[];order=E.orders(torch)
    expected={0:0,1:16,128:2048,10337:165392,10338:165400,10339:165416,20676:330800,31014:496200}
    for step,count in expected.items():
        mask=F.expected_coverage(step,order,torch)
        assert int(mask.sum())==count==F.prefix_counts(step)[2]
    checks.append('actual_full165400_perpass_coverage_first_last_and_pass_boundaries')
    for bad in (-1,31015,1.5):
        try:F.prefix_counts(bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid prefix accepted')
    checks.append('invalid_or_fractional_update_prefix_refused')
    flat=torch.cat(order);chunks=[]
    boundaries=[0,128,777,10338,10339,20676,30000,31014]
    for low,high in zip(boundaries,boundaries[1:]):
        items=[]
        for step in range(low,high):
            epoch,batch,unused=F.prefix_counts(step);items.append(order[epoch][batch*16:batch*16+16])
        chunks.append(torch.cat(items))
    assert torch.equal(torch.cat(chunks),flat)
    checks.append('planned_segments_recover_exact3pass_order_with8query_tail_batches')
    with tempfile.TemporaryDirectory(prefix='HIVE_full1654_checkpoint_fixture_') as directory:
        root=Path(directory).resolve()
        def state(step,segment):
            return dict(updates=step,exposures=F.prefix_counts(step)[2],segment=segment,
                        CPU_FIXTURE_ONLY=True,model={'dummy':torch.arange(4,dtype=torch.float32)+step},
                        noise_rng=torch.Generator().manual_seed(1729+step).get_state(),coverage=F.expected_coverage(step,order,torch))
        first=F.save_checkpoint(root,state(128,0),torch)
        assert not list((root/'checkpoints').glob('*.pending'))
        source=Path(first['checkpoint']['path']).read_bytes()
        second=F.save_checkpoint(root,state(512,1),torch)
        restored,pointer=F.load_checkpoint(root,torch)
        assert restored['updates']==512 and pointer==second
        assert Path(first['checkpoint']['path']).read_bytes()==source
        assert len(list((root/'checkpoints').glob('*/commit.json')))==2
        checks.append('actual_immutable_checkpoints_atomic_latest_and_old_bytes_preserved')
        payload=Path(second['checkpoint']['path']);valid=payload.read_bytes();payload.write_bytes(valid[:-1])
        try:F.load_checkpoint(root,torch)
        except AssertionError:pass
        else:raise AssertionError('Corrupt checkpoint accepted')
        checks.append('truncated_latest_checkpoint_refused_no_silent_older_fallback')
        payload.write_bytes(valid)
        snapshot,_=F.load_checkpoint(root,torch)
        generator=torch.Generator().set_state(snapshot['noise_rng']);expected_draw=torch.rand(32,generator=generator)
        other=torch.Generator().set_state(snapshot['noise_rng']);actual=torch.rand(32,generator=other)
        assert torch.equal(expected_draw,actual)
        checks.append('actual_CPU_private_rng_state_roundtrip_not_GPU_determinism_claim')
        # The actual existing engineering protocol MUST fail the future fit gate
        # before any Tensor/model import by the real CLI.
        profile=(Path(engineering_protocol) if engineering_protocol is not None else
                 Path(__file__).parent/'audit/HIVE_full1654_original_ordinary_profile_preparation_20261101_v1/actual/protocol.json')
        entry=Path(__file__).parent/'train_HIVE_full1654_original_ordinary_GPU_20261101_v1.py'
        guard="""import builtins,runpy,sys
original=builtins.__import__
def blocked(name,*args,**kwargs):
 if name.split('.')[0] in ('torch','numpy','diffusers','torchvision','brainhive_native_models_v1'):
  raise AssertionError('TENSOR_IMPORT_BEFORE_FULL_LIFECYCLE_REVIEW')
 return original(name,*args,**kwargs)
builtins.__import__=blocked
sys.argv=[sys.argv[1],'--protocol',sys.argv[2],'--segment','0']
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        child=subprocess.run([sys.executable,'-B','-c',guard,str(entry),str(profile)],
                             cwd=entry.parent,text=True,capture_output=True,timeout=30)
        assert child.returncode and 'TENSOR_IMPORT_BEFORE_FULL_LIFECYCLE_REVIEW' not in child.stderr
        assert 'AssertionError' in child.stderr and 'ModuleNotFoundError' not in child.stderr
        checks.append('real_existing_engineering_protocol_refuses_actual_fit_CLI_before_tensor_import')
    assert not torch.cuda.is_initialized()
    return dict(status='PASS_FUTURE_ORIGINAL_ORDINARY_FIT_STATE_CPU_CHECKS_ONLY',checks=checks,
                sources=[K.ref(Path(__file__).parent/name) for name in
                         ('brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1.py',
                          'train_HIVE_full1654_original_ordinary_GPU_20261101_v1.py',
                          'audit_HIVE_full1654_original_ordinary_fit_CPU_20261101_v1.py',
                          'check_HIVE_full1654_ordinary_fit_state_CPU_20261101_v1.py')],
                CPU_fixture_only=True,actual_formal_training_updates=0,actual_frozen_UNet_calls=0,
                actual_GPU_jobs=0,actual_full_lifecycle_admission=False,
                all31k_gradient_transitions_replayed=False,future_GPU_bit_exactness_proved=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--engineering-protocol',type=Path)
    args=parser.parse_args();result=run(args.engineering_protocol);args.receipt.parent.mkdir(parents=True,exist_ok=True)
    assert not args.receipt.exists();args.receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result))
