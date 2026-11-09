"""Actual v3 receipt normalization and functional UNKNOWN regression checks."""
from pathlib import Path
import json
import tempfile
from unittest.mock import patch
import brainhive_paired_study_contract_20261025_v1 as C
import queue_BrainHIVE_paired_effect_20261026_v3 as Q3
from evaluate_BrainHIVE_paired_effect_CPU_20261025_v2 import functional_readout


def main():
    checked=[];spec=dict(name='probe',command=['python','probe.py'])
    with patch.object(Q3.Q,'live',return_value=False):
        for raw in (0,1,7,-15,-9):
            receipt=dict(name='probe',wrapper_pid=111,pid=112,command=spec['command'],exit_code=raw,
                status='COMPLETED' if raw==0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE')
            normalized=0 if raw==0 else 1
            result=Q3.close_GPU_receipt(receipt,spec,111,normalized)
            assert result['child_exit_code']==raw and result['wrapper_exit_code']==normalized
            checked.append('raw_child_'+str(raw)+'_wrapper_mapping')
        bad=receipt|dict(exit_code=-15)
        try:Q3.close_GPU_receipt(bad,spec,111,-15)
        except AssertionError:checked.append('raw_child_exit_cannot_be_mistaken_for_wrapper_exit')
        else:raise AssertionError('Unnormalized wrapper exit accepted')
    with tempfile.TemporaryDirectory(prefix='hive_signal_unknown_') as folder:
        root=Path(folder);(root/'audit').mkdir();(root/'functional_T80').mkdir()
        (root/'functional_T80/summary.json').write_text('{"status":"partial"}')
        with patch.object(C,'ROOT',root),patch.object(C,'DATA',root):
            for raw in (-15,-9,7):
                (root/'audit/probe_job.json').write_text(json.dumps(dict(status='FAILED_OR_JOB_TIMEOUT_RESUMABLE',exit_code=raw)))
                result=functional_readout(dict(functional_job_name='probe'))
                assert not result['closed'] and result['result']=='UNKNOWN'
                checked.append('signal_or_arbitrary_exit_'+str(raw)+'_functional_UNKNOWN')
    assert C.bytes_used is Q3.safe_bytes_used
    checked.append('safe_counter_installed_only_in_supervisor_process')
    print(json.dumps(dict(status='V3_CONTROLLER_CPU_REGRESSIONS_PASSED',checks=checked,
        GPU_used=False,new_fit=False,scientific_core_changed=False)))


if __name__=='__main__':main()
