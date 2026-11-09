"""Source-derived0/1 wrapper mapping and real CPU-only signal child check."""
import ast
import os
from pathlib import Path
import subprocess
import sys
from brainhive_gpu_exit_contract_CPU_20261101_v1 import verify,TIMEOUT


def run(wrapper):
    source=ast.parse(Path(wrapper).read_bytes())
    main=next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    returns=[node for node in main.body if isinstance(node,ast.Return)]
    assert len(returns)==1
    expression=compile(ast.Expression(returns[0].value),str(wrapper),'eval')
    for child in (0,1,7,-9,-15):assert eval(expression,{'code':child})==(0 if child==0 else 1)
    def receipt(child,timeout=False):
        value=dict(exit_code=child,wrapper_pid=51,status='COMPLETED' if child==0 else 'FAILED_OR_JOB_TIMEOUT_RESUMABLE')
        if timeout:value['termination_reason']=TIMEOUT
        return value
    checks=['actual_pinned_wrapper_return_expression_is0_or1_not_child_signal']
    assert verify(0,receipt(0),51)['success']
    assert verify(1,receipt(-15,True),51)==dict(wrapper_exit_code=1,child_exit_code=-15,success=False,timed_out=True)
    assert not verify(1,receipt(7),51)['timed_out']
    checks.append('success_timeout_and_non_timeout_are_distinct')
    for code,row,pid in [(0,receipt(7),51),(1,receipt(0),51),(1,receipt(-15,True),52)]:
        try:verify(code,row,pid)
        except AssertionError:pass
        else:raise AssertionError('Invalid wrapper/child/PID accepted')
    checks.append('wrong_wrapper_exit_or_wrong_owner_PID_refused')
    result=subprocess.run([sys.executable,'-B','-c','raise SystemExit(7)'],capture_output=True,timeout=20)
    assert result.returncode==7 and verify(eval(expression,{'code':result.returncode}),receipt(7),51)['child_exit_code']==7
    checks.append('actual_CPU_only_nonzero_child_exit_contract')
    if os.name=='posix':
        result=subprocess.run([sys.executable,'-B','-c','import os,signal;os.kill(os.getpid(),signal.SIGTERM)'],capture_output=True,timeout=20)
        assert result.returncode==-15
        assert verify(eval(expression,{'code':result.returncode}),receipt(result.returncode,True),51)['timed_out']
        checks.append('actual_Linux_CPU_only_SIGTERM_child_maps_to_wrapper1_and_ledger_minus15')
    return dict(status='PASS_SOURCE_DERIVED_WRAPPER_CHILD_EXIT_CONTRACT_CPU_ONLY',checks=checks,
                CUDA_imports=0,new_GPU_jobs=0,training_updates=0,scientific_threshold_changes=False)
