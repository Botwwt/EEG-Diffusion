"""Narrow pre-execution numerical/refusal checks, no pretrained GPU claim."""
import json
import numpy as np
from audit_VAR_full_native_batch64_top0_CPU_20261101_v1 import posterior,fidelity_decision
from VAR_full_native_throughput_contract_CPU_20261101_v1 import storage_requirement


def run():
    checks=[]
    chunks=[(lo,min(lo+64,1000)) for lo in range(0,1000,64)]
    assert len(chunks)==16 and [c for lo,hi in chunks for c in range(lo,hi)]==list(range(1000)) and chunks[-1]==(960,1000)
    checks.append('ALL1000_CLASSES_NO_DROPPED_FINAL40_OR_DUPLICATES')
    x=np.array([[1.,2.,-5.],[1001.,1002.,995.]])
    p=posterior(x);direct=np.exp(x[0]-2);direct/=direct.sum()
    assert np.allclose(p,np.broadcast_to(direct,(2,3)),atol=1e-14,rtol=0)
    checks.append('INDEPENDENT_DIRECT_PROBABILITIES_AND_COMMON_SHIFT')
    assert fidelity_decision(.0009,True,.001) and not fidelity_decision(.0011,True,.001)
    assert not fidelity_decision(0.,False,.001)
    checks.append('FIXED_POSTERIOR_TOLERANCE_AND_ARGMAX_REFUSAL')
    for value in (np.nan,-1.,np.inf):
        try:fidelity_decision(value,True,.001)
        except AssertionError:pass
        else:raise AssertionError('Invalid numerical comparison accepted')
    checks.append('NONFINITE_OR_NEGATIVE_ERROR_REFUSAL')
    sizes=[1,4,9,16,25,36,64,100,169,256];assert sum(sizes)==680
    assert 8*1000*680*4+8*1000*10*8+2*3*256*256*4+(1<<20)<25<<20
    checks.append('FULL_OUTPUT_DIMENSIONS_AND_PAYLOAD_PLANNING_NOT_RUNTIME_PROOF')
    for old in (0,98<<20,110<<20):
        for new in (0,1<<20,17<<20):
            required=storage_requirement(90<<30,old,new,25<<20,128<<20)
            assert required+old+new==(90<<30)+(128<<20)
    for old,new in ((128<<20,1),(0,26<<20)):
        try:storage_requirement(90<<30,old,new,25<<20,128<<20)
        except AssertionError:pass
        else:raise AssertionError('Shared native or new ownership cap bypassed')
    checks.append('LEGACY_PLUS_SHARED_CAP_CONSERVED_NO_DOUBLE_CHARGE_OR_OVERSPEND')
    return dict(status='PASS_LOCAL_NARROW_FULL_SCOPE_ARITHMETIC_GUARDS_ONLY',checks=checks,
        pretrained_model_or_GPU_tested=False,new_training_admitted=False)


if __name__=='__main__':print(json.dumps(run()))
