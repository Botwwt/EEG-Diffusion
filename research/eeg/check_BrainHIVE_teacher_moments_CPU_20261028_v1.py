"""Check the finite Jensen identity and both symmetric gradient directions."""
import argparse
import json
from pathlib import Path
import torch
import brainhive_teacher_moments_20261028_v1 as M

def run(output):
    torch.set_num_threads(2);torch.manual_seed(1729)
    c=torch.randn(9,7,dtype=torch.float64,requires_grad=True)
    draws=torch.nn.functional.normalize(torch.randn(8,9,7,dtype=torch.float64),dim=-1)
    det=torch.nn.functional.normalize(torch.randn(9,7,dtype=torch.float64),dim=-1)
    checks=[];errors=[]
    for targets in [draws[0],draws.mean(0)]:
        logits=3.7*torch.nn.functional.normalize(c,dim=-1)@targets.T
        labels=torch.arange(len(c))
        loss=.5*(torch.nn.functional.cross_entropy(logits,labels,reduction='sum')+
                  torch.nn.functional.cross_entropy(logits.T,labels,reduction='sum'))
        expected=torch.autograd.grad(loss,c)[0]
        _,actual=M.symmetric_readout(c.detach(),targets,3.7)
        error=float((expected-actual).abs().max());assert error<1e-12;errors.append(error)
    checks.append('symmetric_gradient_matches_independent_autograd_for_unit_and_nonunit_mean_targets')
    result=M.finite_moment_summary(c.detach(),det,draws,3.7)
    assert float(result['jensen_gap'].min())>=-1e-13
    checks.append('finite_sample_Jensen_gap_nonnegative_including_both_directions')
    assert not torch.allclose(result['mean_gradient'],result['mean_target_gradient'],atol=1e-8)
    checks.append('gradient_of_mean_target_does_not_replace_mean_of_stochastic_gradients')
    zero=M.finite_moment_summary(c.detach(),det,det.unsqueeze(0).repeat(8,1,1),3.7)
    assert zero['jensen_gap'].abs().max()<1e-14
    torch.testing.assert_close(zero['mean_gradient'],zero['mean_target_gradient'],atol=1e-14,rtol=0)
    checks.append('zero_teacher_variance_returns_deterministic_objective_and_gradient')
    bad=M.symmetric_readout(c.detach(),torch.nn.functional.normalize(draws.mean(0),dim=-1),3.7)[0]
    correct=M.symmetric_readout(c.detach(),draws.mean(0),3.7)[0]
    assert not torch.allclose(bad,correct,atol=1e-8)
    checks.append('empirical_mean_normalization_would_change_Jensen_comparison')
    value=dict(status='D36_FROZEN_TEACHER_JENSEN_AND_SYMMETRIC_GRADIENT_CPU_CHECKS_PASSED',tests=len(checks),checks=checks,
               maximum_gradient_error=max(errors),actual_EEG_read=False,CUDA_initialized=torch.cuda.is_initialized())
    assert not value['CUDA_initialized']
    with output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.output)
