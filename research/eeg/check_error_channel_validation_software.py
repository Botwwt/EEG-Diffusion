"""Analytical/synthetic tests for the CPU validator; never fit EEG errors."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch

from error_channel_validation_metrics import (energy_score,binary_auc,
    classification_summary,fit_two_sample,confusion_js,distribution_summary)
from paired_analysis import paired_summary


def main():
    torch.set_num_threads(2);g=torch.Generator().manual_seed(1729)
    samples=torch.tensor([[[0.],[1.],[3.]]]);truth=torch.tensor([[2.]])
    assert abs(float(energy_score(samples,truth))-1/3)<1e-6
    constant=torch.full((5,16,3),2.);target=torch.zeros(5,3)
    assert torch.allclose(energy_score(constant,target),torch.full((5,),math.sqrt(12)))
    normal=torch.randn(20000,16,1,generator=g);obs=torch.randn(20000,1,generator=g)
    score=float(energy_score(normal,obs).mean())
    assert abs(score-1/math.sqrt(math.pi))<.02,(score,1/math.sqrt(math.pi))
    shifted=float(energy_score(normal+2,obs).mean());assert shifted>score+.5
    assert binary_auc([0,0,1,1],[0,0,1,1])==1
    assert binary_auc([0,0,1,1],[1,1,0,0])==0
    assert binary_auc([0,0,1,1],[1,1,1,1])==.5
    assert classification_summary([0,0,1,1],[0,0,0,0])['balanced_accuracy']==.5
    plan=dict(architectures=['linear','mlp64'],seed=1729,epochs=40,batch_size=512,
              learning_rate=.001,weight_decay=.01)
    train=torch.randn(320,16,generator=g);test=torch.randn(160,16,generator=g)
    shifted_report,shifted_state=fit_two_sample(train+8,train-8,test+8,test-8,plan)
    for report in shifted_report.values():
        assert report['Dcheck']['AUC_real_positive']>.99
        assert report['Dcheck']['balanced_accuracy']>.98
    null_report,null_state=fit_two_sample(train,train,test,test,plan)
    for report in null_report.values():
        assert report['Dcheck']['AUC_real_positive']==.5
        assert report['Dcheck']['balanced_accuracy']==.5
    _,changed_test_state=fit_two_sample(train,train,test+100,test-100,plan)
    for arch in null_state:
        for key in ['mean','std']:assert torch.equal(null_state[arch][key],changed_test_state[arch][key])
        for key,value in null_state[arch]['model'].items():
            assert torch.equal(value,changed_test_state[arch]['model'][key]),'Dcheck changed fitted classifier weights'
    residual=torch.randn(64,16,generator=g)
    description=distribution_summary(residual,residual)
    assert description['bias_difference_norm']==description['covariance_relative_Frobenius']==0
    assert description['covariance_trace_ratio']==1
    js=confusion_js([0,1,2,3],[0,1,2,3],['a','a','b','b'],['a','a','b','b'])
    assert js['jensen_shannon_nats']==0
    changed=confusion_js([0,1,2,3],[2,3,0,1],['a','a','b','b'],['a','a','b','b'])
    assert abs(changed['jensen_shannon_nats']-math.log(2))<1e-12
    shifted_moments=distribution_summary(residual,residual+2)
    assert abs(shifted_moments['bias_difference_norm']-8)<1e-6
    assert abs(shifted_moments['covariance_trace_ratio']-1)<1e-6
    # Data uncertainty must cluster concepts/subjects, not draw count.
    delta=np.full((2*4*3,2),[-.25,.125]);subjects=[];concepts=[]
    for s in range(2):
        for c in range(4):
            for i in range(3):subjects.append(str(s));concepts.append(str(c))
    paired=paired_summary(delta,subjects,concepts,200,1729)
    np.testing.assert_allclose(paired['CI95'],[[-.25,-.25],[.125,.125]],atol=1e-12)
    assert not torch.cuda.is_initialized()
    root=Path(__file__).parent
    report=dict(status='ANALYTICAL_AND_SYNTHETIC_VALIDATOR_CHECKS_PASSED',
        tests=['norm_energy_closed_form','Gaussian_energy_analytic_expectation','shifted_distribution_worse',
            'AUC_direction_ties','controlled_classifier_positive_and_identical_null',
            'Dcheck_cannot_change_classifier_fit','identical_moments_and_confusion',
            'crossed_subject_concept_intervals'],
        Gaussian_energy_estimate=score,Gaussian_energy_analytic=1/math.sqrt(math.pi),
        shifted_energy_estimate=shifted,classifier_positive_control=shifted_report,
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ['error_channel_validation_metrics.py','validate_native_error_channel.py']},
        actual_EEG_conditions_used=False,scientific_OOF_cache_verified=False,
        actual_error_channel_fitted=False,GPU_initialized=False,images_generated=0,
        limits=['Software checks only. Full actual cache provenance and complete Dcheck run remain required.'])
    path=root/'audit/error_channel_validation_software_checks.json'
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
