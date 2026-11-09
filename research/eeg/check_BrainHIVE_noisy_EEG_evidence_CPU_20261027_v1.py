"""D13 analytic math, production gradients and pre-result decision checks."""
import argparse
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_noisy_EEG_evidence_20261027_v1 as V


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True); args=p.parse_args()
    math=V.self_test(); queries={split:[dict(concept_id=str(i//10)) for i in range(n)] for split,n in [('Dfit',200),('Dcheck',400)]}
    def fixture(mode):
        logits={}; retrieval={}
        for split,n in [('Dfit',200),('Dcheck',400)]:
            logits[split]={}; retrieval[split]={}
            for arm in V.ARMS:
                value=1. if mode=='base' or mode=='raw' and arm=='raw_EEG' else 0.
                logits[split][arm]=np.tile(np.array([value,-value]),(n,4,1))
                retrieval[split][arm]=dict(top1=.2 if arm=='raw_EEG' else .15)
        return V.readout(logits,retrieval,queries)
    assert fixture('null')['decision']['action']=='STOP_NO_HELD_PREDICTIVE_EVIDENCE'
    assert fixture('raw')['raw_EEG_increment_supported'] is True
    base=fixture('base'); assert base['decision']['action']=='BASE_C_ONLY_EVIDENCE_SUPPORTS_SEPARATE_TEST_NO_RAW_INCREMENT'
    assert base['decision']['automatic_image_admission'] is False
    assert not torch.cuda.is_initialized()
    result=dict(status='D13_ALL11_CPU_MATH_AND_DECISION_CHECKS_PASSED',math=math,checks=math['checks']+3,
        decision_cases=['null rejects','raw evidence hierarchy','base-only cannot claim raw increment'],source=U.ref(__file__),
        CUDA_initialized=False,real_EEG_or_visual_data_loaded=False,real_model_fitted=False)
    U.atomic(args.output,result); print(result)


if __name__=='__main__': main()
