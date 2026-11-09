"""Run the unchanged public extractor with finite peak-memory enforcement."""
import argparse
from pathlib import Path
import os
import runpy
import sys
import threading
import time
import brainhive_paired_study_contract_20261025_v1 as C


class PeakGuard:
    def __init__(self,limits,peak_CPU,peak_GPU,headroom):
        self.limits=limits;self.cpu=peak_CPU;self.gpu=peak_GPU;self.headroom=headroom
        self.baseline=peak_CPU();self.started=time.monotonic();self.CPU=0;self.GPU=0
    def check(self):
        self.CPU=max(self.CPU,self.cpu()-self.baseline);self.GPU=max(self.GPU,self.gpu())
        assert self.CPU<=self.limits['CPU'],'Feature extra CPU peak exceeded'
        assert self.GPU<=self.limits['GPU'],'Feature GPU allocated peak exceeded'
        assert self.headroom()>=self.limits['minimum_headroom'],'Feature CPU headroom exhausted'
        assert time.monotonic()-self.started<self.limits['seconds'],'Finite feature deadline'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--arm',choices=C.ARMS);parser.add_argument('--engineering',action='store_true')
    args=parser.parse_args();config=C.read(args.registration);C.validate(config)
    if args.engineering:
        manifest=Path(config['review_feature_profile_manifest']['path'])
        assert C.sha(manifest)==config['review_feature_profile_manifest']['sha256']
        output=C.DATA/'review_v2/feature_profile';job=config['review_feature_profile_job'];maximum=1800
    else:
        C.require_admission(config);assert args.arm in C.ARMS
        root=C.DATA/'arms'/args.arm;done=C.read(root/'artifact/completion.json')
        assert done['status']=='FULL16600_PAIRED_ARTIFACT_AND_EEG_IDENTITY_AUDIT_COMPLETE'
        manifest=Path(done['manifest']['path']);assert C.sha(manifest)==done['manifest']['sha256']
        output=root/'features';job='BrainHIVE_paired_effect_v1_'+args.arm+'_V_features_20261025';maximum=14400
    proof=output.with_name(output.name+'_guard');proof.mkdir(parents=True,exist_ok=False)
    import torch
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_functional_training_contract_v2 import resource_headroom
    D.torch=torch
    wrapper,_=C.verify_wrapper(config,job,maximum,proof)
    guard=PeakGuard(dict(CPU=config['maximum_CPU_extra_bytes'],GPU=config['maximum_GPU_bytes'],
        minimum_headroom=10<<30,seconds=maximum-30),D.process_rss,
        lambda:torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
        lambda:resource_headroom()['effective_headroom_bytes'])
    stopped=threading.Event()
    def monitor():
        while not stopped.wait(.25):
            try:guard.check()
            except BaseException as error:
                C.atomic(proof/'failure.json',dict(status='FEATURE_GUARD_FAILED',error=repr(error),
                    peak_extra_CPU_bytes=guard.CPU,peak_GPU_bytes=guard.GPU))
                os._exit(2)  # No orphan worker; this is the extractor's own process.
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        sys.argv=[str(C.ROOT/'extract_image_features.py'),'--manifest',str(manifest),'--output',str(output)]
        runpy.run_path(sys.argv[0],run_name='__main__')
        guard.check();done=C.read(output/'completion.json');assert done['status']=='COMPLETED'
        C.atomic(proof/'completion.json',dict(status='UNCHANGED_EXTRACTOR_WITH_REAL_PEAK_GUARD_COMPLETE',
            queries=done['queries'],extractor_sha256=C.sha(C.ROOT/'extract_image_features.py'),
            feature_completion=C.ref(output/'completion.json'),peak_extra_CPU_bytes=guard.CPU,
            peak_GPU_bytes=guard.GPU,wrapper=wrapper,engineering_only=args.engineering,
            existing_images_only=True,new_scores=False))
    finally:
        stopped.set();worker.join(timeout=5);assert not worker.is_alive()


if __name__=='__main__':main()
