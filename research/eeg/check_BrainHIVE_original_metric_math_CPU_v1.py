"""Independent analytic checks for pinned source metrics; no real image scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_original_metrics_runtime_v1 import *
from brainhive_native_models_v1 import write


if __name__=='__main__':
    torch.set_num_threads(2)
    output=WORK/'audit/original_metric_math_CPU_v1.json'
    assert not output.exists()
    checks=numerical_source_checks()
    assert not torch.cuda.is_initialized()
    write(output,dict(status='PINNED_AUTHOR_METRIC_ANALYTIC_CHECKS_PASSED_CPU',checks=checks,
        original_metric_source_sha256=sha(ROOT/'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py'),
        runtime_source_sha256=sha(ROOT/'brainhive_original_metrics_runtime_v1.py'),
        GPU_initialized=False,real_images_evaluated=0,component_gain_established=False))
    print(checks,flush=True)
