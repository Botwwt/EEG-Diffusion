"""Exercise the actual progress wrapper, including the failed stage keyword."""
import json,tempfile
from pathlib import Path
import run_ATMS_standard_ordinary_pipeline_CPU_20261028_v1 as V
from resume_ATMS_ordinary_images_CPU_20261028_v2 import progress_compatible
with tempfile.TemporaryDirectory() as folder:
    previous=V.AUDIT;V.AUDIT=Path(folder)
    try:
        report=progress_compatible(V.progress)
        for event,values,expected in [('MATCHED_ATMS_IMAGES_COMMITTED',dict(stage=80,images=40),dict(status='MATCHED_ATMS_IMAGES_COMMITTED',image_stage=80,images=40)),('RUNNING_FINITE_CPU_STAGE',dict(name='audit'),dict(status='RUNNING_FINITE_CPU_STAGE',name='audit'))]:
            report(event,**values);actual=V.read(V.AUDIT/'pipeline_progress.json');assert all(actual[k]==v for k,v in expected.items())
    finally:V.AUDIT=previous
print(json.dumps(dict(status='X01_REAL_CONTROLLER_PROGRESS_COLLISION_REGRESSION_PASSED',cases=2,new_GPU_jobs=0,new_training=0)))
