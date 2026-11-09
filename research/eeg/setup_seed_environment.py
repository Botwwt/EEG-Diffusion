"""Isolated evaluator dependencies; never replace the existing PyTorch runtime."""
from pathlib import Path
import json
import importlib.metadata
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
TARGET=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/runtime/evaluation')
RUNTIME='/root/mced_representation_runtime_v1/lib/python3.12/site-packages'
SOURCE='/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/code/SEED_5327a134'
WHEEL='https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/mmcv-2.2.0-cp312-cp312-manylinux1_x86_64.whl'
# Exact versions are validated against PyPI before this script is launched.
PACKAGES=['mmengine==0.10.7','sentence-transformers==5.1.1','fairscale==0.4.13',
          'pycocotools==2.0.10','shapely==2.1.2','terminaltables==3.1.10',
          'addict==2.4.0','yapf==0.43.0','opencv-python-headless==4.11.0.86',
          'rich==14.2.0','termcolor==3.1.0','platformdirs==4.4.0',
          'scikit-learn==1.5.2','joblib==1.4.2','threadpoolctl==3.5.0']


def main():
    TARGET.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    env=dict(os.environ,PYTHONPATH=RUNTIME,PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_CACHE_DIR='1')
    installed={d.metadata['Name'].lower().replace('_','-'):d.version
               for d in importlib.metadata.distributions(path=[str(TARGET)])}
    needed=[p for p in PACKAGES if installed.get(p.split('==')[0])!=p.split('==')[1]]
    if installed.get('mmcv')!='2.2.0':needed.insert(0,WHEEL)
    command=[sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',
             '--target',str(TARGET),'--index-url','https://pypi.org/simple']+needed
    if needed:
        result=subprocess.run(command,env=env,timeout=1200)
        assert result.returncode==0,'Evaluator dependency installation failed'
    sys.path[:0]=[str(TARGET),RUNTIME,SOURCE]
    import torch
    assert torch.__version__=='2.3.0+cu121'
    assert torch.__file__.startswith('/root/miniconda3/')
    import mmcv,mmengine,mmdet
    from mmcv.ops import MultiScaleDeformableAttention
    import sentence_transformers
    torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
    model=MultiScaleDeformableAttention(embed_dims=256,num_heads=8,num_levels=2,num_points=4,batch_first=True).cuda().eval()
    with torch.no_grad():
        q=torch.randn(1,3,256,device='cuda');value=torch.randn(1,5,256,device='cuda')
        reference=torch.full((1,3,2,2),.5,device='cuda')
        shapes=torch.tensor([[2,2],[1,1]],device='cuda');starts=torch.tensor([0,4],device='cuda')
        out=model(q,value=value,reference_points=reference,spatial_shapes=shapes,level_start_index=starts)
        reference_out=model.cpu()(q.cpu(),value=value.cpu(),reference_points=reference.cpu(),
                    spatial_shapes=shapes.cpu(),level_start_index=starts.cpu())
    assert out.shape==(1,3,256) and torch.isfinite(out).all()
    assert torch.allclose(out.cpu(),reference_out,atol=1e-4,rtol=1e-4)
    report=dict(status='EVALUATOR_DEPENDENCY_IMPORT_AND_CUDA_OP_PASSED',torch=torch.__version__,
                torch_path=torch.__file__,mmcv=mmcv.__version__,mmengine=mmengine.__version__,
                mmdet=mmdet.__version__,sentence_transformers=sentence_transformers.__version__,
                target=str(TARGET),source=SOURCE,source_commit='5327a134b90b04fb1b37066d98b309c9cc533a02',
                command=command,seconds=time.monotonic()-started,
                evaluator_weights_loaded=False,formal_metric_calls=0,existing_torch_changed=False)
    report['CUDA_vs_CPU_attention_max_abs_error']=float((out.cpu()-reference_out).abs().max())
    (ROOT/'audit/seed_environment_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (ROOT/'audit/seed_environment_failure.json').write_text(json.dumps(
            dict(status='FAILED_REPAIRABLE',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),indent=2)+'\n')
        raise
