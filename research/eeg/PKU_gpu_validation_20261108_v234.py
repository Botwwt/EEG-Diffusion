"""Continue previously pending numerical checks inside ONE Slurm GPU allocation."""
import json,os,socket,time,traceback
from pathlib import Path
job=os.environ.get('SLURM_JOB_ID');assert job and 'login' not in socket.gethostname().lower()
project=Path('/dfs/share/maleiLab/wwt/eeg_sota_autoresearch_20261018_v1')
out=project/'runs'/('GPU_validation_'+job);out.mkdir(mode=0o700)
def write(name,value):
    p=out/name;temp=p.with_suffix('.pending');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
started=time.monotonic();stages=[]
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    gpu=torch.cuda.get_device_properties(0)
    identity=dict(job_id=job,hostname=socket.gethostname(),GPU_name=gpu.name,total_GPU_bytes=gpu.total_memory,
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,CUDA_runtime=torch.version.cuda)
    write('identity.json',identity)
    from check_eeg_cross_host_gpu_runtime_CPU_20261106_v231 import checks as first
    from check_eeg_margin_matrix_free_CPU_20261106_v233 import checks as matrix_free
    write('CPU_reference_checks.json',first('cpu'));stages.append('CPU_reference_checks')
    write('GPU_reference_checks.json',first('cuda'));stages.append('GPU_reference_checks')
    write('matrix_free_CPU_checks.json',matrix_free());stages.append('matrix_free_CPU_checks')
    import eeg_margin_matrix_free_20261106_v233 as M
    from check_eeg_cross_host_gpu_runtime_CPU_20261106_v231 import serial
    from torch import nn
    from torch.nn import functional as F
    rows=[]
    for dtype in (torch.float64,torch.float32):
        net=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,24)).to(device='cuda',dtype=dtype).eval()
        x=torch.randn(8,32,device='cuda',dtype=dtype)
        visual=F.normalize(torch.randn(40,24,device='cuda',dtype=dtype),dim=-1)
        ids=torch.arange(8,device='cuda');groups=torch.arange(40,device='cuda')//4
        _,J,_,_=serial(net(x),visual,ids,groups,tuple(net.parameters()))
        encoder,theta=M.functional_encoder_snapshot(net,list(dict(net.named_parameters())),(x,))
        operator=M.MatrixFreeMarginMap(encoder,theta,visual,ids,groups)
        v=tuple(torch.randn_like(p) for p in theta);vf=torch.cat([t.reshape(-1) for t in v]).double()
        w=torch.randn(16,device='cuda',dtype=dtype)
        jv=operator.forward(v).double();jtw=torch.cat([g.reshape(-1) for g in operator.transpose(w)]).double()
        gram=operator.gram(w).double()
        errors=dict(Jv=float((jv-J@vf).abs().max()),JT_w=float((jtw-J.T@w.double()).abs().max()),
                    Gram=float((gram-J@(J.T@w.double())).abs().max()),adjoint=float(abs(jv@w.double()-vf@jtw)))
        bound=1e-9 if dtype==torch.float64 else 5e-4
        assert max(errors.values())<bound,(dtype,errors)
        rows.append(dict(dtype=str(dtype),errors=errors))
    stages.append('matrix_free_GPU_checks');write('matrix_free_GPU_checks.json',dict(results=rows))
    torch.cuda.synchronize()
    result=dict(status='COMPLETE_SINGLE_GPU_NUMERICAL_VALIDATION_ONLY',identity=identity,stages=stages,
                seconds=time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                new_model_training_updates=0,new_images=0,actual_pretrained_host_or_speedup_proved=False)
    write('completion.json',result);print(json.dumps(result),flush=True)
except BaseException as exc:
    write('failure.json',dict(status='FAILED_PRESERVED',stages=stages,error=repr(exc),traceback=traceback.format_exc()))
    raise
