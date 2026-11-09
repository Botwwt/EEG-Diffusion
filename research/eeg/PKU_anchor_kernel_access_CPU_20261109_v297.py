"""Actual native projection null direction: distinguish available raw information from a post-c head."""
import json,math,os,socket,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_anchor_kernel_access_v297';out.mkdir();start=time.monotonic()
def write(n,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode()
    with (out/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
try:
    import torch
    from PKU_anchored_encoder_core_20261109_v286 import make_model,posterior_indices,sha
    torch.set_num_threads(4);torch.manual_seed(1729);m=make_model('C_pair',device='cpu').double();W=m.anchor.proj_in.weight;v=torch.randn(4250,dtype=torch.float64);gram=W@W.T;alpha=torch.linalg.solve(gram,W@v);v=v-W.T@alpha;v=v/v.norm();residual=float((W@v).norm());assert residual<1e-10
    split=json.loads((R/'data/recovery_20261108_v236/current_split_manifest.json').read_text());rows=split['image_rows']['T'][:4];raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);post,_=posterior_indices();x=raw[rows][:,post,:].double().reshape(4,-1);del raw
    parameter=m.candidate.proj_in.weight;original=parameter.detach().clone();direction=torch.randn(1024,dtype=torch.float64);direction/=direction.norm()
    with torch.no_grad():
        native0=m.anchor(x);native1=m.anchor(x+v);assert torch.allclose(native0,native1,atol=1e-10,rtol=1e-10)
        before=m(x);assert torch.equal(before,native0.float())
        # Analytic construction only: not an optimizer step, trained EEG effect, or data-derived gain.
        parameter.add_(.5*direction[:,None]*v[None,:]);changed0=m(x);changed1=m(x+v);difference=float((changed1-changed0).norm());assert difference>1e-5
        parameter.copy_(original);assert torch.equal(m(x),before)
    assert not torch.cuda.is_initialized()
    write('completion.json',dict(status='PASS_REAL_NATIVE_NULL_DIRECTION_AND_ANCHORED_ENCODER_FUNCTION_CLASS_ACCESS',job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,
        EEG_shape=[17,250],actual_W_shape=list(W.shape),nullspace_dimension_lower_bound=3226,native_projection_null_residual_norm=residual,native_condition_pair_max_abs=float((native1-native0).abs().max()),
        constructed_anchored_condition_pair_L2=difference,temporary_parameter_construction_fully_restored=True,reference_native_null_check_precision="FP64",anchored_coordinate_execution_precision="ProductionFP32",original_failed_fixture_preserved="v295 compared FP64 reference bytes to intentionallyFP32 production output",new_optimizer_updates=0,new_images=0,
        interpretation='A deterministic post-c head sees equal native c for this algebraic perturbation. The raw-input anchored encoder can represent different outputs while keeping the same cap. Constructed direction is not evidence of physiological signal, naturally occurring EEG differences, learned use or image gains.',source_sha256=sha(__file__)))
    print('PASS_ARCHITECTURAL_RAW_INPUT_ACCESS_NOT_EEG_EFFECT',flush=True)
except BaseException as e:
    write('failure.json',dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
