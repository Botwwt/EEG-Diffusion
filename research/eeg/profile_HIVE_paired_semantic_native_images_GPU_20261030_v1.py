"""Four discarded engineering images, native observation hooks and CPU math."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import atomic,checked,read,ref
def main(path):
    s=read(path);assert s['status']=='REGISTERED_PAIRED_SEMANTIC_FULL_LIFECYCLE_PREFLIGHT_ONLY'
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import numpy as np,torch
    import brainhive_native_models_v1 as N
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from run_HIVE_paired_semantic_images_20261030_v1 import sample_observed,save_image
    from audit_HIVE_paired_semantic_study_CPU_20261030_v1 import manual_EEG,manual_ordinary,manual_tokens
    from PIL import Image
    import torch.nn.functional as F
    assert 'run_gpu_job_v2.py' in Path('/proc',str(__import__('os').getppid()),'cmdline').read_text()
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,(20<<30)/torch.cuda.get_device_properties(0).total_memory))
    out=Path(s['preflight_image_root']);assert not out.exists();out.mkdir();start=time.monotonic();(out/'loader').mkdir();audit=ConvertedWeightAudit(torch,out/'loader');audit.install()
    try:pipe,A,P,prompts,host=load_native(training=False)
    finally:audit.restore()
    loading=audit.complete();w=torch.load(checked(s['parent']),weights_only=True,map_location='cpu')['model'];O=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model']
    model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);model.load_state_dict(w,strict=True)
    ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False);ordinary.load_state_dict(O,strict=True)
    acpu={n:v.cpu() for n,v in A.state_dict().items()};pcpu={n:v.cpu() for n,v in P.state_dict().items()};cache=read(checked(s['cache_completion']));x=np.load(checked(cache['EEG_mean4']),allow_pickle=False,mmap_mode='r');proofs=[];error=[0.,0.,0.]
    for i in range(2):
        with torch.inference_mode():
            eeg=torch.from_numpy(x[i:i+1].copy());c=model(eeg.cuda());co=ordinary(c,w['block.layers.0.norm.weight'][None].cuda(),w['block.layers.0.norm.bias'][None].cuda())[0];tk=P(A(co.half()))
            expected=manual_EEG(eeg,w,torch,F);expected_o=manual_ordinary(expected,O,w['block.layers.0.norm.weight'].double()[None],w['block.layers.0.norm.bias'].double()[None],torch,F)
            for k,a,b in [(0,expected,c.cpu()),(1,expected_o,co.cpu())]:error[k]=max(error[k],float((a-b).abs().max()/(1+a.abs().max())));assert error[k]<3e-5
            expected_t=manual_tokens(co.cpu(),acpu,pcpu,torch,F).float();diff=expected_t-tk.cpu().float();error[2]=max(error[2],float(diff.abs().max()/(1+expected_t.abs().max())));assert error[2]<.02 and float(diff.norm()/expected_t.norm())<.005
            a,p1=sample_observed(pipe,prompts,tk,1729+i,torch);b,p2=sample_observed(pipe,prompts,tk,1729+i,torch);assert p1==p2 and np.array_equal(np.asarray(a),np.asarray(b))
            save_image(out/f'{i}_ordinary.webp',a,np,Image);save_image(out/f'{i}_repeat.webp',b,np,Image);proofs.append(p1)
    atomic(out/'completion.json',dict(status='COMPLETE_ACTUAL_NATIVE_IMAGE_EXPORT_MATH_AND_PIXEL_REPLAY_ENGINEERING',protocol=ref(path),maximum_scaled_EEG_O_token_errors=error,all4_noise_and_IP_native_observations_passed=True,repeat_pixels_bit_exact=True,engineering_images=4,research_images=0,formal_updates=0,host=host,loader=loading,seconds=time.monotonic()-start,peak_GPU_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
