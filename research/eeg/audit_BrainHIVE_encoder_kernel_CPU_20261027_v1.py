"""Independent CPU source and projection challenges of all30 learned kernels."""
import argparse,signal,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U

def main():
    p=argparse.ArgumentParser()
    for k in ('completion','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Fixed 240second CPU audit limit')));signal.alarm(240)
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(C.read(a.native_registration))
    from brainhive_native_models_v1 import classes
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.set_num_threads(2);scope=classes();done=U.read(a.completion);spec=U.read(U.checked(done['spec']))
    for ref in spec['source_pins']:U.checked(ref)
    raw=U.read(U.checked(spec['raw_T_inputs']));bindings=U.read(U.checked(spec['raw_bindings']))
    records=U.read(U.checked(spec['Dfit_metadata']))['records'];rows=[int(x['array_row']) for x in records]
    ordinary=BoundedConditionAdapter().eval().requires_grad_(False)
    ordinary.load_state_dict(torch.load(U.checked(spec['ordinary_checkpoint']),weights_only=False,map_location='cpu')['model'],strict=True)
    rand=np.load(U.checked(done['random_basis']),allow_pickle=False)
    columns=np.linspace(0,1177,32).round().astype(int)
    ti=np.linspace(0,13229,20).round().astype(int);hi=np.linspace(0,819,20).round().astype(int)
    challenges=[]
    for j,(name,files) in enumerate(done['subjects'].items()):
        null=np.load(U.checked(files['common_kernel_basis.npy']),allow_pickle=False)
        n=torch.from_numpy(null);nc=n[:,columns];assert (nc.T@nc-torch.eye(32)).abs().max()<1e-11
        item=raw['subjects'][name];e=np.load(U.checked(dict(path=item['path'],sha256=item['sha256'])),mmap_mode='r').reshape(13230,4250)
        bind=bindings['subjects'][name];released=torch.load(U.checked(dict(path=bind['train_path'],sha256=bind['train_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        held=released[[rows[i] for i in hi]][:,bindings['selected_indices'],:].reshape(20,4250).clone().numpy();del released
        projection_errors={}
        for key,values,basis,ix in [('T_common_kernel.npy',e[ti],null,ti),('T_random_control.npy',e[ti],rand,ti),
                                    ('Dfit_common_kernel.npy',held,null,hi),('Dfit_random_control.npy',held,rand,hi)]:
            saved=np.load(U.checked(files[key]),mmap_mode='r')[ix]
            expected=values.astype(np.float64)@basis
            rel=float(np.linalg.norm(saved-expected)/max(np.linalg.norm(expected),1e-20));assert rel<1e-6
            projection_errors[key]=rel
        # Fresh independent CPU perturbations, not a bitwise replay of CUDA RNG.
        rng=np.random.default_rng(1729+j);delta=(rng.normal(size=(16,1178))@null.T).astype(np.float32)
        source=torch.from_numpy(np.array(e[:16],copy=True));d=torch.from_numpy(delta)
        d=d/d.norm(dim=1,keepdim=True)*source.norm(dim=1,keepdim=True)*.1
        proof=next(x for x in done['proofs'] if x['subject']==name);checks={}
        with torch.no_grad():
            for role,ref in proof['source_checkpoints'].items():
                m=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False)
                m.load_state_dict(torch.load(U.checked(ref),weights_only=True,map_location='cpu')['model'],strict=True)
                residual=float((m.proj_in.weight.double()@nc).norm()/m.proj_in.weight.double().norm());assert residual<1e-11
                c0=m(source);c1=m(source+d);rel=float(((c1-c0).norm(dim=1)/c0.norm(dim=1)).max());assert rel<3e-6
                w=m.block.layers[0].norm.weight.expand(16,-1);b=m.block.layers[0].norm.bias.expand(16,-1)
                o0=ordinary(c0,w,b)[0];o1=ordinary(c1,w,b)[0];orel=float(((o1-o0).norm(dim=1)/o0.norm(dim=1)).max());assert orel<3e-6
                checks[role]=dict(kernel32_column_relative_residual=residual,native_condition_relative_max=rel,ordinary_condition_relative_max=orel,perturbation_queries=16)
        challenges.append(dict(subject=name,projection_errors=projection_errors,checks=checks))
    signal.alarm(0)
    U.atomic(a.output/'completion.json',dict(status='ALL30_CPU_KERNEL_CHALLENGES480_FRESH_PERTURBATIONS_AND800_PROJECTION_ROW_CHECKS_PASSED',
        source=U.ref(__file__),GPU_completion=U.ref(a.completion),basis_columns_checked=columns.tolist(),T_rows_per_subject=ti.tolist(),Dfit_rows_per_subject=hi.tolist(),
        full_basis_CPU_QR_recomputed=False,full_basis_GPU_residual_already_checked=True,all30_model_SHA_checked=True,all_projection_file_SHA_checked=True,
        challenges=challenges,seconds=time.monotonic()-start,new_GPU_jobs=0,new_fit=0,new_images=0,visual_information_or_image_gain_established=False))
    print('ALL30_CPU_KERNEL_SOURCE_AND_NUMERICAL_CHALLENGES_PASSED',flush=True)

if __name__=='__main__':main()
