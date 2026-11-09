"""Construct a common input blind subspace of actual full/oof0/oof1 BrainMLPs.

This is a source-bound architecture/data audit, not a new image method. QR's
last1178 columns are a guaranteed shared-kernel SUBSPACE, not necessarily the
entire shared kernel. No claim that the subspace contains useful brain signal.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_bounded_component_v1 import BoundedConditionAdapter

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==180 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stage='INPUTS';stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True))
    try:
        spec=U.read(a.spec)
        for ref in spec['source_pins']:U.checked(ref)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration)
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=180,maximum_output_bytes=2048<<20,maximum_extra_CPU_bytes=10<<30,
                 maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-start<165;budget.check()
        def log(x):
            x=x|dict(seconds=time.monotonic()-start)
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
            print(json.dumps(x),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        from brainhive_native_models_v1 import classes
        scope=classes();contract=U.read(U.checked(spec['component_input_contract']));raw=U.read(U.checked(spec['raw_T_inputs']));bindings=U.read(U.checked(spec['raw_bindings']))
        held=U.read(U.checked(spec['Dfit_metadata']))['records'];heldrows=[int(x['array_row']) for x in held];assert len(heldrows)==820
        ordinary=BoundedConditionAdapter().cuda().eval().requires_grad_(False)
        saved=torch.load(U.checked(spec['ordinary_checkpoint']),weights_only=False,map_location='cpu');ordinary.load_state_dict(saved['model'],strict=True)
        assert saved['arm']=='ordinary' and saved['calibration']['selected_strength']==1.
        # Independent same-dimension random control, fixed once in raw4250 axes.
        random=torch.linalg.qr(torch.randn(4250,1178,device='cuda',dtype=torch.float64),mode='reduced').Q
        U.save_array(a.output/'fixed_random_basis.npy',random.cpu().numpy());subjects={};proofs=[]
        for subject in contract['subjects']:
            check();name=subject['subject'];stage='SHARED_KERNEL_'+name;models={};ws=[];refs={}
            for role in ('full','oof0','oof1'):
                source=next(x for x in subject['sources'] if x['role']==role);ref=dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])
                saved=torch.load(U.checked(ref),weights_only=True,map_location='cpu')['model']
                m=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);m.load_state_dict(saved,strict=True)
                assert m.proj_in.weight.shape==(1024,4250);models[role]=m;ws.append(m.proj_in.weight.double());refs[role]=ref
            stack=torch.cat(ws);q,_=torch.linalg.qr(stack.T,mode='complete');null=q[:,3072:].contiguous();assert null.shape==(4250,1178)
            residual=float(torch.linalg.vector_norm(stack@null)/torch.linalg.vector_norm(stack));orth=float((null.T@null-torch.eye(1178,device='cuda')).abs().max())
            assert residual<1e-11 and orth<1e-11
            item=raw['subjects'][name];eeg=torch.from_numpy(np.array(np.load(U.checked(dict(path=item['path'],sha256=item['sha256'])),mmap_mode='r'),copy=True).reshape(13230,4250)).cuda()
            coordinates=eeg.double()@null;random_coordinates=eeg.double()@random
            energy=float((coordinates-coordinates.mean(0)).square().sum()/(eeg.double()-eeg.double().mean(0)).square().sum())
            bind=bindings['subjects'][name];released=torch.load(U.checked(dict(path=bind['train_path'],sha256=bind['train_sha256'])),weights_only=True,map_location='cpu',mmap=True)
            he=released[heldrows][:,bindings['selected_indices'],:].reshape(820,4250).clone().cuda();del released
            source=eeg[:16];coeff=torch.randn(16,1178,device='cuda',dtype=torch.float64)
            delta=(coeff@null.T).float();delta=delta/delta.norm(dim=1,keepdim=True)*source.norm(dim=1,keepdim=True)*.1
            interventions={}
            with torch.no_grad():
                for role,m in models.items():
                    before=m(source);after=m(source+delta)
                    error=float(((after-before).norm(dim=1)/before.norm(dim=1)).max());assert error<3e-6
                    w=m.block.layers[0].norm.weight.expand(16,-1);b=m.block.layers[0].norm.bias.expand(16,-1)
                    cb=ordinary(before,w,b)[0];ca=ordinary(after,w,b)[0]
                    cerror=float(((ca-cb).norm(dim=1)/cb.norm(dim=1)).max());assert cerror<3e-6
                    interventions[role]=dict(actual_FP32_native_condition_relative_max=error,actual_FP32_ordinary_condition_relative_max=cerror,
                                             delta_input_relative_norm=.1,queries=16,not_a_physiological_intervention=True)
            folder=a.output/name;folder.mkdir()
            for filename,value in [('common_kernel_basis.npy',null),('T_common_kernel.npy',coordinates.float()),('T_random_control.npy',random_coordinates.float()),
                                   ('Dfit_common_kernel.npy',(he.double()@null).float()),('Dfit_random_control.npy',(he.double()@random).float())]:
                U.save_array(folder/filename,value.cpu().numpy())
            proof=dict(subject=name,source_checkpoints=refs,stack_shape=[3072,4250],guaranteed_common_kernel_dimensions=1178,
                       exact_full_shared_kernel_dimension_established=False,complete_QR_residual=residual,basis_orthogonality=orth,
                       centered_T_EEG_energy_fraction=energy,energy_is_not_neural_signal_or_visual_information=True,actual_input_perturbations=interventions)
            proofs.append(proof);subjects[name]={p.name:U.ref(p) for p in folder.glob('*.npy')};log(proof)
            del models,ws,stack,q,null,eeg,coordinates,random_coordinates,he;torch.cuda.empty_cache();check()
        result=dict(status='ALL30_NATIVE_ENCODERS_COMMON_KERNEL_AND_ALL10_T_DFIT_PROJECTIONS_AUDITED',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
                    subjects=subjects,proofs=proofs,random_basis=U.ref(a.output/'fixed_random_basis.npy'),raw_T_rows=raw['source_rows'],Dfit_rows=heldrows,
                    no_visual_F_latent_or_image_scores_loaded=True,no_Dcheck_V_or_official_values_read=True,new_optimizer_updates=0,new_UNet_calls=0,new_images=0,
                    useful_visual_information_in_kernel_established=False,new_image_method_admitted=False,formal_D_resumed=False,
                    seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(a.output/'completion.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('subjects','proofs','raw_T_rows','Dfit_rows')}))
    except BaseException as exc:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

if __name__=='__main__':main()
