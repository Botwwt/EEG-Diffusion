"""T-only latent basis; complete Dfit VAE encoding with original T preprocessing."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,hashlib,io,json,signal,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V

def main():
    p=argparse.ArgumentParser()
    for k in ('spec','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    assert a.max_seconds==180 and not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));stage='INPUTS'
    try:
        spec=U.read(a.spec);assert spec['schema']==V.SCHEMA
        for x in spec['source_pins']:U.checked(x)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(a.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;cfg=C.read(a.native_registration)
        doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=180,maximum_output_bytes=192<<20,
                 maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def check():assert not stopped[0] and time.monotonic()-start<165;budget.check()
        def log(x):
            with (a.output/'progress.jsonl').open('a') as f:f.write(json.dumps(x|dict(seconds=time.monotonic()-start))+'\n')
            print(json.dumps(x),flush=True)
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        split=U.read(U.checked(spec['split']));receipt=U.read(U.checked(spec['T_latent_receipt']))
        ids=split['image_rows']['T'];assert ids==receipt['array_rows'] and len(ids)==13230
        native=np.load(U.checked(dict(path=receipt['path'],sha256=receipt['sha256'])),mmap_mode='r',allow_pickle=False)
        assert native.shape==(13230,4,64,64) and native.dtype==np.float32
        stage='T_ONLY_RANDOMIZED_PC128';x=torch.from_numpy(np.array(native,copy=True)).cuda().flatten(1);mean=x.mean(0);centered=x-mean
        _,_,basis=torch.pca_lowrank(centered,q=144,center=False,niter=4)
        basis=torch.linalg.qr(basis[:,:128].double(),mode='reduced').Q
        pc=(x.double()-mean.double())@basis;orth=float((basis.T@basis-torch.eye(128,device='cuda')).abs().max());assert orth<1e-10
        energy=float(pc.square().sum()/centered.double().square().sum());assert 0<energy<1
        torch.save(dict(latent_mean=mean.cpu(),basis=basis.cpu(),rank=128,retained_energy=energy,orthogonality_max=orth,
                        randomized_subspace=True,oversampling=16,power_iterations=4,T_rows=ids,seed=1729),a.output/'T_basis.pt')
        U.save_array(a.output/'T_projected.npy',pc.cpu().numpy());del x,centered,pc;torch.cuda.empty_cache();check()
        stage='READ_ONLY_DFIT820_METADATA';metadata={int(x['array_row']):dict(x,array_row=int(x['array_row'])) for x in csv.DictReader(U.checked(spec['image_manifest']).open()) if x['dataset_split']=='train'}
        concepts=sorted(split['d_fit_concepts']);records=[]
        for c in concepts:
            rows=sorted(i for i,v in metadata.items() if v['concept_id']==c);assert len(rows)==10
            records.extend([metadata[i] for i in rows])
        assert len(records)==820 and set(x['array_row'] for x in records).isdisjoint(ids)
        U.atomic(a.output/'Dfit_metadata.json',dict(records=records,concepts=concepts,partition='Dfit',query_targets_for_diagnostic_only=True))
        from brainhive_native_vae_subset_20261027_v1 import load
        pipe,vaeproof=load(U.checked(spec['VAE_subset']));check()
        from torchvision import transforms
        preprocess=transforms.Compose([transforms.Resize(512,interpolation=transforms.InterpolationMode.BILINEAR),transforms.CenterCrop(512),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
        containers={}
        def pixels(record):
            path=record['image_zip']
            if path not in containers:containers[path]=zipfile.ZipFile(path)
            raw=containers[path].read(record['image_member']);assert hashlib.sha256(raw).hexdigest()==record['image_sha256']
            with Image.open(io.BytesIO(raw)) as image:return preprocess(image.convert('RGB'))
        stage='NATIVE_T4_ENCODER_COORDINATE_REPLAY'
        with torch.no_grad():
            replay=pipe.vae.encode(torch.stack([pixels(metadata[i]) for i in ids[:4]]).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor
        expected=torch.from_numpy(np.array(native[:4],copy=True));difference=float(torch.linalg.vector_norm(replay.cpu()-expected)/torch.linalg.vector_norm(expected));assert difference<3e-5
        stage='ALL820_DFIT_NATIVE_VAE_ENCODING';encoded=np.empty((820,4,64,64),np.float32)
        with torch.no_grad():
            for lo in range(0,820,4):
                check();value=pipe.vae.encode(torch.stack([pixels(v) for v in records[lo:lo+4]]).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor
                assert torch.isfinite(value).all();encoded[lo:lo+len(value)]=value.cpu().numpy()
                if lo%100==0:log(dict(stage=stage,completed=lo+len(value)))
            projected=(torch.from_numpy(encoded.reshape(820,-1)).cuda().double()-mean.double())@basis
        for z in containers.values():z.close()
        U.save_array(a.output/'Dfit_latents.npy',encoded);U.save_array(a.output/'Dfit_projected.npy',projected.cpu().numpy());check()
        from diffusers import DDPMScheduler
        scheduler=DDPMScheduler.from_config(U.read(U.checked(spec['scheduler_config'])))
        assert scheduler.config.num_train_timesteps==1000 and scheduler.config.prediction_type=='epsilon'
        time_values={str(t):dict(alpha=float(scheduler.alphas_cumprod[t].sqrt()),sigma=float((1-scheduler.alphas_cumprod[t]).sqrt())) for t in V.TIMES}
        result=dict(status='ALL13230_T_BASIS_AND820_DFIT_NATIVE_VAE_LATENTS_COMPLETE',schema=V.SCHEMA,source=U.ref(__file__),spec=U.ref(a.spec),wrapper=wrapper,
                    basis=U.ref(a.output/'T_basis.pt'),T_projected=U.ref(a.output/'T_projected.npy'),Dfit_projected=U.ref(a.output/'Dfit_projected.npy'),Dfit_latents=U.ref(a.output/'Dfit_latents.npy'),
                    Dfit_metadata=U.ref(a.output/'Dfit_metadata.json'),T_randomized_PC128_retained_energy=energy,basis_orthogonality_max=orth,
                    native_T4_VAE_encode_relative_L2=difference,preprocessing='Original T Resize512 BILINEAR CenterCrop512 RGB[-1,1], not D19 ImageOps BICUBIC',VAE=vaeproof,time_values=time_values,
                    no_Dcheck_V_or_official_image_pixels_read=True,no_new_research_images=True,no_UNet_loaded=True,full_latent_dimensions=16384,diagnostic_dimensions=128,
                    seconds=time.monotonic()-start,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,formal_D_resumed=False)
        U.atomic(a.output/'completion.json',result);print(json.dumps(result))
    except BaseException as exc:
        U.atomic(a.output/'failure.json',dict(stage=stage,error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,no_automatic_retry=True));raise

if __name__=='__main__':main()
