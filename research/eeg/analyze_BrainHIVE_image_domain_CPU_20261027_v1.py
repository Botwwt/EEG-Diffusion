"""D19 full held logits/all400 latent edit replays and selected CPU VAE encodings."""
import argparse,csv,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_image_domain_20261027_v1 as V

def compare(candidate,reference,queries):
    difference=np.asarray(candidate,np.float64)-np.asarray(reference,np.float64);concepts=sorted({q['concept_id'] for q in queries})
    means=np.array([difference[[i for i,q in enumerate(queries) if q['concept_id']==concept]].mean() for concept in concepts]);rng=np.random.default_rng(1729)
    samples=means[rng.integers(len(means),size=(1000,len(means)))].mean(1)
    return dict(difference=float(difference.mean()),descriptive_concept_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),concepts=len(concepts),negative_improves_log_loss=True)

def calibration(logits):
    probability=torch.sigmoid(torch.from_numpy(logits)).numpy().reshape(-1);labels=np.tile([1.,0.],logits.size//2);rows=[];ece=0.
    for i in range(10):
        mask=(probability>=i/10)&(probability<(i+1)/10 if i<9 else probability<=1.)
        if mask.any():
            p=float(probability[mask].mean());y=float(labels[mask].mean());count=int(mask.sum());ece+=count/len(labels)*abs(p-y);rows.append(dict(bin=i,count=count,mean_probability=p,actual_real_fraction=y))
    assert sum(x['count'] for x in rows)==len(labels);return dict(ECE=float(ece),bins=rows,examples=len(labels))

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not a.output.exists();a.output.mkdir(parents=True)
    done=U.read(a.completion);assert done['status']=='D19_THREE_FIXED_IMAGE_DOMAIN_HEADS_AND_ALL400_LOCKED_ENDPOINT_EDITS_COMPLETE'
    spec=U.read(U.checked(done['spec']));prepared=U.read(U.checked(done['prepared_inputs']));mapping=U.read(U.checked(prepared['mapping']))
    saved=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu',mmap=True);assert saved['epoch']==20 and saved['updates_per_head']==260 and saved['query_exposures_per_head']==8000
    cond={name:torch.from_numpy(np.load(U.checked(item))) for name,item in prepared['conditions'].items()};mean=cond['T400'].mean(0);scale=(cond['T400']-mean).square().mean().sqrt()
    assert torch.allclose(mean,saved['condition_mean'],atol=5e-7,rtol=5e-7) and torch.allclose(scale,saved['condition_scale'],atol=5e-7,rtol=5e-7)
    cond={name:(value-saved['condition_mean'])/saved['condition_scale'] for name,value in cond.items()}
    latent={name:torch.from_numpy(np.load(U.checked(item))) for name,item in done['encoded_latents'].items()}
    assert latent['T400'].shape==(400,2,4,64,64) and latent['Dfit200'].shape==(200,2,4,64,64) and latent['Dcheck400'].shape==(400,4,64,64)
    assert all(torch.isfinite(x).all() and x.dtype==torch.float32 for x in latent.values())
    noise=torch.from_numpy(np.load(U.checked(done['Dfit_noise'])));states=V.corrupt(latent['Dfit200'][:,0],latent['Dfit200'][:,1],noise).reshape(-1,4,64,64)
    sigma=torch.tensor(V.SIGMAS)[None,:,None].expand(200,4,2).reshape(-1);c=cond['Dfit200'][:,None,None].expand(200,4,2,1024).reshape(-1,1024)
    expected=np.load(U.checked(done['Dfit_logits']));assert expected.shape==(200,3,4,2)
    lock=U.read(U.checked(done['edit_lock']));bank=np.load(U.checked(lock['latents']));assert bank.shape==(400,4,4,64,64) and np.array_equal(bank[:,0],latent['Dcheck400'].numpy())
    diagnostics=U.read(U.checked(done['edit_diagnostics']));metrics={};cal={};replay=[];max_logit_error=0.;radius={}
    for j,arm in enumerate(V.FIELDS):
        model=V.DomainHead().eval().requires_grad_(False);model.load_state_dict(saved['models'][arm],strict=True)
        assert sum(x.numel() for x in model.parameters())==841121 and set(int(x['step']) for x in saved['optimizers'][arm]['state'].values())=={260}
        cc=torch.zeros_like(c) if arm=='unconditional' else c
        with torch.no_grad():actual=torch.cat([model(states[lo:lo+128],sigma[lo:lo+128],cc[lo:lo+128]) for lo in range(0,1600,128)]).reshape(200,4,2)
        error=float(np.abs(actual.numpy()-expected[:,j]).max());max_logit_error=max(max_logit_error,error);assert error<5e-4
        assert torch.equal(actual[:,3,0],actual[:,3,1]);metrics[arm]={key:value.numpy() for key,value in V.proper(actual).items()};cal[arm]=[calibration(actual[:,k].numpy()) for k in range(4)]
        parts=[]
        for lo in range(0,400,32):
            cc=cond['Dcheck400'][lo:lo+32]
            if arm=='unconditional':cc=torch.zeros_like(cc)
            actual_edit,detail=V.edit(model,latent['Dcheck400'][lo:lo+32],cc);parts.append(actual_edit)
        actual_edit=torch.cat(parts);reference=torch.from_numpy(bank[:,j+1]);relative=(actual_edit-reference).flatten(1).norm(dim=-1)/reference.flatten(1).norm(dim=-1)
        assert relative.max()<3e-5
        change=(reference-latent['Dcheck400']).square().mean((1,2,3)).sqrt()/latent['Dcheck400'].square().mean((1,2,3)).sqrt();assert change.max()<=.100001
        radius[arm]=dict(all400_max_relative_RMS=float(change.max()),all400_mean_relative_RMS=float(change.mean()))
        replay.append(dict(arm=arm,all400_eight_step_edits_replayed=True,maximum_relative_L2=float(relative.max()),held_logits_queries=200,times=4,classes=2,max_logit_error=error))
    means={arm:{key:value.mean(0).tolist() for key,value in values.items()} for arm,values in metrics.items()}
    contrasts={arm:dict(clean_sigma0=compare(metrics['conditional']['log_loss'][:,0],metrics[arm]['log_loss'][:,0],mapping['Dfit200']),
        all4_sigma_mean=compare(metrics['conditional']['log_loss'].mean(1),metrics[arm]['log_loss'].mean(1),mapping['Dfit200'])) for arm in ('unconditional','wrong_condition')}
    out=a.output/'Dfit_proper_per_query.csv'
    with out.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['query_id','subject_id','concept_id','stimulus_id','array_row','arm','sigma','real_logit','generated_logit','log_loss','Brier','accuracy']);writer.writeheader()
        for i,q in enumerate(mapping['Dfit200']):
            for j,arm in enumerate(V.FIELDS):
                for k,s in enumerate(V.SIGMAS):writer.writerow({key:q[key] for key in ['query_id','subject_id','concept_id','stimulus_id','array_row']}|dict(arm=arm,sigma=s,
                    real_logit=float(expected[i,j,k,0]),generated_logit=float(expected[i,j,k,1]),**{key:float(value[i,k]) for key,value in metrics[arm].items()}))
    logs=[json.loads(x) for x in (a.completion.parent/'training_log.jsonl').read_text().splitlines()];train=[x for x in logs if x['stage']=='COMPLETE_T400_PASS'];held=[x for x in logs if x['stage']=='FIXED_DFIT200_FOUR_SIGMA_PROPER_LOG_ONLY']
    assert len(train)==20 and [x['epoch'] for x in train]==list(range(1,21)) and len(held)==21
    assert all(x['full_coverage_min']==x['full_coverage_max']==1 and x['updates_per_head']==13*x['epoch'] and x['query_exposures_per_head']==400*x['epoch'] for x in train)
    # Read only native VAE tensors from the original checkpoint; no UNet allocation.
    import brainhive_paired_study_contract_20261025_v1 as C
    root=Path(__file__).resolve().parent;config=C.read(root/'configs/paired_effect_20261025_v1/study_review_v3.json');C.setup_runtime(config)
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors import safe_open
    import fit_BrainHIVE_image_domain_GPU_20261027_v2 as FIT
    reg=U.read(root/'configs/ATM_public_decoder_registration_20261020_v1.json');payload=next(x for x in reg['generator_assets'] if 'sdxl_turbo_fp16' in x['path']);U.checked(payload)
    vae_config=U.read(Path(reg['generator_asset_root'])/'sdxl_config/vae/config.json')
    with safe_open(payload['path'],framework='pt',device='cpu') as f:state={key:f.get_tensor(key) for key in f.keys() if key.startswith('first_stage_model.')}
    assert state;converted=convert_ldm_vae_checkpoint(state,vae_config);vae=AutoencoderKL.from_config(vae_config).eval().requires_grad_(False);vae.load_state_dict(converted,strict=True)
    provenance=U.read(a.completion.parent/'provenance.json');assert FIT.state_sha({key:value.to(torch.float16) for key,value in vae.state_dict().items()})==provenance['VAE']['original_FP16_VAE_state_sha256']
    vae_replays=[]
    for name,index,real in [('T400',0,True),('T400',0,False),('Dfit200',0,True),('Dcheck400',0,False)]:
        row=mapping[name][index]
        if real:
            import zipfile
            item=row['target']
            with zipfile.ZipFile(item['image_zip']) as archive:raw=archive.read(item['image_member'])
            assert hashlib.sha256(raw).hexdigest()==item['image_sha256']
        else:raw=U.checked(row['generated']).read_bytes()
        with torch.no_grad():actual=vae.encode(FIT.image_tensor(raw)[None]).latent_dist.mode()*vae.config.scaling_factor
        expected_latent=latent[name][index:index+1,0 if real else 1] if name!='Dcheck400' else latent[name][index:index+1]
        relative=float((actual-expected_latent).norm()/expected_latent.norm());assert relative<5e-4
        vae_replays.append(dict(partition=name,index=index,real=real,relative_L2=relative))
    plot=None
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,3.8))
        for arm in V.FIELDS:
            axes[0].plot(range(1,21),[x['balanced_BCE'][arm] for x in train],label=arm)
            axes[1].plot(range(21),[x['metrics_by_sigma'][arm]['log_loss'][0] for x in held],label=arm)
            axes[2].plot([x['mean_probability'] for x in cal[arm][0]['bins']],[x['actual_real_fraction'] for x in cal[arm][0]['bins']],marker='o',label=arm)
        axes[2].plot([0,1],[0,1],'k--');axes[0].set(title='T400 balanced BCE, all4 sigma',xlabel='Fixed complete passes');axes[1].set(title='Dfit200 clean sigma0 BCE, logs only',xlabel='Fixed complete passes');axes[2].set(title='Dfit200 sigma0 reliability',xlabel='Predicted real probability',ylabel='Actual real frequency')
        for ax in axes:ax.legend(fontsize=7)
        fig.tight_layout();path=a.output/'actual_training_proper_loss_and_reliability.png';fig.savefig(path,dpi=160);plt.close(fig);plot=U.ref(path)
    except ImportError:pass
    for ref in spec['source_pins']:U.checked(ref)
    assert not torch.cuda.is_initialized();result=dict(status='D19_ALL4800_HELD_LOGITS_ALL1200_EIGHT_STEP_EDITS_AND_FOUR_NATIVE_VAE_CPU_REPLAYS_VERIFIED',completion=U.ref(a.completion),source=U.ref(__file__),
        all20_T400_full_passes_verified=True,all21_Dfit_logs_retained=True,all3_optimizers_steps260=True,maximum_CPU_GPU_logit_error=max_logit_error,all1200_edited_latents_CPU_replay=replay,
        all400_radius_bounds_per_head=radius,proper_means_by_sigma=means,conditional_minus_each_control_proper_loss=contrasts,calibration_by_sigma=cal,
        same_noise_sigma1_class_inputs_and_logits_identical=True,selected_CPU_VAE_replays=vae_replays,CPU_VAE_scope='4 actual images: T real/generated, Dfit real, Dcheck generated. Not a full1600 CPU VAE replay.',
        original_VAE_subset_load_and_widened_FP32_weights_exact=True,per_query=U.ref(out),per_query_rows=2400,actual_plot=plot,
        Dcheck_target_images_or_F_used=False,proper_loss_does_not_establish_input_gradient_usefulness_or_image_gain=True,seconds=time.monotonic()-start,new_GPU_jobs=0,new_images=0,new_parameter_updates=0,
        exact_diffusion_score_or_posterior=False,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],proper_means=means,comparisons=contrasts)))

if __name__=='__main__':main()
