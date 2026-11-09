"""Frozen host-value FP32 derivatives; original ordinary never updated."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
from brainhive_overlap_gradient_contract_CPU_20261103_v1 import setup,resources,DOMAINS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--job-name',required=True);parser.add_argument('--part',required=True)
    cli=parser.parse_args();started=time.monotonic();spec,base,G=setup(cli.protocol);G.wrapper(base,cli.job_name)
    indices=spec['engineering_queries'] if cli.part=='engineering' else spec['full_chunks'][int(cli.part)]
    assert cli.part=='engineering' or cli.part in ('0','1','2','3')
    root=Path(spec['output_root']);out=root/('GPU_'+cli.part);assert not out.exists();out.mkdir()
    before=resources(spec,base,G,start=True);base_rss=G.rss();stage='LOAD';torch=None
    try:
        import torch
        from brainhive_component_generator_v1 import load_native
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction((20<<30)/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats()
        def budget(refresh=False):
            assert time.monotonic()-started<(570 if cli.part=='engineering' else 870)
            assert G.rss()-base_rss<10<<30 and torch.cuda.max_memory_allocated()<20<<30
            if refresh:resources(spec,base,G)
        prep=G.read(root/'prepared/completion.json');states=np.load(G.checked(prep['states']),mmap_mode='r',allow_pickle=False)
        bank=dict(np.load(G.checked(prep['geometry']),allow_pickle=False));times=bank['timesteps'];geometry=bank['geometry']
        prior=G.read(G.checked(prep['old_inputs']));data=dict(np.load(G.checked(prior['arrays']),allow_pickle=False))
        saved=torch.load(G.checked(base['ordinary']),weights_only=True,map_location='cpu')
        head=BoundedConditionAdapter().eval();head.load_state_dict(saved['model'],strict=True);head=head.cuda()
        parameters=tuple(head.parameters());parameter_names=[name for name,_ in head.named_parameters()]
        assert sum(p.numel() for p in parameters)==525568 and len(parameters)==4
        def digest(module,half_values=False):
            h=hashlib.sha256()
            for name,value in module.state_dict().items():
                t=value.detach()
                if half_values and t.is_floating_point():t=t.half()
                t=t.cpu().contiguous();h.update(name.encode());h.update(str(t.dtype).encode());h.update(t.numpy().tobytes())
            return h.hexdigest()
        head_before=digest(head)
        with torch.no_grad():
            inputs=torch.from_numpy(data['condition']).cuda();weights=torch.from_numpy(data['weight']).cuda();biases=torch.from_numpy(data['bias']).cuda()
            changed=head(inputs.reshape(160,1024),weights.reshape(160,1024),biases.reshape(160,1024))[0].reshape(80,2,1024)
            assert torch.allclose(changed.cpu(),torch.from_numpy(data['corrected_CPU']),atol=1e-5,rtol=5e-6)
        loader_path=out/'loader';loader_path.mkdir();loader=ConvertedWeightAudit(torch,loader_path);loader.install()
        try:pipe,A,P,prompts,host=load_native(training=False)
        finally:loader.restore()
        loading=loader.complete();host_before={name:digest(m,True) for name,m in dict(UNet=pipe.unet,A=A,P=P).items()}
        def native_forward(sample,time_id,tokens):
            count=len(sample)
            return pipe.unet(sample,time_id,
                encoder_hidden_states=prompts[0].expand(count,-1,-1).contiguous(),
                added_cond_kwargs=dict(text_embeds=prompts[1].expand(count,-1).contiguous(),time_ids=prompts[2].expand(count,-1).contiguous()),
                cross_attention_kwargs=dict(ip_hidden_states=tokens),return_dict=False)[0]
        # Same native half inputs, actual contiguous conditioning and all four times.
        prior_states=np.load(G.checked(spec['prior_states']),mmap_mode='r',allow_pickle=False)
        prior_predictions=np.load(G.checked(spec['prior_predictions']),mmap_mode='r',allow_pickle=False)
        parity=[]
        with torch.no_grad():
            native_tokens=P(A(changed[:2,0].half()))
            for ti,time_id in enumerate(times):
                sample=torch.from_numpy(np.array(prior_states[0,:2,ti],copy=True)).cuda()
                tt=torch.tensor(float(time_id),device='cuda',dtype=torch.float32)
                with torch.autocast('cuda',dtype=torch.float16):prediction=native_forward(sample,tt,native_tokens)
                assert np.array_equal(prediction.cpu().numpy(),prior_predictions[0,:2,ti,0])
                parity.append(dict(t=int(time_id),exact=True))
        # Numerically faithful derivative profile, not the BF16 training or FP16 image path.
        # Released modules are unused by any subsequent UNet-only call; original files remain.
        pipe.register_modules(vae=None,text_encoder=None,text_encoder_2=None);torch.cuda.empty_cache()
        pipe.unet.float().eval();A.float().eval();P.float().eval();prompts=tuple(v.float() for v in prompts)
        assert all(not p.requires_grad for m in (pipe.unet,A,P) for p in m.parameters())
        precision=[]
        with torch.no_grad():
            fp32_tokens=P(A(changed[:2,0]))
            for ti,time_id in enumerate(times):
                sample=torch.from_numpy(np.array(prior_states[0,:2,ti],copy=True)).cuda().float()
                value=native_forward(sample,torch.tensor(float(time_id),device='cuda'),fp32_tokens).cpu().numpy()
                ref=prior_predictions[0,:2,ti,0].astype(np.float64)
                rel=float(np.linalg.norm(value.astype(np.float64)-ref)/max(np.linalg.norm(ref),1e-12));assert rel<=.02
                precision.append(dict(t=int(time_id),FP32_vs_original_FP16_relative_L2=rel))
        del prior_states,prior_predictions,native_tokens,fp32_tokens,sample,prediction,value
        split=G.read(G.checked(base['input_pins']['split']));position={r:i for i,r in enumerate(split['image_rows']['T'])}
        latent=np.load(G.checked(spec['T_latent']),mmap_mode='r',allow_pickle=False)
        count=len(indices);shape=(3,count,4,2,4,64,64)
        eeg=np.lib.format.open_memmap(out/'EEG_predictions.npy',mode='w+',dtype=np.float32,shape=shape)
        teacher=np.lib.format.open_memmap(out/'F_predictions.npy',mode='w+',dtype=np.float32,shape=shape)
        input_grad=np.empty((3,count,4,2,2,1024),np.float32);metrics=np.empty((3,count,4,23),np.float64)
        raw_parameters=[];raw_ids=[];identities=[];FD=[];cell_times=[];forward_samples=16;backward_calls=0
        def flatten(grads):return torch.cat([g.reshape(-1) for g in grads[:-1]])
        def VJP(loss,condition,retain):
            nonlocal backward_calls
            grads=torch.autograd.grad(loss,(*parameters,condition),retain_graph=retain,allow_unused=False)
            assert all(torch.isfinite(v).all() for v in grads);backward_calls+=1
            return flatten(grads),grads[-1]
        def paired_forward(condition,weight,bias,sample,tt):
            corrected=head(condition,weight,bias)[0];tokens=P(A(corrected))
            return native_forward(sample,tt,tokens)
        stage='ALL_FIXED_EEG_INPUT_AND_PARAMETER_GRADIENTS'
        for local_index,global_index in enumerate(indices):
            row=spec['queries'][global_index];zi=np.asarray(latent[position[row['array_row']]],np.float64);zj=np.asarray(latent[position[row['donor_array_row']]],np.float64)
            original_state_positive_gradients={}
            for domain in range(3):
                for ti,time_id in enumerate(times):
                    budget(refresh=ti==0);cell_started=time.monotonic()
                    state=np.asarray(states[domain,global_index,ti],np.float64);a=float(np.sqrt(bank['alpha_cumprod'][int(time_id)]));sig=float(np.sqrt(1-bank['alpha_cumprod'][int(time_id)]))
                    target=np.stack(((state-a*zi)/sig,(state-a*zj)/sig));contrast=target[0]-target[1]
                    target_t=torch.from_numpy(target).cuda();difference=torch.from_numpy(contrast).cuda()
                    condition=inputs[global_index].detach().clone().requires_grad_(True)
                    weight=weights[global_index];bias=biases[global_index]
                    sample=torch.from_numpy(np.array(state,np.float32,copy=True)).cuda().unsqueeze(0).repeat(2,1,1,1).contiguous()
                    tt=torch.tensor(float(time_id),device='cuda',dtype=torch.float32)
                    prediction=paired_forward(condition,weight,bias,sample,tt);values=prediction.double()
                    lossP=(values[0]-target_t[0]).square().mean();lossC=(values[0]-values[1]-difference).square().mean()
                    ri,rj=map(float,geometry[domain,global_index,ti,2:4]);residual=values-target_t
                    lossR=ri*residual[0].square().mean()+rj*residual[1].square().mean()
                    lossM=(ri*residual[0]+rj*residual[1]).square().mean()
                    assert abs(float(lossR-lossM-ri*rj*lossC))<1e-10
                    special=cli.part=='engineering' or (local_index==0 and ti==0)
                    gP,cP=VJP(lossP,condition,True);gC,cC=VJP(lossC,condition,special)
                    if special:
                        gM,cM=VJP(lossM,condition,True);gR,cR=VJP(lossR,condition,False)
                        discrepancy=gR.double()-gM.double()-ri*rj*gC.double()
                        absolute=float(discrepancy.abs().max());relative=float(discrepancy.norm()/gR.double().norm().clamp_min(1e-24))
                        assert relative<=5e-4 or absolute<=1e-7
                        identities.append(dict(query=global_index,domain=domain,time_index=ti,parameter_max_abs=absolute,parameter_relative_L2=relative,
                            input_max_abs=float((cR.double()-cM.double()-ri*rj*cC.double()).abs().max())))
                    with torch.no_grad():
                        ft=P(A(torch.from_numpy(data['target'][global_index]).cuda()))
                        visual=native_forward(sample,tt,ft);forward_samples+=4
                    eeg[domain,local_index,ti]=prediction.detach().cpu().numpy();teacher[domain,local_index,ti]=visual.cpu().numpy()
                    input_grad[domain,local_index,ti]=torch.stack((cP,cC)).detach().cpu().numpy()
                    pp=float(gP.double().square().sum());cc=float(gC.double().square().sum());dot=float((gP.double()*gC.double()).sum())
                    cosine=dot/max(np.sqrt(pp*cc),1e-24);head_alignment=dot/max(pp,1e-24)
                    if domain==0:original_state_positive_gradients[ti]=gP.detach().clone()
                    original_gradient=original_state_positive_gradients[ti];original_norm2=float(original_gradient.double().square().sum())
                    cross_dot=float((original_gradient.double()*gC.double()).sum())
                    wrong=float((values[1]-target_t[0]).square().mean());f0=visual[0].double();f1=visual[1].double()
                    target_energy=float(difference.square().mean());fdiff=f0-f1
                    metrics[domain,local_index,ti]=[float(lossP),wrong,float(lossC),float(lossR),float(lossM),ri*rj*float(lossC),pp,cc,dot,cosine,head_alignment,
                        float((f0-target_t[0]).square().mean()),float((f1-target_t[0]).square().mean()),target_energy,
                        float(((values[0]-values[1])*difference).mean()),float((fdiff*difference).mean()),float((fdiff-difference).square().mean()),float(lossR-lossM-ri*rj*lossC),
                        original_norm2,cross_dot,cross_dot/max(np.sqrt(original_norm2*cc),1e-24),cross_dot/max(original_norm2,1e-24),ri*rj]
                    if global_index in spec['full_raw_parameter_queries'] and ti in spec['full_raw_parameter_time_indices']:
                        raw_ids.append([global_index,domain,ti]);raw_parameters.append(torch.stack((gP,gC,original_gradient)).detach().cpu().numpy())
                    if cli.part=='engineering' and global_index==0 and ti==0 and domain==2:
                        originals=[p.detach().clone() for p in parameters]
                        for label,gradient in (('positive',gP),('contrast',gC)):
                            norm=float(gradient.double().norm());assert norm>1e-12;direction=gradient/norm;measured=[]
                            for sign in (-1.,1.):
                                cursor=0
                                with torch.no_grad():
                                    for parameter,original in zip(parameters,originals):
                                        n=parameter.numel();parameter.copy_(original+sign*.001*direction[cursor:cursor+n].reshape_as(parameter));cursor+=n
                                    trial=paired_forward(condition.detach(),weight,bias,sample,tt).double();forward_samples+=2
                                    loss=(trial[0]-target_t[0]).square().mean() if label=='positive' else (trial[0]-trial[1]-difference).square().mean()
                                    measured.append(float(loss))
                            with torch.no_grad():
                                for parameter,original in zip(parameters,originals):parameter.copy_(original)
                            derivative=(measured[1]-measured[0])/.002;err=abs(derivative-norm);assert err<=1e-4+.02*norm
                            FD.append(dict(loss=label,step=.001,autograd_directional=norm,central_difference=derivative,absolute_error=err))
                        del originals
                    cell_times.append(time.monotonic()-cell_started)
                    del prediction,values,residual,lossP,lossC,lossR,lossM,condition,gP,gC,cP,cC,visual,ft,target_t,difference,sample
            print(json.dumps(dict(stage=stage,query=global_index,completed=local_index+1,expected=count,seconds=time.monotonic()-started)),flush=True)
            del original_state_positive_gradients
        eeg.flush();teacher.flush();np.savez_compressed(out/'gradients_and_metrics.npz',input_gradients=input_grad,metrics=metrics,
            parameter_gradients=np.stack(raw_parameters) if raw_parameters else np.empty((0,3,525568),np.float32),parameter_query_domain_time=np.array(raw_ids,np.int64).reshape(-1,3))
        assert digest(head)==head_before
        assert {name:digest(m,True) for name,m in dict(UNet=pipe.unet,A=A,P=P).items()}==host_before
        G.checked(base['ordinary']);budget(True)
        outputs={p.name:G.ref(p) for p in out.glob('*.npy')};outputs['gradients_and_metrics.npz']=G.ref(out/'gradients_and_metrics.npz')
        complete=dict(status='COMPLETE_FROZEN_NATIVE_VALUE_FP32_EEG_OVERLAP_PARAMETER_GRADIENT_PROFILE',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),
            part=cli.part,indices=indices,protocol=G.ref(cli.protocol),source=G.ref(__file__),outputs=outputs,
            original_FP16_allfourtime_parity=parity,FP32_precision_comparison=precision,converted_loader=loading,
            parameter_names=parameter_names,parameter_count=525568,gradient_losses=['correct_EEG_noise_MSE','EEG_pair_point_target_contrast'],
            parameter_snapshot_losses=['positive_current_state','contrast_current_state','positive_original_DDPM_state'],
            metric_keys=['positive_MSE','donor_on_correct_target_MSE','contrast_MSE','weighted_R','marginal_M','conditional_C','positive_parameter_norm2','contrast_parameter_norm2','parameter_dot','parameter_cosine','contrast_projection_on_positive_gradient',
                'correct_F_MSE','donor_F_on_correct_MSE','point_target_contrast_energy','EEG_target_contrast_dot','F_target_contrast_dot','F_target_contrast_MSE','weighted_loss_identity_error',
                'original_state_positive_parameter_norm2','original_state_positive_dot_current_contrast','cross_state_parameter_cosine','current_contrast_projection_on_original_state_positive','responsibility_product'],
            actual_gradient_identities=identities,finite_differences=FD,temporary_FD_head_perturbations=4 if cli.part=='engineering' else 0,
            head_all_original_FP32_values_restored=True,host_all_original_FP16_values_exactly_preserved_in_FP32=True,host_dtype_profile='FP32_frozen_pretrained_values/no_autocast',
            original_training_BF16_or_sampling_FP16_equality_not_claimed=True,forward_sample_calls=forward_samples,backward_VJP_calls=backward_calls,
            cell_seconds_mean=float(np.mean(cell_times)),cell_seconds_max=float(max(cell_times)),seconds=time.monotonic()-started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),extra_CPU_bytes=G.rss()-base_rss,resources_before=before,resources_after=resources(spec,base,G),
            new_parameter_updates=0,new_models=0,new_images=0,optimizer_created=False,NEW_contacted=False,host_original_inference_receipt=host)
        G.write(out/'completion.json',complete);print(json.dumps(complete),flush=True)
    except BaseException as error:
        G.write(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),stage=stage,seconds=time.monotonic()-started,
            peak_GPU_bytes=torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_initialized() else None,automatic_retry=False));raise
if __name__=='__main__':main()
