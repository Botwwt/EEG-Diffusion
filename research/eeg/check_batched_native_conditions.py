"""Real native vectors and UNet: prove query isolation after fixing embedding repetition."""
from profile_standard_generator import *
from diffusers import StableDiffusionXLPipeline
from diffusers.utils.torch_utils import randn_tensor


def main():
    started=time.monotonic()
    rows=[r['array_row'] for r in json.loads((ROOT/'audit/budget_sampler_profile.json').read_text())['files']]
    f=torch.load('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2/train_image_embeds.pt',weights_only=True,mmap=True)
    native=f[rows].clone();packed=torch.cat([torch.zeros_like(native),native])[:,None,:]
    legacy=StableDiffusionXLPipeline.prepare_ip_adapter_image_embeds(None,None,[packed],'cpu',4,True)[0]
    assert legacy.shape==(32,1,1024)
    # The original attention processor views the repeated leading dimension in
    # the UNet's eight-query CFG batch, making four native conditions per query.
    original_query_conditions=legacy.reshape(8,4,1024)[4:]
    assert all(torch.equal(original_query_conditions[0],x) for x in original_query_conditions[1:])
    pipe,infos,receipt,(prompt,negative,pool,negative_pool)=load_pipeline();pipe.set_progress_bar_config(disable=True)
    captured=[];projection_shapes=[]
    def record(module,args,kwargs,output):
        if not captured:captured.append(output[0].detach().float().cpu())
    def project_record(module,args):projection_shapes.append(list(args[0][0].shape))
    hook=pipe.unet.register_forward_hook(record,with_kwargs=True)
    phook=pipe.unet.encoder_hid_proj.register_forward_pre_hook(project_record)
    seeds=[1729+i for i in range(4)]
    @torch.inference_mode()
    def run(value,indices):
        captured.clear();n=len(value);c=value.to(device='cuda',dtype=torch.bfloat16)
        generators=[torch.Generator(device='cuda').manual_seed(seeds[i]) for i in indices]
        noise=randn_tensor((n,4,64,64),generator=generators,device='cuda',dtype=torch.bfloat16)
        pipe(prompt_embeds=prompt.repeat(n,1,1),negative_prompt_embeds=negative.repeat(n,1,1),
            pooled_prompt_embeds=pool.repeat(n,1),negative_pooled_prompt_embeds=negative_pool.repeat(n,1),
            ip_adapter_image_embeds=[torch.cat([torch.zeros_like(c),c])[:,None,:]],height=512,width=512,
            num_inference_steps=2,guidance_scale=5.,latents=noise,output_type='latent')
        return captured[0].clone()
    baseline=run(native,list(range(4)))
    changed=native.clone();changed[0]=0;intervention=run(changed,list(range(4)))
    unaffected=list(range(4))+[5,6,7]
    unaffected_delta=float((baseline[unaffected]-intervention[unaffected]).abs().max())
    changed_delta=float((baseline[4]-intervention[4]).abs().max())
    assert unaffected_delta==0 and changed_delta>0,(unaffected_delta,changed_delta)
    permuted=run(native[[1,0,2,3]],list(range(4)))
    perm_unaffected=float((baseline[[0,1,2,3,6,7]]-permuted[[0,1,2,3,6,7]]).abs().max())
    assert perm_unaffected==0
    single=run(native[:1],[0])
    relative_RMSE=float((single[1]-baseline[4]).square().mean().sqrt()/baseline[4].square().mean().sqrt())
    assert relative_RMSE<.02,'Batch/single first epsilon differs beyond predeclared bf16 numeric tolerance'
    hook.remove();phook.remove()
    report=dict(status='PASSED_ACTUAL_UNET_QUERY_ISOLATION',legacy_embedding_shape=list(legacy.shape),
        expected_CFG_batch=8,legacy_positive_queries_received_identical_four_condition_sets=True,
        corrected_projection_input_shapes=projection_shapes,unaffected_query_max_abs_delta=unaffected_delta,
        changed_query_max_abs_delta=changed_delta,permutation_unaffected_query_max_abs_delta=perm_unaffected,
        batch_vs_single_first_epsilon_relative_RMSE=relative_RMSE,rows=rows,
        old_partial_formal_images_valid_for_query_content=False,training_checkpoints_affected=False,
        historical_single_query_image_engine_affected=False,images_per_query=1,seed_search=False,
        seconds=time.monotonic()-started)
    (ROOT/'audit/batched_native_condition_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
