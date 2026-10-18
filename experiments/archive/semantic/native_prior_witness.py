"""Full1024D author ATM embedding prior after the fixed encoder-objective gate.

Retains native150-epoch optimization and DDPM sampler. No image reconstruction.
"""
import argparse
import ast
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,'/root/avde_rime_20261015')
from run_fast_learning import digest,write
from paper_main_runtime import load_prior,add_prior_epoch_hook
from train_paper_main import seed,save,rng,finite

TRAIN=Path('/root/autodl-tmp/atm_native_prior_witness_seed0_20261017_v1')
GEN=Path('/root/atm_native_prior_generation_20261017_v1')
SEEDS=[2026101700,2026101701,2026101702]
SOURCE=Path('/root/atm_paper_main_20261016')


def enable_native_batch_sampler(pipe):
    """Only fix scalar timestep extraction; source expands t before .item()."""
    path=SOURCE/'Generation/diffusion_prior.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Pipe')
    original=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='generate');adapted=copy.deepcopy(original)
    calls=[n for n in ast.walk(adapted) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step']
    assert len(calls)==1 and ast.unparse(calls[0].args[1])=='t.long().item()'
    calls[0].args[1]=ast.parse('t[0].long().item()',mode='eval').body
    restored=copy.deepcopy(adapted);call=next(n for n in ast.walk(restored) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step')
    call.args[1]=ast.parse('t.long().item()',mode='eval').body
    assert ast.dump(restored,include_attributes=False)==ast.dump(original,include_attributes=False)
    scope=dict(type(pipe).generate.__globals__);exec(compile(ast.fix_missing_locations(ast.Module(body=[adapted],type_ignores=[])),str(path),'exec'),scope)
    pipe.generate=types.MethodType(scope['generate'],pipe)


def sampler_check(module,model,cond):
    from diffusers import DDIMScheduler
    with torch.no_grad():
        original=module.Pipe(model,device='cuda');batched=module.Pipe(model,device='cuda');enable_native_batch_sampler(batched)
        a=original.generate(cond[:1],generator=torch.Generator(device='cuda').manual_seed(2026101705))
        b=batched.generate(cond[:1],generator=torch.Generator(device='cuda').manual_seed(2026101705))
        assert torch.equal(a,b),'N1 native DDPM replay'
        initial=torch.randn(3,1024,device='cuda',generator=torch.Generator(device='cuda').manual_seed(2026101706))
        actual_randn=torch.randn
        def run(c,noise,batch_fix):
            p=module.Pipe(model,scheduler=DDIMScheduler(),device='cuda')
            if batch_fix:enable_native_batch_sampler(p)
            calls=[0]
            def fixed_noise(*shape,**kwargs):
                calls[0]+=1;assert calls[0]==1 and tuple(shape)==tuple(noise.shape);return noise.clone()
            try:
                torch.randn=fixed_noise
                value=p.generate(c)
                assert calls[0]==1;return value
            finally:torch.randn=actual_randn
        together=run(cond[:3],initial,True);separate=torch.cat([run(cond[i:i+1],initial[i:i+1],False) for i in range(3)])
        error=float((together-separate).abs().max());relative=float((together-separate).norm()/separate.norm())
        assert error<1e-3 and relative<1e-4
    return dict(status='completed',native_DDPM_N1_bitwise_identity=True,deterministic_DDIM_companion_batch3_vs_three_N1_max_error=error,
        companion_relative_error=relative,only_AST_change='scheduler.step timestep t.long().item() -> t[0].long().item(); native t is constant across batch')


def select_encoder():
    assert json.loads(Path('/root/encoder_objective_queue_20261017_v1/results.json').read_text())['status']=='completed'
    checked=[]
    for name in ['retrieval_unit','reconstruction_unit']:
        path=Path('/root/autodl-tmp/atm_encoder_'+name+'_seed0_20261017_v1');r=json.loads((path/'results.json').read_text());assert r['status']=='completed'
        score=r['retrieval_summary']['development/waveform_mean/paired']['Recall1'];interval=r['paired_Recall1_intervals']['development']['interval95']
        checked.append(dict(name=name,Recall1=score,pairing_interval95=interval))
        if score>=.1 and interval[0]>0:return path,checked
    raise RuntimeError('No encoder passed the predeclared engineering gate')


def fp32_check(module):
    seed(0);m=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1).eval();g=torch.Generator().manual_seed(2026101703)
    x=torch.randn(4,1024,generator=g);c=torch.randn(4,1024,generator=g);target=torch.randn(4,1024,generator=g);t=torch.tensor([0,100,500,999])
    a=m(x,t,c);(a-target).square().mean().backward();ca=torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).clone()
    m=m.cuda();m.zero_grad();b=m(x.cuda(),t.cuda(),c.cuda());(b-target.cuda()).square().mean().backward();cb=torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None]).cpu()
    error=float((a.detach()-b.detach().cpu()).abs().max());grad=float((ca-cb).norm()/ca.norm());assert error<5e-5 and grad<1e-4
    return dict(status='completed',max_output_error=error,gradient_relative_error=grad,AMP=False,TF32=False)


def train(out,report):
    assert out==TRAIN and torch.cuda.is_available()
    encoder,checked=select_encoder();manifest=json.loads((encoder/'EEG_only_cache_manifest.json').read_text());er=json.loads((encoder/'results.json').read_text())
    assert digest(encoder/'encoder_final_10.pth')==er['weights']['sha256']
    cfg=json.loads(Path('/root/atm_diffusion_20261015/mced_research/paper_development_task_v1.json').read_text());assets=json.loads((Path(cfg['cache_validation'])/'asset_manifest.json').read_text())
    assert digest(assets['features']['path'])==assets['features']['sha256']
    fit=manifest['caches']['fit'];assert digest(fit['single_trials']['path'])==fit['single_trials']['sha256']
    c=torch.from_numpy(np.array(np.load(fit['single_trials']['path'],mmap_mode='r')).reshape(-1,1024))
    ids=torch.tensor(fit['image_ids']);h=torch.load(assets['features']['path'],map_location='cpu',weights_only=True)['img_features'][ids].float().repeat_interleave(4,dim=0)
    assert c.shape==h.shape==(46520,1024) and len(torch.unique(ids//10))==1163
    module=load_prior(SOURCE);write(out/'FP32_semantic_check.json',fp32_check(module));seed(0)
    prior=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1);pipe=module.Pipe(prior,device='cuda')
    loader=DataLoader(module.EmbeddingDataset(c_embeddings=c,h_embeddings=h),batch_size=1024,shuffle=True,num_workers=0,drop_last=False);assert len(loader)==46
    contract=dict(status='completed',encoder=str(encoder),encoder_sha256=er['weights']['sha256'],encoder_selection_checked=checked,
        rule='Predeclared preference for author unit-retrieval if gate passes, else unit-reconstruction; no maximum-development-score selection',
        training_images=11630,training_concepts=1163,training_trials=46520,training_R=1,inference_R=4,
        shapes=dict(condition=[46520,1024],raw_CLIP_target=[46520,1024]),dtype='float32',condition_cache=fit['single_trials'],
        feature_source_sha256=assets['features']['sha256'],split_sha256=er['diagnostic_split_sha256'],
        input_feature_file_contains_other_rows_but_only_fitting_ids_indexed=True,calibration_and_development_not_optimized=True,
        preprocessing='Native published250 samples; upstream train-only whitening not established',official_test_read=False)
    write(out/'data_contract.json',contract);report.update(stage='native_prior_training',encoder=str(encoder),epochs=[],parameters=sum(p.numel() for p in prior.parameters()),
        epochs_planned=150,steps_per_epoch=46,training_seed=0,source_sha256=digest(SOURCE/'Generation/diffusion_prior.py'),
        objective='Unchanged native epsilon MSE/Adam1e-3/cosine500warmup/clipgrad1; native scheduler-before-optimizer order retained',
        query_targets_or_gallery_read=False)
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic()
    def callback(epoch,loss,optimizer,scheduler):
        finite(pipe.diffusion_prior.state_dict());steps=sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})
        output_step=int(optimizer.state[pipe.diffusion_prior.output_layer.weight]['step']);assert output_step==(epoch+1)*46
        row=dict(epoch=epoch+1,epsilon_MSE=loss,optimizer_calls=output_step,parameter_Adam_steps_min=min(steps),parameter_Adam_steps_max=max(steps),
                 note='Whole-batch conditioning dropout legitimately skips conditional-embedding Adam states',seconds=time.monotonic()-tick)
        report['epochs'].append(row);write(out/'progress.json',report)
        if (epoch+1)%10==0 or epoch==149:save(out/'prior_resume_last.pt',dict(model=pipe.diffusion_prior.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,RNG=rng()))
    add_prior_epoch_hook(pipe,SOURCE,callback);pipe.train(loader,num_epochs=150,learning_rate=1e-3)
    save(out/'prior_final_150.pth',pipe.diffusion_prior.state_dict())
    report.update(status='completed',stage='full1024_native_prior150_completed',optimizer_updates=6900,training_seconds=time.monotonic()-tick,
        weight_sha256=digest(out/'prior_final_150.pth'),resume_sha256=digest(out/'prior_resume_last.pt'),data_contract_sha256=digest(out/'data_contract.json'),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),actual_images_generated=0)
    write(out/'weight_manifest.json',report)


def generate(out,report):
    assert out==GEN and torch.cuda.is_available();r=json.loads((TRAIN/'results.json').read_text());assert r['status']=='completed'
    assert digest(TRAIN/'prior_final_150.pth')==r['weight_sha256'];contract=json.loads((TRAIN/'data_contract.json').read_text());encoder=Path(contract['encoder'])
    manifest=json.loads((encoder/'EEG_only_cache_manifest.json').read_text());query=manifest['caches']['development'];assert manifest['R']==4 and len(query['image_ids'])==200
    assert digest(query['waveform_mean']['path'])==query['waveform_mean']['sha256']
    cond=torch.from_numpy(np.array(np.load(query['waveform_mean']['path']))).cuda();assert cond.shape==(200,1024)
    module=load_prior(SOURCE);model=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1);model.load_state_dict(torch.load(TRAIN/'prior_final_150.pth',map_location='cpu',weights_only=True));model=model.cuda().eval().requires_grad_(False)
    write(out/'native_sampler_batch_semantics.json',sampler_check(module,model,cond))
    samples={};sampling_start=time.monotonic()
    with torch.no_grad():
        for clipped in [True,False]:
            pipe=module.Pipe(model,device='cuda');enable_native_batch_sampler(pipe);pipe.scheduler.register_to_config(clip_sample=clipped);name='native_clip' if clipped else 'no_clip'
            for guidance,condition in [(0.,'prior'),(1.,'paired'),(1.,'permuted'),(5.,'paired'),(5.,'permuted')]:
                c=cond.roll(1,0) if condition=='permuted' else cond if condition=='paired' else torch.zeros_like(cond)
                for s in SEEDS:
                    rng_gen=torch.Generator(device='cuda').manual_seed(s)
                    value=pipe.generate(c_embeds=c,num_inference_steps=50,guidance_scale=guidance,generator=rng_gen)
                    assert value.shape==(200,1024) and torch.isfinite(value).all();samples[name+'/cfg'+str(int(guidance))+'/'+condition+'/'+str(s)]=value.cpu().numpy()
        for condition,value in [('paired',cond),('permuted',cond.roll(1,0))]:
            for s in SEEDS:samples['direct/none/'+condition+'/'+str(s)]=value.cpu().numpy()
    np.savez_compressed(out/'EEG_only_full1024_semantic_samples.npz',**samples)
    report.update(status='completed',query_count=200,R=4,generation_seeds=SEEDS,native_DDPM_steps=50,
        configurations='Native DDPM with N1-verified scalar timestep batching fix; clipTrue / same weights clipFalse; fixed CFG0,1,5; not topK or best-of',
        source_sampler_sha256=digest(SOURCE/'Generation/diffusion_prior.py'),training_weight_sha256=r['weight_sha256'],
        sampling_seconds=time.monotonic()-sampling_start,actual_images_generated=0,stochastic_semantic_vectors=6000,
        distinct_deterministic_encoder_vectors=400,direct_arrays_repeated_across_seeds_are_aliases=True,
        null_implemented='Guidance0 with dummy200-row condition fixes native N=1 default while UNet receives c=None',
        targets_gallery_captions_image_tokens_read=False,original_VAR_risk_bound_claimed=False,
        sample_sha256=digest(out/'EEG_only_full1024_semantic_samples.npz'))


def evaluate(out,report):
    assert not torch.cuda.is_available();gen=json.loads((GEN/'results.json').read_text());assert gen['status']=='completed'
    contract=json.loads((TRAIN/'data_contract.json').read_text());encoder=Path(contract['encoder']);manifest=json.loads((encoder/'EEG_only_cache_manifest.json').read_text());ids=manifest['caches']['development']['image_ids']
    cfg=json.loads(Path('/root/atm_diffusion_20261015/mced_research/paper_development_task_v1.json').read_text());assets=json.loads((Path(cfg['cache_validation'])/'asset_manifest.json').read_text())
    target=torch.load(assets['features']['path'],map_location='cpu',weights_only=True)['img_features'][ids].float().numpy();norm=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    samples=np.load(GEN/'EEG_only_full1024_semantic_samples.npz');summary={};per={};matrices={}
    for name in sorted({k.rsplit('/',1)[0] for k in samples.files}):
        values=[samples[name+'/'+str(s)] for s in SEEDS];wins=[];cos=[];ranks=[];mse=[]
        for value in values:
            sim=norm(target).astype(np.float64)@norm(value).astype(np.float64).T;win=(np.diag(sim)[None,:]>sim).astype(float);np.fill_diagonal(win,0)
            wins.append(win);cos.append(np.diag(sim));ranks.append((sim>=np.diag(sim)[None,:]).sum(0));mse.append(((value-target)**2).mean(1))
        matrices[name]=np.mean(wins,0);per[name]=dict(cosine=np.mean(cos,0).tolist(),MSE=np.mean(mse,0).tolist(),Recall1=np.mean(np.array(ranks)==1,0).tolist())
        diversity=float(np.mean([np.mean((values[i]-values[j])**2) for i in range(3) for j in range(i+1,3)]))
        summary[name]=dict(GeneratedSemanticFeatureCosineTwoWay=float(matrices[name].sum()/(200*199)),Recall1=float((np.array(ranks)==1).mean()),
            Recall5=float((np.array(ranks)<=5).mean()),matched_cosine=float(np.mean(cos)),raw_feature_MSE=float(np.mean(mse)),
            mean_output_L2=float(np.mean([np.linalg.norm(v,axis=1).mean() for v in values])),three_seed_latent_diversity=diversity,NLL=None)
    sys.path.insert(0,'/root/avde_reproduction_20261016');from native_statistics import concept_bootstrap_counts,paired_identification_interval,paired_query_interval
    counts=concept_bootstrap_counts(200,repetitions=20000,seed=2026101704);comparisons={}
    contrasts=[('native_clip/cfg5/paired','native_clip/cfg0/prior'),('native_clip/cfg5/paired','native_clip/cfg5/permuted'),
        ('no_clip/cfg5/paired','native_clip/cfg5/paired'),('native_clip/cfg1/paired','native_clip/cfg5/paired'),
        ('no_clip/cfg5/paired','direct/none/paired'),('native_clip/cfg5/paired','direct/none/paired')]
    for a,b in contrasts:
        value=dict(identification=paired_identification_interval(matrices[a],matrices[b],counts))
        for metric in per[a]:value[metric]=paired_query_interval(per[a][metric],per[b][metric],counts)
        comparisons[a+'__minus__'+b]=value
    np.savez_compressed(out/'identification_matrices.npz',**matrices);np.savez_compressed(out/'evaluation_targets_only.npz',target_raw=target,image_ids=np.array(ids));write(out/'per_query_metrics.json',per)
    report.update(status='completed',summary=summary,comparisons=comparisons,training_seed=0,training_optimizer_updates=6900,
        generation_seeds=SEEDS,actual_images_generated=0,
        caveats=['Full1024 rawCLIP semantic features, not generated images or Inception','CLIP is supervision, not independent image evaluation','One subject/one training seed/reused200 development concepts; descriptive intervals','Not full paper reconstruction: encoder10epochs and1163 concepts; native prior150epochs alone is reproduced','No tractable likelihood or original VAR token risk bound claimed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','generate','evaluate'],required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(status='running',mode=a.mode,implementation_sha256=digest(__file__),official_test_read=False)
    write(out/'preregistration.json',dict(**report,UTC=datetime.now(timezone.utc).isoformat(),native_prior_epochs=150,training_seed=0,generation_seeds=SEEDS,
        inference='ExactlyR4; all configurations reported; no development configuration selection or image quality claim'))
    try:globals()[a.mode](out,report)
    except BaseException as e:report.update(status='failed',error_type=type(e).__name__,message=str(e));raise
    finally:report.update(seconds=time.monotonic()-start);write(out/'results.json',report);write(out/'progress.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
