"""Single-intervention native prior R4 condition-matching control; no images."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sort_endpoint import load,write,digest,unit,rank_metrics,verified_asset
from native_prior_witness import enable_native_batch_sampler,SOURCE,fp32_check
from paper_main_runtime import load_prior,add_prior_epoch_hook
from train_paper_main import seed,save,rng,finite

BASE=Path('/root/autodl-tmp/r4_prior_match_20261017_v1')
CFG=Path('/root/atm_diffusion_20261015/mced_research/sort_endpoint_config_v1.json')


def train(out,report):
    assert torch.cuda.is_available()
    cfg=load(CFG);encoder=Path(cfg['encoder']);manifest=load(encoder/'EEG_only_cache_manifest.json')
    old=Path(cfg['prior']);oldcontract=load(old/'data_contract.json');oldreport=load(old/'results.json')
    f=manifest['caches']['fit']['waveform_mean'];assert digest(f['path'])==f['sha256']
    assert manifest['encoder_sha256']==oldcontract['encoder_sha256']
    data=load(cfg['data_config']);assets=load(Path(data['cache_validation'])/'asset_manifest.json')
    assert digest(assets['features']['path'])==assets['features']['sha256']
    ids=np.asarray(manifest['caches']['fit']['image_ids']);assert len(ids)==11630
    c=torch.from_numpy(np.array(np.load(f['path']))).repeat_interleave(4,dim=0)
    h=torch.load(assets['features']['path'],map_location='cpu',weights_only=True)['img_features'][ids].float().repeat_interleave(4,dim=0)
    assert c.shape==h.shape==(46520,1024)
    contract=dict(status='completed',unique_R4_queries=11630,duplicated_rows=46520,R=4,inference_R=4,
                  duplicate_factor=4,independent_R4_queries=11630,encoder_sha256=manifest['encoder_sha256'],condition_cache=f,
                  target_hash=assets['features']['sha256'],fit_image_ids=ids.tolist(),split_sha256=oldcontract['split_sha256'],
                  baseline_R1_prior_sha256=oldreport['weight_sha256'],preprocessing=oldcontract['preprocessing'],official_test_read=False)
    write(out/'data_contract.json',contract)
    module=load_prior(SOURCE);write(out/'FP32_semantic_check.json',fp32_check(module));seed(0)
    prior=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1);pipe=module.Pipe(prior,device='cuda')
    loader=DataLoader(module.EmbeddingDataset(c_embeddings=c,h_embeddings=h),batch_size=1024,shuffle=True,num_workers=0,drop_last=False)
    assert len(loader)==46 and sum(p.numel() for p in prior.parameters())==9675648
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic();epochs=[]
    def callback(epoch,loss,optimizer,scheduler):
        finite(prior.state_dict());steps=int(optimizer.state[prior.output_layer.weight]['step']);assert steps==(epoch+1)*46
        epochs.append(dict(epoch=epoch+1,epsilon_MSE=loss,optimizer_calls=steps,seconds=time.monotonic()-tick))
        report.update(epochs=epochs);write(out/'progress.json',report)
        if (epoch+1)%10==0 or epoch==149:
            save(out/'prior_resume_last.pt',dict(model=prior.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,RNG=rng()))
    add_prior_epoch_hook(pipe,SOURCE,callback);pipe.train(loader,num_epochs=150,learning_rate=1e-3)
    save(out/'prior_final_150.pth',prior.state_dict())
    report.update(status='completed',training_seed=0,optimizer_updates=6900,parameters=9675648,epochs=epochs,
                  training_seconds=time.monotonic()-tick,weight_sha256=digest(out/'prior_final_150.pth'),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  baseline_R1_prior_sha256=oldreport['weight_sha256'],new_images=0)
    write(out/'weight_manifest.json',report)


def generate(out,report):
    assert torch.cuda.is_available();cfg=load(CFG);trained=load(BASE/'train/results.json');assert trained['status']=='completed'
    weights=BASE/'train/prior_final_150.pth';assert digest(weights)==trained['weight_sha256']
    packet=np.load(verified_asset(Path(cfg['out'])/'prepare','EEG_ONLY.npz'))
    eeg=torch.from_numpy(np.array(packet['development/e0'])).cuda();assert eeg.shape==(200,1024)
    module=load_prior(SOURCE);model=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval()
    model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True)
    pipe=module.Pipe(model,device='cuda');enable_native_batch_sampler(pipe);assert pipe.scheduler.config.clip_sample
    samples={};tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for name,c in [('paired',eeg),('permuted',eeg.roll(1,0))]:
            for s in cfg['generation_seeds']:
                x=pipe.generate(c_embeds=c,num_inference_steps=50,guidance_scale=5.,generator=torch.Generator(device='cuda').manual_seed(s))
                assert torch.isfinite(x).all();samples[name+'/'+str(s)]=x.cpu().numpy()
    np.savez_compressed(out/'R4_PRIOR_EEG_ONLY.npz',**samples)
    report.update(status='completed',actual_stochastic_vectors=1200,new_images=0,generation_seeds=cfg['generation_seeds'],
                  targets_or_gallery_read=False,weight_sha256=digest(weights),sample_sha256=digest(out/'R4_PRIOR_EEG_ONLY.npz'),
                  sampling_seconds=time.monotonic()-tick,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())


def evaluate(out,report):
    assert not torch.cuda.is_available();cfg=load(CFG);base=Path(cfg['out']);seeds=cfg['generation_seeds']
    r4=np.load(BASE/'generate/R4_PRIOR_EEG_ONLY.npz');r1=np.load(base/'generate/PRIOR_EEG_ONLY.npz')
    target=np.load(verified_asset(base/'prepare','development_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'))['raw']
    packet=np.load(verified_asset(base/'prepare','EEG_ONLY.npz'));summary={};per={};mats={}
    for method in ['R1_trained','R4_trained','a0']:
        for name in ['paired','permuted']:
            wins=[];ranks=[];cos=[];mse=[]
            for s in seeds:
                if method=='R1_trained':x=r1['development/'+name+'/'+str(s)]
                elif method=='R4_trained':x=r4[name+'/'+str(s)]
                else:x=np.roll(packet['development/e0'],1,axis=0) if name=='permuted' else packet['development/e0']
                rank,win,sim=rank_metrics(x,target);ranks.append(rank);wins.append(win);cos.append(sim);mse.append(((x-target)**2).mean(1))
            k=method+'/'+name;mats[k]=np.mean(wins,axis=0)
            per[k]=dict(Recall1=np.mean(np.asarray(ranks)==1,axis=0).tolist(),Recall5=np.mean(np.asarray(ranks)<=5,axis=0).tolist(),
                        cosine=np.mean(cos,axis=0).tolist(),MSE=np.mean(mse,axis=0).tolist(),ranks=np.asarray(ranks).tolist())
            summary[k]={m:float(np.mean(per[k][m])) for m in ['Recall1','Recall5','cosine','MSE']}
            summary[k]['SemanticFeatureCosineTwoWay']=float(mats[k].sum()/(200*199))
    sys.path.insert(0,'/root/avde_reproduction_20261016')
    from native_statistics import concept_bootstrap_counts,paired_query_interval,paired_identification_interval
    counts=concept_bootstrap_counts(200,repetitions=20000,seed=2026101714);comparisons={}
    for a,b in [('R4_trained/paired','R1_trained/paired'),('R4_trained/paired','a0/paired'),('R4_trained/paired','R4_trained/permuted')]:
        comparisons[a+'__minus__'+b]={m:paired_query_interval(per[a][m],per[b][m],counts) for m in ['Recall1','Recall5','cosine','MSE']}
        comparisons[a+'__minus__'+b]['two_way']=paired_identification_interval(mats[a],mats[b],counts)
    write(out/'per_query.json',per);np.savez_compressed(out/'identification_matrices.npz',**mats)
    report.update(status='completed',summary=summary,comparisons=comparisons,new_images=0,
                  reused_development=True,training_seed=0,generation_seeds=seeds,
                  inference_R=4,formal_confirmatory_claim=False,SOTA_established=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['train','generate','evaluate'],required=True);a=p.parse_args()
    out=BASE/a.mode;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(status='running',mode=a.mode,source_sha256=digest(__file__),official_test_read=False)
    try:globals()[a.mode](out,report)
    except BaseException as e:report.update(status='failed',error_type=type(e).__name__,message=str(e));raise
    finally:report.update(seconds=time.monotonic()-start);write(out/'results.json',report);print(report,flush=True)


if __name__=='__main__':main()
