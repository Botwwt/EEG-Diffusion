"""Read-only scheduler instrumentation; replay identity is mandatory."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from sort_endpoint import load,write,digest,verified_asset,rank_metrics
from paper_main_runtime import load_prior
from native_prior_witness import SOURCE,enable_native_batch_sampler

BASE=Path('/root/autodl-tmp/native_prior_trace_20261017_v1')
CFG=Path('/root/atm_diffusion_20261015/mced_research/sort_endpoint_config_v1.json')
POINTS=[0,9,19,29,39,49]


def generate(out,report):
    assert torch.cuda.is_available();cfg=load(CFG);sort=Path(cfg['out'])
    packet=np.load(verified_asset(sort/'prepare','EEG_ONLY.npz'));c=torch.from_numpy(np.array(packet['development/e0'])).cuda()
    original=np.load(sort/'generate/PRIOR_EEG_ONLY.npz');r4=Path('/root/autodl-tmp/r4_prior_match_20261017_v1')
    matched=np.load(r4/'generate/R4_PRIOR_EEG_ONLY.npz');stored={};meta={}
    for method,directory in [('R1',Path(cfg['prior'])),('R4',r4/'train')]:
        trained=load(directory/'results.json');weights=directory/'prior_final_150.pth';assert digest(weights)==trained['weight_sha256']
        module=load_prior(SOURCE);model=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval()
        model.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True));pipe=module.Pipe(model,device='cuda');enable_native_batch_sampler(pipe)
        assert pipe.scheduler.config.prediction_type=='epsilon' and pipe.scheduler.config.clip_sample and not pipe.scheduler.config.thresholding
        step=pipe.scheduler.step
        for seed in cfg['generation_seeds']:
            records=[];seen=[]
            def capture(eps,t,x,*args,**kwargs):
                alpha=pipe.scheduler.alphas_cumprod[t].to(x.device);beta=1-alpha
                pre=(x-beta**.5*eps)/alpha**.5;result=step(eps,t,x,*args,**kwargs)
                assert torch.allclose(result.pred_original_sample,pre.clamp(-1,1),rtol=1e-5,atol=1e-6)
                i=len(records);records.append(dict(step=i+1,timestep=int(t),fraction_outside_clip=float((pre.abs()>1).float().mean())))
                if i in POINTS:
                    stored[method+'/'+str(seed)+'/'+str(i+1)+'/pre_clip']=pre.cpu().numpy()
                    stored[method+'/'+str(seed)+'/'+str(i+1)+'/post_clip']=result.pred_original_sample.cpu().numpy()
                return result
            pipe.scheduler.step=capture
            with torch.no_grad():x=pipe.generate(c_embeds=c,num_inference_steps=50,guidance_scale=5.,generator=torch.Generator(device='cuda').manual_seed(seed))
            pipe.scheduler.step=step;assert len(records)==50
            ref=original['development/paired/'+str(seed)] if method=='R1' else matched['paired/'+str(seed)]
            assert np.array_equal(x.cpu().numpy(),ref),'Instrumentation changed original final output'
            meta[method+'/'+str(seed)]=dict(final_bitwise_identity=True,steps=records,weight_sha256=digest(weights))
        del model,pipe;torch.cuda.empty_cache()
    np.savez_compressed(out/'CLEAN_ESTIMATES_EEG_ONLY.npz',**stored);write(out/'trajectory.json',meta)
    report.update(status='completed',targets_or_gallery_read=False,replayed_vectors=1200,unique_new_samples=0,new_images=0,
                  all_final_outputs_bitwise_identity=True,stored_tensor_bytes=sum(x.nbytes for x in stored.values()),
                  source_trajectory_sha256=digest(out/'CLEAN_ESTIMATES_EEG_ONLY.npz'))


def evaluate(out,report):
    assert not torch.cuda.is_available();cfg=load(CFG);base=Path(cfg['out'])/'prepare'
    y=np.load(verified_asset(base,'development_targets_EVALUATION_OR_CALIBRATION_ONLY.npz'))['raw']
    vectors=np.load(BASE/'generate/CLEAN_ESTIMATES_EEG_ONLY.npz');assert digest(BASE/'generate/CLEAN_ESTIMATES_EEG_ONLY.npz')==load(BASE/'generate/results.json')['source_trajectory_sha256']
    traces=load(BASE/'generate/trajectory.json');summary={};per={}
    for method in ['R1','R4']:
        for i in POINTS:
            for variant in ['pre_clip','post_clip']:
                values=[vectors[method+'/'+str(seed)+'/'+str(i+1)+'/'+variant] for seed in cfg['generation_seeds']]
                results=[rank_metrics(x,y) for x in values];rr=np.array([r[0] for r in results]);w=np.mean([r[1] for r in results],axis=0)
                key=method+'/'+str(i+1)+'/'+variant
                summary[key]=dict(Recall1=float((rr==1).mean()),Recall5=float((rr<=5).mean()),semantic_two_way=float(w.sum()/(200*199)),
                     cosine=float(np.mean([r[2] for r in results])),raw_MSE=float(np.mean([(x-y)**2 for x in values])),
                     output_L2=float(np.mean([np.linalg.norm(x,axis=1) for x in values])),
                     timestep=traces[method+'/'+str(cfg['generation_seeds'][0])]['steps'][i]['timestep'])
                per[key]=dict(ranks=rr.tolist(),Recall1=(rr==1).mean(0).tolist())
    write(out/'summary.json',summary);write(out/'per_query.json',per)
    report.update(status='completed',summary=summary,target_fraction_outside_clip=float((abs(y)>1).mean()),
                  target_mean_L2=float(np.linalg.norm(y,axis=1).mean()),
                  single_subject=True,training_seeds_per_model=1,official_test_read=False,new_images=0,
                  caveat='Observational generated-trajectory diagnostics; no best-timestep selection, no likelihood claim, no independent image evaluation')


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['generate','evaluate'],required=True);a=p.parse_args()
    out=BASE/a.mode;out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report=dict(status='running',mode=a.mode,source_sha256=digest(__file__),official_test_read=False)
    try:globals()[a.mode](out,report)
    except BaseException as e:report.update(status='failed',error=repr(e));raise
    finally:report.update(seconds=time.monotonic()-tick);write(out/'results.json',report)


if __name__=='__main__':main()
