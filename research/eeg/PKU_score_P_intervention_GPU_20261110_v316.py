"""All fixed images, frozen evaluators, bounded metric prefixes, no method/epoch selection."""
import argparse,hashlib,json,os,socket,subprocess,sys,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);ap.add_argument('--metric',required=True);args=ap.parse_args();assert 1<=args.stage<=32
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_P_intervention_scores_v316';start=time.monotonic();jid=os.environ['SLURM_JOB_ID'];sys.path.insert(0,str(R/'package_cache/evaluation_runtime_v275'))
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
try:
    import numpy as np
    import torch
    from scipy.spatial.distance import correlation
    from PKU_EEG_evaluation_core_20261109_v279 import METRICS,sha,load_model,features,pixels,ranked,paper_source_scope
    assert args.metric in METRICS and torch.cuda.device_count()==1;torch.set_num_threads(4);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    policy_path=R/'code/EEG_P_intervention_scoring_policy_v316.json';policy=json.loads(policy_path.read_text());assert policy['metrics_in_fixed_order']==METRICS
    ready=json.loads((out/'readiness.json').read_text());assert ready['status']=='PASS_ALL3320_IMAGES_830_TARGETS_AND_ORIGINAL_RANK_FIXTURES'
    assert sha(out/'gallery.json')==ready['gallery_sha256'] and sha(out/'all_image_records.json')==ready['all_records_sha256']
    gallery=json.loads((out/'gallery.json').read_text())['rows'];records=json.loads((out/'all_image_records.json').read_text());arms=records['arms'];records=records['records'];assert len(arms)==4 and len(gallery)==830 and len(records)==3320
    paths=[r['path'] for r in gallery];concepts=[r['concept_id'] for r in gallery];stages=out/'stages';stages.mkdir(exist_ok=True);folder=out/args.metric;folder.mkdir(exist_ok=True)
    assert not (stages/('stage_'+str(args.stage)+'.json')).exists()
    if args.stage>1:
        previous=json.loads((stages/('stage_'+str(args.stage-1)+'.json')).read_text());assert previous['next_job_id']==jid and previous['next_metric']==args.metric
    if args.metric=='low':scope=paper_source_scope();model=None;real=None
    else:
        model=load_model(args.metric);real=torch.cat([features(paths[lo:lo+16],args.metric,model,real=True).cpu() for lo in range(0,830,16)]).cuda()
        assert torch.isfinite(real).all()
    torch.cuda.reset_peak_memory_stats();new_arms=0;completed=[]
    for ai,arm in enumerate(arms):
        target=folder/(arm+'.pt');receipt=folder/(arm+'.json')
        if receipt.exists():
            proof=json.loads(receipt.read_text());assert sha(target)==proof['data_sha256'] and proof['policy_sha256']==sha(policy_path);completed.append(arm);continue
        assert not target.exists()
        if new_arms and time.monotonic()-start>700:break
        rows=[records[qi*4+ai] for qi in range(830)];fake_paths=[v['image_path'] for v in rows]
        # The shared full audit already decoded every image. Recheck byte identity at actual evaluator access.
        for r in rows:assert sha(r['image_path'])==r['file_sha256']
        data={};matrices=[];checks=[]
        if args.metric=='low':
            values=dict(PixCorr425=[],SSIM425=[])
            with torch.inference_mode():
                for qi in range(830):
                    actual=pixels(paths[qi])[None].cuda();generated=pixels(fake_paths[qi])[None].cuda()
                    values['PixCorr425'].append(float(scope['pixcorr'](actual,generated)));values['SSIM425'].append(float(scope['ssim'](actual,generated)))
            data={k:torch.tensor(v,dtype=torch.float64) for k,v in values.items()}
        else:
            fields={}
            if args.metric in ['Alex2','Alex5','Inception','CLIP']:
                predicted=torch.cat([features(fake_paths[lo:lo+16],args.metric,model,real=False).cpu() for lo in range(0,830,16)])
                v,scores=ranked(predicted,real.cpu(),range(830),concepts,'paper',return_scores=True);matrices.append(scores)
                for k,value in v.items():fields[k]=[value]
                if ai==0:
                    original=paper_source_scope();fn=lambda x:x.flatten(1)
                    counts,den=original['two_way_identification'](predicted[:,None,None,:],real.cpu()[:,None,None,:],fn,lambda x:x,return_avg=False)
                    error=float(np.abs(np.asarray(counts)/den-v['two_afc'].numpy()).max());assert error<1e-7
                    checks.append(dict(all830_author_function_rank_max_abs=error,full_NumPy_gallery_correlation=True))
                del predicted
            for lo in range(0,830,16) if args.metric in ['DINO','EffNet','SwAV'] else []:
                q=features(fake_paths[lo:lo+16],args.metric,model,real=False);n=len(q)
                if args.metric in ['EffNet','SwAV']:
                    r=real[lo:lo+n].float().cpu().numpy();x=q.float().cpu().numpy();v=torch.tensor([correlation(a,b) for a,b in zip(r,x)],dtype=torch.float64)
                    fields.setdefault('distance',[]).append(v)
                else:
                    v,scores=ranked(q,real,range(lo,lo+n),concepts,args.metric,return_scores=True);matrices.append(scores)
                    for k,value in v.items():fields.setdefault(k,[]).append(value)
                    if lo==0 and ai==0:
                        if args.metric=='DINO':
                            a=torch.nn.functional.normalize(q.float(),dim=-1);b=torch.nn.functional.normalize(real.float(),dim=-1);cpu=(a.cpu().numpy()@b.cpu().numpy().T);error=float(np.abs(cpu-scores.numpy()).max());assert error<3e-6
                        else:
                            # Independent NumPy Pearson on fixed4 generated queries versus the full real gallery.
                            r=real.float().cpu().numpy();x=q[:4].float().cpu().numpy();cpu=np.corrcoef(r,x)[:830,830:].T;error=float(np.abs(cpu-scores[:4].numpy()).max());assert error<1e-10
                        checks.append(dict(first_batch_CPU_gallery_similarity_max_abs=error,queries=4 if args.metric!='DINO' else n))
            data={k:torch.cat(v) for k,v in fields.items()}
            if matrices:data['full_gallery_scores']=torch.cat(matrices);assert data['full_gallery_scores'].shape==(830,830)
        assert all(torch.isfinite(v).all() for v in data.values()) and all(len(v)==830 for v in data.values())
        payload=dict(metric=args.metric,arm=arm,fields=data,array_rows=[r['array_row'] for r in gallery],stimulus_ids=[r['stimulus_id'] for r in gallery],policy_sha256=sha(policy_path),readiness_sha256=sha(out/'readiness.json'),batch_feature_route='Frozen eval, CPU PIL inputs, GPU author Tensor resize; DINO historical realCPU/generatedCUDA resize')
        temp=target.with_name(target.name+'.pending')
        with temp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        temp.replace(target)
        proof=dict(status='COMPLETE_ALL830_FIXED_IMAGES_ONE_ARM_ONE_METRIC',metric=args.metric,arm=arm,queries=830,data_sha256=sha(target),policy_sha256=sha(policy_path),checks=checks,job_id=jid,source_sha256=sha(__file__))
        write(receipt,proof);new_arms+=1;completed.append(arm);write(out/'progress.json',dict(stage=args.stage,metric=args.metric,completed_arms=completed,job_id=jid,seconds=time.monotonic()-start))
        assert torch.cuda.max_memory_allocated()<20<<30 and sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<2<<30
    assert new_arms>0;complete=len(completed)==4;last=complete and args.metric==METRICS[-1]
    stage=dict(stage=args.stage,metric=args.metric,job_id=jid,new_arms=new_arms,complete_arms=completed,metric_complete=complete,seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    if last:
        stage.update(status='COMPLETE_DINO_ALL7_PAPER_METRICS_AND_EFFNET_SEED_STILL_PENDING',SEED_caption_and_detector_complete=False,new_method_effect_established=False);write(out/'scoring_completion.json',stage)
        p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+jid,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_analyze_P_intervention_CPU_20261110_v316.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
        analysis=p.stdout.strip().split(';')[0];assert p.returncode==0 and analysis.isdigit(),(p.returncode,p.stdout,p.stderr);stage['analysis_job']=analysis
    elif args.stage==32:
        stage['status']='FINITE_32_STAGE_CAP_SAFE_METRIC_PREFIX';write(out/'cap_stop.json',stage)
    else:
        metric=METRICS[METRICS.index(args.metric)+1] if complete else args.metric
        write(stages/('dispatch_after_'+str(args.stage)+'.json'),dict(parent_job=jid,next_stage=args.stage+1,next_metric=metric))
        p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+jid,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_score_P_intervention_GPU_20261110_v316.sbatch'),str(args.stage+1),metric],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
        next_job=p.stdout.strip().split(';')[0];assert p.returncode==0 and next_job.isdigit(),(p.returncode,p.stdout,p.stderr);stage.update(status='METRIC_PREFIX_COMPLETE_NEXT_SUBMITTED',next_job_id=next_job,next_metric=metric)
    write(stages/('stage_'+str(args.stage)+'.json'),stage);write(out/'progress.json',stage);print(json.dumps(stage),flush=True)
except BaseException as e:
    if out.exists():write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED',stage=args.stage,metric=args.metric,error=repr(e),traceback=traceback.format_exc()))
    raise
