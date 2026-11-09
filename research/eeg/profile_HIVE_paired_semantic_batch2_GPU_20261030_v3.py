"""All-five-arm actual batch2/Adam engineering; no production weights."""
import argparse,datetime,gc,hashlib,io,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
def read(p):return json.loads(Path(p).read_bytes())
def ref(p):
    p=Path(p)
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def main(path):
    s=read(path);assert s['status']=='REGISTERED_ALL5_BATCH2_EEG_P_ADAM_ENGINEERING' and s['host']=='old'
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    for pin in s['source_pins']+s['payloads']:assert ref(pin['path'])==pin
    out=Path(s['output_root']);out.mkdir();start=time.monotonic();memory=rss();phase='IMPORT'
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent));sys.path.insert(1,s['original_profile_source_root'])
    import numpy as np,torch
    import brainhive_paired_semantic_runtime_20261030_v2 as U
    from brainhive_paired_semantic_images_streamed_20261030_v3 import image_features
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():assert time.monotonic()-start<800 and rss()-memory<s['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes'] and shutil.disk_usage(out).free>4<<30
    try:
        phase='LOAD';context=U.load_context(s,out);before={n:U.fingerprint(m) for n,m in context['frozen'].items()}
        cache=read(s['cache_completion']['path']);EEG=np.load(cache['EEG_mean4']['path'],allow_pickle=False,mmap_mode='r');V=np.load(cache['target_features']['path'],allow_pickle=False,mmap_mode='r');pairs=read(cache['pairs']['path'])
        records=[];stored={};isolation=None;state_roundtrip=False
        for arm in U.ARMS:
            phase=arm;U.reset_initial(context);opt=U.optimizer(context);row=[]
            for step in range(2):
                budget();pair=pairs[step];ids=pair['T_indices'];x=torch.from_numpy(EEG[ids].copy()).cuda();target=torch.from_numpy(V[ids].copy()).cuda();tick=time.monotonic()
                tk,_,_=U.tokens(context,x)
                if step==0:
                    baseline,proof0=U.sample_pair(context,tk.detach(),1729,gradient=False)
                latent,proof=U.sample_pair(context,tk,1729+step,gradient=True)
                if step==0:assert torch.equal(baseline,latent.detach()) and proof0==proof
                feature=image_features(context,latent,gradient=True)
                if arm==U.ARMS[0] and step==0:
                    cross=torch.autograd.grad((feature[0]*target[0]).sum(),tk,retain_graph=True)[0]
                    assert cross[0].norm()>0 and int(cross[1].count_nonzero())==0
                    isolation=dict(own_token_gradient_norm=float(cross[0].norm()),other_query_token_nonzero=0)
                    stored['token_isolation_gradient']=cross.detach().cpu().numpy();del cross
                loss,terms=U.objective(context,feature,target,arm);loss.backward();grad=U.step(context,opt,arm)
                budget();row.append(dict(step=step,pair_index=pair['pair_index'],loss=float(loss.detach()),absolute=float(terms['absolute'].detach()),pairing=float(terms['pairing'].detach()),effect=float(terms['effect'].detach()),gradients=grad,seconds=time.monotonic()-tick,noise_and_steps=proof))
                stored[arm+f'_step{step}_features']=feature.detach().cpu().numpy();stored[arm+f'_step{step}_targets']=target.detach().cpu().numpy()
                del loss,terms,feature,latent,tk,x,target;gc.collect()
            if arm==U.ARMS[0]:
                order=np.arange(6615,dtype=np.int64);state=U.snapshot(context,opt,arm=arm,updates=2,order=order,registration_sha256=ref(path)['sha256'],log_sha256='0'*64)
                stream=io.BytesIO();torch.save(state,stream);stream.seek(0);restored=torch.load(stream,map_location='cpu',weights_only=False)
                assert U.restore(context,opt,restored,arm=arm,order=order,registration_sha256=ref(path)['sha256'])==2
                assert torch.equal(torch.get_rng_state(),state['CPU_RNG']) and all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['CUDA_RNG']))
                state_roundtrip=True;del state,restored,stream
            if arm=='P_only_correct':assert all(torch.equal(p.detach().cpu(),context['initial_E'][n]) for n,p in context['model'].named_parameters())
            if arm=='EEG_only_correct':assert all(torch.equal(context['master'][n],p.float()) for n,p in context['P'].named_parameters())
            records.append(dict(arm=arm,steps=row));del opt;gc.collect();torch.cuda.empty_cache()
        assert before=={n:U.fingerprint(m) for n,m in context['frozen'].items()}
        for pin in s['payloads']:assert ref(pin['path'])==pin
        np.savez_compressed(out/'actual_batch2.npz',**stored);budget()
        result=dict(status='COMPLETE_ALL5_NATIVE_BATCH2_PAIRED_SEMANTIC_ADAM_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(path),records=records,
            temporary_updates=10,all_5_control_branches_checked=True,all4_step_latents_including_final_parity=True,batch_query_isolation=isolation,
            actual_model_optimizer_CPU_CUDA_RNG_serialization_restore=state_roundtrip,future_GPU_bit_exact_continuation_NOT_claimed=True,
            production_models_saved=0,new_research_images=0,full_fit_admitted=False,ordinary_unchanged=True,NEW_contacted=False,
            program_seconds=time.monotonic()-start,extra_CPU_peak=rss()-memory,peak_GPU_bytes=torch.cuda.max_memory_allocated())
        write(out/'completion.json',result);print(json.dumps(result))
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();main(a.protocol)
