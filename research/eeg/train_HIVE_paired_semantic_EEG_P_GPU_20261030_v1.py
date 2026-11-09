"""Full fixed-T native EEG/P coadaptation with immutable resumable prefixes."""
import argparse,datetime,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,atomic,checked,read,ref,scan_checkpoints,publish_checkpoint
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def main(path,arm,segment):
    s=read(path);assert s['status']=='ADMITTED_FULL_T_EEG_P_PAIRED_SEMANTIC_STUDY_V1' and s['host']=='old' and arm in ARMS
    assert s['arms']==list(ARMS) and s['training_seed']==1729
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    for pin in s['source_pins']+s['payloads']:checked(pin)
    root=Path(s['output_root']);root.mkdir(exist_ok=True);folder=root/arm;folder.mkdir(exist_ok=True);(folder/'segments').mkdir(exist_ok=True)
    attempt=folder/'segments'/f'{segment:03d}';assert not attempt.exists();attempt.mkdir();start=time.monotonic();memory=rss();phase='IMPORT'
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    sys.path.insert(0,str(Path(path).parent))
    import numpy as np,torch
    import brainhive_paired_semantic_runtime_20261030_v2 as U
    from brainhive_paired_semantic_images_streamed_20261030_v3 import image_features
    assert ARMS==U.ARMS
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():
        assert time.monotonic()-start<820 and rss()-memory<s['maximum_extra_CPU_bytes'] and torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes']
        assert shutil.disk_usage(root).free>s['reserved_legacy_full1654_bytes']+s['disk_buffer_bytes']
    try:
        phase='LOAD_ACTUAL_MODEL_AND_LAST_COMMIT'
        cache=read(checked(s['cache_completion']));EEG=np.load(checked(cache['EEG_mean4']),mmap_mode='r',allow_pickle=False);V=np.load(checked(cache['target_features']),mmap_mode='r',allow_pickle=False)
        pairs=read(checked(cache['pairs']));order=np.load(checked(s['training_order']),allow_pickle=False)
        assert EEG.shape==(13230,17,250) and V.shape==(13230,768) and sorted(order.tolist())==list(range(6615))
        committed=scan_checkpoints(folder,path,arm,order,pairs);previous=committed[-1] if committed else None
        context=U.load_context(s,attempt);opt=U.optimizer(context,lr=s['optimizer']['lr']);updates=0
        if previous:
            state=torch.load(Path(previous['directory'])/'state.pt',map_location='cpu',weights_only=False)
            assert state['log_sha256']==previous['log_sha256']
            updates=U.restore(context,opt,state,arm=arm,order=order,registration_sha256=ref(path)['sha256']);del state
        original_versions={n:[p._version for p in m.parameters()] for n,m in context['frozen'].items()}
        initial_affine=[context['LN'].weight.detach().clone(),context['LN'].bias.detach().clone()]
        start_update=updates;pending=[];features=[];loop_start=time.monotonic();budget()
        phase='FULL_T_PAIR_UPDATES'
        while updates<6615 and updates-start_update<s['maximum_updates_per_segment']:
            budget()
            if time.monotonic()-start>s['stop_work_after_seconds'] and updates>start_update:break
            index=int(order[updates]);pair=pairs[index];ids=pair['T_indices'];seed=int(hashlib.sha256(f'1729|{index}|fixed_pair_semantic_v1'.encode()).hexdigest()[:16],16)%(2**63-1)
            x=torch.from_numpy(EEG[ids].copy()).cuda();target=torch.from_numpy(V[ids].copy()).cuda();tick=time.monotonic()
            tk,c,corrected=U.tokens(context,x);latent,proof=U.sample_pair(context,tk,seed,gradient=True)
            feature=image_features(context,latent,gradient=True)
            loss,terms=U.objective(context,feature,target,arm);loss.backward();gradients=U.step(context,opt,arm)
            updates+=1
            pending.append(dict(update=updates,pair_index=index,T_indices=ids,array_rows=pair['array_rows'],seed=seed,
                primary_EEG_exposures=2*updates,proof=proof,loss=float(loss.detach()),absolute=float(terms['absolute'].detach()),
                pairing=float(terms['pairing'].detach()),effect=float(terms['effect'].detach()),gradients=gradients,seconds=time.monotonic()-tick))
            features.append(feature.detach().cpu().numpy())
            del x,target,tk,c,corrected,latent,feature,loss,terms
            assert all(p.grad is None for m in context['frozen'].values() for p in m.parameters())
            assert torch.equal(context['LN'].weight,initial_affine[0]) and torch.equal(context['LN'].bias,initial_affine[1])
            if updates%128==0:
                log=b''.join((json.dumps(v,allow_nan=False)+'\n').encode() for v in pending)
                state=U.snapshot(context,opt,arm=arm,updates=updates,order=order,registration_sha256=ref(path)['sha256'],log_sha256=hashlib.sha256(log).hexdigest())
                previous=publish_checkpoint(folder,state,pending,np.stack(features),previous,path,pairs,torch,np);del state;pending=[];features=[]
                assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<s['output_cap_bytes']
                atomic(folder/'progress.json',dict(arm=arm,updates=updates,total_updates=6615,primary_EEG_exposures=2*updates,last_commit=previous['commit'],segment=segment,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),replace=True)
            if updates%16==0:print(json.dumps(dict(arm=arm,updates=updates,total_updates=6615,seconds=time.monotonic()-start)),flush=True)
        assert updates>start_update
        if pending:
            log=b''.join((json.dumps(v,allow_nan=False)+'\n').encode() for v in pending)
            state=U.snapshot(context,opt,arm=arm,updates=updates,order=order,registration_sha256=ref(path)['sha256'],log_sha256=hashlib.sha256(log).hexdigest())
            previous=publish_checkpoint(folder,state,pending,np.stack(features),previous,path,pairs,torch,np);del state
        assert original_versions=={n:[p._version for p in m.parameters()] for n,m in context['frozen'].items()}
        assert not context['LN'].weight.requires_grad and not context['LN'].bias.requires_grad
        if arm=='P_only_correct':assert all(torch.equal(p.detach().cpu(),context['initial_E'][n]) for n,p in context['model'].named_parameters())
        if arm=='EEG_only_correct':assert all(torch.equal(context['master'][n],p.float()) for n,p in context['P'].named_parameters())
        for key in ('ordinary','parent'):checked(s[key])
        if updates==6615:
            final=folder/'model_final.pt';assert not final.exists()
            payload=dict(EEG=U.cpu_tree(context['model'].state_dict()),P=U.cpu_tree(context['master']),arm=arm,seed=1729,updates=6615,
                primary_EEG_exposures=13230,registration=ref(path),original_ordinary=s['ordinary'],last_commit=previous['commit'],final_LN_affine_fixed=True,
                checkpoint_rule='fixed last complete T pass',CLIP_critic_shared_with_author_CLIP_metric=True)
            with final.with_suffix('.pt.pending').open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            final.with_suffix('.pt.pending').rename(final)
            atomic(folder/'completion.json',dict(status='COMPLETE_ONE_FULL_T_PAIRED_IMAGE_SEMANTIC_EEG_P_FIT',arm=arm,model=ref(final),registration=ref(path),updates=6615,primary_EEG_exposures=13230,pair_count=6615,last_commit=previous['commit']))
        budget();assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<s['output_cap_bytes']
        atomic(folder/'progress.json',dict(arm=arm,updates=updates,total_updates=6615,primary_EEG_exposures=2*updates,last_commit=previous['commit'],segment=segment,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()),replace=True)
        result=dict(status='COMPLETE_IMMUTABLE_TRAINING_PREFIX_SEGMENT',arm=arm,range=[start_update,updates],full_arm_complete=updates==6615,last_commit=previous['commit'],
            seconds=time.monotonic()-start,training_loop_seconds=time.monotonic()-loop_start,extra_CPU_peak=rss()-memory,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            frozen_parameter_versions_unchanged=True,all_frozen_parameter_gradients_none=True,future_GPU_bit_exactness_not_claimed=True,NEW_contacted=False)
        atomic(attempt/'completion.json',result);print(json.dumps(result))
    except BaseException as e:
        atomic(attempt/'failure.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--segment',type=int,required=True);a=p.parse_args();main(a.protocol,a.arm,a.segment)
