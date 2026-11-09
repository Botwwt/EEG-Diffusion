"""Path-portable, finite EEG-only OOF c/local-memory export from fixed finals.

No fitted parameters or images. The original clinical/downstream encoder is
loaded strictly. Every T row uses the role excluding its concept and its own
role-only normalization. Full-role weights are used only for Dcheck EEG.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,hashlib,json,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from eeg_labram_pretrained_20261028_v1 import build_encoder
from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform,global_and_patch_conditions

CAMPAIGN_SHA='aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56'
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def checked(ref):
    path=Path(ref['path']);assert path.is_file() and sha(path)==ref['sha256'],str(path)
    if 'bytes' in ref:assert path.stat().st_size==ref['bytes']
    return path
def reference(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024

def run(spec_path):
    began=time.monotonic();spec=json.loads(spec_path.read_bytes());out=Path(spec['output']);assert not out.exists();out.mkdir(parents=True)
    phase='PINNED_EEG_AND_FINAL_CHECKPOINTS';base_CPU=peak()
    try:
        for item in spec['source_pins']:checked(item)
        assert shutil.disk_usage(out).free>8<<30
        uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==spec['GPU_UUID']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        cpu=json.loads(checked(spec['spatial_CPU']).read_bytes());assert cpu['status']=='D40_COMPLETE_T_TRUE63_ELECTRODES_REPEAT_NOISE_INPUTS_VERIFIED'
        files=cpu['files'];raw=np.load(checked(files['T_mean63']),mmap_mode='r',allow_pickle=False);dcheck=np.load(checked(files['Dcheck_mean63']),mmap_mode='r',allow_pickle=False)
        queries=json.loads(checked(files['queries']).read_bytes());identities=json.loads(checked(spec['T_identities']).read_bytes());folds=np.array(identities['folds'],dtype=np.int64)
        assert raw.shape==(13230,63,250) and raw.dtype==np.float32 and dcheck.shape==(400,63,250) and dcheck.dtype==np.float32
        assert queries['T']==identities['T'] and len(queries['Dcheck'])==400 and folds.shape==(13230,) and set(folds)=={0,1}
        assert len({q['concept_id'] for q in queries['T']})==1323
        assert not ({q['concept_id'] for q in queries['T']}&{q['concept_id'] for q in queries['Dcheck']})
        def guard():
            assert time.monotonic()-began<840 and peak()-base_CPU<10<<30
            assert not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated()<20<<30
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        t_c=np.lib.format.open_memmap(out/'T_OOF_c.npy',mode='w+',dtype=np.float32,shape=(13230,1024))
        t_h=np.lib.format.open_memmap(out/'T_OOF_memory.npy',mode='w+',dtype=np.float32,shape=(13230,63,1024))
        d_c=np.lib.format.open_memmap(out/'Dcheck400_c.npy',mode='w+',dtype=np.float32,shape=(400,1024))
        d_h=np.lib.format.open_memmap(out/'Dcheck400_memory.npy',mode='w+',dtype=np.float32,shape=(400,63,1024))
        seen=np.zeros(13230,np.int8);role_proofs={};replays=[];max_mean_error=0.;torch.cuda.reset_peak_memory_stats()
        for role in ('oof0','oof1','full'):
            phase='ROLE_'+role
            item=spec['roles'][role];done=json.loads(checked(item['completion']).read_bytes());assert done['status']=='COMPLETE_SUB01_FIXED_50_EPOCH_T_ENCODER' and done['role']==role and done['epochs']==50
            assert done['spec_sha256']==CAMPAIGN_SHA and item['model']['sha256']==done['final_model']['sha256']
            state=torch.load(checked(item['model']),weights_only=True,map_location='cpu');assert state['role']==role and state['epoch']==50 and state['spec_sha256']==CAMPAIGN_SHA
            model=build_encoder(spec['author_source'],spec['vendor'],seed=1729)
            assert len(model.state_dict())==224 and set(state['model'])==set(model.state_dict())
            for key,value in state['model'].items():assert value.shape==model.state_dict()[key].shape and value.dtype==model.state_dict()[key].dtype and torch.isfinite(value).all(),key
            model.load_state_dict(state['model'],strict=True);model.eval().requires_grad_(False)
            stats=json.loads(checked(item['normalization']).read_bytes());names=tuple(stats['channel_names']);assert len(names)==63 and stats['role']==role
            expected=np.arange(13230) if role=='full' else np.flatnonzero(folds!=int(role[-1]));assert stats['training_positions']==expected.tolist()
            transform=CacheNativeEEGTransform(names,np.array(stats['center']),np.array(stats['scale']))
            from models.labram import get_input_chans
            positions=get_input_chans([name.upper() for name in names]);assert len(positions)==64
            source=dcheck if role=='full' else raw;ids=np.arange(400) if role=='full' else np.flatnonzero(folds==int(role[-1]))
            assert role=='full' or not np.intersect1d(ids,expected).size
            first=ids[:2];cpu_x=torch.from_numpy(transform(source[first],channel_names=names))
            with torch.inference_mode():cpu_c,cpu_h=global_and_patch_conditions(model,cpu_x,input_chans=positions)
            model.cuda();role_start=time.monotonic();max_author_error=0.
            with torch.inference_mode():
                for lo in range(0,len(ids),128):
                    guard();rows=ids[lo:lo+128];x=torch.from_numpy(transform(source[rows],channel_names=names)).cuda()
                    c,h=global_and_patch_conditions(model,x,input_chans=positions)
                    assert c.shape==(len(rows),1024) and h.shape==(len(rows),63,1024) and torch.isfinite(c).all() and torch.isfinite(h).all()
                    mean_error=float((h.mean(1)-c).abs().max());max_mean_error=max(max_mean_error,mean_error);assert torch.allclose(h.mean(1),c,atol=2e-5,rtol=2e-5)
                    if lo==0:
                        author=model(x[:2],input_chans=positions)[0];max_author_error=float((c[:2]-author).abs().max())
                        assert torch.equal(c[:2],author) or torch.allclose(c[:2],author,atol=2e-5,rtol=2e-4)
                        dc=float((c[:2].cpu()-cpu_c).abs().max());dh=float((h[:2].cpu()-cpu_h).abs().max())
                        assert torch.allclose(c[:2].cpu(),cpu_c,atol=3e-5,rtol=3e-4) and torch.allclose(h[:2].cpu(),cpu_h,atol=3e-5,rtol=3e-4)
                        replays.append(dict(role=role,rows=first.tolist(),CPU_GPU_c_max_abs=dc,CPU_GPU_memory_max_abs=dh,author_c_max_abs=max_author_error))
                    if role=='full':d_c[rows]=c.cpu().numpy();d_h[rows]=h.cpu().numpy()
                    else:t_c[rows]=c.cpu().numpy();t_h[rows]=h.cpu().numpy();seen[rows]+=1
            role_proofs[role]=dict(model=item['model'],completion=item['completion'],normalization=item['normalization'],exported_queries=len(ids),training_queries=len(expected),heldout_fold=None if role=='full' else int(role[-1]),seconds=time.monotonic()-role_start)
            model.cpu();del model,state;torch.cuda.empty_cache();guard()
        assert (seen==1).all()
        for value in (t_c,t_h,d_c,d_h):value.flush()
        del t_c,t_h,d_c,d_h
        write(out/'identities.json',dict(T=queries['T'],T_OOF_fold=folds.tolist(),Dcheck=queries['Dcheck'],normal_inference_inputs='Physical R4 EEG only; labels/identities are audit joins, never encoder inputs'))
        phase='COMPLETE_EXPORTED_ARRAY_CPU_RECHECK';cpu_mean_error=0.;scanned=0
        for tag,count in [('T_OOF',13230),('Dcheck400',400)]:
            c=np.load(out/(tag+'_c.npy'),mmap_mode='r',allow_pickle=False);h=np.load(out/(tag+'_memory.npy'),mmap_mode='r',allow_pickle=False)
            for lo in range(0,count,64):
                assert np.isfinite(c[lo:lo+64]).all() and np.isfinite(h[lo:lo+64]).all()
                error=float(np.abs(h[lo:lo+64].mean(1,dtype=np.float64)-c[lo:lo+64]).max());cpu_mean_error=max(cpu_mean_error,error)
                assert np.allclose(h[lo:lo+64].mean(1,dtype=np.float64),c[lo:lo+64],atol=2e-5,rtol=2e-5);scanned+=len(c[lo:lo+64]);guard()
        payloads={p.stem:reference(p) for p in out.iterdir() if p.is_file()}
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<5<<30
        completion=dict(status='COMPLETE_REAL_EEG_T13230_GENUINE_OOF_AND_DCHECK400_LOCAL_MEMORY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=reference(spec_path),files=payloads,roles=role_proofs,T_queries=13230,T_concepts=1323,Dcheck_queries=400,channels=63,memory_dimension=1024,one_second_temporal_window_per_channel=True,physical_repeats=4,all_T_rows_exactly_once_from_concept_excluded_encoder=True,T_statistics_fit_only_role_allowed_concepts=True,CPU_array_rows_rechecked=scanned,CPU_mean_memory_error=cpu_mean_error,GPU_mean_memory_error=max_mean_error,independent_CPU_forward_replays=replays,source_campaign_sha256=CAMPAIGN_SHA,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base_CPU,seconds=time.monotonic()-began,trained_models=0,optimizer_updates=0,new_images=0,visual_targets_read=False,global_c_preserved=True,source_weights_modified=False)
        guard();write(out/'completion.json',completion);print(json.dumps(completion))
    except BaseException as error:
        write(out/'failure.json',dict(status='EEG_LOCAL_MEMORY_EXPORT_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
