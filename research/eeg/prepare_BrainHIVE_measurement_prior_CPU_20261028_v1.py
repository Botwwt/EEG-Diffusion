"""Complete-T physical-repeat JVP covariance, with native encoder fixed."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import brainhive_measurement_prior_20261028_v1 as M

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256'],str(p)
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def run(sp):
    start=time.monotonic();baseline=rss();spec=read(sp);assert spec['status']=='D39_NATIVE_EEG_MEASUREMENT_PRECISION_CPU_LOCKED'
    for x in spec['source_pins']:checked(x)
    B=Path(spec['output']);assert not B.exists();B.mkdir(parents=True);torch.set_num_threads(2)
    stage='INPUTS'
    try:
        d30=read(checked(spec['D30_spec']));d33=read(checked(spec['D33_spec']))
        physical=np.load(checked(d33['T_trials']),mmap_mode='r',allow_pickle=False)
        index=[json.loads(s) for s in checked(d33['T_index']).read_text().splitlines()]
        assert physical.shape==(13230,4,17,250) and len(index)==13230
        assert all(v['row']==i and v['partition']=='T' and v['subject_id']=='sub-01' for i,v in enumerate(index))
        state=torch.load(checked(d30['roles']['full']['checkpoint']),map_location='cpu',weights_only=True)['model']
        raw=np.lib.format.open_memmap(B/'T13230_mean4.npy',mode='w+',dtype=np.float32,shape=(13230,4250))
        units=np.lib.format.open_memmap(B/'T13230_original_units.npy',mode='w+',dtype=np.float32,shape=(13230,1024))
        directions=np.lib.format.open_memmap(B/'T13230x4_measurement_JVP.npy',mode='w+',dtype=np.float32,shape=(13230,4,1024))
        N=torch.zeros((1024,1024),dtype=torch.float64)
        max_mean_error=0.
        def budget():
            assert time.monotonic()-start<840 and rss()-baseline<10<<30
            assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<1<<30
            assert not torch.cuda.is_initialized()
        with torch.no_grad():
            for lo in range(0,13230,128):
                stage='COMPLETE_T_REAL_MEASUREMENT_JVP'
                t=np.asarray(physical[lo:lo+128]);x=t.mean(1,dtype=np.float32).reshape(len(t),4250)
                perturb=(t.reshape(len(t),4,4250)-x[:,None])/math.sqrt(12)
                unit,du=M.unit_and_jvp(torch.from_numpy(x.copy()),torch.from_numpy(perturb.copy()),state)
                raw[lo:lo+len(t)]=x;units[lo:lo+len(t)]=unit.numpy();directions[lo:lo+len(t)]=du.numpy()
                d=du.double().flatten(0,1);N+=d.T@d/13230
                max_mean_error=max(max_mean_error,float(du.sum(1).abs().max()))
                budget()
                if lo%2048==0:print(json.dumps(dict(stage=stage,queries=lo+len(t),seconds=time.monotonic()-start)),flush=True)
        raw.flush();units.flush();directions.flush()
        Q=M.normalized_precision(N);permutation=torch.randperm(1024,generator=torch.Generator().manual_seed(1729))
        permuted=Q[permutation][:,permutation]
        for name,value in [('measurement_covariance.npy',N),('measurement_precision.npy',Q),('permuted_precision.npy',permuted),('coordinate_permutation.npy',permutation)]:
            with (B/name).open('xb') as f:np.save(f,value.numpy(),allow_pickle=False)
        stage='REAL_JVP_AD_CHECK'
        selected=np.linspace(0,13229,16,dtype=np.int64);xx=torch.from_numpy(np.array(raw[selected],copy=True)).double()
        trials=np.asarray(physical[selected]).astype(np.float64).reshape(16,4,4250)
        dd=torch.from_numpy((trials-np.asarray(raw[selected],np.float64)[:,None])/math.sqrt(12))
        state64={k:v.double() for k,v in state.items()}
        uu,manual=M.unit_and_jvp(xx,dd,state64)
        def original(x):
            h=F.linear(x,state64['proj_in.weight'],state64['proj_in.bias'])
            z=h+F.linear(F.gelu(F.linear(h,state64['block.layers.0.ff.w1.weight'])),state64['block.layers.0.ff.w2.weight'])
            return F.layer_norm(z,(1024,),eps=1e-6)
        ad_error=0.
        for r in range(4):
            _,truth=torch.autograd.functional.jvp(original,xx,dd[:,r]);ad_error=max(ad_error,float((truth-manual[:,r]).abs().max()))
        assert ad_error<1e-10
        trace=float(Q.trace()/1024);eig=torch.linalg.eigvalsh(Q);rotation=float((Q-permuted).norm()/Q.norm())
        budget()
        write(B/'completion.json',dict(status='D39_COMPLETE_T13230_PHYSICAL_REPEAT_MEASUREMENT_PRECISION_PREPARED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          source=ref(Path(__file__).resolve()),spec=ref(sp),subject='sub-01',parent_role='full',not_whole_pipeline_OOF=True,
          queries=13230,concepts=1323,real_repeat_input_JVPs=52920,physical_repeats=4,
          normalization='N=mean_query sum_r(J_u(x_r-mean4)/sqrt12)^T(...); C=.9*N/(trN/1024)+.1I; Q=C^-1 normalizedtrQ/1024=1',
          Q_mean_eigenvalue=trace,Q_min_eigenvalue=float(eig.min()),Q_max_eigenvalue=float(eig.max()),
          Q_permuted_Frobenius_relative_difference=rotation,raw_noise_covariance_trace=float(N.trace()),
          all4_direction_sum_max=float(max_mean_error),real16x4_FP64_JVP_vs_AD_max_error=ad_error,
          covariance_assumes_repeat_error_exchangeability_not_proven=True,local_linearized_noise_not_full_nonlinear_posterior=True,
          precision_normalized_for_direction_test_not_calibrated_absolute_noise=True,source_encoder_unchanged=True,
          new_neural_updates=0,new_GPU_jobs=0,new_images=0,seconds=time.monotonic()-start,peak_extra_CPU_bytes=rss()-baseline,
          files=[ref(p) for p in B.iterdir() if p.is_file()]))
        print(json.dumps(dict(status='D39_COMPLETE_CPU_METRIC_READY',seconds=time.monotonic()-start,precision_permutation_difference=rotation)),flush=True)
    except BaseException as e:
        write(B/'failure.json',dict(status='D39_CPU_PREPARATION_FAILURE_PRESERVED',stage=stage,error=repr(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('D39 finite CPU stop')))
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
