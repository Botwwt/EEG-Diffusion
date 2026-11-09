"""Independent full covariance/precision and sampled NumPy native JVP replay."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(x):
    p=Path(x['path']);assert p.is_file() and sha(p)==x['sha256']
    if 'bytes' in x:assert p.stat().st_size==x['bytes']
    return p
def run(sp):
    started=time.monotonic();spec=read(sp);B=Path(spec['output']);complete=read(B/'completion.json')
    assert complete['real_repeat_input_JVPs']==52920
    for x in spec['source_pins']:checked(x)
    files={Path(x['path']).name:x for x in complete['files']}
    arrays={name:np.load(checked(files[name]),mmap_mode='r',allow_pickle=False) for name in files}
    du=arrays['T13230x4_measurement_JVP.npy'];N=np.asarray(arrays['measurement_covariance.npy'])
    full=np.zeros((1024,1024),np.float64)
    for lo in range(0,13230,256):
        d=np.asarray(du[lo:lo+256],np.float64).reshape(-1,1024);full+=d.T@d/13230
    cov_error=float(np.linalg.norm(full-N)/np.linalg.norm(N));assert cov_error<2e-12
    C=.9*full/(np.trace(full)/1024)+.1*np.eye(1024)
    Q=np.linalg.inv(C);Q=(Q+Q.T)/2;Q/=np.trace(Q)/1024
    saved=np.asarray(arrays['measurement_precision.npy']);qerror=float(np.linalg.norm(Q-saved)/np.linalg.norm(saved));assert qerror<2e-11
    permutation=np.asarray(arrays['coordinate_permutation.npy']);assert np.array_equal(np.sort(permutation),np.arange(1024))
    assert np.array_equal(np.asarray(arrays['permuted_precision.npy']),saved[permutation][:,permutation])
    assert abs(float(np.trace(saved))/1024-1)<1e-12
    assert np.max(np.abs(np.linalg.eigvalsh(saved)-np.linalg.eigvalsh(arrays['permuted_precision.npy'])))<1e-11
    d30=read(checked(spec['D30_spec']));d33=read(checked(spec['D33_spec']))
    trials=np.load(checked(d33['T_trials']),mmap_mode='r',allow_pickle=False)
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    state={k:v.numpy().astype(np.float64) for k,v in torch.load(checked(d30['roles']['full']['checkpoint']),weights_only=True,map_location='cpu')['model'].items()}
    rows=np.linspace(0,13229,16,dtype=np.int64);physical=np.asarray(trials[rows])
    x=physical.mean(1,dtype=np.float32).reshape(16,4250);assert np.array_equal(x,arrays['T13230_mean4.npy'][rows])
    perturb=((physical.reshape(16,4,4250)-x[:,None])/math.sqrt(12)).astype(np.float64)
    h=x.astype(np.float64)@state['proj_in.weight'].T+state['proj_in.bias'];dh=perturb@state['proj_in.weight'].T
    a=h@state['block.layers.0.ff.w1.weight'].T
    erf=np.fromiter((math.erf(v/math.sqrt(2)) for v in a.ravel()),dtype=np.float64,count=a.size).reshape(a.shape)
    slope=.5*(1+erf)+a*np.exp(-.5*a*a)/math.sqrt(2*math.pi)
    z=h+(.5*a*(1+erf))@state['block.layers.0.ff.w2.weight'].T
    dz=dh+((dh@state['block.layers.0.ff.w1.weight'].T)*slope[:,None])@state['block.layers.0.ff.w2.weight'].T
    centered=z-z.mean(-1,keepdims=True);inverse=1/np.sqrt(np.square(centered).mean(-1,keepdims=True)+1e-6);unit=centered*inverse
    truth=(dz-dz.mean(-1,keepdims=True)-unit[:,None]*(dz*unit[:,None]).mean(-1,keepdims=True))*inverse[:,None]
    uerror=float(np.max(np.abs(unit-arrays['T13230_original_units.npy'][rows])))
    jerror=float(np.max(np.abs(truth-du[rows])));assert uerror<3e-5 and jerror<3e-5
    result=dict(status='D39_FULL52920_COVARIANCE_PRECISION_AND_REAL16x4_NUMPY_JVP_AUDIT_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      source_sha256=sha(__file__),CPU_completion_sha256=sha(B/'completion.json'),all_JVP_vectors_covariance_replayed=52920,
      full_covariance_relative_error=cov_error,full_precision_relative_error=qerror,
      sampled_raw_queries=16,sampled_raw_JVPs=64,native_unit_NumPy64_max_error=uerror,native_JVP_NumPy64_max_error=jerror,
      same_NumPy_BLAS_backend_not_all_raw_EEG_recomputed=True,original_sources_or_weights_changed=False,
      no_GPU_or_neural_fit_or_images=True,seconds=time.monotonic()-started)
    with (B/'independent_CPU_audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
