"""Post-result FIT/held error and exact update-energy analysis; no refitting."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,datetime,hashlib,json,time
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(sp):
    start=time.monotonic();spec=read(sp);B=Path(spec['output']);complete=read(B/'completion.json')
    assert read(B/'independent_CPU_audit.json')['all_scalar_metrics']==396900
    D30=read(spec['D30_spec']['path']);inp=read(D30['physical_R4_input_contract']['path'])['files']
    trials=np.load(inp['trials.npy']['path'],mmap_mode='r',allow_pickle=False)
    y=np.asarray(np.load(inp['targets.npy']['path'],allow_pickle=False),np.float64)
    pred=np.load(complete['predictions']['path'],mmap_mode='r',allow_pickle=False)
    raw=np.empty((13230,4250),np.float64)
    for lo in range(0,13230,256):raw[lo:lo+256]=np.asarray(trials[lo:lo+256]).mean(1,dtype=np.float32).reshape(-1,4250)
    records=[]
    for fold in (0,1):
        f=np.load(B/f'fold{fold}_features.npz',allow_pickle=False);fit,held=f['fit_rows'],f['held_rows']
        common=np.concatenate((f['native_conditions'],raw,f['scatter']),axis=1)
        by_arm={}
        for arm in (0,2,3):
            record=next(x for x in complete['model_records'] if x['held_fold']==fold and x['arm']==complete['arms'][arm])
            assert sha(record['model']['path'])==record['model']['sha256']
            model=np.load(record['model']['path'],allow_pickle=False);dim=model['common_coefficient'].shape[0]
            fit_error=0.
            for low in range(0,len(fit),256):
                positions=fit[low:low+256]
                p=((common[positions,:dim]-model['input_mean'])/model['input_scale'])@model['common_coefficient']+model['target_mean']
                if arm==3:p+=((f['area'][positions]-model['area_mean'])/model['area_scale'])@model['area_coefficient']
                fit_error+=float(np.square(p-y[positions]).sum())
            by_arm[complete['arms'][arm]]=dict(FIT_MSE=fit_error/(len(fit)*1024),held_MSE=float(np.square(np.asarray(pred[held,arm],np.float64)-y[held]).mean()))
        records.append(dict(fold=fold,fit_rows=len(fit),held_rows=len(held),errors=by_arm))
        del common,f
        assert time.monotonic()-start<800
    base=np.asarray(pred[:,2],np.float64);delta=np.asarray(pred[:,3],np.float64)-base
    energy=np.square(delta).mean(1);alignment=2*(delta*(y-base)).mean(1)
    error_change=np.square(base+delta-y).mean(1)-np.square(base-y).mean(1)
    identity_error=float(np.max(np.abs(error_change-(energy-alignment))));assert identity_error<1e-12
    result=dict(status='D38_POSTHOC_FIT_HELD_AND_ENERGY_IDENTITY_COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),role_errors=records,
      held_primary_minus_common_update_energy=float(energy.mean()),held_twice_residual_alignment=float(alignment.mean()),
      held_MSE_change=float(error_change.mean()),identity_max_error=identity_error,
      queries_MSE_improved=int((error_change<0).sum()),queries=13230,
      post_result_analysis_not_preregistered_gate=True,no_strength_or_ridge_scan=True,
      no_models_refit_or_method_gate_changed=True,new_GPU_jobs=0,new_images=0,seconds=time.monotonic()-start)
    with (B/'posthoc_risk_decomposition.json').open('x') as file:json.dump(result,file,indent=2);file.write('\n')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
