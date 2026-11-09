"""Independent fixed80 geometry for a terminal noO study. No efficacy gate changes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np

ARMS=('joint_correct','joint_swapped_targets','joint_absolute_only')
METHODS=('parent','ordinary')+ARMS+tuple(a+'_plus_ordinary' for a in ARMS)

def geometry(z):
    z=np.asarray(z,dtype=float).reshape(80,-1); assert np.isfinite(z).all()
    norms=np.linalg.norm(z,axis=1); assert norms.min()>0
    u=z/norms[:,None]; c=u-u.mean(0); var=float((c*c).sum()/80)
    pairs=float(sum(np.dot(u[i]-u[j],u[i]-u[j]) for i in range(80) for j in range(i))/80**2)
    identity=abs(var-(1-np.dot(u.mean(0),u.mean(0))))
    assert max(abs(pairs-var),identity)<2e-12
    return dict(mean_norm=float(norms.mean()),centred_angular_energy=var,unique_exact_rows=len({r.tobytes() for r in z}),moment_identity_max_error=max(abs(pairs-var),identity))

def main(receipt,terminal,output):
    receipt=Path(receipt); envelope=json.loads(receipt.read_bytes()); assert envelope['exit_code']==0
    s=json.loads(envelope['stdout']); assert s['status']=='COMPLETE_CLOSED_NOO_THREE_FIT_EIGHT_PATH_TRANSMISSION_EXTRACTION'
    terminal=Path(terminal); independent=json.loads((terminal/'independent_local_statistics_and_provenance.json').read_bytes())
    # The independent terminal receiver/auditor must already have finished.
    assert (terminal/'received_manifest.json').is_file()
    assert independent['status']=='PASS_LOCAL_CLOSED_CORE_RANKS_42_CONTRASTS_RAW_CELLS_AND_PROVENANCE'
    done=json.loads((terminal/'actual/controller_completion.json').read_bytes())
    assert done['registration']==s['protocol'] and done['final_audit']['stage']==s['stage']
    data=base64.b64decode(s['NPZ_base64'],validate=True)
    assert len(data)==s['NPZ_bytes'] and hashlib.sha256(data).hexdigest()==s['NPZ_sha256']
    out=Path(output); out.mkdir(parents=True,exist_ok=False); (out/'observed_arrays.npz').write_bytes(data)
    x=np.load(out/'observed_arrays.npz',allow_pickle=False); gamma=x['LN_gamma']; beta=x['LN_beta']; results={}
    maximum=0.; feature_error=0.; selected=np.r_[0:8,504:512]
    for arm in METHODS:
        raw=x[f'heldout80__{arm}__raw']; c=x[f'heldout80__{arm}__conditions']; tk=x[f'heldout80__{arm}__tokens']
        assert raw.shape==c.shape==(80,1024) and tk.shape==(80,4,2048)
        if arm in ('parent','ordinary'): assert np.array_equal(raw,x['heldout80__parent__raw'])
        if arm in ('parent',)+ARMS: assert np.array_equal(raw,c)
        if arm.endswith('_plus_ordinary'): assert np.array_equal(raw,x[f'heldout80__{arm.removesuffix("_plus_ordinary")}__raw'])
        f=np.load(terminal/'actual/images'/f'features80_{arm}.npy',allow_pickle=False)
        records=[json.loads((terminal/'actual/images/records'/f'{i:03d}_{arm}.json').read_bytes()) for i in range(80)]
        results[arm]=dict(undo_affine_EEG=geometry((raw.astype(float)-beta)/gamma),corrected_undo_affine=geometry((c.astype(float)-beta)/gamma),actual_tokens=geometry(tk),DINO_features=geometry(f),unique_pixel_hashes=len({r['generated_pixel_sha256'] for r in records}))
        for value in results[arm].values():
            if isinstance(value,dict): maximum=max(maximum,value['moment_identity_max_error'])
    logs={}
    for arm in ARMS:
        logs[arm]={}
        for block in ('first512','last512'):
            z=x[f'training__{arm}__{block}']; assert z.shape==(512,11)
            description=s['training_blocks'][arm][block]; assert np.max(np.abs(z.mean(0)-description['mean']))<2e-12
            g=x[f'features__{arm}__{block}']; v=x[f'targets__{arm}__{block}']; dg=g[:,0]-g[:,1]; dv=v[:,0]-v[:,1]
            gs=(dg*dg).sum(-1); vs=(dv*dv).sum(-1); cosine=(dg*dv).sum(-1)/np.maximum(np.sqrt(gs*vs),1e-15)
            expected=np.column_stack([gs,vs,cosine,1-(g*v).sum(-1).mean(-1)])
            err=float(np.max(np.abs(expected-z[selected,7:11]))); assert err<2e-12; feature_error=max(feature_error,err)
            logs[arm][block]={key:float(z[:,i].mean()) for i,key in enumerate(description['columns'])}
    record=dict(status='COMPLETE_LOCAL_CLOSED_NOO_TRANSMISSION_DIAGNOSIS',protocol=s['protocol'],source_completion=s['completion'],fixed_first80=True,full_image_stage=s['stage'],geometry=results,online_training_windows=logs,moment_identity_max_error=maximum,fixed_feature_replay_max_error=feature_error,
                all512_scalar_columns_aggregated=True,actual_feature_pairs_replayed_per_arm_window=16,other496_feature_pairs_not_locally_recomputed=True,variance_not_information_bits=True,not_frozen_model_validation=True,actual_parameter_Jacobians_not_measured=True,no_unique_cause_claim=True,primary_gates_unchanged=True,new_updates=0,new_images=0,new_GPU_jobs=0)
    s.pop('NPZ_base64')
    for name,value in [('source_and_scope.json',s),('local_transmission_and_log_geometry.json',record)]:
        (out/name).write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=record['status'],methods=len(results),fits=len(logs),maximum_identity_error=maximum,feature_replay_error=feature_error)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--terminal',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.receipt,a.terminal,a.output)
