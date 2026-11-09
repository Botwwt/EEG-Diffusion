"""Local independent moments and fixed feature replays after original STOP."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,base64,hashlib,json
from pathlib import Path
import numpy as np

def geometry(z):
    z=np.asarray(z,dtype=np.float64).reshape(len(z),-1);assert len(z)==80 and np.isfinite(z).all()
    norms=np.linalg.norm(z,axis=1);assert norms.min()>0
    unit=z/norms[:,None];centred=unit-unit.mean(0);gram=centred@centred.T
    eigen=np.maximum(np.linalg.eigvalsh(gram),0);trace=eigen.sum()
    energy=float(np.mean(np.einsum('ij,ij->ij',unit,unit).sum(1)))
    fraction=float(np.dot(unit.mean(0),unit.mean(0))/energy)
    # Alternate trace identity, independent of the direct centred matrix sum.
    via_pairs=float(sum(np.dot(unit[i]-unit[j],unit[i]-unit[j]) for i in range(80) for j in range(i))/80**2)
    variance=float((centred*centred).sum()/80)
    assert abs(via_pairs-variance)<2e-12 and abs(1-fraction-variance)<2e-12
    return dict(mean_norm=float(norms.mean()),norm_std=float(norms.std()),common_angular_energy=fraction,centred_angular_energy=variance,centred_participation_rank=float(trace**2/np.dot(eigen,eigen)) if trace>0 else 0.,unique_exact_rows=len({row.tobytes() for row in z}),independent_variance_identity_max_error=max(abs(via_pairs-variance),abs(1-fraction-variance)))

def main(receipt,output):
    r=json.loads(Path(receipt).read_bytes());assert r['exit_code']==0;s=json.loads(r['stdout']);assert s['status']=='COMPLETE_POSTHOC_CLOSED_FIVE_ARM_TRANSMISSION_AND_LOG_EXTRACTION'
    b=base64.b64decode(s.pop('NPZ_base64'),validate=True);assert len(b)==s['NPZ_bytes'] and hashlib.sha256(b).hexdigest()==s['NPZ_sha256']
    out=Path(output);out.mkdir(exist_ok=False,parents=True);(out/'observed_arrays.npz').write_bytes(b)
    x=np.load(out/'observed_arrays.npz',allow_pickle=False);g=x['LN_gamma'];beta=x['LN_beta'];methods=('parent','ordinary','joint_correct','EEG_only_correct','P_only_correct','joint_swapped_targets','joint_absolute_only')
    results={};maximum=0.
    for arm in methods:
        raw=x[f'heldout80__{arm}__raw'];c=x[f'heldout80__{arm}__conditions'];tk=x[f'heldout80__{arm}__tokens'];assert raw.shape==c.shape==(80,1024) and tk.shape==(80,4,2048)
        u=(raw.astype(float)-beta)/g;co=(c.astype(float)-beta)/g
        record=dict(raw_EEG_output=geometry(raw),undo_affine_EEG=geometry(u),corrected_undo_affine=geometry(co),actual_FP16_tokens=geometry(tk))
        if arm in ('parent','P_only_correct','ordinary'):assert np.array_equal(raw,x['heldout80__parent__raw'])
        if arm=='parent':assert np.array_equal(raw,c)
        for v in record.values():maximum=max(maximum,v['independent_variance_identity_max_error'])
        results[arm]=record
    local=Path(__file__).resolve().parent/'audit/HIVE_paired_semantic_full_study_complete_20261030_v1'
    for arm in methods:
        f=np.load(local/'actual/images'/f'features80_{arm}.npy',allow_pickle=False);results[arm]['final_DINO_features']=geometry(f)
        records=[json.loads((local/'actual/images/records'/f'{i:03d}_{arm}.json').read_bytes()) for i in range(80)]
        results[arm]['unique_pixel_hashes']=len({r['generated_pixel_sha256'] for r in records})
    logs={};selected=np.r_[0:8,504:512];replay_maximum=0.
    for a in methods[2:]:
        logs[a]={}
        for block in ('first512','last512'):
            z=x[f'training__{a}__{block}'];assert z.shape==(512,11)
            summary=s['training_blocks'][a][block];assert np.max(np.abs(z.mean(0)-summary['mean']))<2e-12
            fg=x[f'features__{a}__{block}'];fv=x[f'targets__{a}__{block}'];dg=fg[:,0]-fg[:,1];dv=fv[:,0]-fv[:,1]
            gsq=np.einsum('ij,ij->i',dg,dg);vsq=np.einsum('ij,ij->i',dv,dv);dot=np.einsum('ij,ij->i',dg,dv)
            cos=dot/np.maximum(np.sqrt(gsq*vsq),1e-15);absolute=1-np.einsum('bij,bij->bi',fg,fv).mean(1)
            expected=np.column_stack([gsq,vsq,cos,absolute]);error=float(np.max(np.abs(expected-z[selected,7:11])));assert error<2e-12;replay_maximum=max(replay_maximum,error)
            value={name:float(z[:,i].mean()) for i,name in enumerate(summary['columns'])};value['EEG_gradient_clipped_fraction']=float((z[:,5]>1).mean());value['P_gradient_clipped_fraction']=float((z[:,6]>1).mean());value['local_feature_pair_replays']=16
            logs[a][block]=value
    result=dict(status='COMPLETE_LOCAL_POSTHOC_TRANSMISSION_AND_LOG_DIAGNOSIS',protocol=s['protocol'],source_completion=s['completion'],source_scope=s['scope'],
        heldout_final_geometry=results,online_training_blocks=logs,maximum_moment_identity_error=maximum,maximum_fixed_feature_replay_error=replay_maximum,
        all80_cached_forward_rows_locally_checked=True,all512_pair_scalar_columns_locally_aggregated=True,actual_training_feature_replays_per_arm_block=16,
        other496_feature_pairs_per_block_not_locally_recomputed=True,full400_raw_EEG_forward_and_all_image_audits_remain_server_evidence=True,
        no_unique_failure_cause_or_causal_gradient_claim=True,not_frozen_model_training_validation=True,original_STOP_not_modified=True,new_GPU_jobs=0,new_updates=0,new_images=0)
    for name,value in [('source_and_scope.json',s),('local_transmission_and_log_geometry.json',result)]:
        (out/name).write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=result['status'],geometry={a:{'u_variance':v['undo_affine_EEG']['centred_angular_energy'],'token_variance':v['actual_FP16_tokens']['centred_angular_energy'],'DINO_variance':v['final_DINO_features']['centred_angular_energy'],'unique_token_rows':v['actual_FP16_tokens']['unique_exact_rows']} for a,v in results.items()},logs={a:{block:{k:v for k,v in value.items() if k in ['loss','absolute','pairing','generated_pair_difference_norm2','true_difference_direction_cosine']} for block,value in log.items()} for a,log in logs.items()},maximum_moment_error=maximum,feature_replay_error=replay_maximum)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.receipt,a.output)
