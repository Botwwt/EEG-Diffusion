"""All full-UNet paired effects, exact symmetric path accounting and fixed CIs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,sys,time
from pathlib import Path
import numpy as np
from brainhive_complete_EEG_influence_CPU_20261103_v1 import vector_stats,symmetric_paths,DOMAINS,CASES

def summarize(cells,rows):
    # Input shape80 x metrics. Equal four times were averaged inside each query.
    subjects=sorted({r['subject'] for r in rows});concepts=sorted({r['concept_id'] for r in rows})
    matrix=np.array([[cells[next(i for i,r in enumerate(rows) if r['subject']==s and r['concept_id']==c)] for c in concepts] for s in subjects])
    assert matrix.shape[:2]==(10,8)
    rng=np.random.default_rng(1729);draws=[]
    for _ in range(1000):
        si=rng.integers(0,10,10);ci=rng.integers(0,8,8);draws.append(matrix[si][:,ci].mean((0,1)))
    ci=np.quantile(np.asarray(draws),[.025,.975],axis=0)
    return dict(mean=cells.mean(0).tolist(),descriptive_subject_concept_bootstrap95=ci.T.tolist(),queries=80,subjects=10,concepts=8)

def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();started=time.monotonic()
    import json
    spec=json.loads(Path(a.protocol).read_bytes());sys.path.insert(0,str(Path(spec['base_protocol']['path']).parent))
    import brainhive_functional_mean_contract_20261103_v1 as G
    base=G.setup(G.checked(spec['base_protocol']));root=Path(spec['output_root']);done=G.read(root/'probe/completion.json')
    assert done['status']=='COMPLETE_FROZEN_FULL_UNET_ALL10_SUBJECT80_T_THREE_DOMAINS'
    for pin in spec['source_pins']:G.checked(pin)
    data={name:np.load(G.checked(pin),mmap_mode='r',allow_pickle=False) for name,pin in done['outputs'].items()}
    pred=data['EEG_predictions.npy'];teacher=data['DDPM_teacher_predictions.npy'];noise=data['DDPM_noise.npy'];layer=data['layer_statistics.npy']
    assert pred.shape==(3,80,4,4,4,64,64) and teacher.shape==(80,4,2,4,64,64)
    assert noise.dtype==np.float32 and layer.shape==(3,80,4,70,10)
    refs=data['rollout_predictions.npy'];assert np.array_equal(refs[0],pred[1,:,:,0]) and np.array_equal(refs[1],pred[2,:,:,2])
    metadata=G.read(G.checked(done['trace']));rows=metadata['rows'];assert rows==spec['queries']
    assert metadata['parity']['status']=='EXACT_FINAL_LATENT_AND_ALL_NOISE_OBSERVER_PARITY'
    assert all(r['ordinary']==r['biased'] for r in metadata['noise_proofs'])
    first=metadata['nodes'].index(metadata['actual_forward_order'][0]);assert np.all(layer[:,:,:,first,[0,1,6]]==0)
    local=np.empty((3,80,4,6));valid=np.empty((3,80,4),bool);path=np.empty((80,4,9));errors=np.empty((80,4,6));path_error=0.
    for domain in range(3):
        for i in range(80):
            for t in range(4):
                p=np.asarray(pred[domain,i,t],np.float64);base_delta=p[0]-p[1];new=p[2]-p[3]
                values=vector_stats(new,base_delta)
                denominator=max(values['reference_energy'],1e-12);valid[domain,i,t]=values['valid_reference']
                local[domain,i,t]=[values[k] for k in ('reference_energy','value_energy','dot','change_energy')]+[values['dot']/denominator,values['change_energy']/denominator]
                if domain==0:
                    truth=np.asarray(noise[i,t],np.float64);f=np.asarray(teacher[i,t],np.float64)
                    errors[i,t]=[np.mean((v-truth)**2) for v in [*p,*f]]
    for i in range(80):
        for t in range(4):
            e0=np.asarray(pred[1,i,t,0],np.float64)-pred[1,i,t,1]
            b0=np.asarray(pred[1,i,t,2],np.float64)-pred[1,i,t,3]
            eb=np.asarray(pred[2,i,t,0],np.float64)-pred[2,i,t,1]
            bb=np.asarray(pred[2,i,t,2],np.float64)-pred[2,i,t,3]
            v=symmetric_paths(e0,b0,eb,bb);base_energy=float(np.mean(e0*e0));denominator=max(base_energy,1e-12)
            direct_dot=float(np.mean(v['direct']*e0));state_dot=float(np.mean(v['state']*e0))
            path[i,t]=[v[k] for k in ('direct_energy','state_energy','cross_energy','total_energy')]+[base_energy,direct_dot,state_dot,direct_dot/denominator,state_dot/denominator]
            path_error=max(path_error,v['identity_error'],v['energy_identity_error'])
    out=root/'readout';assert not out.exists();out.mkdir()
    np.savez_compressed(out/'all_network_path_noise_statistics.npz',network=local,path=path,DDPM_errors=errors,layer=layer,valid_reference=valid)
    metrics=['reference_energy','value_energy','dot','change_energy','epsilon_floored_aligned_gain','epsilon_floored_relative_change_energy']
    readout={domain:summarize(local[j].mean(1),rows) for j,domain in enumerate(DOMAINS)}
    result=dict(status='COMPLETE_ALL10_T80_FULL_UNET_FIXED_STATE_AND_TWO_PATH_READOUT',UTC=G.datetime.datetime.now(G.datetime.timezone.utc).isoformat(),
                protocol=G.ref(a.protocol),probe=G.ref(root/'probe/completion.json'),source=G.ref(__file__),raw=G.ref(out/'all_network_path_noise_statistics.npz'),
                metrics=metrics,fixed_state_results=readout,path_metrics=['direct_energy','state_energy','cross_energy','total_energy','reference_energy','direct_dot','state_dot','direct_aligned_gain_change','state_aligned_gain_change'],
                symmetric_path_result=summarize(path.mean(1),rows),symmetric_vector_and_energy_identity_max_error=path_error,
                DDPM_condition_order=list(CASES)+['correct_F','same_concept_donor_F'],DDPM_MSE=summarize(errors.mean(1),rows),
                DDPM_correct_minus_donor_MSE=summarize((errors[:,:,[0,2,4]]-errors[:,:,[1,3,5]]).mean(1),rows),
                first_layer_actual_Q_and_pre_bias_output_pair_exactly_preserved=True,
                valid_reference_counts={domain:int(valid[j].sum()) for j,domain in enumerate(DOMAINS)},expected_reference_cells_per_domain=320,
                ratio_denominator_floor=1e-12,zero_reference_cells_retained=True,ratio_interpretation_requires_all_reference_cells_valid=True,
                layer_statistics_all70_mean=layer.mean((1,2)).tolist(),layer_metric_keys=metadata['layer_keys'],actual_forward_order=metadata['actual_forward_order'],
                no_layer_time_or_subject_selection=True,all_times_averaged_within_query_before_subject_concept_CI=True,
                raw_network_forward_scope='Frozen nativeFP16; exact actual raw forward arrays kept OLD. No new SGD or images.',
                scientific_scope='Post-v90 frozen diagnostic. Algebraic direct/state order-averaged decomposition is not causal image-gain attribution or proof of a novel loss.',
                all_original_ordinary_unchanged=True,new_GPU_jobs=0,new_images=0,new_fits=0,NEW_contacted=False,seconds=time.monotonic()-started)
    G.write(out/'completion.json',result);print(json.dumps(result))
if __name__=='__main__':main()
