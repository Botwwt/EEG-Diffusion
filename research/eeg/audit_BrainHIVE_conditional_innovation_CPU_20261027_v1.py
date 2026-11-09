"""All8200 held forward replays; fixed16-column independent T fit challenges."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V
from brainhive_bounded_component_v1 import BoundedConditionAdapter

def main():
    p=argparse.ArgumentParser();p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(2)
    done=U.read(a.completion);assert done['fitted_subjects']==10 and done['fit_models']==20 and done['all_Dfit_queries']==8200
    spec=U.read(U.checked(done['spec']));prepared=U.read(U.checked(done['prepared']))
    for x in spec['source_pins']:U.checked(x)
    from brainhive_component_data_v1 import load_training
    train=load_training('cpu');del train['target']
    contract=U.read(U.checked(spec['component_input_contract']));bindings=U.read(U.checked(spec['raw_bindings']));raw=U.read(U.checked(spec['raw_T_inputs']))
    metadata=U.read(U.checked(prepared['Dfit_metadata']));rows=[q['array_row'] for q in metadata['records']];concepts=[q['concept_id'] for q in metadata['records']]
    models={key:torch.load(U.checked(ref),weights_only=True,map_location='cpu') for key,ref in done['models'].items()}
    target=np.load(U.checked(prepared['T_projected']),allow_pickle=False);heldtarget=torch.from_numpy(np.load(U.checked(prepared['Dfit_projected']),allow_pickle=False))
    basis=torch.load(U.checked(prepared['basis']),weights_only=True,map_location='cpu')
    assert float((basis['basis'].T@basis['basis']-torch.eye(128)).abs().max())<1e-10
    heldlatent=np.load(U.checked(prepared['Dfit_latents']),mmap_mode='r',allow_pickle=False).reshape(820,-1)
    actual=(torch.from_numpy(np.array(heldlatent,copy=True)).double()-basis['latent_mean'].double())@basis['basis']
    assert torch.allclose(actual,heldtarget,atol=1e-9,rtol=1e-9)
    noise=torch.from_numpy(np.load(U.checked(done['noise']),allow_pickle=False))
    saved_scores=np.load(U.checked(done['scores']),allow_pickle=False);pred=np.load(U.checked(done['predictions']),mmap_mode='r',allow_pickle=False)
    variance=np.load(U.checked(done['posterior_covariances']),mmap_mode='r',allow_pickle=False)
    source=U.read(U.checked(done['source_replay']));perm=np.asarray(source['training_sameconcept_permutation']);heldperm=np.asarray(source['Dfit_sameconcept_permutation'])
    assert sorted(perm.tolist())==list(range(13230)) and np.all(perm!=np.arange(13230))
    saved=torch.load(U.checked(spec['ordinary_checkpoint']),weights_only=False,map_location='cpu')
    component=BoundedConditionAdapter().eval().requires_grad_(False);component.load_state_dict(saved['model'],strict=True)
    def correct(c,w,b):
        with torch.no_grad():return torch.cat([component(c[i:i+256],w[i:i+256],b[i:i+256])[0] for i in range(0,len(c),256)])
    rebuilt=np.full_like(saved_scores,np.nan);maxima=dict(ordinary_relative_L2=0.,prediction_abs=0.,covariance_abs=0.,score_abs=0.)
    challenges=[];columns=np.linspace(0,4249,16).round().astype(int)
    for j,name in enumerate(spec['subjects']):
        assert time.monotonic()-started<240,'finite CPU full replay limit'
        subjects=next(x for x in contract['subjects'] if x['subject']==name);full=next(x for x in subjects['sources'] if x['role']=='full')
        cache=torch.load(U.checked(dict(path=full['condition_cache_path'],sha256=full['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        lookup={int(r):i for i,r in enumerate(cache['array_rows'])};ix=[lookup[r] for r in rows]
        checkpoint=torch.load(U.checked(dict(path=full['checkpoint_path'],sha256=full['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model']
        hc=correct(cache['condition'][ix].clone(),checkpoint['block.layers.0.norm.weight'].expand(820,-1),checkpoint['block.layers.0.norm.bias'].expand(820,-1));del cache,checkpoint
        gpu_hc=np.load(a.completion.parent/(name+'_Dfit_ordinary_conditions.npy'),allow_pickle=False)
        rel=np.linalg.norm(hc.numpy()-gpu_hc,axis=1)/np.linalg.norm(gpu_hc,axis=1);assert rel.max()<1e-6
        maxima['ordinary_relative_L2']=max(maxima['ordinary_relative_L2'],float(rel.max()));hc=torch.from_numpy(gpu_hc)
        b=bindings['subjects'][name];released=torch.load(U.checked(dict(path=b['train_path'],sha256=b['train_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        he=released[rows][:,bindings['selected_indices'],:].reshape(820,4250).clone();del released
        pair=models[name+'_paired'];wrong=models[name+'_wrong'];base,joint,innovation=V.clean_means(pair,hc,he)
        _,wtrain,_=V.clean_means(wrong,hc,he);_,winfer,_=V.clean_means(pair,hc,he[heldperm])
        stored=np.load(a.completion.parent/(name+'_Dfit_raw_innovation.npy'),allow_pickle=False);assert np.allclose(innovation.numpy(),stored,atol=1e-10,rtol=1e-10)
        expected=(heldtarget-pair['image_mean'])/pair['image_std']
        for ti,t in enumerate(V.TIMES):
            alpha=prepared['time_values'][str(t)]['alpha'];sigma=prepared['time_values'][str(t)]['sigma']
            noisy=alpha*expected+sigma*noise[ti]/pair['image_std']
            for ai,(mean,cov) in enumerate(zip([base,joint,wtrain,winfer],[pair['prior_cov'],pair['joint_cov'],wrong['joint_cov'],pair['joint_cov']])):
                prediction,pcov=V.noisy_update(mean,cov,noisy,alpha,sigma,pair['image_std'])
                error=float(np.max(np.abs(prediction.numpy()-pred[j,:,ti,ai])));assert error<1e-8
                ce=float(np.max(np.abs(pcov.numpy()-variance[j,ti,ai])));assert ce<1e-9
                metric=V.score(prediction,expected,pcov,pair['image_std'])
                rebuilt[j,:,ti,ai]=torch.stack(list(metric.values()),1).numpy()
                maxima['prediction_abs']=max(maxima['prediction_abs'],error);maxima['covariance_abs']=max(maxima['covariance_abs'],ce)
        # Recompute every T ordinary condition for challenges, but verify only
        # the declared16 EEG normal-equation columns, not a second full fit.
        lo=j*13230;ids=train['affine_ids'][lo:lo+13230]
        tc=correct(train['condition'][lo:lo+13230],train['weight'][ids],train['bias'][ids]).double().numpy()
        item=raw['subjects'][name];te=np.array(np.load(U.checked(dict(path=item['path'],sha256=item['sha256'])),mmap_mode='r'),dtype=np.float64).reshape(13230,4250)
        for arm,m in [('paired',pair),('wrong',wrong)]:
            mm={k:v.numpy() if torch.is_tensor(v) else v for k,v in m.items()}
            cu=(target-mm['image_mean'])/mm['image_std'];cc=(tc-mm['condition_mean'])/mm['condition_std']
            ee=((te if arm=='paired' else te[perm])-mm['eeg_mean'])/mm['eeg_std']
            scc=.9*(cc.T@cc)/13230+.1*np.eye(1024);suc=.9*(cu.T@cc)/13230;sec=.9*(ee[:,columns].T@cc)/13230
            auc=mm['mean_image_from_c'];aec=mm['mean_eeg_from_c'];gain=mm['EEG_gain']
            auc_error=np.linalg.norm(auc@scc-suc)/np.linalg.norm(suc)
            aec_error=np.linalg.norm(aec[columns]@scc-sec)/np.linalg.norm(sec)
            rcols=.9*(ee.T@ee[:,columns])/13230+.1*np.eye(4250)[:,columns]-aec@sec.T
            crosscols=.9*(cu.T@ee[:,columns])/13230-auc@sec.T
            gain_error=np.linalg.norm(gain@rcols-crosscols)/np.linalg.norm(crosscols)
            assert max(auc_error,aec_error,gain_error)<1e-4
            challenges.append(dict(subject=name,arm=arm,conditional_c_image_normal_equation=auc_error,selected_EEG_c_normal_equation=aec_error,selected_conditional_gain_equation=gain_error))
        del tc,te,cu,cc,ee,rcols,models[name+'_paired'],models[name+'_wrong'];print(json.dumps(dict(subject=name,CPU_seconds=time.monotonic()-started)),flush=True)
    maxima['score_abs']=float(np.max(np.abs(rebuilt-saved_scores)));assert np.allclose(rebuilt,saved_scores,atol=1e-8,rtol=1e-9)
    assert V.readout(saved_scores,concepts,spec['subjects'])==U.read(U.checked(done['readout']))
    for x in spec['source_pins']:U.checked(x)
    assert not torch.cuda.is_initialized()
    value=dict(status='ALL8200_DFIT_ALL4TIME_ALL4ARM_CPU_PREDICTIONS_AND20_FIT_CHALLENGES_VERIFIED',source=U.ref(__file__),completion=U.ref(a.completion),
               all8200_ordinary_conditions_CPU_replayed=True,all131200_vector_predictions_and_covariances_CPU_replayed=True,
               all820_Dfit_latent_projections_CPU_replayed=True,T_ordinary_conditions_CPU_recomputed=132300,
               all20_models_all128_image_conditional_equations_checked=True,EEG_columns_checked=columns.tolist(),EEG_normal_equations_all4250_columns_CPU_recomputed=False,
               challenges=challenges,maxima=maxima,all_actual_readout_and_bootstraps_recomputed=True,all_sources_unchanged=True,
               GPU_jobs=0,new_fit=0,new_images=0,formal_D_resumed=False,seconds=time.monotonic()-started)
    U.atomic(a.output/'summary.json',value);print(json.dumps(value))

if __name__=='__main__':main()
