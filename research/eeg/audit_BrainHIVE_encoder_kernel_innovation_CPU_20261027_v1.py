"""Replay every D22 held prediction; independently challenge30 fitted systems."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V
import brainhive_encoder_kernel_innovation_20261027_v1 as K

def main():
    p=argparse.ArgumentParser()
    for k in ('completion','native-registration','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True);start=time.monotonic();torch.set_num_threads(2)
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(C.read(a.native_registration))
    from brainhive_component_data_v1 import load_training
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    done=U.read(a.completion);s=U.read(U.checked(done['spec']))
    for ref in s['source_pins']:U.checked(ref)
    oldfit=U.read(U.checked(s['D21_fit']));prep=U.read(U.checked(s['D21_prepared']));oldspec=U.read(U.checked(oldfit['spec']))
    kernel=U.read(U.checked(s['kernel_completion']));replay=U.read(U.checked(oldfit['source_replay']));perm=np.asarray(replay['training_sameconcept_permutation']);hp=np.asarray(replay['Dfit_sameconcept_permutation'])
    metadata=U.read(U.checked(prep['Dfit_metadata']))['records'];concepts=[x['concept_id'] for x in metadata]
    target=np.load(U.checked(prep['T_projected']),allow_pickle=False);heldtarget=torch.from_numpy(np.load(U.checked(prep['Dfit_projected']),allow_pickle=False))
    noise=torch.from_numpy(np.load(U.checked(oldfit['noise']),allow_pickle=False))
    train=load_training('cpu');del train['target'];component=BoundedConditionAdapter().eval().requires_grad_(False)
    component.load_state_dict(torch.load(U.checked(oldspec['ordinary_checkpoint']),weights_only=False,map_location='cpu')['model'],strict=True)
    saved=np.load(U.checked(done['scores']),allow_pickle=False);pred=np.load(U.checked(done['predictions']),mmap_mode='r',allow_pickle=False);variances=np.load(U.checked(done['covariances']),mmap_mode='r',allow_pickle=False)
    rebuilt=np.full_like(saved,np.nan);maxima=dict(prediction_abs=0.,covariance_abs=0.,score_abs=0.);columns=np.linspace(0,1177,16).round().astype(int);challenges=[]
    for j,name in enumerate(train['subjects']):
        assert time.monotonic()-start<240
        files=kernel['subjects'][name]
        def array(key):return np.load(U.checked(files[key]),allow_pickle=False)
        tk=array('T_common_kernel.npy');tr=array('T_random_control.npy');hk=torch.from_numpy(array('Dfit_common_kernel.npy'));hr=torch.from_numpy(array('Dfit_random_control.npy'))
        hc=torch.from_numpy(np.load(U.checked(s['Dfit_c_refs'][name]),allow_pickle=False))
        models={arm:torch.load(U.checked(done['models'][name+'_'+arm]),weights_only=True,map_location='cpu') for arm in ('kernel','random','wrong_kernel')}
        m=models['kernel'];mr=models['random'];mw=models['wrong_kernel']
        base,pair,_=V.clean_means(m,hc,hk);_,random,_=V.clean_means(mr,hc,hr);_,wrong,_=V.clean_means(mw,hc,hk);_,wi,_=V.clean_means(m,hc,hk[hp])
        truth=(heldtarget-m['image_mean'])/m['image_std']
        for ti,t in enumerate(V.TIMES):
            v=prep['time_values'][str(t)];alpha=v['alpha'];sigma=v['sigma'];noisy=alpha*truth+sigma*noise[ti]/m['image_std']
            for ai,(mean,cov) in enumerate(zip([base,pair,random,wrong,wi],[m['prior_cov'],m['joint_cov'],mr['joint_cov'],mw['joint_cov'],m['joint_cov']])):
                pp,pc=V.noisy_update(mean,cov,noisy,alpha,sigma,m['image_std']);pe=float(np.abs(pp.numpy()-pred[j,:,ti,ai]).max());ce=float(np.abs(pc.numpy()-variances[j,ti,ai]).max())
                assert pe<1e-8 and ce<1e-9;maxima['prediction_abs']=max(maxima['prediction_abs'],pe);maxima['covariance_abs']=max(maxima['covariance_abs'],ce)
                rebuilt[j,:,ti,ai]=torch.stack(list(V.score(pp,truth,pc,m['image_std']).values()),1).numpy()
        lo=j*13230;ids=train['affine_ids'][lo:lo+13230];parts=[]
        with torch.no_grad():
            for i in range(0,13230,256):
                ii=ids[i:i+256];parts.append(component(train['condition'][lo+i:lo+i+256],train['weight'][ii],train['bias'][ii])[0])
        tc=torch.cat(parts).double().numpy()
        for arm,raw in [('kernel',tk),('random',tr),('wrong_kernel',tk[perm])]:
            mm={k:v.numpy() if torch.is_tensor(v) else v for k,v in models[arm].items()};cu=(target-mm['image_mean'])/mm['image_std'];cc=(tc-mm['condition_mean'])/mm['condition_std'];ee=(raw.astype(np.float64)-mm['eeg_mean'])/mm['eeg_std']
            scc=.9*(cc.T@cc)/13230+.1*np.eye(1024);suc=.9*(cu.T@cc)/13230;sec=.9*(ee[:,columns].T@cc)/13230
            auc=mm['mean_image_from_c'];aec=mm['mean_eeg_from_c'];gain=mm['EEG_gain']
            rcols=.9*(ee.T@ee[:,columns])/13230+.1*np.eye(1178)[:,columns]-aec@sec.T;cross=.9*(cu.T@ee[:,columns])/13230-auc@sec.T
            errs=[float(np.linalg.norm(auc@scc-suc)/np.linalg.norm(suc)),float(np.linalg.norm(aec[columns]@scc-sec)/np.linalg.norm(sec)),float(np.linalg.norm(gain@rcols-cross)/np.linalg.norm(cross))];assert max(errs)<1e-4
            challenges.append(dict(subject=name,arm=arm,image_normal_equation=errs[0],selected_EEG_normal_equation=errs[1],selected_gain_equation=errs[2]))
        print(json.dumps(dict(subject=name,seconds=time.monotonic()-start)),flush=True)
    maxima['score_abs']=float(np.abs(rebuilt-saved).max());assert np.allclose(rebuilt,saved,atol=1e-8,rtol=1e-9)
    assert K.readout(saved,concepts,train['subjects'])==U.read(U.checked(done['readout']))
    for ref in s['source_pins']:U.checked(ref)
    assert not torch.cuda.is_initialized()
    U.atomic(a.output/'summary.json',dict(status='ALL164000_DFIT_PREDICTIONS30_FIT_CHALLENGES_AND_FROZEN_READOUT_CPU_VERIFIED',
        source=U.ref(__file__),completion=U.ref(a.completion),T_ordinary_conditions_CPU_recomputed=132300,
        all164000_128dim_predictions_covariances_scores_replayed=True,all30_full_image_conditional_equations_checked=True,
        EEG_columns_checked=columns.tolist(),all1178_EEG_normal_equation_columns_CPU_recomputed=False,
        maxima=maxima,challenges=challenges,all_sources_unchanged=True,seconds=time.monotonic()-start,new_GPU_jobs=0,new_fit=0,new_images=0,formal_D_resumed=False))

if __name__=='__main__':main()
