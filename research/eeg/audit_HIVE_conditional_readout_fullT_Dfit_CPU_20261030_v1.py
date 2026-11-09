"""Frozen natural-direction/conditional-moment diagnostics; no model fitting."""
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path


def main(protocol_path):
    p=json.loads(Path(protocol_path).read_bytes());assert p['host']=='old'
    def checked(item):
        path=Path(item['path'])
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
        return path
    for item in p['input_pins']:checked(item)
    sys.path.insert(0,p['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=p['runtime_manifest']));sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    from brainhive_semantic_teacher_Torch_20261029_v1 import F_native_coordinates
    from brainhive_conditional_moment_readout_CPU_20261030_v1 import conditional_moments,latent_chordal_action,model_chordal_risk
    from brainhive_conditional_vmf_sampler_CPU_20261030_v1 import sample_conditional_mixture
    from brainhive_EEG_joint_density_M_step_CPU_20261030_v1 import joint_value_and_gradients,Torch_joint_first_order_loss
    torch.set_num_threads(2);started=time.monotonic()
    prior=np.load(checked(p['prior']),allow_pickle=False);T8=np.load(checked(p['T8_inputs']),allow_pickle=False)
    contract=json.loads(checked(p['input_contract']).read_bytes());split=json.loads(checked(p['split_manifest']).read_bytes())
    assert len(contract['subjects'])==10
    target=torch.load(checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),map_location='cpu',weights_only=True,mmap=True)
    weight=torch.from_numpy(prior['F_weight']);bias=torch.from_numpy(prior['F_bias'])
    def unit(c):
        with torch.no_grad():u=F_native_coordinates(c,weight,bias);u=u/u.norm(dim=-1,keepdim=True)
        return u.numpy()
    first_full=next(source for source in contract['subjects'][0]['sources'] if source['role']=='full')
    rows_by_split={'T':split['image_rows']['T'],'Dfit':[item['array_row'] for item in first_full['groups']['calibration_only_Dfit']]}
    assert len(rows_by_split['Dfit'])==820 and len(set(rows_by_split['Dfit']))==820
    assert set(rows_by_split['Dfit']).issubset(split['image_rows']['D']) and not set(rows_by_split['Dfit'])&set(rows_by_split['T'])
    targets={name:unit(target[rows_by_split[name]]) for name in ('T','Dfit')}
    all_stats=[];all_subject=[];all_split=[];all_rows=[];summaries=[];counts={'T':0,'Dfit':0};min_risk_improvement=0.
    t8_moments=[];sample_checks=[];scale_checks=[]
    for subject_index,subject in enumerate(contract['subjects']):
        scale=float(T8['subject_strengths'][subject_index]);full=next(source for source in subject['sources'] if source['role']=='full')
        for name in ('T','Dfit'):
            rows=rows_by_split[name];position={row:i for i,row in enumerate(rows)};condition=torch.empty(len(rows),1024,dtype=torch.float64);coverage=np.zeros(len(rows),dtype=np.int16)
            sources=[source for source in subject['sources'] if source['role'] in ('oof0','oof1')] if name=='T' else [full]
            group='component_train' if name=='T' else 'calibration_only_Dfit'
            for source in sources:
                checked(dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256']))
                cache=torch.load(source['condition_cache_path'],map_location='cpu',mmap=True,weights_only=True)
                assert cache['model_sha256']==source['checkpoint_sha256']
                records=source['groups'][group];dst=np.array([position[item['array_row']] for item in records]);src=np.array([item['cache_row'] for item in records])
                assert cache['array_rows'][src].tolist()==[item['array_row'] for item in records]
                condition[dst]=cache['condition'][src].double();coverage[dst]+=1
                del cache
            assert np.all(coverage==1)
            mu=unit(condition);moment,information=conditional_moments(prior['weights'],prior['means'],prior['concentrations'],mu,scale)
            action,readout=latent_chordal_action(moment)
            rawcos=np.einsum('ij,ij->i',mu,targets[name]);meancos=np.einsum('ij,ij->i',action,targets[name]);samplecos=np.einsum('ij,ij->i',moment,targets[name])
            advantage=model_chordal_risk(moment,mu)-model_chordal_risk(moment,action);assert advantage.min()>-1e-10
            min_risk_improvement=min(min_risk_improvement,float(advantage.min()))
            array=np.column_stack((rawcos,meancos,samplecos,readout['moment_norm'],advantage,information['log_evidence']))
            all_stats.append(array);all_subject.extend([subject_index]*len(rows));all_split.extend([0 if name=='T' else 1]*len(rows));all_rows.extend(rows);counts[name]+=len(rows)
            summaries.append(dict(subject=subject['subject'],split=name,queries=len(rows),evidence_F_cosine=float(rawcos.mean()),conditional_mean_direction_F_cosine=float(meancos.mean()),single_predictive_draw_expected_F_cosine=float(samplecos.mean()),conditional_mean_norm=float(readout['moment_norm'].mean()),latent_model_risk_reduction_of_mean=float(advantage.mean()),canonical_zero_moment_ties=int(readout['canonical_zero_moment_tie'].sum())))
            if name=='T':
                subset=moment[T8['positions']];t8_moments.append(subset)
                for j,point in enumerate(T8['positions']):
                    values,draw=sample_conditional_mixture(prior['weights'],prior['means'],prior['concentrations'],mu[point],scale,512,np.random.default_rng(1729+subject_index*100+j))
                    error=float(np.linalg.norm(values.mean(axis=0)-moment[point]));SE=float(np.sqrt((1-np.sum(moment[point]**2))/512));assert error<5*SE
                    sample_checks.append(dict(subject=subject['subject'],position=int(point),samples=512,mean_vector_error=error,reference_SE=SE,proposals=draw['proposals']))
                    # Geometry/oracle direction capacity check only: not an EEG result.
                    teacher=targets[name][point]*(100000./101022.)
                    c=targets[name][point]
                    value,unused,log_gradient,unused2=joint_value_and_gradients(c,teacher,prior['weights'],prior['means'],prior['concentrations'],np.log(scale))
                    step=1e-5
                    plus=joint_value_and_gradients(c,teacher,prior['weights'],prior['means'],prior['concentrations'],np.log(scale)+step)[0]
                    minus=joint_value_and_gradients(c,teacher,prior['weights'],prior['means'],prior['concentrations'],np.log(scale)-step)[0]
                    error=abs((plus-minus)/(2*step)-log_gradient);assert error<1e-5
                    scale_checks.append(dict(subject=subject['subject'],position=int(point),aligned_visual_direction_is_privileged_not_EEG=True,existing_log_scale_gradient=log_gradient,independent_finite_difference_error=error))
            del condition,mu,moment,action,information,array
            assert not torch.cuda.is_initialized() and time.monotonic()-started<100
    assert counts=={'T':132300,'Dfit':8200}
    # Autograd test includes a real 1023dim prior, a shared scalar, and teacher detach.
    raw=torch.tensor(T8['individual_EEG_unit'][0],requires_grad=True)
    teacher=torch.tensor(T8['visual_latent_mean']*(100000./101022.),requires_grad=True)
    log_scale=torch.tensor(np.log(float(T8['subject_strengths'][0])),dtype=torch.float64,requires_grad=True)
    loss=Torch_joint_first_order_loss(raw,teacher,log_scale,prior['weights'],prior['means'],prior['concentrations']);loss.mean().backward()
    reference=[joint_value_and_gradients(c,q,prior['weights'],prior['means'],prior['concentrations'],float(log_scale.detach())) for c,q in zip(raw.detach().numpy(),teacher.detach().numpy())]
    assert teacher.grad is None and torch.isfinite(raw.grad).all() and torch.isfinite(log_scale.grad)
    assert np.max(abs(raw.grad.numpy()-np.stack([record[1] for record in reference])/8))<1e-12
    scalar_error=abs(float(log_scale.grad)-np.mean([record[2] for record in reference]));assert scalar_error<1e-10
    out=Path(p['output_root']);np.savez_compressed(out/'all140500_conditional_readout_statistics.npz',values=np.concatenate(all_stats),subject=np.array(all_subject,dtype=np.int16),split=np.array(all_split,dtype=np.int8),array_row=np.array(all_rows,dtype=np.int32),T8_conditional_moments=np.stack(t8_moments))
    result=dict(status='COMPLETE_FULL_T_DFIT_FROZEN_CONDITIONAL_READOUT_AND_SCALE_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=counts,summaries=summaries,sampling_checks=sample_checks,scale_gradient_checks=scale_checks,shared_log_scale_autograd_error=scalar_error,statistics_columns=['evidence_F_cosine','mean_action_F_cosine','single_predictive_draw_expected_F_cosine','conditional_mean_norm','latent_model_risk_raw_minus_mean','log_evidence'],minimum_model_latent_risk_reduction=min_risk_improvement,new_fitted_parameters=0,new_EEG_optimizer_updates=0,new_research_images=0,new_GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),NEW_contacted=False,T_prior_is_in_sample=True,Dfit_not_used_to_fit_or_select_readout=True,Dcheck_or_V_or_official_rows_used=False,main_draft_readout='one normalized conditional-mixture draw with RNG independent of generatornoise; mean only latent-risk diagnostic',calibrated_posterior_or_image_gain_established=False,scope='frozen old encoders/workprior, not trained new EEG model or actual G efficacy; aligned-direction scalar checks use privilegedTvisual directions',seconds=time.monotonic()-started)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);main(parser.parse_args().protocol)
