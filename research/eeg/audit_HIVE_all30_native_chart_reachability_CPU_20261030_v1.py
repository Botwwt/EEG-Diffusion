"""Actual 30 EEG output affines and real LayerNorm feasibility, CPU only."""
import argparse,datetime,hashlib,json,sys,time
from pathlib import Path


def main(protocol_path):
    p=json.loads(Path(protocol_path).read_bytes());assert p['host']=='old'
    def checked(item):
        path=Path(item['path'])
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
        return path
    for item in p['input_pins']:checked(item)
    sys.path.insert(0,p['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=p['runtime_manifest']))
    sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    import brainhive_native_models_v1 as N
    from brainhive_native_chart_inverse_CPU_20261030_v1 import chart_parameters,inverse_direction,layernorm_preimage
    from brainhive_semantic_teacher_Torch_20261029_v1 import native_F_condition
    torch.set_num_threads(2);started=time.monotonic()
    prior=np.load(checked(p['prior']),allow_pickle=False);T8=np.load(checked(p['T8_inputs']),allow_pickle=False)
    contract=json.loads(checked(p['input_contract']).read_bytes())
    radius=float(prior['reference_pre_affine_radius']);Fweight=prior['F_weight'];Fbias=prior['F_bias']
    assert 0<radius<32 and contract['training_queries']==132300
    generator=np.random.default_rng(1729);random=generator.normal(size=(64,1023));random/=np.linalg.norm(random,axis=1,keepdims=True)
    intrinsic=np.concatenate((T8['visual_latent_mean'],T8['individual_EEG_unit'].reshape(-1,1023),prior['means'],random))
    assert intrinsic.shape==(184,1023)
    # The existing native map supplies Qv; the inverse is an independent NumPy formula.
    with torch.no_grad():
        Fcondition=native_F_condition(torch.from_numpy(intrinsic),torch.from_numpy(Fweight),torch.from_numpy(Fbias),1.).numpy()
    common=(Fcondition-Fbias)/Fweight
    records=[];weights=[];biases=[];distances=[];epsilons=[];OOF_count=0
    for subject in contract['subjects']:
        for source in subject['sources']:
            checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256']))
            payload=torch.load(source['checkpoint_path'],map_location='cpu',weights_only=True,mmap=True)
            model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False)
            model.load_state_dict(payload['model'],strict=True)
            norm=model.block.layers[0].norm.double()
            gamma=norm.weight.detach().numpy().copy();beta=norm.bias.detach().numpy().copy();epsilon=float(norm.eps)
            parameters=chart_parameters(gamma,beta,Fweight,Fbias,radius)
            origin=parameters['origin'];extra=origin-origin.mean()
            extra=extra/np.linalg.norm(extra) if np.linalg.norm(extra)>1e-12 else common[0]
            directions=np.concatenate((common,extra[None],-extra[None]))
            condition,u,certificate=inverse_direction(directions,gamma,beta,Fweight,Fbias,radius)
            pre=layernorm_preimage(u,epsilon)
            with torch.no_grad():actual=norm(torch.from_numpy(pre)).numpy()
            native_error=float(np.max(abs(actual-condition)))
            assert native_error<1e-9
            Fraw=(actual-Fbias)/Fweight;Fraw-=Fraw.mean(axis=1,keepdims=True);Fraw/=np.linalg.norm(Fraw,axis=1,keepdims=True)
            direction_error=float(np.max(abs(Fraw-directions)));assert direction_error<1e-12
            zero_error=float(np.max(abs(u.mean(axis=1))));radius_error=float(np.max(abs(np.linalg.norm(u,axis=1)-radius)))
            assert zero_error<1e-12 and radius_error<1e-10
            empirical_radius=None
            if source['role'] in ('oof0','oof1'):
                checked(dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256']))
                cache=torch.load(source['condition_cache_path'],map_location='cpu',weights_only=True,mmap=True)
                assert cache['model_sha256']==source['checkpoint_sha256']
                group=source['groups']['component_train'];indices=[row['cache_row'] for row in group]
                assert cache['array_rows'][indices].tolist()==[row['array_row'] for row in group]
                c=cache['condition'][indices].double().numpy();old=(c-beta)/gamma;old_radius=np.linalg.norm(old,axis=-1)
                empirical_radius=dict(queries=len(indices),minimum=float(old_radius.min()),median=float(np.median(old_radius)),maximum=float(old_radius.max()))
                assert float(np.linalg.norm(origin))<old_radius.min()
                OOF_count+=len(indices)
                del cache,c,old,old_radius
            else:
                assert source['role']=='full' and 'component_train' not in source['groups']
            record=dict(subject=subject['subject'],role=source['role'],checkpoint_SHA256=source['checkpoint_sha256'],gamma_min=float(gamma.min()),gamma_max=float(gamma.max()),output_LayerNorm_epsilon=epsilon,EEG_sphere_witness_radius=radius,origin_norm=float(np.linalg.norm(origin)),all_direction_coverage_margin_squared=parameters['positive_radius_gap'],sampled_directions=len(directions),ray_min=float(certificate['ray_distance'].min()),ray_max=float(certificate['ray_distance'].max()),actual_LayerNorm_condition_maxabs=native_error,F_direction_roundtrip_maxabs=direction_error,EEG_pre_affine_zero_mean_maxabs=zero_error,EEG_pre_affine_radius_maxabs=radius_error,actual_OOF_T_radius=empirical_radius,whole_EEG_network_reachability_or_predictability_proven=False)
            records.append(record);weights.append(gamma);biases.append(beta);distances.append(certificate['ray_distance']);epsilons.append(epsilon)
            assert not torch.cuda.is_initialized() and time.monotonic()-started<95
            del model,norm,payload
    assert len(records)==30 and OOF_count==132300
    for subject in contract['subjects']:
        for source in subject['sources']:checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256']))
    out=Path(p['output_root'])
    np.savez_compressed(out/'all30_affines_and_direction_certificates.npz',EEG_weights=np.stack(weights),EEG_biases=np.stack(biases),F_weight=Fweight,F_bias=Fbias,intrinsic_common_directions=intrinsic,ray_distances=np.stack(distances),epsilons=np.array(epsilons),witness_radius=np.array(radius))
    result=dict(status='COMPLETE_ALL30_AFFINE_LAYERNORM_CHART_FEASIBILITY_CPU',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,existing_models=30,actual_OOF_T_conditions=OOF_count,analytic_positive_ray_coverage_criterion_passed=30,actual_LayerNorm_forward_samples=30*186,new_EEG_optimizer_updates=0,new_models_saved=0,new_research_images=0,new_GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),NEW_contacted=False,scope='affine-output-normalization chart attainability only; not EEG network capacity, neural predictability, calibrated posterior or image gain',seconds=time.monotonic()-started)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);main(parser.parse_args().protocol)
