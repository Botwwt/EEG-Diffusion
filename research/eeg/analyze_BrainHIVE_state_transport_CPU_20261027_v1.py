"""D18 full source/numeric/endpoint checks and explicitly selected CPU ODE replays."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_state_transport_20261027_v1 as V
import brainhive_state_transport_data_20261027_v1 as DATA
import brainhive_cross_encoder_20261027_v1 as M

def main():
    p=argparse.ArgumentParser()
    for key in ('completion','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();torch.set_num_threads(2);assert not torch.cuda.is_initialized() and not a.output.exists();a.output.mkdir(parents=True)
    done=U.read(a.completion);assert done['status']=='D18_MATCHED_TWO_GLOBAL_FIELDS_FULL_T10_AND_ALL_SOURCE_DIAGNOSTICS_COMPLETE'
    spec=U.read(U.checked(done['spec']));saved=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu',mmap=True)
    assert saved['epoch']==10 and saved['updates_per_field']==1300 and saved['exposures_per_field']==1323000 and saved['steps']==64
    for arm in V.ARMS:
        steps=[int(x['step']) for x in saved['optimizers'][arm]['state'].values()];assert len(steps)>0 and set(steps)=={1300}
    train,metadata=DATA.training_inputs('cpu');mu,scale=V.shared_coordinates(train['condition'],train['target'])
    assert torch.allclose(mu,saved['mean'],atol=2e-7,rtol=2e-7) and torch.allclose(scale,saved['scale'],atol=2e-7,rtol=2e-7)
    coupling=U.read(U.checked(done['endpoint_coupling']));assert coupling['T_array_rows']==train['array_rows']
    expected=V.concept_endpoint_permutation(train['array_rows'],metadata);assert expected.tolist()==coupling['sameconcept_wrong_endpoint_positions']
    del train
    dev=DATA.development_inputs(spec,'cpu');queries=dev['queries'];groups=dev['groups']
    parent=np.load(U.checked(done['Dfit_source']),allow_pickle=False);target=np.load(U.checked(done['Dfit_targets']),allow_pickle=False)
    assert np.array_equal(dev['raw'].numpy(),parent) and np.array_equal(dev['target'].numpy(),target)
    modelmap=U.read(U.checked(done['Dfit_queries']));assert modelmap['queries']==queries and modelmap['groups']==groups
    condition=np.load(U.checked(done['Dfit_conditions']),allow_pickle=False);back=np.load(U.checked(done['Dfit_roundtrip']),allow_pickle=False)
    lock=U.read(U.checked(done['condition_lock']));dc=np.load(U.checked(lock['source_conditions']),allow_pickle=False);bank=np.load(U.checked(lock['conditions']),allow_pickle=False)
    dcback=np.load(U.checked(done['Dcheck_roundtrip']),allow_pickle=False);refine=np.load(U.checked(done['Dcheck_refined128']),allow_pickle=False)
    assert condition.shape==back.shape==(8200,2,1024) and bank.shape==dcback.shape==refine.shape==(400,2,1024)
    assert all(x.dtype==np.float32 and np.isfinite(x).all() for x in [parent,target,condition,back,dc,bank,dcback,refine])
    d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']))
    assert np.array_equal(dc,np.load(U.checked(d09['check_conditions']))[:,0]) and lock['query_ids']==[q['query_id'] for q in panel['Dcheck']]
    diagnostic=U.read(U.checked(done['diagnostics']));numbers={};geometry={};replays=[];allpass=True
    selected=[index for subject in range(10) for index in [subject*820,subject*820+819]];dcselected=list(range(0,80,10))
    for j,arm in enumerate(V.ARMS):
        errors=dict(Dfit_max_relative_roundtrip=float((np.linalg.norm(back[:,j]-parent,axis=-1)/np.linalg.norm(parent,axis=-1)).max()),
            Dcheck_max_relative_roundtrip=float((np.linalg.norm(dcback[:,j]-dc,axis=-1)/np.linalg.norm(dc,axis=-1)).max()),
            Dcheck_max_relative64_vs128_refinement=float((np.linalg.norm(refine[:,j]-bank[:,j],axis=-1)/np.linalg.norm(dc,axis=-1)).max()))
        for key,value in errors.items():assert abs(value-diagnostic['arms'][arm][key])<2e-6
        allpass &= max(errors.values())<=.01;numbers[arm]=errors
        pairs=np.triu_indices(10,1);x=parent.reshape(-1,10,1024);y=condition[:,j].reshape(-1,10,1024)
        ratios=np.linalg.norm(y[:,pairs[0]]-y[:,pairs[1]],axis=-1)/np.linalg.norm(x[:,pairs[0]]-x[:,pairs[1]],axis=-1)
        quantiles=np.quantile(ratios,[0.,.01,.05,.5,.95,.99,1.]);assert ratios.size==36900
        assert np.allclose(quantiles,diagnostic['geometry'][arm]['distance_ratio_quantiles'],atol=3e-6,rtol=3e-6)
        geometry[arm]=dict(within_concept_pairs=36900,distance_ratio_quantiles=quantiles.tolist(),fraction_below_point1=float((ratios<.1).mean()),
            raw_condition_norm_mean=float(np.linalg.norm(condition[:,j],axis=-1).mean()),original_norm_mean=float(np.linalg.norm(parent,axis=-1).mean()))
        model=V.StateFlow().eval().requires_grad_(False);model.load_state_dict(saved['models'][arm],strict=True)
        for name,value,indices,expected_value,reverse in [
            ('Dfit_forward64',parent,selected,condition[:,j],False),('Dfit_reverse64',condition[:,j],selected,back[:,j],True),
            ('Dcheck_forward64',dc,dcselected,bank[:,j],False),('Dcheck_forward128',dc,dcselected,refine[:,j],False)]:
            actual=V.transport(model,torch.from_numpy(np.array(value[indices],copy=True)),saved['mean'],saved['scale'],128 if name.endswith('128') else 64,reverse).numpy()
            error=np.linalg.norm(actual-expected_value[indices],axis=-1)/np.linalg.norm(expected_value[indices],axis=-1)
            assert error.max()<3e-5
            replays.append(dict(arm=arm,operation=name,query_indices=indices,maximum_relative_L2=float(error.max())))
    assert bool(allpass)==done['numerical_gate_passed']==diagnostic['numerical_gate_passed']
    truth=torch.from_numpy(target);values={'original_parent':M.metrics(torch.from_numpy(parent),truth,groups)}
    for j,arm in enumerate(V.ARMS):values[arm]=M.metrics(torch.from_numpy(condition[:,j]),truth,groups)
    means={arm:{key:float(v.mean()) for key,v in value.items()} for arm,value in values.items()}
    subjects=[q['subject_id'] for q in queries];concepts=[q['concept_id'] for q in queries]
    contrasts={arm:M.paired_difference(values[V.ARMS[0]],value,subjects,concepts) for arm,value in values.items() if arm!=V.ARMS[0]}
    fields=list(queries[0])+['arm']+list(values['original_parent']);out=a.output/'Dfit_per_query.csv'
    with out.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,q in enumerate(queries):
            for arm,value in values.items():writer.writerow(q|dict(arm=arm,**{key:float(v[i]) for key,v in value.items()}))
    logs=[json.loads(x) for x in (a.completion.parent/'training_log.jsonl').read_text().splitlines()]
    train=[x for x in logs if x['stage']=='COMPLETE_T_PASS'];held=[x for x in logs if x['stage']=='FIXED_DFIT200_CORRECT_VELOCITY_LOG_ONLY']
    assert len(train)==10 and [x['epoch'] for x in train]==list(range(1,11)) and len(held)==11 and [x['epoch'] for x in held]==list(range(11))
    assert all(x['full_coverage_min']==x['full_coverage_max']==1 and x['updates_per_field']==130*x['epoch'] and x['exposures_per_field']==132300*x['epoch'] for x in train)
    plot=None
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for arm in V.ARMS:
            axes[0].plot(range(1,11),[x['velocity_MSE'][arm] for x in train],label=arm)
            axes[1].plot(range(11),[x['velocity_MSE'][arm] for x in held],label=arm)
        for ax,title in zip(axes,['All10 OOF-T velocity MSE','Dfit200 correct-path velocity MSE, logs only']):ax.set(title=title,xlabel='Fixed complete T passes');ax.legend(fontsize=7)
        fig.tight_layout();path=a.output/'actual_training_and_held_curves.png';fig.savefig(path,dpi=160);plt.close(fig);plot=U.ref(path)
    except ImportError:pass
    for item in spec['source_pins']:U.checked(item)
    assert not torch.cuda.is_initialized()
    result=dict(status='D18_FULL_SOURCE_NUMERICAL_AND_ALL24600_ENDPOINT_SCORES_WITH_SELECTED_CPU_ODE_REPLAYS_VERIFIED',completion=U.ref(a.completion),source=U.ref(__file__),
        all10_original_Dfit_source_and_F_identities_bitwise_verified=True,shared_T_only_coordinates_recomputed=True,full13230_endpoint_permutation_verified=True,
        all8200x2_and400x2_stored_roundtrip_errors_recomputed=True,all400x2_stored64_vs128_errors_recomputed=True,numerical_gate_passed=bool(allpass),
        independent_CPU_ODE_replays=replays,CPU_ODE_scope='20 Dfit(first/last per10subjects) perfield forward64/reverse64;8 Dcheck(first of8concepts) perfield forward64/128. This is not a full8200 CPU ODE rerun.',
        numerical_errors=numbers,full36900_sameconcept_distance_ratios_per_field=geometry,full_Dfit_means=means,paired_minus_references=contrasts,
        contrast_sign='candidate minus reference; negative improves errors',all10_full_T_passes_and_11_held_logs_verified=True,optimizer_steps_per_field=1300,
        per_query=U.ref(out),per_query_rows=24600,feature_gate_applied=False,held_interpolation_states_contain_F_and_are_not_inference=True,
        actual_plot=plot,seconds=time.monotonic()-start,new_GPU_jobs=0,new_images=0,new_parameter_updates=0,
        numerical_inverse_does_not_establish_information_or_image_gain=True,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],numbers=numbers,means=means)))

if __name__=='__main__':main()
