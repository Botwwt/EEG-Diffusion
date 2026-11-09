"""D16 frozen readout, native sphere, full CSV, T normalizer and CPU forward audit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_response_residual_20261027_v1 as V
import brainhive_response_residual_data_20261027_v1 as DATA


def main():
    p=argparse.ArgumentParser()
    for n in ('completion','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True);torch.set_num_threads(2)
    assert not torch.cuda.is_initialized();done=U.read(a.completion);assert done['status']=='D16_FOUR_MATCHED_HEADS_FULL_T10_AND_ALL8200_DFIT_COMPLETE'
    spec=U.read(U.checked(done['spec']));exported=U.read(U.checked(done['response_export']))
    for item in spec['source_pins']:U.checked(item)
    saved=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu',mmap=True);state=saved['state']
    assert saved['epoch']==10 and saved['updates_per_head']==1330 and saved['query_exposures_per_head']==1323000
    old=U.read(U.checked(spec['D15_probe']));prior=torch.load(U.checked(old['model']),weights_only=True,map_location='cpu',mmap=True)
    for key in ('c_mean','c_scale','ATMS_mean','ATMS_scale','F_mean','F_scale','F_weight','F_bias'):assert torch.equal(state[key],prior[key])
    assert torch.equal(state['Bc'],prior['models']['c_to_y'])
    # Independent streaming float64 reconstruction of ALL T response statistics.
    bank=np.load(U.checked(exported['T_responses']),allow_pickle=False,mmap_mode='r');assert bank.shape==(132300,4,1024)
    means=[];scales=[]
    for si in range(10):
        total=np.zeros((4,1024),np.float64);squares=np.zeros(4,np.float64)
        for lo in range(0,13230,1024):
            value=np.array(bank[si*13230+lo:si*13230+min(lo+1024,13230)],dtype=np.float64,copy=True)
            total+=value.sum(0);squares+=(value*value).sum((0,2))
        mean=total/13230;scale=np.sqrt(squares/(13230*1024)-(mean*mean).mean(-1));means.append(mean);scales.append(scale[:,None])
    mean=np.stack(means);scale=np.stack(scales)
    mean_error=float(np.abs(mean-state['response_mean'].numpy()).max());scale_error=float(np.abs(scale-state['response_scale'].numpy()).max())
    assert np.allclose(mean,state['response_mean'].numpy(),rtol=1e-5,atol=1e-5) and np.allclose(scale,state['response_scale'].numpy(),rtol=1e-5,atol=1e-5)
    data=DATA.load_Dfit(spec,state,'cpu');mapping=U.read(U.checked(done['queries_and_controls']));assert data['queries']==mapping['queries'] and data['groups']==mapping['groups']
    assert np.array_equal(data['permutation'].numpy(),mapping['permutation'])
    target=np.load(U.checked(done['targets']),allow_pickle=False);assert np.array_equal(target,data['target'].numpy())
    predictions=np.load(U.checked(done['predictions']),allow_pickle=False,mmap_mode='r');conditions=np.load(U.checked(done['conditions']),allow_pickle=False,mmap_mode='r')
    radius=np.load(U.checked(done['used_radius']),allow_pickle=False);assert predictions.shape==conditions.shape==(8200,4,1024) and radius.shape==(8200,4)
    assert np.isfinite(predictions).all() and np.isfinite(conditions).all() and radius.min()>=0 and radius.max()<=1+1e-6
    rows=list(csv.DictReader(U.checked(done['per_query']).open()));assert len(rows)==49200
    names=list(V.ARMS)+['linear_c','linear_joint'];values={arm:{k:np.empty(8200,np.float64 if k=='MSE' else np.float32) for k in ('MSE','target_cosine','within10_log_loss','within10_top1','within10_cosine_margin')} for arm in names}
    for ordinal,row in enumerate(rows):
        i,j=divmod(ordinal,6);arm=names[j];q=data['queries'][i];assert row['arm']==arm
        for key,value in q.items():assert row[key]==str(value)
        for key in values[arm]:values[arm][key][i]=float(row[key])
    metric_errors={}
    for j,arm in enumerate(names):
        prediction=torch.from_numpy(np.array(predictions[:,j],copy=True)) if j<4 else data['linear_predictions'][:,0 if arm=='linear_c' else 2]
        actual=V.OLD.metrics(prediction,data['target'],data['groups']);metric_errors[arm]={key:float(np.abs(value-values[arm][key]).max()) for key,value in actual.items()}
        assert all(x==0. for x in metric_errors[arm].values())
    rebuilt=V.readout({a:values[a] for a in V.ARMS},{a:values[a] for a in names[4:]},[q['subject_id'] for q in data['queries']],[q['concept_id'] for q in data['queries']])
    assert rebuilt==U.read(U.checked(done['readout']))
    # Actual native conditions obey the inherited sphere, independently of proxy metrics.
    base_u=data['u'].numpy();modified=(conditions-data['bias'].numpy()[:,None])/data['weight'].numpy()[:,None]
    norm=np.linalg.norm(base_u,axis=-1);modified_norm=np.linalg.norm(modified,axis=-1)
    norm_error=float(np.abs(modified_norm/norm[:,None]-1).max());cosine=(modified*base_u[:,None]).sum(-1)/(modified_norm*norm[:,None])
    assert norm_error<1e-5 and cosine.min()>=1/np.sqrt(1+.25**2)-1e-5
    # CPU float32 head replay: first stimulus of every concept and subject, all4.
    selected=np.array([group[0] for group in data['groups']]);assert len(selected)==820
    ix=torch.as_tensor(selected);sid=data['subject'][ix];u=data['u'][ix];c=(u-state['c_mean'][sid])/state['c_scale'][sid];atms=data['atms'][ix]
    replay={};interventions={}
    for j,arm in enumerate(V.ARMS):
        model=V.ResponseAdapter().eval().requires_grad_(False);model.load_state_dict(saved['models'][arm],strict=True)
        response=data['responses'][data['permutation'][ix]] if arm=='shuffled_responses' else data['responses'][ix]
        if arm=='c_only':context=c[:,None].expand(-1,5,-1)
        elif arm=='final_ATMS':context=atms[:,None].expand(-1,5,-1)
        else:context=torch.cat([atms[:,None],response],1)
        with torch.no_grad():correction,_=model(c,context,u);native=data['condition'][ix]+data['weight'][ix]*correction
        reference=np.array(conditions[selected,j],copy=True);actual=native.numpy();relative=np.linalg.norm(actual-reference,axis=-1)/np.linalg.norm(reference,axis=-1)
        cs=(actual*reference).sum(-1)/(np.linalg.norm(actual,axis=-1)*np.linalg.norm(reference,axis=-1))
        replay[arm]=dict(maximum_relative_L2=float(relative.max()),minimum_cosine=float(cs.min()),queries=820)
        assert relative.max()<.03 and cs.min()>.999
        if arm=='paired_responses':
            # Fixed inputs, no refitting or image generation; potentially off
            # the C/A/response joint support, so sensitivity only.
            for label,other_a,other_r in [
                ('correct',atms,data['responses'][ix]),
                ('shuffled_response_only',atms,data['responses'][data['permutation'][ix]]),
                ('shuffled_final_ATMS_only',data['atms'][data['permutation'][ix]],data['responses'][ix]),
                ('both_auxiliary_views_shuffled',data['atms'][data['permutation'][ix]],data['responses'][data['permutation'][ix]])]:
                ctx=torch.cat([other_a[:,None],other_r],1)
                with torch.no_grad():delta,_=model(c,ctx,u);yhat=((u+delta-state['c_mean'][sid])/state['c_scale'][sid])@state['Bc']
                native_changed=(data['condition'][ix]+data['weight'][ix]*delta).numpy()
                interventions[label]=dict(queries=820,proxy_MSE=float((yhat-data['target'][ix]).double().square().mean()),
                    native_condition_relative_change_mean=float((np.linalg.norm(native_changed-actual,axis=-1)/np.linalg.norm(actual,axis=-1)).mean()))
    logs=[json.loads(x) for x in (Path(done['model']['path']).parent/'training_log.jsonl').read_text().splitlines()]
    passes=[x for x in logs if x['stage']=='COMPLETE_T_PASS'];held=[x for x in logs if x['stage']=='FIXED_DFIT200_LOG_ONLY']
    assert len(passes)==10 and len(held)==11 and [x['updates_per_head'] for x in passes]==[133*i for i in range(1,11)]
    assert [x['exposures_per_head'] for x in passes]==[132300*i for i in range(1,11)]
    provenance=U.read(Path(done['model']['path']).parent/'provenance.json');assert len(set(provenance['initial_state_sha256'].values()))==1
    branch=U.read(Path(done['model']['path']).parent/'actual_response_branch_gradient.json');assert branch['finite_nonzero'] is True and branch['token_projection_gradient_norm']>0
    plot=None
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,3.8));epochs=[x['epoch'] for x in passes]
        for arm in V.ARMS:
            axes[0].plot(epochs,[x['means'][arm]['MSE'] for x in passes],label=arm)
            axes[1].plot([x['epoch'] for x in held],[x['values'][arm]['MSE'] for x in held],label=arm)
            axes[2].plot([x['epoch'] for x in held],[x['values'][arm]['within10_CE'] for x in held],label=arm)
        for ax,title in zip(axes,['Complete T pass MSE','Fixed Dfit200 MSE (logging)','Fixed Dfit200 within10 CE']):ax.set(title=title,xlabel='Complete T passes');ax.legend(fontsize=7)
        fig.tight_layout();path=a.output/'actual_training_and_Dfit200_curves.png';fig.savefig(path,dpi=160);plt.close(fig);plot=U.ref(path)
    except ImportError:pass
    result=dict(status='D16_COMPLETE_ALL_DFIT_METRICS_GATE_NATIVE_GEOMETRY_AND_SELECTED_HEAD_CPU_AUDIT',source=U.ref(__file__),completion=U.ref(a.completion),
        all132300_T_response_normalizers_recomputed_float64=True,response_mean_max_absolute_error=mean_error,response_scale_max_absolute_error=scale_error,
        all8200_targets_bitwise_equal=True,all49200_perquery_identities_verified=True,all_six_arm_metrics_exact=metric_errors,
        full_readout_and_subject_concept_bootstrap_recomputed_exactly=True,all_native_conditions_preserve_norm_max_relative_error=norm_error,
        native_direction_min_cosine=float(cosine.min()),CPU_selected_head_forward_replay=replay,head_replayed_queries_per_arm=820,
        paired_head_fixed_input_auxiliary_interventions=interventions,auxiliary_interventions_are_postresult_sensitivity_not_causal_or_calibration_proof=True,
        original_head_forward_dtype='BF16, FP32 native geometry/readout',CPU_head_forward_dtype='FP32',all8200_heads_forward_CPU_replayed=False,
        all10_T_passes_logged=passes,all11_Dfit200_logs=held,initial_states_all4_bitwise_identical=True,response_branch_gradient=branch,
        radius_summary={arm:dict(mean=float(radius[:,j].mean()),maximum=float(radius[:,j].max()),fraction_at_least99percent=float((radius[:,j]>=.99).mean())) for j,arm in enumerate(V.ARMS)},
        actual_curve=plot,prediction_gate_passed=rebuilt['prediction_gate_passed'],decision=rebuilt['decision'],
        no_model_or_condition_modified=True,new_GPU_jobs=0,new_images=0,seconds=time.monotonic()-start,independent_confirmation=False,SOTA_established=False)
    for item in spec['source_pins']:U.checked(item)
    assert not torch.cuda.is_initialized();U.atomic(a.output/'summary.json',result)
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],CPU_head_replay=replay,radius=result['radius_summary'],decision=result['decision'])))


if __name__=='__main__':main()
