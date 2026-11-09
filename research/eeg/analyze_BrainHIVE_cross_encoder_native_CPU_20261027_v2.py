"""D15 native-CLIP coordinate recheck: full8200x4 CPU replay and fixed gate audit."""
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
import brainhive_cross_encoder_20261027_v1 as V

R=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    for n in ('completion','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();start=time.monotonic();assert not a.output.exists();a.output.mkdir(parents=True)
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    done=U.read(a.completion);assert done['status']=='D15_ALL10_COMPLETE_T_FIT_AND_ALL_DFIT_LINEAR_PROBE_COMPLETE'
    spec=U.read(U.checked(done['spec']));exported=U.read(U.checked(done['ATMS_export']));inputs=U.read(U.checked(exported['inputs']))
    for item in spec['source_pins']:U.checked(item)
    saved=torch.load(U.checked(done['model']),weights_only=True,map_location='cpu');models=saved['models']
    mapping=U.read(U.checked(done['queries_and_controls']));queries=mapping['queries'];groups=mapping['groups'];permutation=np.array(mapping['shuffled_innovation_source_indices'])
    assert len(queries)==8200 and len(groups)==820 and sorted(np.concatenate(groups).tolist())==list(range(8200))
    split=U.read(R/'configs/split_manifest.json');metadata={int(x['array_row']):x for x in csv.DictReader((R/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    expected=set(split['d_fit_concepts']);assert {q['concept_id'] for q in queries}==expected and len(expected)==82
    assert not {q['array_row'] for q in queries}&set(split['image_rows']['T']+split['image_rows']['V'])
    for i,q in enumerate(queries):
        assert q['subject_id']==f'sub-{i//820+1:02d}' and q['concept_id']==metadata[q['array_row']]['concept_id'] and q['stimulus_id']==metadata[q['array_row']]['stimulus_id']
        other=queries[permutation[i]];assert other['subject_id']==q['subject_id'] and other['concept_id']!=q['concept_id']
    contract=U.read('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
    rawF=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    ad=np.load(U.checked(exported['D_ATMS_full']),allow_pickle=False,mmap_mode='r');dl={row:i for i,row in enumerate(inputs['D_array_rows'])}
    affines=torch.load(U.checked(exported['affines']),weights_only=True,map_location='cpu');cx=[];ax=[];targets=[]
    for si,subject in enumerate(contract['subjects']):
        name=subject['subject'];source=next(x for x in subject['sources'] if x['role']=='full')
        cache=torch.load(U.checked(dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
        state=torch.load(U.checked(dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])),weights_only=True,map_location='cpu')['model']
        rows=source['groups']['calibration_only_Dfit'];ids=[q['array_row'] for q in rows];ix=[q['cache_row'] for q in rows]
        assert cache['array_rows'][ix].tolist()==ids==[q['array_row'] for q in queries[si*820:(si+1)*820]]
        c=(cache['condition'][ix]-state['block.layers.0.norm.bias'])/state['block.layers.0.norm.weight']
        aa=torch.from_numpy(np.array(ad[[si*1650+dl[row] for row in ids]],copy=True));affine=affines[name+'/full']
        assert saved['ATMS_coordinate']=='original_raw_pooled_CLIP_H_output'  # keep raw shared CLIP coordinates
        cx.append((c-saved['c_mean'][si])/saved['c_scale'][si]);ax.append((aa-saved['ATMS_mean'][si])/saved['ATMS_scale'][si])
        targets.append(((rawF[ids]-saved['F_bias'])/saved['F_weight']-saved['F_mean'])/saved['F_scale'])
    c=torch.cat(cx);atms=torch.cat(ax);y=torch.cat(targets)
    true=np.load(U.checked(done['Dfit_targets']),allow_pickle=False);assert np.array_equal(y.numpy(),true)
    with torch.no_grad():predictions=V.predict(models,c,atms,torch.as_tensor(permutation))
    stored=np.load(U.checked(done['Dfit_predictions']),allow_pickle=False,mmap_mode='r');relative={};maximum_absolute={}
    for j,arm in enumerate(V.ARMS):
        diff=predictions[arm].numpy()-stored[:,j];norm=np.linalg.norm(stored[:,j],axis=-1)
        relative[arm]=float((np.linalg.norm(diff,axis=-1)/np.maximum(norm,1e-8)).max());maximum_absolute[arm]=float(np.abs(diff).max())
        assert relative[arm]<1e-4 and maximum_absolute[arm]<1e-4
    cpu_metrics={arm:V.metrics(value,y,groups) for arm,value in predictions.items()}
    csv_rows=list(csv.DictReader(U.checked(done['per_query']).open()));assert len(csv_rows)==32800
    values={arm:{key:np.empty(8200,dtype=np.float64 if key=='MSE' else np.float32) for key in cpu_metrics[arm]} for arm in V.ARMS}
    for ordinal,row in enumerate(csv_rows):
        i,j=divmod(ordinal,4);arm=V.ARMS[j];q=queries[i]
        assert row['arm']==arm
        for key,v in q.items():assert row[key]==str(v)
        for key in values[arm]:values[arm][key][i]=float(row[key])
    metric_error={arm:{key:float(np.abs(cpu_metrics[arm][key]-value).max()) for key,value in values[arm].items()} for arm in V.ARMS}
    assert all(error<1e-4 for metrics in metric_error.values() for error in metrics.values())
    reread=V.readout(values,[q['subject_id'] for q in queries],[q['concept_id'] for q in queries]);assert reread==U.read(U.checked(done['readout']))
    base=predictions['c_only'];residual=y-base;decomposition={}
    for arm in V.ARMS[2:]:
        update=predictions[arm]-base
        energy=update.double().square().mean(-1).numpy();alignment=(residual.double()*update.double()).mean(-1).numpy()
        identity=energy-2*alignment
        observed=(predictions[arm]-y).double().square().mean(-1).numpy()-(base-y).double().square().mean(-1).numpy()
        error=float(np.abs(identity-observed).max());assert error<1e-6
        decomposition[arm]=dict(update_energy_mean=float(energy.mean()),twice_residual_alignment_mean=float(2*alignment.mean()),
            predicted_MSE_increase=float(identity.mean()),actual_MSE_increase=float(observed.mean()),identity_max_absolute_error=error)
    innovation=atms-c@models['c_to_a'];normalization={}
    for si in range(10):
        ix=slice(si*820,(si+1)*820)
        normalization[f'sub-{si+1:02d}']=dict(Dfit_c_feature_RMS=float(c[ix].square().mean().sqrt()),
            Dfit_ATMS_feature_RMS=float(atms[ix].square().mean().sqrt()),Dfit_innovation_RMS=float(innovation[ix].square().mean().sqrt()))
    innovation_T_m2=models['regularization']['innovation']/.1
    parities=U.read(U.checked(exported['CPU_GPU_parity']));assert len(parities['rows'])==30
    assert parities['status']=='ALL30_FROZEN_MODELS_60_INPUTS_CPU_GPU_PARITY_PASSED'
    perconcept=V.paired_difference(values['c_only'],values['c_plus_ATMS_innovation'],[q['subject_id'] for q in queries],[q['concept_id'] for q in queries])
    plot=None
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        concepts=sorted(set(q['concept_id'] for q in queries));labels=['c','ATMS','c + innovation','shuffled innovation']
        fig,axs=plt.subplots(1,3,figsize=(13,3.7))
        for ax,key,title in zip(axs[:2],['MSE','within10_log_loss'],['All10 Dfit visual MSE (lower better)','Within10 log loss (lower better)']):
            ax.bar(np.arange(4),[reread['means'][arm][key] for arm in V.ARMS],color=['#577b9e','#9ba7ac','#b55d4c','#a9a09d']);ax.set_xticks(np.arange(4),labels,rotation=24,ha='right');ax.set_title(title)
        difference=values['c_only']['MSE']-values['c_plus_ATMS_innovation']['MSE'];concept_ids=np.array([q['concept_id'] for q in queries])
        effect=np.array([difference[concept_ids==name].mean() for name in concepts]);axs[2].bar(np.arange(82),effect,color=np.where(effect>0,'#438b71','#b55d4c'));axs[2].axhline(0,color='black',linewidth=.6)
        axs[2].set(title='MSE improvement from ATMS by concept',xlabel='All82 concepts, fixed sorted order',ylabel='c-only MSE minus joint MSE')
        fig.tight_layout();path=a.output/'actual_Dfit_effects.png';fig.savefig(path,dpi=160);plt.close(fig);plot=U.ref(path)
    except ImportError:pass
    for item in spec['source_pins']:U.checked(item)
    assert not torch.cuda.is_initialized()
    result=dict(status='D15_ALL8200_FOUR_ARM_CPU_PREDICTIONS_METRICS_AND_GATE_AUDIT_COMPLETE',source=U.ref(__file__),completion=U.ref(a.completion),
        ATMS_coordinate='original_raw_pooled_CLIP_H_output',full_Dfit_predictions_replayed=8200*4,CPU_max_relative_L2=relative,CPU_max_absolute_error=maximum_absolute,
        all32800_CSV_row_identities_verified=True,all_metrics_CPU_recomputed_max_absolute_error=metric_error,
        readout_and_all_bootstrap_recomputed_exactly_from_actual_perquery_values=True,all_Dfit_targets_bitwise_replayed=True,
        no_Dcheck_values_selected=True,conditional_increment_supported=reread['conditional_increment_supported'],decision=reread['decision'],
        innovation_update_MSE_decomposition=decomposition,T_innovation_RMS_from_registered_training_Gram=float(innovation_T_m2**.5),
        Dfit_view_normalization_by_subject=normalization,all30_exported_model_CPU_GPU_parities_verified=True,
        actual_plot=plot,no_models_or_predictions_modified=True,new_GPU_jobs=0,new_fit_updates=0,new_images=0,
        seconds=time.monotonic()-start,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'summary.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],decomposition=decomposition,
        T_innovation_RMS=result['T_innovation_RMS_from_registered_training_Gram'],Dfit_normalization=normalization)))


if __name__=='__main__':main()
