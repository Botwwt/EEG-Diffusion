"""Read-only CPU audit of both fixed D07 ten-epoch training histories.

No model selection, fit, checkpoint read, image scoring or CUDA operation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SCHEMA='BrainHIVE_D07_complete_training_log_CPU_analysis_20261027_v1'
FIELDS=('train_symmetric_loss','train_EEG_to_unique_F_loss','train_unique_F_to_all_EEG_loss',
        'held_batched2048_symmetric_loss','paired_cosine','sameconcept10_top1',
        'sameconcept10_margin','prediction_norm_mean','target_norm_mean','raw_target_MSE')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def flatten(row):return {**row,**row['held']}


def trend(values):
    values=np.asarray(values,dtype=np.float64)
    assert len(values)==10 and np.isfinite(values).all()
    return dict(first=float(values[0]),fixed_final_epoch10=float(values[-1]),
        final_minus_first=float(values[-1]-values[0]),
        relative_final_minus_first=float((values[-1]-values[0])/abs(values[0])) if values[0] else None,
        early_epochs1_to3_mean=float(values[:3].mean()),late_epochs8_to10_mean=float(values[-3:].mean()),
        late_minus_early=float(values[-3:].mean()-values[:3].mean()),
        all10_linear_slope=float(np.polyfit(np.arange(1,11),values,1)[0]),
        consecutive_decreases=int(np.sum(np.diff(values)<0)),
        consecutive_increases=int(np.sum(np.diff(values)>0)),
        last5_linear_slope=float(np.polyfit(np.arange(6,11),values[-5:],1)[0]))


def analyze(input_dir,output):
    started=time.monotonic();input_dir=Path(input_dir);output=Path(output)
    assert not output.exists(),'Preserve every analysis attempt'
    summary_path=input_dir/'actual_fit_summary.json';progress_path=input_dir/'actual_fit_progress.jsonl'
    bindings=[ref(summary_path),ref(progress_path),ref(__file__)]
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['status']=='COMPLETE_SHARED_TEMPORAL_EEG_ENCODER_SCREEN'
    rows=[json.loads(line) for line in progress_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    epochs=[r for r in rows if r['stage']=='SHARED_TEMPORAL_EPOCH_COMPLETE']
    assert len(rows)==23 and len(epochs)==20
    assert sum(r['stage']=='SHARED_TEMPORAL_FOLD_COMPLETE' for r in rows)==2
    assert rows[-1]['stage']=='SHARED_TEMPORAL_FIXED_SCREEN_COMPLETE'
    assert summary['completed_updates']==660 and summary['query_exposures']==1323000
    assert summary['epochs_per_outer_fold']==10 and summary['fixed_final_epoch'] is True
    assert summary['held_metrics_used_for_selection'] is False
    assert summary['query_subject']=='sub-01' and len(summary['training_subjects'])==10
    assert summary['arms']['shared_temporal']['oof_coverage_min']==summary['arms']['shared_temporal']['oof_coverage_max']==1
    assert all(np.isfinite(float(r['seconds'])) for r in rows)
    assert all(a['seconds']<=b['seconds'] for a,b in zip(rows,rows[1:]))
    fold_results=[];durations=[];plotted={};global_updates=0;global_exposures=0
    for fold,stimuli,tail in ((0,6610,564),(1,6620,664)):
        selected=[r for r in epochs if r['outer_fold']==fold]
        assert [r['epoch'] for r in selected]==list(range(1,11))
        fit_queries=stimuli*10
        for i,row in enumerate(selected,1):
            cover=row['coverage'];assert row['train_query_count']==fit_queries
            assert cover['query_min']==cover['query_max']==1
            assert cover['per_subject_queries']==[stimuli]*10
            assert cover['per_fit_stimulus_queries']==10 and cover['held_stimulus_training_queries']==0
            assert cover['unique_fit_stimuli']==stimuli and cover['tail_batch']==tail
            assert row['updates']==i*33
            global_updates+=33;global_exposures+=fit_queries
            assert row['global_updates']==global_updates and row['global_query_exposures']==global_exposures
            assert row['checkpoint_selection_uses_held'] is False
            assert row['held']['held_rows']==13230-stimuli
            assert row['held']['held_loss_is_fixed_batches_not_full_gallery_softmax'] is True
            values=flatten(row)
            assert all(np.isfinite(float(values[key])) for key in FIELDS)
            assert abs(row['train_symmetric_loss']-.5*(row['train_EEG_to_unique_F_loss']+row['train_unique_F_to_all_EEG_loss']))<1e-6
        elapsed=np.diff([r['seconds'] for r in selected]);durations.extend(elapsed.tolist())
        values={key:[float(flatten(r)[key]) for r in selected] for key in FIELDS};plotted[fold]=values
        final=summary['outer_folds'][fold]
        assert final['outer_fold']==fold and final['completed_epochs']==10 and final['completed_updates']==330
        assert final['query_exposures']==fit_queries*10
        for key in FIELDS:
            if key in final['final_held']:assert final['final_held'][key]==values[key][-1]
        fold_results.append(dict(outer_fold=fold,fit_stimuli=stimuli,queries_per_epoch=fit_queries,
            total_query_exposures=fit_queries*10,final_updates=330,tail_batch=tail,
            complete_all10_epochs=True,all10_subjects_and_every_fit_stimulus_retained=True,
            metrics={key:trend(value) for key,value in values.items()},
            fixed_final_condition_readout=final['final_held'],
            consecutive_epoch_time_seconds_including_held_evaluation=dict(count=len(elapsed),
                values=elapsed.tolist(),median=float(np.median(elapsed)),minimum=float(elapsed.min()),maximum=float(elapsed.max())),
            timing_first_epoch_excluded_from_steady_intervals=True))
    assert global_updates==660 and global_exposures==1323000
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.22})
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),constrained_layout=True)
    x=np.arange(1,11);colors=('#1764ab','#dc7f22')
    for fold,color in enumerate(colors):
        v=plotted[fold];label=f'Fold {fold}'
        axes[0,0].plot(x,v['train_symmetric_loss'],color=color,label=label+' train')
        axes[0,0].plot(x,v['held_batched2048_symmetric_loss'],color=color,ls='--',label=label+' held')
        for ax,key in ((axes[0,1],'paired_cosine'),(axes[0,2],'sameconcept10_top1'),
                       (axes[1,0],'raw_target_MSE'),(axes[1,1],'prediction_norm_mean'),(axes[1,2],'sameconcept10_margin')):
            ax.plot(x,v[key],marker='o',markersize=3,color=color,label=label)
        axes[1,1].plot(x,v['target_norm_mean'],color=color,ls=':',alpha=.7,label=label+' F target')
    titles=('Contrastive loss (batch definitions differ)','Held paired F cosine','Held within-concept10 Top1',
            'Held raw F-target MSE','Held prediction / target norm','Held within-concept10 margin')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title);ax.set_xlabel('Fixed epoch');ax.set_xticks(x);ax.legend(fontsize=8)
        ax.axvline(10,color='black',lw=.7,alpha=.25)
    axes[0,2].axhline(.1,color='gray',ls=':',label='uniform chance')
    fig.suptitle('D07: all 20 epochs, fixed final epoch10; condition readout is not image quality',fontsize=13)
    figure=output/'D07_complete_training_curves.png';fig.savefig(figure,dpi=180,facecolor='white');plt.close(fig)
    for binding in bindings:assert sha(binding['path'])==binding['sha256']
    duration=np.asarray(durations)
    result=dict(schema=SCHEMA,status='COMPLETE_D07_ALL20_EPOCH_CPU_LOG_ANALYSIS',source=ref(__file__),
        input_bindings=bindings,progress_events=len(rows),epoch_events=len(epochs),folds=fold_results,
        complete_coverage_verified=True,updates=global_updates,query_exposures=global_exposures,
        all_logged_numeric_metrics_finite=True,no_hidden_epoch_or_query_exclusion=True,
        per_update_or_gradient_logs_available=False,checkpoint_parameters_independently_decoded=False,
        original_method_source_or_preregistration_changed=False,fixed_final_epoch10_preserved=True,
        training_seconds_reported=summary['seconds'],peak_extra_CPU_bytes_reported=summary['peak_extra_CPU_bytes'],
        peak_GPU_bytes_reported=summary['peak_GPU_bytes'],
        steady_epoch_interval_seconds=dict(count=len(duration),median=float(np.median(duration)),minimum=float(duration.min()),maximum=float(duration.max()),
            scope='Consecutive epoch events2..10 per fold; includes33 train updates, held evaluation and logging, not kernel-only throughput'),
        figure=ref(figure),epochs_by_fold={str(fold):values for fold,values in plotted.items()},
        interpretive_limits=[
            'Train loss uses mixed known-subject batches with deduplicated targets and uniform positive EEG labels; held loss uses sub01 fixed batches. Their absolute difference is not a calibrated generalization gap.',
            'Held loss decreases early and rises slightly late while train loss continues falling; report this trajectory without selecting a different epoch.',
            'Held same-concept retrieval improves while absolute cosine falls and raw MSE rises; MSE and cosine are related geometry, not independent success evidence.',
            'Norm near32 is largely imposed by LayerNorm and is not semantic recovery evidence.',
            'The final full13230 F-gallery metrics are condition readout, not generated-image metrics.',
            'Do not compare sub01 OOF readout against a historical all10-subject mean or a different gallery; use matched nativebank CPU replay.',
            'No conclusion about image benefit, SOTA, convergence optimum or a new stopping epoch follows from this analysis.'],
        new_GPU_calls=0,new_method_fit=False,image_effects_read=False,SOTA_established=False,CPU_analysis_seconds=time.monotonic()-started)
    path=output/'analysis.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(args.input_dir,args.output)
    print(json.dumps(dict(status=result['status'],epoch_events=result['epoch_events'],updates=result['updates'],
        query_exposures=result['query_exposures'],figure=result['figure'],new_GPU_calls=0)))


if __name__=='__main__':main()
