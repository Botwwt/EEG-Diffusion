"""D07: shared known-subject temporal EEG encoder, fixed10epochs/two folds.

All ten subjects train on the same permitted outer-FIT concepts. Only sub01
held rows are evaluated/exported. Same-stimulus EEG queries are multi-positives,
not false negatives. No generator or visual encoder is loaded or optimized.
"""
import argparse
import contextlib
import gzip
import io
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
import tempfile
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import fit_BrainHIVE_raw_neural_recoverability_20261027_v1 as U

ROOT=Path(__file__).resolve().parent
SCHEMA='BrainHIVE_shared_temporal_eeg_encoder_20261027_v1'
STATUS='COMPLETE_SHARED_TEMPORAL_EEG_ENCODER_SCREEN'
COORDINATES='raw learned shared temporal EEG encoder LayerNorm output; no L2 normalization or target-derived scaling'
SUBJECTS=tuple(f'sub-{i:02d}' for i in range(1,11))
sha,ref,atomic=U.sha,U.ref,U.atomic


class TemporalResidual(nn.Module):
    def __init__(self):
        super().__init__();self.depthwise=nn.Conv1d(64,64,7,padding=3,groups=64);self.pointwise=nn.Conv1d(64,64,1)
    def forward(self,x):return x+self.pointwise(F.gelu(self.depthwise(x)))


class SharedTemporalEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.subject_gain=nn.Parameter(torch.ones(10,17));self.subject_bias=nn.Parameter(torch.zeros(10,17))
        self.stem=nn.Conv1d(17,64,15,stride=5,padding=7)
        self.temporal=nn.Sequential(TemporalResidual(),TemporalResidual())
        self.projection=nn.Linear(64*50,1024);self.norm=nn.LayerNorm(1024,eps=1e-6)
    def calibrate(self,x,subjects):
        return x*self.subject_gain[subjects,:,None]+self.subject_bias[subjects,:,None]
    def forward(self,x,subjects):
        assert x.ndim==3 and x.shape[1:]==(17,250) and subjects.shape==(len(x),)
        value=self.temporal(F.gelu(self.stem(self.calibrate(x,subjects))))
        assert value.shape[1:]==(64,50)
        return self.norm(self.projection(value.flatten(1)))


def multi_positive_loss(predictions,stimuli,target_bank,temperature=.07):
    """Each visual row has a uniform soft label over ALL matching EEG rows."""
    unique,inverse=torch.unique(stimuli,sorted=True,return_inverse=True)
    logits=F.normalize(predictions.float(),dim=1)@F.normalize(target_bank[unique].float(),dim=1).T/temperature
    eeg_loss=F.cross_entropy(logits,inverse)
    positive=(torch.arange(len(unique),device=inverse.device)[:,None]==inverse[None,:]).float()
    counts=positive.sum(1);assert (counts>0).all()
    visual_loss=(-(F.log_softmax(logits.T,dim=1)*positive).sum(1)/counts).mean()
    return .5*(eeg_loss+visual_loss),dict(eeg_to_unique_F=eeg_loss,unique_F_to_all_EEG=visual_loss,
        unique_stimuli=len(unique),maximum_positive_count=int(counts.max()))


def fit_standardizer(fit_values):
    return dict(mean=fit_values.mean(0),std=fit_values.std(0,unbiased=False).clamp_min(1e-4))


def fold_positions(items,fold):
    train=[i for i,row in enumerate(items) if row['fold']!=fold];held=[i for i,row in enumerate(items) if row['fold']==fold]
    assert set(train).isdisjoint(held) and len(train)+len(held)==len(items)
    assert {items[i]['concept_id'] for i in train}.isdisjoint({items[i]['concept_id'] for i in held})
    return train,held


def optimizer_for(model):
    norm_ids={id(p) for module in model.modules() if isinstance(module,nn.LayerNorm) for p in module.parameters()}
    groups=[[],[]]
    for name,parameter in model.named_parameters():
        no_decay=name.endswith('bias') or id(parameter) in norm_ids or name in ('subject_gain','subject_bias')
        groups[int(no_decay)].append(parameter)
    return torch.optim.AdamW([dict(params=groups[0],weight_decay=.05),dict(params=groups[1],weight_decay=0.)],
        lr=5e-4,betas=(.9,.995),eps=1e-8)


def paired_readout(predictions,targets,items):
    prediction=F.normalize(predictions.float(),dim=1);truth=F.normalize(targets.float(),dim=1)
    groups={}
    for i,item in enumerate(items):groups.setdefault(item['concept_id'],[]).append(i)
    assert all(len(ids)==10 for ids in groups.values())
    ids=torch.tensor(list(groups.values()),device=predictions.device)
    similarity=torch.bmm(prediction[ids],truth[ids].transpose(1,2))
    expected=torch.arange(10,device=predictions.device)[None,:].expand(len(ids),-1)
    top=(similarity.argmax(2)==expected).float()
    positive=similarity.diagonal(dim1=1,dim2=2)
    mask=torch.eye(10,device=predictions.device,dtype=torch.bool)[None]
    margin=positive-similarity.masked_fill(mask,-torch.inf).amax(2)
    cosine=(prediction*truth).sum(1)
    return dict(paired_cosine=float(cosine.mean()),prediction_norm_mean=float(predictions.norm(dim=1).mean()),
        target_norm_mean=float(targets.norm(dim=1).mean()),raw_target_MSE=float((predictions-targets).square().mean()),
        sameconcept10_top1=float(top.mean()),sameconcept10_margin=float(margin.mean()))


@torch.inference_mode()
def evaluate(model,values,target_bank,stimuli,items,check):
    model.eval();pieces=[];loss_sum=0.;count=0
    for start in range(0,len(values),2048):
        check();batch=values[start:start+2048];ids=stimuli[start:start+2048]
        prediction=model(batch,torch.zeros(len(batch),dtype=torch.long,device=batch.device))
        loss,_=multi_positive_loss(prediction,ids,target_bank)
        assert torch.isfinite(prediction).all() and torch.isfinite(loss)
        pieces.append(prediction);loss_sum+=float(loss)*len(batch);count+=len(batch)
    predictions=torch.cat(pieces)
    return predictions,dict(held_batched2048_symmetric_loss=loss_sum/count,held_rows=count,
        held_loss_is_fixed_batches_not_full_gallery_softmax=True,**paired_readout(predictions,target_bank[stimuli],items))


@torch.inference_mode()
def full_gallery_readout(predictions,stimuli,target_bank,check):
    gallery=F.normalize(target_bank.float(),dim=1);top=[];afc=[]
    for start in range(0,len(predictions),256):
        check();batch=F.normalize(predictions[start:start+256].float(),dim=1);labels=stimuli[start:start+256]
        scores=batch@gallery.T;correct=scores.gather(1,labels[:,None])
        top.append((scores.argmax(1)==labels).float().cpu())
        afc.append(((scores<correct).sum(1)+.5*((scores==correct).sum(1)-1)).float().cpu()/(len(gallery)-1))
    return dict(full_F_gallery_size=len(gallery),full_F_gallery_top1=float(torch.cat(top).mean()),
        full_F_gallery_two_afc=float(torch.cat(afc).mean()),all_query_targets_retained=True,
        interpretation='condition-space readout, not image reconstruction performance')


def self_test():
    torch.set_num_threads(2);torch.manual_seed(1729)
    prediction=torch.tensor([[1.,0.],[1.,0.],[0.,1.]],requires_grad=True)
    targets=torch.zeros(10,2);targets[7]=torch.tensor([1.,0.]);targets[9]=torch.tensor([0.,1.])
    stimulus=torch.tensor([7,7,9]);loss,parts=multi_positive_loss(prediction,stimulus,targets)
    logits=F.normalize(prediction,dim=1)@targets[[7,9]].T/.07
    forward=F.cross_entropy(logits,torch.tensor([0,0,1]));soft=torch.tensor([[.5,.5,0.],[0.,0.,1.]])
    reverse=-(soft*F.log_softmax(logits.T,dim=1)).sum(1).mean()
    assert torch.allclose(loss,.5*(forward+reverse)) and float(parts['unique_F_to_all_EEG'])>.34
    loss.backward();assert torch.isfinite(prediction.grad).all()
    perm=torch.tensor([2,0,1]);other,_=multi_positive_loss(prediction.detach()[perm],stimulus[perm],targets)
    assert torch.allclose(loss.detach(),other)
    items=[dict(fold=i//10,concept_id=f'c{i//10}') for i in range(20)]
    train,held=fold_positions(items,0);assert train==list(range(10,20)) and held==list(range(10))
    values=torch.randn(20,17,250);fit=fit_standardizer(values[train]);changed=values.clone();changed[held]+=1000
    fitted_again=fit_standardizer(changed[train]);assert all(torch.equal(fit[k],fitted_again[k]) for k in fit)
    for n,tail in ((66100,564),(66200,664)):
        order=torch.randperm(n,generator=torch.Generator().manual_seed(1729)).numpy();coverage=np.zeros(n,np.int8)
        sizes=[]
        for start in range(0,n,2048):coverage[order[start:start+2048]]+=1;sizes.append(len(order[start:start+2048]))
        assert np.all(coverage==1) and len(sizes)==33 and sizes[-1]==tail
    model=SharedTemporalEEG().eval();assert sum(p.numel() for p in model.parameters())==3305940
    x=torch.randn(2,17,250);subjects=torch.zeros(2,dtype=torch.long)
    assert torch.equal(model.calibrate(x,subjects),x)
    with torch.inference_mode():
        first=model(x,subjects)
        model.subject_gain[1:].fill_(10);model.subject_bias[1:].fill_(100)
        assert torch.equal(first,model(x,subjects)) and first.shape==(2,1024) and first.norm(dim=1).min()>20
    # Exercise the real typed-pack adapter with a synthetic complete bank. This
    # proves schema compatibility, not the validity of a trained EEG model.
    import brainhive_rapid_candidate_pack_20261027_v1 as P
    with tempfile.TemporaryDirectory(prefix='brainhive_D07_CPU_') as temporary:
        directory=Path(temporary);bank=directory/'predictions.npy'
        np.save(bank,np.repeat(first[:1].detach().numpy(),13230,axis=0),allow_pickle=False)
        index=directory/'index.jsonl'
        with index.open('w',encoding='utf-8') as stream:
            for i in range(13230):stream.write(json.dumps(dict(row=i,source_row=i,subject_id='sub-01'))+'\n')
        native=directory/'native.json';atomic(native,dict(status='COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS',prediction_index_sha256=sha(index),
            predictions_path=str(bank),predictions_sha256=sha(bank),predictions_shape=[13230,1024],oof_coverage_min=1,oof_coverage_max=1))
        baseline=directory/'baseline.json';atomic(baseline,dict(schema=P.C.BASELINE_SCHEMA,status=P.C.BASELINE_STATUS,queries=400,
            normal_inference_target_access=False,host={'CPU_fixture':True},native_parent_binding=ref(native)))
        fitted=directory/'summary.json';atomic(fitted,dict(status=STATUS,prediction_index=str(index),prediction_index_sha256=sha(index),
            arms={'shared_temporal':dict(predictions_path=str(bank),predictions_sha256=sha(bank),predictions_shape=[13230,1024],
                prediction_coordinates=COORDINATES,oof_coverage_min=1,oof_coverage_max=1)}))
        pack=directory/'pack.json'
        with contextlib.redirect_stdout(io.StringIO()):
            P.build(SimpleNamespace(fit_summary=fitted,baseline_completion=baseline,native_parent_binding=native,output=pack,
                mode='native_conditions',arm='D07_fixture',source_arm='shared_temporal',expected_status=STATUS))
        assert P.validate_pack(pack)[0]['mode']=='native_conditions'
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='D07_SIX_CRITICAL_CPU_CHECKS_PASSED',checks=6,CUDA_initialized=False,
        groups=['duplicatepositive_uniform_soft_CE','fold_and_trainonly_standardization','complete_permutation_and_tail_batches',
            'identity_subject_adapter','single_subject_inference_permission','raw1024_LayerNorm_shape_and_parameter_count'],
        parameter_count=3305940,raw_output_coordinates=COORDINATES,typed_pack_mode='native_conditions',
        typed_pack_source_arm='shared_temporal',scope='synthetic math/architecture checks, not a trained model or real GPU profile')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','targets','spec','output'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--gpu-job-name');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--max-seconds',type=float,default=180);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test();return 0
    assert all((args.inputs,args.targets,args.spec,args.output,args.gpu_job_name)) and args.device=='cuda:0' and 0<args.max_seconds<=180
    assert not args.output.exists();args.output.mkdir(parents=True);invoked=time.monotonic();stopped=[False]
    signal.signal(signal.SIGTERM,lambda number,frame:stopped.__setitem__(0,True))
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch=torch
    document=dict(gpu_job_name=args.gpu_job_name,maximum_seconds=args.max_seconds,maximum_output_bytes=256<<20,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,
        resource_authorization=ref(ROOT/'configs/resource_authorization_20261020_v3.json'))
    wrapper=D.verify_wrapper(document);budget=D.Budget(document)
    def check():
        assert not stopped[0] and time.monotonic()-invoked<args.max_seconds,'D07 finite180s limit; do not lower epochs'
        budget.check()
    def log(value):
        value=value|dict(seconds=time.monotonic()-invoked)
        with (args.output/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
        print(json.dumps(value),flush=True)
    active=None;folds=[];updates=0;exposures=0
    try:
        spec=json.loads(args.spec.read_bytes());assert spec['schema']==SCHEMA
        assert spec['training']['epochs_per_outer_fold']==10 and spec['training']['batch']==2048 and spec['training']['temperature']==.07
        assert spec['architecture']['parameters']==3305940 and spec['limits']['maximum_seconds']==180
        inputs=json.loads(args.inputs.read_bytes());assert inputs['status']=='COMPLETE_RAW_NEURAL_T17_ALL10_INPUTS'
        assert set(inputs['subjects'])==set(SUBJECTS) and inputs['rows']==13230 and inputs['concepts']==1323
        index=Path(inputs['prediction_index']);assert sha(index)==inputs['prediction_index_sha256']==spec['prediction_index_sha256']
        assert sha(args.targets)==spec['targets_sha256']
        items=U.read_index(index);assert len(items)==13230 and [row['row'] for row in items]==list(range(13230))
        assert all(row['subject_id']=='sub-01' and row['partition']=='T' and row['target_row']==row['row'] for row in items)
        source_bindings={name:{key:inputs['subjects'][name][key] for key in ('path','sha256')} for name in SUBJECTS}
        for binding in source_bindings.values():check();assert sha(binding['path'])==binding['sha256']
        controls={key:ref(path) for key,path in [('inputs',args.inputs),('targets',args.targets),('index',index),('spec',args.spec),('source',Path(__file__)),('D04_utilities',Path(U.__file__))]}
        atomic(args.output/'provenance.json',dict(schema=SCHEMA,controls=controls,source_bindings=source_bindings,wrapper=wrapper,
            query_subject='sub-01',training_subjects=list(SUBJECTS),known_subject_joint_training=True,unseen_subject_zero_shot=False,
            standardization_only_outer_fit=True,held_other_subject_EEG_values_loaded=False,held_metrics_select_checkpoint=False,
            cached_F_is_deterministic=True,original_BrainMLP_affine_retained=False,normal_inference_target_access=False,
            additional_subject_EEG_at_inference=False,no_prediction_L2_normalization=True,no_query_target_scaling=True,
            added_subject_training_and_architecture_confounded=True,no_generator_or_visual_encoder_loaded=True))
        torch.set_num_threads(2);torch.manual_seed(1729);torch.cuda.set_device(torch.device(args.device));torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        target_array=np.load(args.targets,mmap_mode='r',allow_pickle=False)
        assert target_array.shape==(13230,1024) and target_array.dtype==np.float32
        target_bank=torch.from_numpy(np.array(target_array,copy=True)).to(args.device);del target_array
        assert torch.isfinite(target_bank).all()
        predictions=np.empty((13230,1024),np.float32);coverage=np.zeros(13230,np.int8)
        for fold in (0,1):
            active=dict(outer_fold=fold,completed_epochs=0,completed_updates=0)
            train_positions,held_positions=fold_positions(items,fold)
            n=len(train_positions);expected_rows=n*10;expected_updates=10*math.ceil(expected_rows/2048)
            assert (n,expected_updates,expected_rows%2048) in ((6610,330,564),(6620,330,664))
            standardized=[];normalizers={};held_values=None
            for name in SUBJECTS:
                check();array=np.load(source_bindings[name]['path'],mmap_mode='r',allow_pickle=False)
                assert array.shape==(13230,17,250) and array.dtype==np.float32
                value=torch.from_numpy(np.array(array[train_positions],copy=True)).to(args.device)
                assert torch.isfinite(value).all()
                fitted=fit_standardizer(value);standardized.append((value-fitted['mean'])/fitted['std'])
                normalizers[name]=fitted
                if name=='sub-01':
                    raw_held=torch.from_numpy(np.array(array[held_positions],copy=True)).to(args.device)
                    held_values=(raw_held-fitted['mean'])/fitted['std'];del raw_held
                del value,array
            training_values=torch.cat(standardized);del standardized
            training_stimuli=torch.tensor(train_positions,device=args.device).repeat(10)
            training_subjects=torch.arange(10,device=args.device).repeat_interleave(n)
            training_stimuli_cpu=np.tile(np.array(train_positions),10);training_subjects_cpu=np.repeat(np.arange(10),n)
            held_stimuli=torch.tensor(held_positions,device=args.device);held_items=[items[i] for i in held_positions]
            torch.manual_seed(1729);model=SharedTemporalEEG().to(args.device);assert sum(p.numel() for p in model.parameters())==3305940
            optimizer=optimizer_for(model);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5*(1+math.cos(math.pi*step/expected_updates)))
            generator=torch.Generator(device=args.device).manual_seed(1729)
            for epoch in range(10):
                check();model.train();order=torch.randperm(expected_rows,device=args.device,generator=generator)
                order_cpu=order.cpu().numpy();epoch_coverage=np.zeros(expected_rows,np.int8)
                seen=0;loss_sum=0.;forward_sum=0.;reverse_sum=0.;tail=0
                for start in range(0,expected_rows,2048):
                    check();ids=order[start:start+2048];tail=len(ids)
                    optimizer.zero_grad(set_to_none=True)
                    prediction=model(training_values[ids],training_subjects[ids]);loss,parts=multi_positive_loss(prediction,training_stimuli[ids],target_bank)
                    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
                    loss.backward();optimizer.step();scheduler.step()
                    epoch_coverage[order_cpu[start:start+2048]]+=1
                    seen+=tail;exposures+=tail;updates+=1;active['completed_updates']+=1
                    loss_sum+=float(loss.detach())*tail;forward_sum+=float(parts['eeg_to_unique_F'].detach())*tail;reverse_sum+=float(parts['unique_F_to_all_EEG'].detach())*tail
                assert seen==expected_rows and np.all(epoch_coverage==1) and tail==expected_rows%2048
                subject_coverage=np.bincount(training_subjects_cpu[order_cpu],minlength=10)
                stimulus_coverage=np.bincount(training_stimuli_cpu[order_cpu],minlength=13230)
                assert np.all(subject_coverage==n) and np.all(stimulus_coverage[train_positions]==10) and np.all(stimulus_coverage[held_positions]==0)
                held_prediction,held_metrics=evaluate(model,held_values,target_bank,held_stimuli,held_items,check)
                active['completed_epochs']=epoch+1
                log(dict(stage='SHARED_TEMPORAL_EPOCH_COMPLETE',outer_fold=fold,epoch=epoch+1,updates=active['completed_updates'],
                    global_updates=updates,global_query_exposures=exposures,train_query_count=seen,
                    train_symmetric_loss=loss_sum/seen,train_EEG_to_unique_F_loss=forward_sum/seen,train_unique_F_to_all_EEG_loss=reverse_sum/seen,
                    held=held_metrics,coverage=dict(query_min=int(epoch_coverage.min()),query_max=int(epoch_coverage.max()),
                        per_subject_queries=subject_coverage.tolist(),per_fit_stimulus_queries=10,held_stimulus_training_queries=0,
                        unique_fit_stimuli=n,tail_batch=tail),checkpoint_selection_uses_held=False))
            assert active['completed_epochs']==10 and active['completed_updates']==expected_updates
            final=held_metrics|full_gallery_readout(held_prediction,held_stimuli,target_bank,check)
            predictions[held_positions]=held_prediction.cpu().float().numpy();coverage[held_positions]+=1
            checkpoint=args.output/f'fold{fold}_encoder.pt'
            torch.save(dict(model={key:value.detach().cpu() for key,value in model.state_dict().items()},
                standardizers=U.cpu_value(normalizers),subjects=list(SUBJECTS),outer_fold=fold,
                fit_source_rows=[items[i]['source_row'] for i in train_positions],held_source_rows=[items[i]['source_row'] for i in held_positions],
                epochs=10,updates=expected_updates,query_exposures=expected_rows*10,seed=1729,
                spec_sha256=controls['spec']['sha256'],raw_output_coordinates=COORDINATES),checkpoint)
            result=dict(outer_fold=fold,fit_stimuli=n,fit_queries_per_epoch=expected_rows,held_sub01_queries=len(held_positions),
                completed_epochs=10,completed_updates=expected_updates,query_exposures=expected_rows*10,final_held=final,checkpoint=ref(checkpoint))
            folds.append(result);atomic(args.output/f'fold{fold}_summary.json',result);log(dict(stage='SHARED_TEMPORAL_FOLD_COMPLETE',**result))
            del model,optimizer,scheduler,training_values,training_stimuli,training_subjects,held_values,held_prediction,normalizers
            torch.cuda.empty_cache()
        assert updates==660 and exposures==1323000 and np.all(coverage==1) and np.isfinite(predictions).all()
        folder=args.output/'shared_temporal';folder.mkdir();prediction_path=folder/'predictions.npy';np.save(prediction_path,predictions,allow_pickle=False)
        for binding in controls.values():check();assert sha(binding['path'])==binding['sha256']
        for binding in source_bindings.values():check();assert sha(binding['path'])==binding['sha256']
        assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file())<256<<20
        norms=np.linalg.norm(predictions.astype(np.float64),axis=1)
        summary=dict(status=STATUS,schema=SCHEMA,rows=13230,concepts=1323,training_subjects=list(SUBJECTS),query_subject='sub-01',
            epochs_per_outer_fold=10,completed_updates=updates,query_exposures=exposures,outer_folds=folds,
            prediction_index=str(index.resolve()),prediction_index_sha256=sha(index),
            arms={'shared_temporal':dict(predictions_path=str(prediction_path.resolve()),predictions_sha256=sha(prediction_path),predictions_shape=[13230,1024],
                prediction_coordinates=COORDINATES,predictions_norm_mean=float(norms.mean()),oof_coverage_min=int(coverage.min()),oof_coverage_max=int(coverage.max()))},
            normal_inference_target_access=False,known_subject_joint_training=True,unseen_subject_zero_shot=False,
            additional_subject_EEG_at_inference=False,no_prediction_L2_normalization=True,no_query_target_scaling=True,
            held_metrics_used_for_selection=False,fixed_final_epoch=True,deterministic_cached_F_targets=True,
            additional_subject_training_and_architecture_confounded=True,prior_original_affine_retained=False,
            no_visual_encoder_or_generator_training=True,provenance=ref(args.output/'provenance.json'),candidate_spec=ref(args.spec),
            seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            development_only=True,independent_confirmation=False,SOTA_established=False)
        atomic(args.output/'summary.json',summary);atomic(args.output/'completion.json',dict(status=STATUS,schema=SCHEMA,summary=ref(args.output/'summary.json')))
        log(dict(stage='SHARED_TEMPORAL_FIXED_SCREEN_COMPLETE',status=STATUS,completed_updates=updates,query_exposures=exposures))
        return 0
    except BaseException as error:
        atomic(args.output/'failure.json',dict(status='INCOMPLETE_SHARED_TEMPORAL_EEG_SCREEN_NO_REDUCED_EPOCH_RETRY',error=repr(error),traceback=traceback.format_exc(),
            active=active,completed_outer_folds=len(folds),completed_updates=updates,query_exposures=exposures,
            full_required_updates=660,full_required_query_exposures=1323000,seconds=time.monotonic()-invoked))
        raise


if __name__=='__main__':raise SystemExit(main())
