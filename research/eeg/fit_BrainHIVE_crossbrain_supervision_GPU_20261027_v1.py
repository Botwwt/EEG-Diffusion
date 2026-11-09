"""D17 all10 native BrainMLP fine-tunes with training-only other-brain targets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_crossbrain_supervision_20261027_v1 as V

R=Path(__file__).resolve().parent

def feature_metrics(prediction,target):
    assert prediction.shape==target.shape and len(prediction)%10==0
    p=F.normalize(prediction.float(),dim=-1);y=F.normalize(target.float(),dim=-1)
    score=torch.bmm(p.reshape(-1,10,1024),y.reshape(-1,10,1024).transpose(1,2))
    positive=score.diagonal(dim1=-2,dim2=-1)
    values=dict(MSE=(prediction-target).double().square().mean(-1),target_cosine=(p*y).sum(-1),
        within10_log_loss=F.cross_entropy((score/.1).reshape(-1,10),torch.arange(10,device=p.device).repeat(len(p)//10),reduction='none'),
        within10_top1=(score.argmax(-1)==torch.arange(10,device=p.device)).float().reshape(-1),
        within10_cosine_margin=(positive-(score.sum(-1)-positive)/9).reshape(-1))
    return {key:value.cpu().numpy() for key,value in values.items()}

def cpu_state(x):
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_state(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_state(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_state(v) for v in x)
    return x

def main():
    p=argparse.ArgumentParser()
    for key in ('spec','native-registration','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();start=time.monotonic();assert a.max_seconds==360 and not a.output.exists();a.output.mkdir(parents=True)
    stop=[False];signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));spec=U.read(a.spec)
    for ref in spec['source_pins']:U.checked(ref)
    teacher=U.read(U.checked(spec['teacher_completion']));assert teacher['neural_teacher_gate_passed'] is True
    assert U.read(U.checked(spec['teacher_CPU_audit']))['neural_teacher_gate_passed'] is True
    mapping=U.read(U.checked(teacher['mapping']));assert np.array(mapping['groups']).shape==(1323,10)
    raw_inputs=U.read(U.checked(teacher['raw_T_inputs']));bindings=U.read(U.checked(spec['original_raw_inputs']));contract=U.read(U.checked(spec['native_input_contract']))
    prepared=U.read(U.checked(spec['prepared_EEG']));assert prepared['all132300_T_input_values_bitwise_original_R4'] is True
    Dinput=np.load(U.checked(prepared['Dfit']),allow_pickle=False,mmap_mode='r');DCinput=np.load(U.checked(prepared['Dcheck']),allow_pickle=False,mmap_mode='r')
    prepared_mapping=U.read(U.checked(prepared['mapping']))
    assert mapping['T_array_rows']==bindings['T_array_rows']==U.read(R/'configs/split_manifest.json')['image_rows']['T']
    import brainhive_paired_study_contract_20261025_v1 as C
    config=C.read(a.native_registration);C.validate(config);C.setup_runtime(config)
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=3072<<20,
        maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=config['resource_authorization'])
    wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
    def check():assert not stop[0] and time.monotonic()-start<330,'D17 finite full training deadline';budget.check()
    def log(x):
        x=x|dict(seconds=time.monotonic()-start)
        with (a.output/'training_log.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(config,training_start=True)
    groups=torch.tensor(mapping['groups'],device='cuda')
    from brainhive_native_models_v1 import classes
    BrainMLP=classes()['BrainMLP']
    Fbank=torch.load(U.checked(dict(path=contract['target_cache_path'],sha256=contract['target_cache_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    targets=Fbank[mapping['T_array_rows']].cuda();prob=torch.from_numpy(np.load(U.checked(teacher['probabilities']),allow_pickle=False)).cuda()
    prior_D=U.read(U.checked(spec['D15_probe']));dmap=U.read(U.checked(prior_D['queries_and_controls']));queries=dmap['queries']
    all_models={};all_optimizers={};initial=[];Dvalues=[];Dtargets=[];Dmeans={};Dconditions=np.empty((8200,3,1024),np.float32)
    d09=U.read(U.checked(spec['D09_completion']));panel=U.read(U.checked(U.read(U.checked(d09['inputs']))['panel']));check_bank=None
    native_full_sources=[]
    for si,subject in enumerate(contract['subjects']):
        check();name=subject['subject'];assert name==mapping['subjects'][si]
        source=next(x for x in subject['sources'] if x['role']=='full');native_full_sources.append(dict(subject=name,cache=dict(path=source['condition_cache_path'],sha256=source['condition_cache_sha256']),checkpoint=dict(path=source['checkpoint_path'],sha256=source['checkpoint_sha256'])))
        old=torch.load(U.checked(native_full_sources[-1]['checkpoint']),weights_only=True,map_location='cpu',mmap=True)
        Tref=raw_inputs['subjects'][name];array=np.load(U.checked(dict(path=Tref['path'],sha256=Tref['sha256'])),allow_pickle=False,mmap_mode='r')
        eeg=torch.from_numpy(np.array(array,copy=True)).cuda();assert eeg.shape==(13230,17,250)
        rows=source['groups']['calibration_only_Dfit'];ids=[x['array_row'] for x in rows];assert ids==[q['array_row'] for q in queries[si*820:(si+1)*820]]
        assert prepared_mapping['Dfit_queries'][si*820:(si+1)*820]==queries[si*820:(si+1)*820]
        development=torch.from_numpy(np.array(Dinput[si],copy=True)).cuda();truth=Fbank[ids].cuda();Dtargets.append(truth.cpu())
        cache=torch.load(U.checked(native_full_sources[-1]['cache']),weights_only=True,map_location='cpu',mmap=True)
        baseline=cache['condition'][[x['cache_row'] for x in rows]].clone();models={};opts={};scales={};params={};curves=[]
        for arm in V.ARMS:
            model=BrainMLP(4250,1024,num_layers=1,dropout=.1).cuda();model.load_state_dict(old['model'],strict=True)
            assert sum(x.numel() for x in model.parameters())==6452224
            scale=nn.Parameter(old['logit_scale'].cuda().clone());normids={id(p) for m in model.modules() if isinstance(m,nn.LayerNorm) for p in m.parameters()}
            decay=[p for n,p in model.named_parameters() if not n.endswith('bias') and id(p) not in normids]+[scale]
            no_decay=[p for n,p in model.named_parameters() if n.endswith('bias') or id(p) in normids]
            opts[arm]=torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=no_decay,weight_decay=0.)],lr=1e-4,betas=(.9,.995))
            models[arm]=model;scales[arm]=scale;params[arm]=list(model.parameters())+[scale]
            model.eval()
            with torch.no_grad():value=model(development[:2]).float().cpu()
            relative=(value-baseline[:2]).norm(dim=-1)/baseline[:2].norm(dim=-1);assert relative.max()<3e-5
            initial.append(dict(subject=name,arm=arm,model_source=native_full_sources[-1]['checkpoint'],CPU_cache_initial_GPU_max_relative_L2=float(relative.max())))
        def evaluate(epoch):
            result={};predictions=[]
            with torch.no_grad():
                for arm,model in models.items():
                    model.eval();prediction=torch.cat([model(x).float() for x in development.split(256)])
                    score=feature_metrics(prediction,truth)
                    result[arm]={k:float(v.mean()) for k,v in score.items()};predictions.append(prediction.cpu())
            log(dict(stage='DFIT820_LOG_ONLY',subject=name,epoch=epoch,values=result))
            return torch.stack(predictions,1).numpy(),result
        evaluate(0);updates=0
        wrong=prob[si].clone()
        for j,concept in enumerate(mapping['concepts']):wrong[j]=prob[si,j,V.within_concept_permutation(concept).cuda()]
        order_gen=torch.Generator().manual_seed(1729+si)
        for epoch in range(10):
            order=torch.randperm(1323,generator=order_gen);coverage=np.zeros(13230,np.int8);sums={arm:np.zeros(3,np.float64) for arm in V.ARMS};seen=0
            for lo in range(0,1323,100):
                check();chosen=order[lo:lo+100].cuda();ix=groups[chosen].reshape(-1);coverage[ix.cpu().numpy()]+=1
                for arm in V.ARMS:
                    model=models[arm];model.train();opt=opts[arm];opt.zero_grad(set_to_none=True)
                    for pg in opt.param_groups:pg['lr']=1e-4*.5*(1+math.cos(math.pi*updates/140))
                    torch.cuda.manual_seed(1729+si*10000+epoch*100+lo//100)
                    target=wrong[chosen] if arm=='shuffled_crossbrain_soft' else prob[si,chosen]
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(eeg[ix])
                    nce,kl,fine=V.student_losses(prediction,targets[ix],target,scales[arm]);loss=nce+(.1*kl if arm!='visual_only' else 0.)
                    assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(params[arm],1.,error_if_nonfinite=True);opt.step()
                    sums[arm]+=np.array([float(nce.detach()),float(kl.detach()),float(fine.detach())])*len(ix)
                updates+=1;seen+=len(ix)
            assert np.all(coverage==1) and seen==13230 and updates==14*(epoch+1)
            log(dict(stage='COMPLETE_T_PASS',subject=name,epoch=epoch+1,updates_per_model=updates,exposures_per_model=(epoch+1)*13230,
                full_T_coverage_min=int(coverage.min()),full_T_coverage_max=int(coverage.max()),group_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                means={arm:dict(NCE=float(x[0]/seen),teacher_KL=float(x[1]/seen),within10_CE=float(x[2]/seen)) for arm,x in sums.items()}))
            final,held_means=evaluate(epoch+1)
        Dconditions[si*820:(si+1)*820]=final;Dmeans[name]=held_means
        all_models[name]={arm:cpu_state(model.state_dict()) for arm,model in models.items()}
        all_optimizers[name]={arm:dict(optimizer=cpu_state(opts[arm].state_dict()),logit_scale=scales[arm].detach().cpu()) for arm in V.ARMS}
        # Freeze all400 sub01 Dcheck EEG conditions before any new image/score.
        if si==0:
            assert prepared_mapping['Dcheck_queries']==panel['Dcheck']
            check_input=torch.from_numpy(np.array(DCinput,copy=True)).cuda();values=[]
            with torch.no_grad():
                for arm in V.ARMS:
                    models[arm].eval();values.append(torch.cat([models[arm](x).float().cpu() for x in check_input.split(256)]))
            check_bank=torch.stack(values,1).numpy()
        del models,opts,params,scales,eeg,development,truth,old,cache
        torch.cuda.empty_cache()
    assert Dconditions.shape==(8200,3,1024) and check_bank.shape==(400,3,1024)
    model_file=a.output/'all10_fixed_final_encoders.pt';torch.save(dict(models=all_models,optimizers=all_optimizers,seed=1729,
        epochs_per_model=10,updates_per_model=140,exposures_per_model=132300,old_epochs_per_model=25,source_spec=U.ref(a.spec),
        CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all()),model_file)
    U.save_array(a.output/'Dfit_conditions.npy',Dconditions);U.save_array(a.output/'Dfit_targets.npy',torch.cat(Dtargets).numpy());U.save_array(a.output/'all400_conditions.npy',check_bank)
    U.atomic(a.output/'initial_cache_parities.json',dict(status='ALL30_OLD_FULL_NATIVE_ENCODER_STATES_AND_INITIAL_INPUT_PARITIES_VERIFIED',rows=initial))
    U.atomic(a.output/'Dfit_queries.json',dict(queries=queries,groups=dmap['groups']));U.atomic(a.output/'Dfit_means_by_subject.json',Dmeans)
    U.atomic(a.output/'condition_lock.json',dict(status='D17_ALL400_THREE_FINAL_ENCODER_CONDITIONS_LOCKED_BEFORE_IMAGES',created_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        conditions=U.ref(a.output/'all400_conditions.npy'),model=U.ref(model_file),spec=U.ref(a.spec),query_ids=[q['query_id'] for q in panel['Dcheck']],arms=V.ARMS,
        Dcheck_target_F_or_images_used=False,Dcheck_scores_used_for_choices=False,final_checkpoint_fixed=True,normal_inference_only_current_query_EEG=True,normal_inference_other_subject_EEG=False,
        training_only_privileged_teacher=True,normal_inference_new_head_parameters=0,normal_inference_NFE=4,original_D_resumed=False))
    for ref in spec['source_pins']:U.checked(ref)
    assert sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())<3072<<20
    check();complete=dict(status='D17_ALL10_THREE_MATCHED_NATIVE_ENCODERS_FULL_T10_AND_ALL8200_DFIT_COMPLETE',spec=U.ref(a.spec),source=U.ref(__file__),wrapper=wrapper,
        teacher_completion=spec['teacher_completion'],model=U.ref(model_file),conditions=U.ref(a.output/'Dfit_conditions.npy'),targets=U.ref(a.output/'Dfit_targets.npy'),
        Dfit_queries=U.ref(a.output/'Dfit_queries.json'),Dfit_means_by_subject=U.ref(a.output/'Dfit_means_by_subject.json'),condition_lock=U.ref(a.output/'condition_lock.json'),
        initial_parities=U.ref(a.output/'initial_cache_parities.json'),subjects=10,arms=list(V.ARMS),parameters_per_encoder=6452224,trained_encoders=30,
        epochs_per_encoder=10,updates_per_encoder=140,exposures_per_encoder=132300,updates_per_arm_all10=1400,query_exposures_per_arm_all10=1323000,
        teacher_F='frozen deterministic nativeF cache, unlike original stochastic teacher; same targets for all3 arms',
        native_generator_or_prior_updated=False,Dcheck_target_access=False,no_Dfit_model_or_epoch_selection=True,original_D_resumed=False,
        normal_inference_only_current_query_EEG=True,training_only_neural_teacher_absent_at_inference=True,normal_inference_additional_parameters=0,
        peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,seconds=time.monotonic()-start,independent_confirmation=False,SOTA_established=False)
    U.atomic(a.output/'completion.json',complete);log(dict(stage='COMPLETE'));return 0

if __name__=='__main__':raise SystemExit(main())
