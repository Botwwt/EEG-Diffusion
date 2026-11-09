"""One fixed architecture/subject: all13230 T,25epochs, native stochastic F.

At most two predeclared finite execution segments preserve exact cursor/RNG.
No data-driven early stopping or hyperparameter changes. Exports DV EEG-only.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,math,signal,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
import brainhive_architecture_campaign_20261027_v1 as M
import brainhive_gain_calibration_20261027_v1 as U

def main():
    p=argparse.ArgumentParser()
    for key in ['spec','output','segment-output','native-registration']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--method',choices=M.TRAIN_METHODS,required=True);p.add_argument('--subject',choices=M.SUBJECTS,required=True);p.add_argument('--profile-only',action='store_true')
    p.add_argument('--gpu-job-name',required=True);p.add_argument('--max-seconds',type=float,required=True);a=p.parse_args()
    invoked=time.monotonic();assert not a.segment_output.exists();a.segment_output.mkdir(parents=True);a.output.mkdir(parents=True,exist_ok=True)
    stopped=[False];signal.signal(signal.SIGTERM,lambda *_:stopped.__setitem__(0,True));active='INPUTS';cursor=0;optimizer=None;model=None
    def log(x):
        x|={'method':a.method,'subject':a.subject,'seconds':time.monotonic()-invoked}
        with (a.segment_output/'progress.jsonl').open('a') as f:f.write(json.dumps(x)+'\n');f.flush();os.fsync(f.fileno())
        print(json.dumps(x),flush=True)
    try:
        spec=U.read(a.spec);assert spec['schema']==M.SCHEMA and spec['epochs']==25 and spec['batch']==1024
        for x in spec['code_and_asset_pins']:U.checked(x)
        source=spec['subjects'][a.subject];rawpath=U.checked(source['R4_EEG']);protocol=U.read(U.checked(spec['split']));ids=protocol['image_rows']['T'];export_ids=sorted(set(protocol['image_rows']['D']+protocol['image_rows']['V']))
        assert len(ids)==13230 and len(export_ids)==3310 and set(ids).isdisjoint(export_ids) and set(ids)|set(export_ids)==set(range(16540))
        import brainhive_paired_study_contract_20261025_v1 as C
        cfg=C.read(a.native_registration);C.validate(cfg);C.setup_runtime(cfg)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        D.torch=torch;doc=dict(gpu_job_name=a.gpu_job_name,maximum_seconds=a.max_seconds,maximum_output_bytes=2<<30,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,minimum_CPU_headroom_bytes=10<<30,resource_authorization=cfg['resource_authorization'])
        wrapper=D.verify_wrapper(doc);budget=D.Budget(doc)
        def resource():budget.check()
        torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);C.resources(cfg,training_start=True)
        from brainhive_native_models_v1 import prior_models,classes
        active='FROZEN_NATIVE_TEACHER_AND_REAL_R4_INPUT'
        teachers=prior_models('cuda');teacherF=teachers['fusion_encoder'].vision_model
        for m in teacherF.modules():
            if isinstance(m,nn.Dropout):m.p=.1
        teacherF.train().requires_grad_(False);teacherA=teachers['fusion_encoder'].adapter.half().eval().requires_grad_(False);teacherP=teachers['proj'].half().eval().requires_grad_(False)
        assert teacherP.norm.eps==1e-5
        visual={k:v[ids].cuda() for k,v in torch.load(U.checked(spec['visual_embeddings']),weights_only=True,map_location='cpu',mmap=True).items()}
        raw=torch.load(rawpath,weights_only=True,map_location='cpu',mmap=True);assert raw.shape==(16540,63,250) and raw.dtype==torch.float32
        channels=list(range(63)) if M.METHODS[a.method]['channels']==63 else spec['original_17_indices']
        eeg=raw[:,channels,:].cuda();assert torch.isfinite(eeg).all();del raw
        torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        model=M.make_model(a.method,classes(),teacherP.norm.weight.float(),teacherP.norm.bias.float()).cuda();logit=nn.Parameter(torch.tensor(2.6592,device='cuda'))
        norms={id(p) for m in model.modules() if isinstance(m,nn.LayerNorm) for p in m.parameters()}
        decay=[p for n,p in model.named_parameters() if not n.endswith('bias') and id(p) not in norms]+[logit]
        clean=[p for n,p in model.named_parameters() if n.endswith('bias') or id(p) in norms]
        parameters=list(model.parameters())+[logit];optimizer=torch.optim.AdamW([dict(params=decay,weight_decay=.05),dict(params=clean,weight_decay=0.)],lr=5e-4,betas=(.9,.995))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.5*(1+math.cos(math.pi*step/325)))
        generator=torch.Generator().manual_seed(1730);orders=torch.stack([torch.randperm(13230,generator=generator) for _ in range(25)]);order_sha=hashlib.sha256(orders.numpy().tobytes()).hexdigest()
        coverage=torch.zeros((25,13230),dtype=torch.bool);train_ids=torch.tensor(ids,device='cuda');registration=dict(schema=M.SCHEMA,method=a.method,subject=a.subject,spec=U.ref(a.spec),EEG=source['R4_EEG'],channels=channels,teacher_visual=spec['visual_embeddings'],order_sha256=order_sha,epochs=25,batch=1024,updates=325,seed=1729,
            train_rows_sha256=hashlib.sha256(np.array(ids,dtype=np.int64).tobytes()).hexdigest(),export_DV_rows=export_ids,teacher_F_training_dropout=.1,teacher_dropout_rng_isolated=True,normal_inference_target_access=False,full_T=True,OOF_predictions=False)
        rp=a.output/'registration.json'
        if rp.exists():assert U.read(rp)==registration
        else:U.atomic(rp,registration)
        resume=a.output/'resume.pt';prior_seconds=0.
        if resume.exists():
            assert not a.profile_only
            cp=torch.load(resume,weights_only=False,map_location='cpu');assert cp['registration_sha256']==U.sha(rp)
            model.load_state_dict(cp['model'],strict=True);logit.data.copy_(cp['logit_scale'].cuda());optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler']);cursor=cp['cursor'];coverage=cp['coverage'];prior_seconds=cp['cumulative_seconds']
            torch.set_rng_state(cp['CPU_rng']);torch.cuda.set_rng_state_all(cp['CUDA_rng']);del cp
        def snapshot():
            M.save_tensor(resume,dict(model=model.state_dict(),logit_scale=logit,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),cursor=cursor,coverage=coverage,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all(),shuffle_rng=generator.get_state(),orders_sha256=order_sha,registration_sha256=U.sha(rp),cumulative_seconds=prior_seconds+time.monotonic()-invoked))
        active='FULL_FIXED_TRAINING';model.train();last_saved=time.monotonic();limits=2 if a.profile_only else 325;training_started=time.monotonic();events=[]
        for step in range(cursor,limits):
            resource()
            if stopped[0] or time.monotonic()-invoked>a.max_seconds-55:
                assert not a.profile_only;snapshot();U.atomic(a.segment_output/'completion.json',dict(status='ARCH_SEGMENT_EXACT_RESUME_SAVED',method=a.method,subject=a.subject,cursor=cursor,resume=U.ref(resume),original_D_resumed=False));return
            epoch=step//13;batch_index=step%13;positions=orders[epoch,batch_index*1024:(batch_index+1)*1024];rows=train_ids[positions.cuda()]
            optimizer.zero_grad(set_to_none=True);seed=M.teacher_seed(a.subject,epoch,batch_index)
            with torch.random.fork_rng(devices=[0]):
                torch.cuda.manual_seed(seed)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):f=teacherF(**{k:v[positions.cuda()] for k,v in visual.items()})
            if M.METHODS[a.method]['output']=='IP4x2048':
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):target=teacherP(teacherA(f.float()))
            else:target=f
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(eeg[rows]);loss=M.loss(prediction,target.detach(),logit)
            assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);assert norm>0
            assert all(p.grad is None for t in teachers.values() for p in t.parameters());optimizer.step();scheduler.step()
            assert not coverage[epoch,positions].any();coverage[epoch,positions]=True;cursor=step+1
            e=dict(stage=active,epoch=epoch+1,update=cursor,rows=len(positions),batch_index=batch_index,InfoNCE=float(loss.detach()),gradient_norm=float(norm),prediction_norm=float(prediction.detach().float().flatten(1).norm(dim=1).mean()),target_norm=float(target.float().flatten(1).norm(dim=1).mean()),teacher_seed=seed)
            log(e);events.append(e)
            if not a.profile_only and (cursor%65==0 or time.monotonic()-last_saved>120):snapshot();last_saved=time.monotonic()
        train_seconds=time.monotonic()-training_started
        if a.profile_only:
            resource();U.atomic(a.segment_output/'completion.json',dict(status='ARCHITECTURE_TWO_FULL_BATCH_NATIVE_TEACHER_GPU_PROFILE_COMPLETE',method=a.method,subject=a.subject,parameters=sum(p.numel() for p in model.parameters()),batch=1024,temporary_updates=2,temporary_model_discarded=True,source=U.ref(__file__),registration=U.ref(rp),wrapper=wrapper,events=events,training_step_seconds=train_seconds/2,seconds=time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=torch.cuda.max_memory_allocated(),normal_inference_target_access=False,original_D_resumed=False));return
        assert cursor==325 and coverage.all();snapshot();active='FIXED_FINAL_MODEL_AND_EEG_ONLY_DV_EXPORT'
        final=a.output/'brain_final.pt';M.save_tensor(final,dict(model=model.state_dict(),logit_scale=logit,epochs=25,updates=325,method=a.method,subject=a.subject,registration_sha256=U.sha(rp)))
        del teachers,teacherF,teacherA,teacherP,visual,target,f,prediction;torch.cuda.empty_cache();model.eval();pieces=[];export_started=time.monotonic()
        with torch.no_grad():
            for lo in range(0,len(export_ids),64):
                resource();batch_rows=torch.tensor(export_ids[lo:lo+64],device='cuda')
                # FP32 fixed inference export: no extra F/target call and no
                # arbitrary unit-normalization replacement at the interface.
                pieces.append(model(eeg[batch_rows]).float().cpu().numpy())
        values=np.concatenate(pieces);expected=(3310,4,2048) if M.METHODS[a.method]['output']=='IP4x2048' else (3310,1024)
        assert values.shape==expected and np.isfinite(values).all();U.save_array(a.output/'DV_predictions.npy',values);U.save_array(a.output/'DV_array_rows.npy',np.array(export_ids,dtype=np.int64));resource()
        done=dict(status='ARCH_ALL25EPOCHS_AND_DV_EEG_ONLY_EXPORT_COMPLETE',method=a.method,subject=a.subject,model=U.ref(final),resume=U.ref(resume),registration=U.ref(rp),spec=U.ref(a.spec),predictions=U.ref(a.output/'DV_predictions.npy'),array_rows=U.ref(a.output/'DV_array_rows.npy'),shape=list(values.shape),
            parameters=sum(p.numel() for p in model.parameters()),epochs=25,updates=325,unique_T_stimuli=13230,query_exposures=330750,all_T_coverage=True,physical_repeats=4,channels=len(channels),output=M.METHODS[a.method]['output'],OOF_predictions=False,
            no_new_adapter=True,frozen_F_A_P_IP_UNet=True,native_A_P_bypassed_at_normal_inference=M.METHODS[a.method]['output']=='IP4x2048',normal_inference_target_access=False,Dfit_or_Dcheck_model_selection=False,original_D_resumed=False,training_seconds_this_segment=train_seconds,EEG_only_DV_export_seconds=time.monotonic()-export_started,EEG_only_DV_export_queries=3310,cached_image_generation_excludes_this_already_charged_encoder_cost=True,cumulative_seconds=prior_seconds+time.monotonic()-invoked,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=torch.cuda.max_memory_allocated(),SOTA_established=False)
        U.atomic(a.output/'completion.json',done);U.atomic(a.segment_output/'completion.json',dict(status='ARCH_SUBJECT_TRAINING_AND_EXPORT_COMPLETE',subject_completion=U.ref(a.output/'completion.json'),cursor=cursor,method=a.method,subject=a.subject,original_D_resumed=False))
    except BaseException as e:
        if optimizer is not None and model is not None:
            M.save_tensor(a.output/'failure_state.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),cursor=cursor,CPU_rng=torch.get_rng_state(),CUDA_rng=torch.cuda.get_rng_state_all()))
        U.atomic(a.segment_output/'failure.json',dict(status='FAILED_ARCHITECTURE_SUBJECT_ATTEMPT_PRESERVED',method=a.method,subject=a.subject,stage=active,error=repr(e),cursor=cursor,seconds=time.monotonic()-invoked,original_D_resumed=False,scientific_negative=False));traceback.print_exc();raise

if __name__=='__main__':main()
