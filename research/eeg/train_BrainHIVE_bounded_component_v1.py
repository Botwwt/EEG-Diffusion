"""Full real-OOF image denoising training; fixed last checkpoint, no V/test reads."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from brainhive_native_models_v1 import *
from brainhive_bounded_component_v1 import BoundedConditionAdapter,risk_target
from brainhive_component_generator_v1 import load_native,denoise
from brainhive_component_data_v1 import load_training,load_development
from brainhive_lifecycle_storage_v1 import verify_remaining


def seed(namespace):return int(hashlib.sha256(('1729|BrainHIVE_bounded_v1|'+namespace).encode()).hexdigest()[:16],16)%(2**63-1)


@torch.no_grad()
def development(model,adapter,projector,arm):
    fit=load_development('Dfit');report={}
    if arm=='ordinary':
        totals=[]
        for strength in [0.,.25,.5,.75,1.]:
            model.radius=.25*strength
            values=[]
            for start in range(0,len(fit['condition']),256):
                sl=slice(start,start+256);c=fit['condition'][sl]
                changed,_=model(c,fit['weight'][sl],fit['bias'][sl],gate_mode='fixed')
                # Calibrate the scalar INSIDE the norm-preserving correction;
                # do not linearly blend two sphere points outside host support.
                c=changed
                with torch.autocast('cuda',dtype=torch.bfloat16):a=projector(adapter(c));b=projector(adapter(fit['target'][sl]))
                values.append((1-F.cosine_similarity(a.float(),b.float(),dim=-1)).mean(-1))
            totals.append(float(torch.cat(values).mean()))
        index=min(range(5),key=lambda i:(totals[i],i));strength=[0.,.25,.5,.75,1.][index]
        model.radius=.25
        report=dict(kind='Dfit_only_global_strength_calibration',strengths=[0.,.25,.5,.75,1.],
            Dfit_token_errors=totals,selected_strength=strength,selected_without_V_or_test_images=True)
    else:
        def predict(data):
            return torch.cat([model(data['condition'][i:i+256],data['weight'][i:i+256],data['bias'][i:i+256])[1]['risk']
                for i in range(0,len(data['condition']),256)])
        predicted=predict(fit);y=fit['risk']
        slope=float(((predicted-predicted.mean())*(y-y.mean())).mean()/predicted.var(unbiased=False).clamp_min(1e-12))
        slope=max(0.,slope);intercept=float(y.mean()-slope*predicted.mean())
        report=dict(kind='Dfit_nonnegative_affine_prediction_risk_calibration',slope=slope,intercept=intercept,
            Dfit_constant_risk=float(y.mean()),Dfit_raw_risk_MSE=float((predicted-y).square().mean()),
            Dfit_calibrated_risk_MSE=float(((slope*predicted+intercept).clamp(0,2)-y).square().mean()),
            risk_is_physiological_SNR=False)
        del fit
        check=load_development('Dcheck');predicted=predict(check);calibrated=(slope*predicted+intercept).clamp(0,2);y=check['risk']
        delta=(calibrated-y).square()-(report['Dfit_constant_risk']-y).square()
        report.update(Dcheck_queries=len(y),Dcheck_raw_risk_MSE=float((predicted-y).square().mean()),
            Dcheck_calibrated_risk_MSE=float((calibrated-y).square().mean()),
            Dcheck_Dfit_constant_MSE=float((report['Dfit_constant_risk']-y).square().mean()),
            Dcheck_MSE_difference_vs_constant=float(delta.mean()),
            Dcheck_used_to_revise_or_select_candidate=False,
            calibration_target='Already-audited FP32 native A/P angular error, no repeated target evaluation')
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['ordinary','risk'],required=True)
    parser.add_argument('--recipe',required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert b'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes()
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    recipe_path=Path(args.recipe);recipe=json.loads(recipe_path.read_text());assert recipe['passes']==3 and recipe['batch_size']==16
    for name,expected in recipe['source_sha256'].items():assert sha(ROOT/name)==expected,name
    storage=json.loads(Path(recipe['storage_lifecycle_receipt']).read_text())
    assert storage['all_three_main_lifecycles_reserved'] and storage['formal_queries_per_arm']==16600
    verify_remaining(recipe)
    inputs=load_training();assert inputs['input_contract_sha256']==recipe['input_contract_sha256']
    latent_receipt=Path(recipe['latent_receipt']);latent=json.loads(latent_receipt.read_text())
    assert latent['status']=='COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS' and sha(latent['path'])==latent['sha256']
    assert latent['array_rows']==inputs['array_rows'] and latent['format']=='numpy_npy_float32_C_contiguous_v2'
    import numpy as np
    array=np.load(latent['path'],mmap_mode='r');assert array.shape==(13230,4,64,64) and array.dtype==np.float32
    image_latents=torch.tensor(array,device='cuda');del array
    pipe,adapter,projector,prompts,host=load_native(training=True)
    from diffusers import DDPMScheduler
    scheduler=DDPMScheduler.from_config(pipe.scheduler.config);assert scheduler.config.prediction_type=='epsilon'
    out=WORK/'components/bounded_v1'/args.arm
    if args.resume:assert out.exists() and (out/'resume.pt').exists()
    else:assert not out.exists();out.mkdir(parents=True)
    assert not (out/'completion.json').exists()
    torch.manual_seed(1729);model=BoundedConditionAdapter(predict_risk=args.arm=='risk').cuda()
    expected=525825 if args.arm=='risk' else 525568;assert sum(p.numel() for p in model.parameters())==expected
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    rng=torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
    orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)]
    provenance=dict(recipe_sha256=sha(recipe_path),input_contract_sha256=inputs['input_contract_sha256'],
        latent_receipt_sha256=sha(latent_receipt),arm=args.arm,parameters=expected,host=host,
        orders_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders],
        model_is_new_and_host_frozen=True,normal_inference_reads_targets=False)
    lock=out/'input_provenance.json'
    if lock.exists():assert json.loads(lock.read_text())==provenance
    else:write(lock,provenance)
    cursor=0;updates=0;seen=0;loss_sum=0.;cumulative_seconds=0.
    coverage=torch.zeros(3,132300,dtype=torch.bool)
    if args.resume:
        saved=torch.load(out/'resume.pt',weights_only=False,map_location='cpu')
        assert saved['input_provenance_sha256']==sha(lock)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        cursor=saved['cursor'];updates=saved['updates'];seen=saved['seen'];loss_sum=saved['loss_sum'];coverage=saved['coverage']
        rng.set_state(saved['noise_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        cumulative_seconds=saved['seconds']
    started=time.monotonic();last_saved=started;torch.cuda.reset_peak_memory_stats()
    def snapshot():
        save_tensor(out/'resume.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            optimizer=optimizer.state_dict(),cursor=cursor,updates=updates,seen=seen,loss_sum=loss_sum,coverage=coverage,
            noise_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            seconds=cumulative_seconds+time.monotonic()-started,input_provenance_sha256=sha(lock)))
    with (out/'training.jsonl').open('a',buffering=1) as log:
        for epoch in range(cursor//132300,3):
            start_at=cursor%132300 if epoch==cursor//132300 else 0
            for offset in range(start_at,132300,16):
                if time.monotonic()-started>recipe['training_job_seconds']-90:
                    snapshot();write(out/'progress.json',dict(status='RESOURCE_STOPPED_RESUMABLE',updates=updates,cursor=cursor))
                    raise RuntimeError('Finite training timeout boundary; exact resume retained')
                indices=orders[epoch][offset:offset+16];n=len(indices);pos=(indices%13230).cuda();idx=indices.cuda()
                affine=inputs['affine_ids'][idx];c=inputs['condition'][idx];w=inputs['weight'][affine];b=inputs['bias'][affine]
                latent=image_latents[pos];noise=torch.randn(latent.shape,device='cuda',generator=rng)
                t=torch.randint(0,1000,(n,),device='cuda',generator=rng);noisy=scheduler.add_noise(latent,noise,t)
                optimizer.zero_grad(set_to_none=True);changed,info=model(c,w,b)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    with torch.no_grad():
                        real=projector(adapter(inputs['target'][pos]))
                        base=projector(adapter(c)) if args.arm=='risk' else None
                    tokens=projector(adapter(changed));predicted=denoise(pipe,prompts,noisy,t,tokens)
                    image_loss=F.mse_loss(predicted.float(),noise)
                    token_loss=(1-F.cosine_similarity(tokens.float(),real.float(),dim=-1)).mean()
                    auxiliary=F.mse_loss(info['risk'],risk_target(base,real)) if args.arm=='risk' else image_loss.new_zeros(())
                    loss=image_loss+.1*token_loss+.1*auxiliary
                assert torch.isfinite(loss);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                assert model.direction.weight.grad is not None
                assert all(p.grad is None for p in pipe.unet.parameters())
                optimizer.step()
                assert not coverage[epoch,indices].any();coverage[epoch,indices]=True
                cursor=epoch*132300+offset+n;updates+=1;seen+=n;loss_sum+=float(loss.detach())*n
                if updates%64==0 or offset+n==132300:
                    event=dict(stage='BRAINHIVE_REAL_OOF_IMAGE_TRAINING',arm=args.arm,pass_number=epoch+1,
                        updates=updates,expected_updates=3*math.ceil(132300/16),seen_queries=seen,expected_queries=396900,
                        cumulative_mean_loss=loss_sum/seen,last_loss=float(loss),epsilon_loss=float(image_loss),
                        token_loss=float(token_loss),risk_auxiliary_loss=float(auxiliary),timesteps=t.tolist(),
                        gradient_norm=float(gradient),correction_RMS=float(info['correction'].square().mean().sqrt()),
                        mean_gate=float(info['gate'].mean()),seconds=cumulative_seconds+time.monotonic()-started)
                    log.write(json.dumps(event)+'\n');write(out/'progress.json',dict(status='RUNNING',**event));print(json.dumps(event),flush=True)
                if time.monotonic()-last_saved>120:snapshot();last_saved=time.monotonic()
    assert coverage.all() and seen==396900 and updates==24807
    snapshot()
    calibration=development(model.eval(),adapter,projector,args.arm)
    write(out/'calibration.json',calibration)
    save_tensor(out/'component_final.pt',dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        arm=args.arm,seed=1729,updates=updates,seen_queries=seen,passes=3,calibration=calibration,
        input_provenance_sha256=sha(lock),recipe_sha256=sha(recipe_path)))
    write(out/'completion.json',dict(status='COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_COMPONENT',
        arm=args.arm,parameters=expected,subjects=10,unique_train_queries=132300,total_seen_queries=seen,updates=updates,
        checkpoint_path=str(out/'component_final.pt'),checkpoint_sha256=sha(out/'component_final.pt'),
        input_provenance_sha256=sha(lock),calibration_sha256=sha(out/'calibration.json'),
        training_seconds=cumulative_seconds+time.monotonic()-started,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_selection='Fixed final update; no V or official test target/score use',
        training_complete_is_not_image_success=True,formal_images_generated=0))
    print('COMPLETE_FULL_COMPONENT_TRAINING',args.arm,flush=True)


if __name__=='__main__':main()
