"""Original-main ATM encoder/prior training, with fixed epochs and a test lock.

This entry point reuses the original epoch functions. It does not import the
old test-driven main loop. GPU training waits for completed AVDE generation and
metrics, real single-trial cache validation, and the declared runtime.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_main_runtime import check_source,digest,load_model_training,load_prior,add_prior_epoch_hook,native_dataset


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.pending');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state_all())


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.pending');torch.save(value,temporary);temporary.replace(path)


def finite(state):
    for value in state.values():
        if isinstance(value,torch.Tensor) and value.is_floating_point():assert torch.isfinite(value).all()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    config_path=Path(args.config).resolve();config=json.loads(config_path.read_text())
    assert config['schema']=='atm-paper-main-single-task-v1'
    assert (config['seed'],config['subject'],config['encoder_epochs'],config['encoder_batch'],
            config['encoder_lr'],config['prior_epochs'],config['prior_batch'],config['prior_lr'])==(0,'sub-01',40,64,3e-4,150,1024,1e-3)
    assert config['precision']=='fp32' and not config['average_training_trials']
    source=check_source(config['author_main']);missing=[];versions={}
    for name,expected in config['runtime_requirements'].items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        versions[name]=actual
        if actual is None or actual.split('+')[0]!=expected:missing.append('runtime:'+name+'=='+expected)
    avde_path=Path(config['completed_avde_progress'])
    avde=json.loads(avde_path.read_text()) if avde_path.exists() else {}
    if avde.get('status')!='completed' or not avde.get('generation_results_available'):
        missing.append('completed AVDE native training/generation/metrics')
    cache_root=Path(config['cache_validation']);cache_report=cache_root/'results.json'
    ready=json.loads(cache_report.read_text()) if cache_report.exists() else {}
    if ready.get('status')!='completed' or ready.get('EEG_rows_verified')!=66160:
        missing.append('real single-trial native-loader validation (66160 rows)')
    runtime_path=Path(config['runtime_check']);runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    if runtime.get('status')!='completed' or not runtime.get('real_input_native_encoder_objective',{}).get('native_raw_CLIP_objective_confirmed'):
        missing.append('exact legacy raw-feature training-objective check')
    out=Path(args.out).resolve()
    if args.validate_only:
        out.mkdir(exist_ok=False)
        plan=dict(status='completed',training_status='planned',validate_only=True,
            formal_training_started=False,official_test_read=False,
            missing_preconditions=missing,runtime=versions,config_sha256=digest(config_path),
            encoder_optimizer_steps=40*(66160//64),prior_optimizer_steps_upper_count=150*((66160+1023)//1024),
            original_model_and_encoder_epoch_function_reused=True,
            native_prior_epoch_function_has_post_epoch_receipt_hook=True,
            original_raw_CLIP_loss_retained=True,original_prior_clip_sample_retained=True,
            checkpoint_selection='fixed final encoder40/prior150; official test never read during training')
        write(out/'results.json',plan);print(json.dumps(plan),flush=True);return
    assert not missing,'Training gates not met: '+repr(missing)
    assert out==Path(config['training_output']).resolve()
    assert torch.cuda.is_available()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),'Another GPU job is live'
    assert shutil.disk_usage(out.parent).free>5*1024**3+1500*1024**2
    contract=json.loads((cache_root/'data_contract.json').read_text());assets=json.loads((cache_root/'asset_manifest.json').read_text())
    cache=Path(contract['output']['path']);assert cache.stat().st_size==contract['output']['bytes'] and digest(cache)==contract['output']['sha256']
    assert digest(assets['features']['path'])==assets['features']['sha256']
    assert digest(assets['image_manifest']['path'])==assets['image_manifest']['sha256']
    out.mkdir(exist_ok=False);started=time.monotonic();device='cuda:0'
    report=dict(status='running',stage='encoder',official_test_read=False,seed=config['seed'],subject=config['subject'],
        config_sha256=digest(config_path),data_contract_sha256=digest(cache_root/'data_contract.json'),
        runtime=versions,author_main_revision='49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d',
        implementation_files={str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('paper_main_runtime.py')]},
        encoder_epochs=[],prior_epochs=[],SOTA_claim=False,images_generated=0)
    write(out/'progress.json',report)
    try:
        torch.set_num_threads(2);seed(config['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        runtime=load_model_training(source)
        dataset=native_dataset(source,cache.parent.parent,config['images'],Path(assets['features']['path']).parent,out/'dataset_runtime')
        model=runtime.ATMS().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['encoder_lr'])
        loader=DataLoader(dataset,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
        assert len(loader)==1033
        for epoch in range(40):
            epoch_started=time.monotonic()
            loss,accuracy,discard=runtime.train_model('sub-01',model,loader,optimizer,device,
                dataset.text_features,dataset.img_features,types.SimpleNamespace(insubject=True))
            del discard
            finite(model.state_dict())
            steps=sorted({float(s['step']) for s in optimizer.state.values() if 'step' in s})
            assert steps==[float((epoch+1)*1033)]
            row=dict(epoch=epoch+1,loss=loss,native_training_class_accuracy=accuracy,
                     actual_Adam_steps=steps[0],seconds=time.monotonic()-epoch_started)
            report['encoder_epochs'].append(row)
            save(out/'encoder_resume_last.pt',dict(epoch=epoch+1,model=model.state_dict(),optimizer=optimizer.state_dict(),RNG=rng()))
            if (epoch+1)%5==0:save(out/f'encoder_epoch_{epoch+1}.pth',model.state_dict())
            write(out/'progress.json',report);print(json.dumps(dict(stage='encoder',**row)),flush=True)
        del optimizer,loader
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        ordered=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=0,drop_last=False)
        conditions=torch.empty(66160,1024);offset=0
        with torch.no_grad():
            for eeg,labels,text,text_features,image,img_features in ordered:
                n=len(eeg);condition=model(eeg.to(device),torch.ones(n,dtype=torch.long,device=device)).float().cpu()
                conditions[offset:offset+n]=condition;offset+=n
        assert offset==66160 and torch.isfinite(conditions).all()
        del ordered,model;torch.cuda.empty_cache()
        condition_path=out/'ordered_training_eeg_features.pt'
        save(condition_path,dict(features=conditions,row_ids=torch.arange(66160),image_ids=torch.arange(66160)//4,
            repeat_ids=torch.arange(66160)%4,encoder_sha256=digest(out/'encoder_epoch_40.pth')))
        # Reset to the declared seed for the independent prior initialization phase.
        seed(config['seed']);module=load_prior(source)
        prior=module.DiffusionPriorUNet(cond_dim=1024,dropout=.1);pipe=module.Pipe(prior,device=device)
        prior_dataset=module.EmbeddingDataset(c_embeddings=conditions,h_embeddings=dataset.img_features.repeat_interleave(4,dim=0))
        prior_loader=DataLoader(prior_dataset,batch_size=1024,shuffle=True,num_workers=0,drop_last=False)
        assert len(prior_loader)==65
        report['stage']='prior';write(out/'progress.json',report)
        def prior_epoch(epoch,loss,optimizer,scheduler):
            finite(pipe.diffusion_prior.state_dict())
            row=dict(epoch=epoch+1,noise_MSE=loss,scheduler_steps=scheduler.last_epoch,
                     optimizer_step_range=[min(float(s['step']) for s in optimizer.state.values()),
                                           max(float(s['step']) for s in optimizer.state.values())])
            report['prior_epochs'].append(row)
            save(out/'prior_resume_last.pt',dict(epoch=epoch+1,model=pipe.diffusion_prior.state_dict(),
                optimizer=optimizer.state_dict(),lr_scheduler=scheduler.state_dict(),RNG=rng()))
            if (epoch+1)%10==0:save(out/f'prior_epoch_{epoch+1}.pth',pipe.diffusion_prior.state_dict())
            write(out/'progress.json',report);print(json.dumps(dict(stage='prior',**row)),flush=True)
        add_prior_epoch_hook(pipe,source,prior_epoch)
        pipe.train(prior_loader,num_epochs=150,learning_rate=1e-3)
        assert len(report['encoder_epochs'])==40 and len(report['prior_epochs'])==150
        report.update(status='completed',stage='native_training_completed',generation_completed=False,
            encoder=dict(path=str(out/'encoder_epoch_40.pth'),sha256=digest(out/'encoder_epoch_40.pth')),
            prior=dict(path=str(out/'prior_epoch_150.pth'),sha256=digest(out/'prior_epoch_150.pth')),
            ordered_features_sha256=digest(condition_path))
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,message=str(error));raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if report['status']=='completed':write(out/'weight_manifest.json',report)
        write(out/'progress.json',report);write(out/'results.json',report)


if __name__=='__main__':main()
