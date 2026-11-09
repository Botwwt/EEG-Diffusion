"""Full-data, per-subject author ATMS + native epsilon-prior training.

Reads resolved inputs and protected splits; never opens official test EEG/images.
No legacy checkpoint is overwritten. Epoch checkpoints permit deterministic resume.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/root/atm_official_20261016')
sys.path[:0]=['/root/mced_representation_runtime_v1/lib/python3.12/site-packages',
             '/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages',
             '/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages',
             str(SOURCE),str(SOURCE/'Generation')]
import numpy as np
import torch
import torch.nn.functional as F
from models.atms import ATMS
from encoder_utils import _compute_loss
from diffusion_prior import DiffusionPriorUNet
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup


def digest(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def checkpoint(path,value):
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])


def reset_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


class BudgetStop(RuntimeError):pass


class Trainer:
    def __init__(self,config,role):
        self.config=config;self.role=role;self.started=time.monotonic()
        self.output=Path(config['output_root'])/role;self.output.mkdir(parents=True,exist_ok=True)
        self.protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
        assert digest(ROOT/'configs/split_manifest.json')==config['split_sha256']
        assert json.loads((ROOT/'audit/data_audit.json').read_text())['complete_for_formal_training']
        self.input_rows=list(csv.DictReader((ROOT/'audit/data_manifest.csv').open()))
        self.image_rows={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
        assert digest(config['visual_features_path'])==config['visual_features_sha256']
        self.visual=torch.load(config['visual_features_path'],weights_only=True,mmap=True).clone().cuda()
        assert self.visual.shape==(16540,1024) and torch.isfinite(self.visual).all()
        t=self.protocol['image_rows']['T']
        if role.startswith('oof'):
            heldout=int(role[-1]);folds=self.protocol['oof_fold_by_concept']
            t=[i for i in t if folds[self.image_rows[i]['concept_id']]!=heldout]
        self.t=torch.tensor(t,dtype=torch.long,device='cuda')
        self.v=torch.tensor(self.protocol['image_rows']['V'],dtype=torch.long,device='cuda')
        assert not set(t)&set(self.protocol['image_rows']['D']+self.protocol['image_rows']['V'])
        self.log=(self.output/'epochs.jsonl').open('a',buffering=1)
        self.seconds_before=0.
        old=self.output/'run_state.json'
        if old.exists():self.seconds_before=json.loads(old.read_text()).get('charged_job_seconds',0.)

    def elapsed(self):return self.seconds_before+time.monotonic()-self.started

    def check_budget(self):
        if self.elapsed()>self.config['role_max_wall_seconds'][self.role] or (ROOT/'STOP_FORMAL_TRAINING').exists():
            raise BudgetStop('Preserve epoch-boundary resume; registered role budget or stop file reached')

    def record(self,value):
        value.update(role=self.role,job_seconds=self.elapsed(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        self.log.write(json.dumps(value)+'\n');print(json.dumps(value),flush=True)
        write(self.output/'run_state.json',value|dict(charged_job_seconds=self.elapsed()))

    @torch.inference_mode()
    def encode(self,model,eeg,subject,rows,batch=128):
        model.eval();result=[]
        for start in range(0,len(rows),batch):
            x=eeg[rows[start:start+batch]]
            ids=torch.full((len(x),),subject,dtype=torch.long,device='cuda')
            y=model(x,ids).float();assert torch.isfinite(y).all();result.append(y)
        return torch.cat(result)

    @torch.inference_mode()
    def validate_encoder(self,model,eeg,subject):
        prediction=self.encode(model,eeg,subject,self.v)
        target=self.visual[self.v];n=len(target);total=0.
        for start in range(0,n,64):
            a=prediction[start:start+64];b=target[start:start+64]
            total+=float(_compute_loss(a,b,model.logit_scale,model.loss_func,'generation',.99))*len(a)
        a=F.normalize(prediction,dim=-1);b=F.normalize(target,dim=-1)
        predicted=(a@b.T).argmax(-1)
        correct=torch.arange(n,device='cuda')
        concepts=[self.image_rows[i]['concept_id'] for i in self.v.tolist()]
        return dict(selection_loss=total/n,native_mse=float(F.mse_loss(prediction,target)),
            native_cosine=float((a*b).sum(-1).mean()),
            native_stimulus_top1=float((predicted==correct).float().mean()),
            native_concept_top1=sum(concepts[p]==concepts[i] for i,p in enumerate(predicted.tolist()))/n,
            gallery_images=n,queries=n)

    def train_encoder(self,eeg,subject,folder):
        reset_seed(self.config['seed']);model=ATMS().cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        resume=folder/'encoder_resume.pt';summary=folder/'encoder_training.json'
        if summary.exists():
            report=json.loads(summary.read_text());assert report['status']=='COMPLETED'
            model.load_state_dict(torch.load(folder/'encoder_best.pt',weights_only=True),strict=True)
            return model,report
        first=0;best=float('inf');best_epoch=0;bad=0;updates=0;seen=0
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            first=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad'];updates=saved['updates'];seen=saved['seen'];restore_rng(saved['rng'])
        epoch=first-1
        for epoch in range(first,self.config['encoder_epochs_max']):
            self.check_budget();tick=time.monotonic();model.train();sums=dict(loss=0.,mse=0.,cosine=0.);count=0;norms=[]
            order=self.t[torch.randperm(len(self.t),device='cuda')]
            assert len(torch.unique(order))==len(self.t)
            for start in range(0,len(order),64):
                rows=order[start:start+64];x=eeg[rows];target=self.visual[rows]
                ids=torch.full((len(rows),),subject,dtype=torch.long,device='cuda')
                prediction=model(x,ids).float()
                loss=_compute_loss(prediction,target,model.logit_scale,model.loss_func,'generation',.9)
                assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.stack([p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]).norm()
                assert torch.isfinite(norm);opt.step()
                with torch.no_grad():
                    sums['loss']+=float(loss)*len(rows);sums['mse']+=float(F.mse_loss(prediction,target))*len(rows)
                    sums['cosine']+=float(F.cosine_similarity(prediction,target).sum());norms.append(float(norm))
                count+=len(rows);updates+=1
            assert count==len(self.t);seen+=count
            validation=self.validate_encoder(model,eeg,subject)
            if validation['selection_loss']<best:
                best=validation['selection_loss'];best_epoch=epoch+1;bad=0
                checkpoint(folder/'encoder_best.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            else:bad+=1
            checkpoint(resume,dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,best=best,best_epoch=best_epoch,bad=bad,updates=updates,seen=seen,rng=rng_state()))
            self.record(dict(stage='encoder',subject=subject,epoch=epoch+1,updates=updates,seen_samples=seen,
                unique_training_rows=len(self.t),effective_epochs=seen/len(self.t),
                mean_loss=sums['loss']/count,mean_native_mse=sums['mse']/count,mean_native_cosine=sums['cosine']/count,
                mean_gradient_norm=float(np.mean(norms)),learning_rate=opt.param_groups[0]['lr'],
                validation=validation,best_epoch=best_epoch,seconds=time.monotonic()-tick))
            if bad>=self.config['encoder_patience']:break
        report=dict(status='COMPLETED',epochs=epoch+1,updates=updates,seen_samples=seen,unique_training_rows=len(self.t),
            best_epoch=best_epoch,best_selection_loss=best,early_stopped=bad>=self.config['encoder_patience'],
            parameters=sum(p.numel() for p in model.parameters()),
            gradient_active_parameters=sum(p.numel() for p in model.parameters() if p.grad is not None),
            checkpoint=str(folder/'encoder_best.pt'),checkpoint_sha256=digest(folder/'encoder_best.pt'))
        write(summary,report)
        model.load_state_dict(torch.load(folder/'encoder_best.pt',weights_only=True),strict=True)
        # Finished jobs retain exact selected weights and epoch records; optimizer is scratch.
        resume.unlink();return model,report

    def train_prior(self,condition,subject,folder):
        reset_seed(self.config['seed']);model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        nsteps=math.ceil(len(self.t)/1024)*self.config['prior_epochs_max']
        lr=get_cosine_schedule_with_warmup(opt,num_warmup_steps=500,num_training_steps=nsteps)
        schedule=DDPMScheduler(clip_sample=False);resume=folder/'prior_resume.pt';summary=folder/'prior_training.json'
        if summary.exists():return json.loads(summary.read_text())
        # Fixed validation noise removes sampling fluctuations from checkpoint selection.
        seed=int(hashlib.sha256(f'{self.config["seed"]}|prior-validation|{self.role}|{subject}'.encode()).hexdigest()[:16],16)%(2**63-1)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        vn=torch.randn((len(self.v),1024),device='cuda',generator=generator)
        vt=torch.randint(0,1000,(len(self.v),),device='cuda',generator=generator)
        vx=schedule.add_noise(self.visual[self.v],vn,vt)
        first=0;best=float('inf');best_epoch=0;updates=0;seen=0
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);lr.load_state_dict(saved['lr'])
            first=saved['epoch'];best=saved['best'];best_epoch=saved['best_epoch'];updates=saved['updates'];seen=saved['seen'];restore_rng(saved['rng'])
        epoch=first-1
        for epoch in range(first,self.config['prior_epochs_max']):
            self.check_budget();tick=time.monotonic();model.train();loss_sum=0.;count=0;norms=[]
            order=self.t[torch.randperm(len(self.t),device='cuda')]
            for start in range(0,len(order),1024):
                rows=order[start:start+1024];target=self.visual[rows]
                c=None if torch.rand(1).item()<.1 else condition[rows]
                noise=torch.randn_like(target);t=torch.randint(0,1000,(len(rows),),device='cuda')
                x=schedule.add_noise(target,noise,t);loss=F.mse_loss(model(x,t,c),noise)
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();lr.step();updates+=1;count+=len(rows);loss_sum+=float(loss)*len(rows);norms.append(float(norm))
            assert count==len(self.t);seen+=count;model.eval();val_sum=0.
            with torch.inference_mode():
                for start in range(0,len(self.v),1024):
                    end=min(start+1024,len(self.v));p=model(vx[start:end],vt[start:end],condition[self.v[start:end]])
                    val_sum+=float(F.mse_loss(p,vn[start:end]))*(end-start)
            val=val_sum/len(self.v)
            if val<best:
                best=val;best_epoch=epoch+1
                checkpoint(folder/'prior_best.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            checkpoint(resume,dict(model=model.state_dict(),optimizer=opt.state_dict(),lr=lr.state_dict(),epoch=epoch+1,
                best=best,best_epoch=best_epoch,updates=updates,seen=seen,rng=rng_state()))
            self.record(dict(stage='prior',subject=subject,epoch=epoch+1,updates=updates,seen_samples=seen,
                effective_epochs=seen/len(self.t),unique_training_rows=len(self.t),mean_loss=loss_sum/count,
                mean_gradient_norm_preclip=float(np.mean(norms)),learning_rate=opt.param_groups[0]['lr'],
                fixed_noise_validation_loss=val,best_epoch=best_epoch,seconds=time.monotonic()-tick))
        report=dict(status='COMPLETED',epochs=epoch+1,updates=updates,seen_samples=seen,unique_training_rows=len(self.t),
            best_epoch=best_epoch,best_validation_loss=best,parameters=sum(p.numel() for p in model.parameters()),
            checkpoint=str(folder/'prior_best.pt'),checkpoint_sha256=digest(folder/'prior_best.pt'),
            scheduler=dict(schedule.config),prediction_type='epsilon',condition_dropout_probability=.1)
        write(summary,report);resume.unlink();return report

    def run(self):
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE).decode().strip()
        assert source_commit==self.config['official_source_commit']
        completed=[]
        for subject_name in self.protocol['subjects']:
            folder=self.output/subject_name;folder.mkdir(exist_ok=True)
            done=folder/'completion.json'
            if done.exists():completed.append(json.loads(done.read_text()));continue
            self.check_budget();subject=int(subject_name[-2:])
            row=next(r for r in self.input_rows if r['subject']==subject_name and r['dataset_split']=='train' and r['repeat_protocol']=='R4')
            assert row['status']=='VERIFIED_READY' and digest(row['path'])==row['sha256']
            eeg=torch.load(row['path'],weights_only=True,mmap=True).clone().cuda()
            assert eeg.shape==(16540,63,250) and torch.isfinite(eeg).all()
            model,enc=self.train_encoder(eeg,subject,folder)
            allrows=torch.arange(16540,device='cuda');condition=self.encode(model,eeg,subject,allrows)
            cache=folder/'encoder_native_all_train_rows.pt'
            checkpoint(cache,dict(condition=condition.cpu(),array_rows=allrows.cpu(),native_visual_sha256=self.config['visual_features_sha256'],
                                  encoder_sha256=enc['checkpoint_sha256'],training_role=self.role))
            del model,eeg;torch.cuda.empty_cache()
            prior=self.train_prior(condition,subject,folder)
            record=dict(status='COMPLETED',subject=subject_name,role=self.role,encoder=enc,prior=prior,
                condition_cache=str(cache),condition_cache_sha256=digest(cache),formal_official_test_accessed=False,
                split_sha256=self.config['split_sha256'],training_seed=self.config['seed'])
            write(done,record);completed.append(record)
            del condition;torch.cuda.empty_cache()
        report=dict(status='COMPLETED',role=self.role,subjects=completed,charged_job_seconds=self.elapsed(),
                    formal_official_test_accessed=False,scientific_image_success_claim=False)
        write(self.output/'completion.json',report);self.record(dict(stage='COMPLETED',subjects_completed=len(completed)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--role',choices=['full','oof0','oof1'],default='full');args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());assert config['formal_training_authorized_within_user_cap']
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trainer=Trainer(config,args.role)
    try:trainer.run()
    except BudgetStop as exc:
        trainer.record(dict(stage='BUDGET_STOPPED_RESUMABLE',reason=str(exc)));raise
    except Exception:
        trainer.record(dict(stage='FAILED_RESUMABLE',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
