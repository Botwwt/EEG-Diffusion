"""One registered same-capacity v-objective repair; reuse every frozen EEG encoder."""
from train_full_reference import *


class VelocityRepair(Trainer):
    def run(self):
        original=Path(self.config['reuse_full_reference_root']);complete=[]
        for subject_name in self.protocol['subjects']:
            self.check_budget();subject=int(subject_name[-2:]);folder=self.output/subject_name;folder.mkdir(exist_ok=True)
            done=folder/'completion.json'
            if done.exists():complete.append(json.loads(done.read_text()));continue
            reference=json.loads((original/subject_name/'completion.json').read_text())
            assert reference['encoder']['checkpoint_sha256']==self.config['frozen_encoder_sha256'][subject_name]
            assert digest(reference['condition_cache'])==reference['condition_cache_sha256']
            saved=torch.load(reference['condition_cache'],weights_only=True,map_location='cpu')
            assert torch.equal(saved['array_rows'],torch.arange(16540))
            c=saved['condition'].cuda();reset_seed(self.config['seed'])
            model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda()
            assert sum(p.numel() for p in model.parameters())==9675648
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
            lr=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=500,
                num_training_steps=math.ceil(len(self.t)/1024)*400)
            scheduler=DDPMScheduler(clip_sample=False,prediction_type='v_prediction')
            validation_seed=int(hashlib.sha256(f'{self.config["seed"]}|prior-validation|full|{subject}'.encode()).hexdigest()[:16],16)%(2**63-1)
            gen=torch.Generator(device='cuda').manual_seed(validation_seed)
            vn=torch.randn((len(self.v),1024),device='cuda',generator=gen)
            vt=torch.randint(0,1000,(len(self.v),),device='cuda',generator=gen)
            vx=scheduler.add_noise(self.visual[self.v],vn,vt)
            vv=scheduler.get_velocity(self.visual[self.v],vn,vt)
            resume=folder/'prior_resume.pt';first=0;updates=0;seen=0;best=float('inf');best_epoch=0
            if resume.exists():
                state=torch.load(resume,map_location='cpu',weights_only=False)
                model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);lr.load_state_dict(state['lr'])
                first=state['epoch'];updates=state['updates'];seen=state['seen'];best=state['best'];best_epoch=state['best_epoch'];restore_rng(state['rng'])
            epoch=first-1
            for epoch in range(first,400):
                self.check_budget();tick=time.monotonic();model.train();total=0.;count=0;norms=[]
                order=self.t[torch.randperm(len(self.t),device='cuda')]
                for start in range(0,len(order),1024):
                    rows=order[start:start+1024];target=self.visual[rows]
                    condition=None if torch.rand(1).item()<.1 else c[rows]
                    noise=torch.randn_like(target);t=torch.randint(0,1000,(len(rows),),device='cuda')
                    noisy=scheduler.add_noise(target,noise,t);velocity=scheduler.get_velocity(target,noise,t)
                    prediction=model(noisy,t,condition);loss=F.mse_loss(prediction,velocity)
                    assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();lr.step();updates+=1;count+=len(rows);total+=float(loss)*len(rows);norms.append(float(norm))
                assert count==len(self.t);seen+=count;model.eval();val_sum=0.;epsilon_sum=0.
                with torch.inference_mode():
                    for start in range(0,len(self.v),1024):
                        stop=min(start+1024,len(self.v));prediction=model(vx[start:stop],vt[start:stop],c[self.v[start:stop]])
                        val_sum+=float(F.mse_loss(prediction,vv[start:stop]))*(stop-start)
                        alpha=scheduler.alphas_cumprod.to('cuda')[vt[start:stop],None]
                        epsilon=alpha.sqrt()*prediction+(1-alpha).sqrt()*vx[start:stop]
                        epsilon_sum+=float(F.mse_loss(epsilon,vn[start:stop]))*(stop-start)
                validation=val_sum/len(self.v)
                if validation<best:
                    best=validation;best_epoch=epoch+1
                    checkpoint(folder/'prior_best.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
                checkpoint(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),lr=lr.state_dict(),epoch=epoch+1,
                    updates=updates,seen=seen,best=best,best_epoch=best_epoch,rng=rng_state()))
                self.record(dict(stage='prior_velocity_repair',subject=subject,epoch=epoch+1,updates=updates,seen_samples=seen,
                    unique_training_rows=len(self.t),effective_epochs=seen/len(self.t),mean_velocity_loss=total/count,
                    fixed_noise_V_velocity_loss=validation,fixed_noise_V_equivalent_epsilon_loss=epsilon_sum/len(self.v),
                    mean_gradient_norm_preclip=float(np.mean(norms)),learning_rate=optimizer.param_groups[0]['lr'],
                    best_epoch=best_epoch,seconds=time.monotonic()-tick))
            prior=dict(status='COMPLETED',epochs=400,updates=updates,seen_samples=seen,unique_training_rows=len(self.t),
                best_epoch=best_epoch,best_validation_velocity_loss=best,parameters=9675648,prediction_type='v_prediction',
                checkpoint=str(folder/'prior_best.pt'),checkpoint_sha256=digest(folder/'prior_best.pt'),scheduler=dict(scheduler.config))
            write(folder/'prior_training.json',prior)
            report=dict(status='COMPLETED',subject=subject_name,encoder=reference['encoder'],condition_cache=reference['condition_cache'],
                condition_cache_sha256=reference['condition_cache_sha256'],prior=prior,
                original_epsilon_prior=reference['prior'],seed=self.config['seed'],formal_test_accessed=False,
                source_supervision='Same frozen EEG encoder, fullT, same9675648 parameter prior, same400epochs5200updates; objective/associated weighting changed to v only')
            write(done,report);complete.append(report);resume.unlink()
            del model,optimizer,lr,c,vn,vt,vx,vv;torch.cuda.empty_cache()
        write(self.output/'completion.json',dict(status='COMPLETED',subjects=complete,charged_job_seconds=self.elapsed(),
            purpose='Engineering repair of an unstable native epsilon prior, not a new EEG mechanism or novelty claim'))
        self.record(dict(stage='COMPLETED',subjects_completed=len(complete)))


def main():
    config=json.loads((ROOT/'configs/prior_velocity_repair_resolved.json').read_text())
    assert config['seed']==1729 and config['prior_epochs_max']==400 and config['prior_prediction_type']=='v_prediction'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trainer=VelocityRepair(config,'full')
    try:trainer.run()
    except Exception:
        trainer.record(dict(stage='FAILED_OR_BUDGET_STOPPED_RESUMABLE',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
