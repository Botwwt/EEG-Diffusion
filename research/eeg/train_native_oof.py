"""Two concept-held-out roles, matching the selected full native EEG chain.

Prepared implementation; no formal run is authorized by the presence of code.
Requires a completed reference selection and a separate full-budget receipt.
"""
from train_full_reference import *
import inspect


class NativeOOFTrainer(Trainer):
    def train_prior(self,condition,subject,folder):
        if self.config['condition_chain']=='REF_REG':
            return None  # The deployed chain has no semantic prior.
        assert self.config['condition_chain']=='REF_PRIOR_V'
        reset_seed(self.config['seed']);model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda()
        assert sum(p.numel() for p in model.parameters())==9675648
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        lr=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=500,
            num_training_steps=math.ceil(len(self.t)/1024)*400)
        scheduler=DDPMScheduler(clip_sample=False,prediction_type='v_prediction')
        summary=folder/'prior_training.json';resume=folder/'prior_resume.pt'
        if summary.exists():
            saved=json.loads(summary.read_text());assert saved['status']=='COMPLETED'
            assert digest(saved['checkpoint'])==saved['checkpoint_sha256'];return saved
        # One fixed training seed; folds are data roles. Role-specific validation
        # streams cannot select on held-out T or Dcheck residuals.
        seed=int(hashlib.sha256(f'{self.config["seed"]}|prior-validation|{self.role}|{subject}'.encode()).hexdigest()[:16],16)%(2**63-1)
        generator=torch.Generator(device='cuda').manual_seed(seed)
        vn=torch.randn((len(self.v),1024),device='cuda',generator=generator)
        vt=torch.randint(0,1000,(len(self.v),),device='cuda',generator=generator)
        vx=scheduler.add_noise(self.visual[self.v],vn,vt)
        vv=scheduler.get_velocity(self.visual[self.v],vn,vt)
        first=0;updates=0;seen=0;best=float('inf');best_epoch=0
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);lr.load_state_dict(saved['lr'])
            first=saved['epoch'];updates=saved['updates'];seen=saved['seen'];best=saved['best'];best_epoch=saved['best_epoch'];restore_rng(saved['rng'])
        for epoch in range(first,400):
            self.check_budget();tick=time.monotonic();model.train();total=0.;count=0;norms=[]
            order=self.t[torch.randperm(len(self.t),device='cuda')]
            assert len(torch.unique(order))==len(self.t)
            for start in range(0,len(order),1024):
                rows=order[start:start+1024];target=self.visual[rows]
                c=None if torch.rand(1).item()<.1 else condition[rows]
                noise=torch.randn_like(target);t=torch.randint(0,1000,(len(rows),),device='cuda')
                noisy=scheduler.add_noise(target,noise,t);velocity=scheduler.get_velocity(target,noise,t)
                loss=F.mse_loss(model(noisy,t,c),velocity);assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();lr.step();updates+=1;count+=len(rows);total+=float(loss)*len(rows);norms.append(float(norm))
            assert count==len(self.t);seen+=count;model.eval();val=0.
            with torch.inference_mode():
                for start in range(0,len(self.v),1024):
                    end=min(start+1024,len(self.v))
                    val+=float(F.mse_loss(model(vx[start:end],vt[start:end],condition[self.v[start:end]]),vv[start:end]))*(end-start)
            val/=len(self.v)
            if val<best:
                best=val;best_epoch=epoch+1
                checkpoint(folder/'prior_best.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
            checkpoint(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),lr=lr.state_dict(),epoch=epoch+1,
                updates=updates,seen=seen,best=best,best_epoch=best_epoch,rng=rng_state()))
            self.record(dict(stage='OOF_native_v_prior',subject=subject,epoch=epoch+1,updates=updates,seen_samples=seen,
                unique_training_rows=len(self.t),effective_epochs=seen/len(self.t),mean_velocity_loss=total/count,
                fixed_noise_V_velocity_loss=val,mean_gradient_norm_preclip=float(np.mean(norms)),
                learning_rate=optimizer.param_groups[0]['lr'],best_epoch=best_epoch,seconds=time.monotonic()-tick))
        report=dict(status='COMPLETED',parameters=9675648,prediction_type='v_prediction',epochs=400,updates=updates,
            seen_samples=seen,unique_training_rows=len(self.t),best_epoch=best_epoch,best_validation_velocity_loss=best,
            checkpoint=str(folder/'prior_best.pt'),checkpoint_sha256=digest(folder/'prior_best.pt'),scheduler=dict(scheduler.config))
        write(summary,report);resume.unlink();return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--role',choices=['oof0','oof1'],required=True);args=parser.parse_args()
    config_path=Path(args.config);config=json.loads(config_path.read_text())
    assert config['seed']==1729 and config['subjects']==[f'sub-{i:02d}' for i in range(1,11)]
    assert config['encoder_epochs_max']==100 and config['encoder_patience']==20 and config['prior_epochs_max']==400
    assert config['oof_trainer_sha256']==digest(__file__)
    assert config['base_trainer_sha256']==digest(ROOT/'train_full_reference.py')
    assert digest(inspect.getfile(ATMS))==config['official_atms_source_sha256']
    assert digest(inspect.getfile(DiffusionPriorUNet))==config['official_prior_source_sha256']
    selection=ROOT/'audit/reference_control_selection.json'
    assert digest(selection)==config['reference_selection_sha256']
    chosen=json.loads(selection.read_text());assert chosen['selected_reference']==config['condition_chain']
    gate=ROOT/'audit/S3_OOF_budget_gate.json';assert digest(gate)==config['budget_gate_sha256']
    assert json.loads(gate.read_text())['full_two_fold_training_and_condition_sampling_funded']
    parent=Path('/proc')/str(os.getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes(),'Use the registered GPU budget wrapper'
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    trainer=NativeOOFTrainer(config,args.role)
    heldout=int(args.role[-1]);protocol=trainer.protocol;folds=protocol['oof_fold_by_concept']
    training=set(trainer.t.tolist());all_t=set(protocol['image_rows']['T'])
    heldout_rows={i for i in all_t if folds[trainer.image_rows[i]['concept_id']]==heldout}
    assert training.isdisjoint(heldout_rows) and training|heldout_rows==all_t
    assert training.isdisjoint(set(protocol['image_rows']['D']+protocol['image_rows']['V']))
    lock=dict(config_sha256=digest(config_path),role=args.role,heldout_fold=heldout,condition_chain=config['condition_chain'],
        training_array_rows=sorted(training),heldout_T_array_rows=sorted(heldout_rows),
        training_concepts=sorted({trainer.image_rows[i]['concept_id'] for i in training}),
        heldout_concepts=sorted({trainer.image_rows[i]['concept_id'] for i in heldout_rows}),
        sampling_contract='Use deployed full chain, original raw native coordinates and target query prior stream; never epsilon residuals for a v deployment',
        official_test_accessed=False)
    path=trainer.output/'OOF_data_roles.json'
    if path.exists():assert json.loads(path.read_text())==lock
    else:write(path,lock)
    try:trainer.run()
    except Exception:
        trainer.record(dict(stage='OOF_FAILED_OR_BUDGET_STOPPED_RESUMABLE',traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
