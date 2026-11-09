"""Independent CPU final/checkpoint/order audit and T-only descriptive readout."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,datetime,hashlib,json,math,time,traceback
from pathlib import Path
import numpy as np
import torch

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(item):
    p=Path(item['path']);assert sha(p)==item['sha256']
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main(spec_path):
    start=time.monotonic();spec=json.loads(spec_path.read_bytes());root=Path(spec['output']);out=root.parent/'EEG_alignment_final_CPU_audit_v1';assert not out.exists();out.mkdir();phase='FINAL_EPOCH_AND_ADAM'
    try:
        assert sha(spec_path)=='aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56'
        assert json.loads((root/'queue_completion.json').read_bytes())['status']=='ALL_THREE_SUB01_FULL_T_ROLES_FIXED_FINAL_COMPLETE'
        inputs=json.loads(checked(spec['input_completion']).read_bytes());identities=json.loads(checked(inputs['identities']).read_bytes());proofs={};wrapper_seconds=0.;training_seconds=0.
        torch.set_num_threads(2)
        for role in ('full','oof0','oof1'):
            done=json.loads((root/role/'completion.json').read_bytes());final=torch.load(checked(done['final_model']),map_location='cpu',weights_only=True)
            norm=json.loads(checked(inputs['roles'][role]['normalization']).read_bytes());selected=np.array(norm['training_positions'],dtype=np.int64);folds=np.asarray(identities['folds']);expected=np.arange(13230) if role=='full' else np.flatnonzero(folds!=int(role[-1]));assert np.array_equal(selected,expected)
            steps=math.ceil(len(selected)/128);rng=torch.Generator().manual_seed(1729);rows=[];commits=[]
            for epoch in range(1,51):
                d=root/role/'epochs'/f'epoch_{epoch:03d}';commit=json.loads((d/'commit.json').read_bytes());row=json.loads((d/'epoch.json').read_bytes())
                assert sha(d/'resume.pt')==commit['resume_sha256'] and sha(d/'epoch.json')==commit['epoch_sha256']
                assert commit['role']==role and commit['epoch']==epoch and commit['updates']==epoch*steps and commit['spec_sha256']==done['spec_sha256']==final['spec_sha256']=='aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56'
                order=selected[torch.randperm(len(selected),generator=rng).numpy()]
                assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()==row['order_sha256']
                assert row['queries']==len(selected) and row['query_exposures']==epoch*len(selected) and row['updates']==epoch*steps
                assert all(math.isfinite(row[k]) for k in ('loss','NCE','MSE','seconds'))
                rows.append(row);commits.append(ref(d/'commit.json'));training_seconds+=row['seconds']
            resume=torch.load(root/role/'epochs/epoch_050/resume.pt',map_location='cpu',weights_only=True)
            assert len(final['model'])==len(resume['model'])==224 and sum(x.numel() for x in final['model'].values())==6025761
            for name,value in final['model'].items():assert torch.isfinite(value).all() and torch.equal(value,resume['model'][name]),name
            optimizer=resume['optimizer'];names=[name for group in optimizer['param_groups'] for name in group['parameter_names']];assert set(names)==set(final['model']) and len(names)==len(set(names))==224
            assert len(optimizer['state'])==224 and torch.equal(rng.get_state(),resume['shuffle_rng']) and len(resume['cuda_rng'])==1
            for group in optimizer['param_groups']:
                assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay'] in (0.,.05)
                for identifier,name in zip(group['params'],group['parameter_names']):
                    state=optimizer['state'][identifier];assert float(state['step'])==50*steps
                    for moment in ('exp_avg','exp_avg_sq'):assert state[moment].shape==final['model'][name].shape and torch.isfinite(state[moment]).all()
            for segment in range(1,10):
                receipt=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1/audit')/f'EEG_multiscale_alignment_{role}_s{segment:02d}_20261028_v1_job.json'
                item=json.loads(receipt.read_bytes());assert item['status']=='COMPLETED' and item['exit_code']==0 and item['max_seconds']==900;wrapper_seconds+=item['charged_seconds']
            proofs[role]=dict(final_model=done['final_model'],epochs=50,updates=50*steps,queries=len(selected),query_exposures=50*len(selected),first_epoch=rows[0],final_epoch=rows[-1],all_epoch_commits=commits,all_224_final_tensors_equal_last_resume=True,all_224_Adam_states_and_RNG_verified=True)
            del resume,final
        phase='T_ONLY_GENUINE_OOF_DESCRIPTIVE_ALIGNMENT'
        memory_root=root.parent/'EEG_memory_old_v1';memory=json.loads((memory_root/'completion.json').read_bytes());assert memory['all_T_rows_exactly_once_from_concept_excluded_encoder']
        c=np.load(checked(memory['files']['T_OOF_c']),allow_pickle=False).astype(np.float64);target=np.load(checked(inputs['visual_targets']),allow_pickle=False).astype(np.float64)
        assert c.shape==target.shape==(13230,1024) and np.isfinite(c).all() and np.isfinite(target).all()
        cn=c/np.maximum(np.linalg.norm(c,axis=1,keepdims=True),1e-12);tn=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12)
        groups={}
        for i,q in enumerate(identities['T']):groups.setdefault(q['concept_id'],[]).append(i)
        assert len(groups)==1323 and all(len(ids)==10 for ids in groups.values())
        top1=[];afc=[];cos=[]
        for ids in groups.values():
            scores=cn[ids]@tn[ids].T;diagonal=np.diag(scores);top1.append(np.mean(np.argmax(scores,axis=1)==np.arange(10)));afc.append(np.mean(((diagonal[:,None]>scores).sum(1))/9));cos.append(np.mean(diagonal))
        rng_np=np.random.default_rng(1729);values=np.array([top1,afc,cos]).T;sample=rng_np.integers(0,1323,size=(1000,1323));boot=values[sample].mean(1)
        readout=dict(status='DESCRIPTIVE_COMPLETE_T_GENUINE_OOF_ALIGNMENT_NOT_IMAGE_EFFECT',queries=13230,concepts=1323,CLIP_H_target_cosine=float(np.mean(cos)),raw_CLIP_H_MSE=float(np.mean((c-target)**2)),same_concept_Top1=float(np.mean(top1)),same_concept_2AFC=float(np.mean(afc)),descriptive_concept_bootstrap_95=dict(zip(['Top1','2AFC','target_cosine'],np.quantile(boot,[.025,.975],axis=0).T.tolist())),condition_mean_energy_fraction=float(np.sum(c.mean(0)**2)/np.mean(np.sum(c*c,axis=1))),checkpoint_selection='No score-based selection; all fixed epoch50 finals',comparison_to_other_architecture_or_image_benefit_claimed=False,Dcheck_scores_read=False,limitations='Shared trained fold models; concept bootstrap does not cover training seed or model uncertainty; target CLIP-H differs from HIVE fused visual coordinates.')
        write(out/'T_OOF_alignment_descriptive.json',readout)
        value=dict(status='PASS_ALL150_EPOCH_TRANSACTIONS_COMPLETE_COVERAGE_FINAL_ADAM_RNG_AND_T_ONLY_READOUT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),roles=proofs,training_loop_seconds=training_seconds,all27_GPU_wrapper_seconds=wrapper_seconds,queue_seconds=json.loads((root/'queue_completion.json').read_bytes())['seconds'],T_OOF_readout=ref(out/'T_OOF_alignment_descriptive.json'),seconds=time.monotonic()-start,GPU_jobs=0,new_images=0,all10_formal_training_complete=False)
        assert not torch.cuda.is_initialized();write(out/'completion.json',value);print(json.dumps(dict(completion=ref(out/'completion.json'),training_loop_seconds=training_seconds,all27_GPU_wrapper_seconds=wrapper_seconds,readout=readout)))
    except BaseException as error:
        write(out/'failure.json',dict(status='FINAL_EEG_CPU_AUDIT_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);main(p.parse_args().spec)
