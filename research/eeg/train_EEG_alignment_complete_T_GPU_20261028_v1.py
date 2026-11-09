"""Fixed sub-01 T encoder campaign, true OOF roles, final epoch only.

This is an explicitly named normalized paper-objective adaptation, not an
unmodified execution of AVDE code. It never reads D/V/official-test targets.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,json,math,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
import torch
from eeg_alignment_training_core_20261028_v1 import optimizer_for,run_epoch,snapshot,restore,publish_epoch,sha,cpu_tree
from eeg_labram_pretrained_20261028_v1 import build_encoder,transfer_clinical

def checked(item):
    p=Path(item['path']);assert sha(p)==item['sha256'],str(p)
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write_once(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024

def run(spec_path,role,segment):
    began=time.monotonic();spec=json.loads(spec_path.read_bytes());spec_hash=sha(spec_path);root=Path(spec['output']);out=root/role;out.mkdir(parents=True,exist_ok=True)
    failure=out/f'failure_segment_{segment:02d}.json';phase='FIXED_INPUTS'
    try:
        assert role in ('full','oof0','oof1') and 1<=segment<=9
        for pin in spec['source_pins']:checked(pin)
        engineering=json.loads(checked(spec['engineering_completion']).read_bytes());assert engineering['status']=='PASS_REAL_T_CLINICAL_FULL_ENCODER_GPU_ENGINEERING'
        inputs=json.loads(checked(spec['input_completion']).read_bytes());assert inputs['T_stimuli']==13230 and inputs['subject']=='sub-01'
        role_ref=inputs['roles'][role];stats=json.loads(checked(role_ref['normalization']).read_bytes())
        eeg=np.load(checked(role_ref['EEG']),allow_pickle=False,mmap_mode='r');targets=np.load(checked(inputs['visual_targets']),allow_pickle=False,mmap_mode='r')
        identities=json.loads(checked(inputs['identities']).read_bytes());folds=np.array(identities['folds']);selected=np.array(stats['training_positions'],dtype=np.int64)
        assert len(selected)==len(set(selected))==role_ref['training_queries']
        expected=np.arange(13230) if role=='full' else np.flatnonzero(folds!=int(role[-1]));assert np.array_equal(selected,expected)
        assert stats['training_array_rows']==[identities['T'][i]['array_row'] for i in selected]
        assert eeg.shape==(13230,63,1,200) and targets.shape==(13230,1024)
        uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==spec['GPU_UUID']
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert shutil.disk_usage(root).free>4<<30
        torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cudnn.benchmark=False
        model=build_encoder(spec['author_source'],spec['vendor'],seed=1729)
        transfer=transfer_clinical(model,checked(spec['clinical_checkpoint']));model=model.cuda()
        from models.labram import get_input_chans
        positions=get_input_chans([x.upper() for x in stats['channel_names']]);optimizer=optimizer_for(model);shuffle=torch.Generator().manual_seed(1729)
        epochs=int(spec['epochs']);batch=int(spec['batch']);assert epochs==50 and batch==128
        steps=math.ceil(len(selected)/batch);total=epochs*steps;warmup=5*steps
        transactions=out/'epochs';transactions.mkdir(exist_ok=True)
        committed=sorted(p for p in transactions.iterdir() if p.is_dir() and '.pending.' not in p.name)
        assert [p.name for p in committed]==[f'epoch_{i:03d}' for i in range(1,len(committed)+1)]
        for i,p in enumerate(committed,1):
            commit=json.loads((p/'commit.json').read_bytes());assert commit['epoch']==i and commit['updates']==i*steps and commit['spec_sha256']==spec_hash and commit['role']==role
            assert sha(p/'epoch.json')==commit['epoch_sha256']
        completed=len(committed);updates=completed*steps
        assert completed==min((segment-1)*6,50) and completed<50
        if committed:
            previous=committed[-1];commit=json.loads((previous/'commit.json').read_bytes());assert sha(previous/'resume.pt')==commit['resume_sha256']
            state=torch.load(previous/'resume.pt',map_location='cpu',weights_only=True)
            completed,updates=restore(state,model,optimizer,shuffle,spec_sha256=spec_hash,role=role,updates_per_epoch=steps,total_epochs=50,device='cuda')
        stop=min(completed+6,50);base_CPU=peak();torch.cuda.reset_peak_memory_stats();phase='COMPLETE_T_EPOCHS'
        def guard():
            assert time.monotonic()-began<820 and torch.cuda.max_memory_allocated()<20<<30 and peak()-base_CPU<10<<30
        for epoch in range(completed+1,stop+1):
            row=run_epoch(model,optimizer,shuffle,eeg,targets,selected,positions,epoch=epoch,updates=updates,total_updates=total,warmup_updates=warmup,batch=batch,device='cuda',guard=guard)
            updates=row['updates'];row.update(subject='sub-01',role=role,expected_updates=total,query_exposures=epoch*len(selected),spec_sha256=spec_hash)
            state=snapshot(model,optimizer,shuffle,epoch=epoch,updates=updates,spec_sha256=spec_hash,role=role,device='cuda')
            commit=publish_epoch(transactions,state,row)
            progress=dict(status='FIXED_COMPLETE_T_ENCODER_TRAINING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),role=role,epoch=epoch,epochs=50,updates=updates,expected_updates=total,queries_per_epoch=len(selected),query_exposures=epoch*len(selected),latest_commit=commit,last_epoch=row)
            temporary=out/f'progress_segment_{segment:02d}_epoch_{epoch:03d}.json';write_once(temporary,progress)
            latest=out/'progress.json';pending=out/f'progress.pending.{os.getpid()}.json';pending.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf8');os.replace(pending,latest)
            assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file())<14<<30 and shutil.disk_usage(root).free>3<<30
            print(json.dumps(row),flush=True);guard()
        if stop==50:
            phase='FIXED_FINAL_PUBLICATION';final=out/'encoder_final.pt';assert not final.exists()
            with final.open('xb') as f:torch.save(dict(model=cpu_tree(model.state_dict()),role=role,subject='sub-01',epoch=50,updates=updates,query_exposures=50*len(selected),spec_sha256=spec_hash,clinical_source_sha256=transfer['source_sha256']),f);f.flush();os.fsync(f.fileno())
            write_once(out/'completion.json',dict(status='COMPLETE_SUB01_FIXED_50_EPOCH_T_ENCODER',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),role=role,subject='sub-01',T_training_queries=len(selected),epochs=50,updates=updates,query_exposures=50*len(selected),heldout_fold=None if role=='full' else int(role[-1]),checkpoint_selection='Fixed final epoch; no D/V/official test score read',objective='.8 normalized symmetric InfoNCE + .2 raw CLIP-H MSE; exp(logit_scale), cap100',author_code_identical_claimed=False,final_model=ref(final),last_epoch_commit=commit,spec_sha256=spec_hash,clinical_transfer=transfer,all10_training_complete=False,new_images=0))
        record=dict(status='COMPLETE_FINITE_ENCODER_SEGMENT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),role=role,segment=segment,epoch=stop,updates=updates,expected_updates=total,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base_CPU,seconds=time.monotonic()-began,spec_sha256=spec_hash)
        write_once(out/f'segment_{segment:02d}_completion.json',record);print(json.dumps(record))
    except BaseException as error:
        if not failure.exists():write_once(failure,dict(status='EEG_ENCODER_SEGMENT_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--role',required=True);p.add_argument('--segment',type=int,required=True);a=p.parse_args();run(a.spec,a.role,a.segment)
