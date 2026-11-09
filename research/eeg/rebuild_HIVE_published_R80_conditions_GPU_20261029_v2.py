"""Re-export fixed published parents from original physicalR80 EEG; no fitting."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path


def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))


def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')


def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024


def run(spec_path,job):
 spec=json.loads(Path(spec_path).read_bytes());assert spec['status']=='REGISTERED_OLD_ONLY_FIXED_HIVE_R80_INPUT_REPLAY_v1'
 assert spec['host']=='old' and spec['subjects']==[f'sub-{i:02d}' for i in range(1,11)]
 for r in spec['source_pins']:assert ref(r['path'])==r
 assert 'run_gpu_job_v2.py' in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().decode()
 assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==spec['GPU_UUID']
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 root=Path(spec['output_root']);root.mkdir();started=time.monotonic();base=peak();phase='PINNED_SOURCE_MODEL_AND_INPUTS'
 try:
  W=Path(spec['W']);R=Path(spec['repository_root']);sys.path.insert(0,str(R))
  import brainhive_native_models_v1 as C
  import torch
  import numpy as np
  torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  ordinary=W/'components/bounded_v1/ordinary/component_final.pt';assert sha(ordinary)==spec['ordinary_sha256']
  bindings=json.loads((W/'data/actual_input_bindings.json').read_bytes());inputs=json.loads((W/'data/generation_inputs/benchmark.json').read_bytes())
  assert inputs['expected_queries']==2000 and inputs['repeat_protocol']=='R80' and not inputs['target_image_or_class_fields_present'] and inputs['subjects']==spec['subjects']
  assert len(inputs['inputs'])==200 and [r['array_row'] for r in inputs['inputs']]==list(range(200))
  indices=bindings['selected_indices'];assert len(indices)==len(set(indices))==17
  manifest=W/'images/native_R80_correct_v1/inference_manifest.jsonl';old=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()];old_by={r['query_id']:r for r in old};assert len(old)==len(old_by)==2000
  scope=C.classes();reports=[];arrays=[];gammas=[];betas=[]
  def guard():assert time.monotonic()-started<800 and peak()-base<10<<30 and torch.cuda.max_memory_allocated()<20<<30 and shutil.disk_usage(root).free>2<<30
  for subject in spec['subjects']:
   guard();phase='EXACT_FROZEN_EEG_FORWARD_'+subject
   model_path=W/'runs/published_source_dropout_v2'/subject/'brain_final.pt';model_sha=sha(model_path);registration=json.loads((model_path.parent/'training_complete.json').read_bytes());assert model_sha==registration['model_sha256']
   raw_ref=bindings['subjects'][subject]['test_R80'];raw_path=Path(raw_ref['path']);assert sha(raw_path)==raw_ref['sha256']
   state=torch.load(model_path,weights_only=True,map_location='cpu');assert state['epochs']==25 and state['seed']==1729
   model=scope['BrainMLP'](4250,1024,num_layers=1,dropout=.1).eval().requires_grad_(False);model.load_state_dict(state['model'],strict=True)
   raw=torch.load(raw_path,weights_only=True,mmap=True);assert raw.shape==(200,63,250) and raw.dtype==torch.float32 and torch.isfinite(raw).all()
   eeg=raw[:,indices,:].clone();model.cuda()
   # Same200-row batch as the old infer(batch256) publication export.
   with torch.inference_mode():condition=model(eeg.cuda()).float().cpu()
   assert condition.shape==(200,1024) and torch.isfinite(condition).all()
   model.cpu();torch.cuda.empty_cache()
   with torch.inference_mode():cpu=model(eeg).float()
   relative=float(torch.linalg.vector_norm(condition-cpu)/torch.linalg.vector_norm(condition));absolute=float(torch.max(torch.abs(condition-cpu)));assert relative<1e-5 and absolute<2e-5
   matched=0;records=[]
   for i,q in enumerate(inputs['inputs']):
    key=f"{subject}/{q['stimulus_id']}/R80";r=old_by[key];assert r['array_row']==i and r['EEG_source_stimulus_id']==q['stimulus_id'] and r['intervention']=='correct'
    h=hashlib.sha256(condition[i:i+1].contiguous().numpy().tobytes()).hexdigest();matched+=int(h==r['condition_sha256']);records.append(dict(query_id=key,stimulus_id=q['stimulus_id'],array_row=i,condition_sha256=h,old_condition_sha256=r['condition_sha256'],image_seed=r['image_seed']))
   directory=root/subject;directory.mkdir();payload=directory/'conditions.pt'
   torch.save(dict(condition=condition,array_rows=torch.arange(200),model_sha256=model_sha,role='published',normalization='raw LayerNorm output'),payload)
   reloaded=torch.load(payload,weights_only=True,map_location='cpu');assert torch.equal(reloaded['condition'],condition)
   write(directory/'rows.json',records)
   arrays.append(condition.numpy());gammas.append(state['model']['block.layers.0.norm.weight'].numpy());betas.append(state['model']['block.layers.0.norm.bias'].numpy())
   assert sha(model_path)==model_sha and sha(ordinary)==spec['ordinary_sha256']
   reports.append(dict(subject=subject,raw_EEG=ref(raw_path),model=ref(model_path),conditions=ref(payload),rows=ref(directory/'rows.json'),old_manifest_raw_condition_SHA_matches=matched,queries=200,CPU_vs_GPU_relative_error=relative,CPU_vs_GPU_max_abs_error=absolute))
   del raw,eeg,model,state,cpu,condition,reloaded;torch.cuda.empty_cache()
  assert sum(r['queries'] for r in reports)==2000
  bank=root/'all10_published_R80_conditions_and_affines.npz';np.savez_compressed(bank,conditions=np.stack(arrays),gammas=np.stack(gammas),betas=np.stack(betas));guard()
  exact=sum(r['old_manifest_raw_condition_SHA_matches'] for r in reports)
  write(root/'completion.json',dict(status='COMPLETE_ORIGINAL10_FROZEN_HIVE_R80_CONDITIONS_REPLAY_AND_CPU_CHECK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),subjects=10,queries=2000,repeat_protocol='physicalR80',reports=reports,bank=ref(bank),
   old_manifest=ref(manifest),old_manifest_raw_condition_SHA_matches=exact,all_old_raw_conditions_bit_exact=exact==2000,
   if_not_bit_exact='Do not reuse old parent images as an exact paired baseline; first investigate numerical context and generate a current paired parent baseline if needed. Do not modify values to force hashes.',
   old_dangling_links_left_untouched=True,models_retrained=0,updates=0,new_images=0,test_labels_or_visual_targets_read=False,ordinary_unchanged=True,seconds=time.monotonic()-started,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak()-base))
 except BaseException as error:
  write(root/'failure.json',dict(status='R80_INPUT_REPLAY_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--job-name',required=True);q=p.parse_args();run(q.spec,q.job_name)
