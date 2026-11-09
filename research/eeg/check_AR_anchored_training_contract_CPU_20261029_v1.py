"""Focused pure-CPU production boundary/provenance checks; no GPU or fit."""
import copy,datetime,importlib,json,subprocess,sys,tempfile
from pathlib import Path
import AR_anchored_training_contract_CPU_20261029_v1 as C
import train_AR_anchored_complete_T_GPU_20261029_v1 as P

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'audit/AR_anchored_training_production_20261029_v1'
OUT.mkdir(exist_ok=True)
checks=[]
def passed(name):checks.append(name)
def rejects(call):
    try:call()
    except (AssertionError,ValueError,KeyError,FileNotFoundError):return
    raise AssertionError('Expected rejection')
assert 'torch' not in sys.modules
passed('Pure stdlib producer/contract import creates no CUDA/Torch context')
actual_rows=ROOT/'audit/AR_anchored_images_20261029_v1/fixed_metadata_v1/actual_first20_EEG_VQ_identity_rows.json'
assert C.sha(actual_rows)=='7b5ffa8c8fc54263962e9229c43bea5c2438f09778461cabb154adf75a881fca'
joined=json.loads(actual_rows.read_bytes())
concepts=C.identity_join(joined['EEG_T_first20'],joined['VQ_T_first20'],expected_queries=20,expected_concepts=2)
assert len(concepts)==20 and all('subject' not in q for q in joined['EEG_T_first20'])
passed('Actual canonical20 EEG/VQ rows,2 complete10stim concepts, tuple identity integration passes without invented subject field')
with tempfile.TemporaryDirectory(prefix='AR_contract_CPU_') as folder:
    temp=Path(folder);draft=temp/'draft.json';draft.write_text(json.dumps(dict(status='DRAFT_NOT_ADMITTED_NO_AUTOMATIC_GPU_DISPATCH',output_root=str(temp/'forbidden_output'))))
    for name,args in [('train_AR_anchored_complete_T_GPU_20261029_v1.py',['--arm','native_global','--attempt-index','1','--maximum-updates','1','--job-name','cpu_must_not_run']),('queue_AR_anchored_complete_T_CPU_20261029_v1.py',[])]:
        result=subprocess.run([sys.executable,'-B',str(ROOT/name),'--admission',str(draft)]+args,capture_output=True,text=True,timeout=15)
        assert result.returncode==64 and 'NOT_ADMITTED_' in result.stdout and not (temp/'forbidden_output').exists()
        passed(name+' DRAFT exit64 before output/GPU/dispatch')
    eeg=[dict(array_row=i,stimulus_id=f's{i}',concept_id=f'c{i//2}') for i in range(4)]
    token=[dict(query_index=i,array_row=i,stimulus_id=f's{i}',concept_id=f'c{i//2}',image_sha256='a'*64) for i in range(4)]
    assert C.identity_join(eeg,token,expected_queries=4,expected_concepts=2)==['c0','c0','c1','c1']
    passed('Tuple identity join permits differing nonidentity metadata')
    for label,mutate in [('wrong array row',lambda x:x[0].update(array_row=9)),('wrong stimulus',lambda x:x[0].update(stimulus_id='bad')),('permuted target order',lambda x:x.reverse()),('wrong concept',lambda x:x[0].update(concept_id='c1'))]:
        bad=copy.deepcopy(token);mutate(bad);rejects(lambda:C.identity_join(eeg,bad,expected_queries=4,expected_concepts=2));passed('Identity rejects '+label)
    attempt=temp/'paired_anchored/attempts/attempt_00001';updates=attempt/'updates';updates.mkdir(parents=True)
    for i in range(1,4):
        (updates/f'update_{i:08d}.json').write_text(json.dumps(dict(arm='paired_anchored',spec_sha256='b'*64,update=i,micro_batch_rows=[2]*8,array_positions=list(range(16)))))
    prefix=P.log_prefix(attempt,1,2,arm='paired_anchored',admission_sha='b'*64)
    chain=[dict(start_update=1,end_update=2,attempt_index=1,prefix_sha256=prefix)]
    P.validate_log_chain(temp/'paired_anchored',chain,updates=2,arm='paired_anchored',admission_sha='b'*64)
    passed('Committed log prefix excludes preserved uncommitted update3')
    original=(updates/'update_00000001.json').read_text();(updates/'update_00000001.json').write_text(original+' ')
    rejects(lambda:P.validate_log_chain(temp/'paired_anchored',chain,updates=2,arm='paired_anchored',admission_sha='b'*64))
    passed('Committed raw log byte tampering rejects before resume')
    (updates/'update_00000001.json').write_text(original)
    bad=copy.deepcopy(chain);bad[0]['start_update']=2
    rejects(lambda:P.validate_log_chain(temp/'paired_anchored',bad,updates=2,arm='paired_anchored',admission_sha='b'*64))
    passed('Committed log chain must cover every optimizer update from1')
    # Use actual OLD received metadata, never a synthetic PASS artifact. Only
    # public asset file access is stubbed, clearly not a strict tensor reload.
    actual=ROOT/'audit/EEG_generative_backend_20261029_v1/OLD_pretrained_received_v4'
    profile=json.loads((actual/'resource_v4/completion.json').read_bytes())
    admission=dict(strict_backend_profile=C.ref(actual/'interface_v2/completion.json'),pretrained_micro2_profile=C.ref(actual/'resource_v4/completion.json'),
        initial_generator_tensor_digest=profile['initial_generator_tensor_digest'],GPU_UUID=profile['GPU_UUID'],resources=dict(maximum_resume_payload_bytes=3750000000),
        public_VQ=dict(path='CPU_METADATA_ONLY_VQ_NOT_OPENED',bytes=0,sha256=C.VQ_SHA),public_VAR=dict(path='CPU_METADATA_ONLY_VAR_NOT_OPENED',bytes=0,sha256=C.VAR_SHA))
    real_checked=C.checked
    def metadata_checked(item):
        if str(item['path']).startswith('CPU_METADATA_ONLY_'):return Path(item['path'])
        return real_checked(item)
    C.checked=metadata_checked
    try:
        C.validate_profiles(admission);passed('Actual OLD pretrained/resource/resume metadata schema accepts; public tensor loading not repeated by this CPU test')
        for label,change in [('random VAR',lambda x:x.update(pretrained_VAR=False)),('unmatched RNG',lambda x:x.update(CUDA_RNG_matched=False)),('partial core',lambda x:x.update(actual_core_source_sha256='0'*64)),('failed Adam restoration',lambda x:x['native_resume_serialization_probe'].update(all_Adam_moments_and_steps_exact=False))]:
            bad=copy.deepcopy(profile);change(bad);path=temp/(label.replace(' ','_')+'.json');path.write_text(json.dumps(bad));spec=copy.deepcopy(admission);spec['pretrained_micro2_profile']=C.ref(path)
            rejects(lambda:C.validate_profiles(spec));passed('Profile gate rejects '+label)
    finally:C.checked=real_checked
assert 'torch' not in sys.modules
receipt=dict(status='PASS_CPU_ONLY_AR_PRODUCTION_BOUNDARIES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,checks_passed=len(checks),
    source_pins=[C.ref(ROOT/name) for name in ('AR_anchored_training_contract_CPU_20261029_v1.py','train_AR_anchored_complete_T_GPU_20261029_v1.py','queue_AR_anchored_complete_T_CPU_20261029_v1.py','check_AR_anchored_training_contract_CPU_20261029_v1.py')],
    scope='Pure CPU DRAFT early rejection, actual received profile metadata contracts, identity join and log prefix; no live GPU, full public tensor reload, formal producer or queue execution',new_training_updates=0,new_research_images=0)
(OUT/'CPU_contract_actual_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(status=receipt['status'],checks=len(checks),receipt=C.ref(OUT/'CPU_contract_actual_v1.json'))))
