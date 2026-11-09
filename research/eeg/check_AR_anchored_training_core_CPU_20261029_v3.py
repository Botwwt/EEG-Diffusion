"""CPU fixtures for matched RNG, donor permissions and partial-epoch resume."""
import datetime,hashlib,json,sys,tempfile,time,types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from unittest.mock import patch

L=Path(__file__).resolve().parent;S=L/'reproduction_sources/direction_switch_20261028_v1/avde_3be06a90ae69'
package=types.ModuleType('models');package.__path__=[str(S/'models')];assert 'models' not in sys.modules;sys.modules['models']=package
sys.path.insert(0,str(S));import dist;dist.__dict__['__device']='cpu'
from models.var_eeg import VAREEG
from models.vqvae import VQVAE
from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
import AR_anchored_training_core_DRAFT_20261029_v2 as prior_v2
from AR_anchored_training_core_DRAFT_20261029_v3 import *

torch.set_num_threads(2);start=time.monotonic();checks={};fixture=np.random.default_rng(1729)
concepts=['concept_'+str(i//3) for i in range(9)]
global_bank=fixture.normal(size=(9,1024)).astype(np.float32)
local=fixture.normal(size=(9,5,1024)).astype(np.float32);local-=local.mean(1,keepdims=True)
memory_bank=global_bank[:,None]+local
targets=fixture.integers(0,32,size=(9,14),dtype=np.int64)
def fresh():
    torch.manual_seed(1729)
    vq=VQVAE(vocab_size=32,z_channels=8,ch=32,test_mode=True,v_patch_nums=(1,2,3))
    decoder=VAREEG(vq,depth=2,embed_dim=64,num_heads=4,cond_drop_rate=.1,drop_path_rate=.1,
        patch_nums=(1,2,3),flash_if_available=False,fused_if_available=False)
    decoder.init_weights(init_adaln_gamma=.1)
    model=EEGAnchoredAttentionDecoder(decoder)
    return model,optimizer_for(model),torch.Generator().manual_seed(1729),new_cursor()
def run(model,opt,shuffle,cursor,arm,limit):
    return run_updates(model,opt,shuffle,cursor,arm=arm,global_bank=global_bank,memory_bank=memory_bank,
        target_bank=targets,concept_ids=concepts,total_epochs=2,batch=4,micro_batch=2,maximum_updates=limit,device='cpu')
donors=donor_derangement(concepts,1)
checks['donor_is_same_concept_different_stimulus_bijection']=bool((donors!=np.arange(9)).all() and np.array_equal(np.sort(donors),np.arange(9)) and all(concepts[i]==concepts[j] for i,j in enumerate(donors)))
before=torch.get_rng_state().clone();repeat=donor_derangement(concepts,1)
checks['donor_map_deterministic_and_consumes_no_training_RNG']=np.array_equal(donors,repeat) and torch.equal(before,torch.get_rng_state())
own=torch.tensor(global_bank);source=torch.tensor(global_bank[donors]);bank=torch.tensor(memory_bank[donors])
transferred=transfer_local_deviations(own,source,bank)
checks['donor_keeps_own_global_condition']=torch.equal(own,torch.tensor(global_bank))
checks['donor_changes_only_source_local_deviation']=torch.equal(transferred,own[:,None]+(bank-source[:,None]))
checks['donor_memory_mean_matches_own_c']=torch.allclose(transferred.mean(1),own,atol=5e-7,rtol=1e-5)

rng_states=[];orders=[]
for arm in ARMS:
    model,opt,shuffle,cursor=fresh();events=run(model,opt,shuffle,cursor,arm,1)
    rng_states.append(torch.get_rng_state().clone());orders.append(cursor['order'].clone())
    assert events[0]['array_positions']==cursor['order'][:4].tolist()
checks['all_three_arms_share_training_RNG_draw_count']=all(torch.equal(rng_states[0],x) for x in rng_states[1:])
checks['all_three_arms_share_query_order']=all(torch.equal(orders[0],x) for x in orders[1:])

model,opt,shuffle,cursor=fresh();continuous_events=run(model,opt,shuffle,cursor,'donor_anchored',6)
expected=snapshot(model,opt,shuffle,cursor,arm='donor_anchored',spec_sha256='CPU_FIXTURE_ONLY',device='cpu')
model,opt,shuffle,cursor=fresh();first=run(model,opt,shuffle,cursor,'donor_anchored',2)
state=snapshot(model,opt,shuffle,cursor,arm='donor_anchored',spec_sha256='CPU_FIXTURE_ONLY',device='cpu')
checks['snapshot_inside_epoch_before_short_last_batch']=state['cursor']['position']==8 and state['cursor']['updates']==2 and state['cursor']['completed_epochs']==0
fixture_parent=L/'audit/AR_anchored_training_DRAFT_20261029_v3';fixture_parent.mkdir(parents=True,exist_ok=True)
transaction_root=Path(tempfile.mkdtemp(prefix='CPU_checkpoint_fixture_v1_',dir=fixture_parent))
def injected_failure(point):raise RuntimeError('injected before checkpoint directory commit')
try:publish_step_checkpoint(transaction_root,state,fault=injected_failure)
except RuntimeError:pass
checks['interrupted_checkpoint_not_published']=not (transaction_root/'update_00000002').exists() and bool(list(transaction_root.glob('*.pending.*')))
commit=publish_step_checkpoint(transaction_root,state);state=read_step_checkpoint(commit)
checks['tensor_only_checkpoint_roundtrip_and_SHA_binding']=state['cursor']['position']==8 and state['cursor']['updates']==2
model,opt,shuffle,discard=fresh();cursor=restore(state,model,opt,shuffle,arm='donor_anchored',spec_sha256='CPU_FIXTURE_ONLY',count=9,batch=4,total_epochs=2,device='cpu')
remaining=run(model,opt,shuffle,cursor,'donor_anchored',4)
actual=snapshot(model,opt,shuffle,cursor,arm='donor_anchored',spec_sha256='CPU_FIXTURE_ONLY',device='cpu')
checks['mid_epoch_resume_same_query_donor_loss_events']=continuous_events==first+remaining
checks['mid_epoch_resume_exact_model']=all(torch.equal(v,actual['model'][key]) for key,v in expected['model'].items())
checks['mid_epoch_resume_exact_Adam_moments']=all(torch.equal(value,actual['optimizer']['state'][key][field]) for key,item in expected['optimizer']['state'].items() for field,value in item.items())
checks['mid_epoch_resume_exact_CPU_and_shuffle_RNG']=torch.equal(expected['torch_rng'],actual['torch_rng']) and torch.equal(expected['shuffle_rng'],actual['shuffle_rng'])
checks['two_full_epochs_all_rows_including_short_batches']=cursor['completed_epochs']==2 and cursor['query_exposures']==18 and cursor['updates']==6 and [len(x['array_positions']) for x in continuous_events]==[4,4,1,4,4,1]
checks['all_VQ_parameters_frozen_no_gradients']=all(not p.requires_grad and p.grad is None for p in model.decoder.vae_proxy[0].parameters())
checks['optimizer_contains_only_original_decoder_parameters']=len(opt.param_groups)==1 and {id(p) for p in opt.param_groups[0]['params']}=={id(p) for p in model.decoder.parameters()}
rolling_model,rolling_opt,rolling_rng,rolling_cursor=fresh();states=[]
for i in range(3):
    run(rolling_model,rolling_opt,rolling_rng,rolling_cursor,'paired_anchored',1)
    states.append(snapshot(rolling_model,rolling_opt,rolling_rng,rolling_cursor,arm='paired_anchored',spec_sha256='CPU_ROLLING_FIXTURE',device='cpu'))
rolling=Path(tempfile.mkdtemp(prefix='CPU_bounded_resume_fixture_',dir=fixture_parent))
create_rotating_store(rolling,arm='paired_anchored',spec_sha256='CPU_ROLLING_FIXTURE',authorize_owned_previous_retirement=True)
for saved in states:manifest=publish_rotating_checkpoint(rolling,saved)
latest,manifest=read_rotating_current(rolling)
checks['two_payload_checkpoint_bound_after_three_commits']=len(list(rolling.glob('slot_*/state.pt')))==2 and latest['cursor']['updates']==3 and manifest['previous']['updates']==2
checks['retired_transient_checkpoint_hash_metadata_kept']=(rolling/'events/retired_00000003.json').is_file()
interrupted=Path(tempfile.mkdtemp(prefix='CPU_bounded_resume_failure_fixture_',dir=fixture_parent))
create_rotating_store(interrupted,arm='paired_anchored',spec_sha256='CPU_ROLLING_FIXTURE',authorize_owned_previous_retirement=True)
publish_rotating_checkpoint(interrupted,states[0]);publish_rotating_checkpoint(interrupted,states[1])
def rolling_failure(point):
    if point=='new_payload_complete_before_manifest_commit':raise RuntimeError('injected rotating checkpoint commit interruption')
try:publish_rotating_checkpoint(interrupted,states[2],fault=rolling_failure)
except RuntimeError:pass
latest,incomplete=read_rotating_current(interrupted)
checks['interrupted_rotation_keeps_verified_current_and_two_payload_peak']=latest['cursor']['updates']==2 and incomplete['status']=='PREPARING_CHECKPOINT' and len(list(interrupted.glob('slot_*/state.pt')))==2
try:publish_rotating_checkpoint(interrupted,states[2])
except AssertionError:checks['interrupted_rotation_cannot_silently_retry_or_cleanup']=True
else:checks['interrupted_rotation_cannot_silently_retry_or_cleanup']=False
unknown=Path(tempfile.mkdtemp(prefix='CPU_bounded_resume_unknown_fixture_',dir=fixture_parent))
create_rotating_store(unknown,arm='paired_anchored',spec_sha256='CPU_ROLLING_FIXTURE',authorize_owned_previous_retirement=True)
publish_rotating_checkpoint(unknown,states[0]);manifest=publish_rotating_checkpoint(unknown,states[1])
sentinel=unknown/manifest['previous']['slot']/'unowned_sentinel.txt';sentinel.write_text('must preserve',encoding='utf8')
try:publish_rotating_checkpoint(unknown,states[2])
except AssertionError:checks['unrecognized_owned_slot_file_stops_before_retirement']=sentinel.read_text(encoding='utf8')=='must preserve'
else:checks['unrecognized_owned_slot_file_stops_before_retirement']=False
event_root=Path(tempfile.mkdtemp(prefix='CPU_immutable_update_log_fixture_',dir=fixture_parent))
for event in continuous_events:write_update_event(event_root,event,arm='donor_anchored',spec_sha256='CPU_FIXTURE_ONLY')
checks['full_immutable_update_log_keeps_own_and_donor_rows']=len(list(event_root.glob('update_*.json')))==6 and json.loads((event_root/'update_00000001.json').read_bytes())['array_positions']==continuous_events[0]['array_positions']

# Production-shape batch COUNTS, using the real author's reduced decoder.
# Three logical batches are16,16,14; exactly8,8,7 micro-forwards and3Adamsteps.
logical_count=46;logical_concepts=['logical_'+str(i//10) for i in range(logical_count)]
logical_global=fixture.normal(size=(logical_count,1024)).astype(np.float32)
logical_local=fixture.normal(size=(logical_count,5,1024)).astype(np.float32);logical_local-=logical_local.mean(1,keepdims=True)
logical_memory=logical_global[:,None]+logical_local
logical_targets=fixture.integers(0,32,size=(logical_count,14),dtype=np.int64)
logical_model,logical_opt,logical_rng,logical_cursor=fresh();step_calls=[];old_step=logical_opt.step
def count_step(*args,**kwargs):step_calls.append(1);return old_step(*args,**kwargs)
logical_opt.step=count_step
with patch('torch.nn.utils.clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clip_spy:
    logical_events=run_updates(logical_model,logical_opt,logical_rng,logical_cursor,arm='donor_anchored',
        global_bank=logical_global,memory_bank=logical_memory,target_bank=logical_targets,concept_ids=logical_concepts,
        total_epochs=1,batch=16,micro_batch=2,maximum_updates=3,device='cpu')
checks['logical16_16_14_use8_8_7_microbatches']=[x['micro_batch_rows'] for x in logical_events]==[[2]*8,[2]*8,[2]*7]
checks['exactly_one_clip_and_Adam_step_per_logical_batch']=len(step_calls)==clip_spy.call_count==3 and all(float(s['step'])==3 for s in logical_opt.state.values())
checks['short14_complete_weight_and_coverage']=logical_cursor['completed_epochs']==1 and logical_cursor['query_exposures']==46 and sorted(sum([x['array_positions'] for x in logical_events],[]))==list(range(46))

gradient_agreement=[]
def raw_loss(model,ids,arm):
    c=torch.tensor(logical_global[ids]);target=torch.tensor(logical_targets[ids],dtype=torch.long)
    memory=None if arm=='native_global' else torch.tensor(logical_memory[ids])
    with torch.no_grad():previous=model.decoder.vae_proxy[0].quantize.idxBl_to_var_input(list(torch.split(target,[1,4,9],dim=1)))
    return F.cross_entropy(model(previous,c,eeg_memory=memory).reshape(-1,32),target.reshape(-1))
for arm in ('native_global','paired_anchored'):
    for rows in (16,14):
        full,_,_,_=fresh();full.eval();full.decoder.cond_drop_rate=0.
        full_loss=raw_loss(full,np.arange(rows),arm);full_loss.backward()
        full_grad=torch.cat([p.grad.reshape(-1) for p in full.parameters()])
        accumulated,_,_,_=fresh();accumulated.eval();accumulated.decoder.cond_drop_rate=0.;weighted_mean=0.
        for lo in range(0,rows,2):
            ids=np.arange(lo,min(lo+2,rows));part=raw_loss(accumulated,ids,arm)
            (part*(len(ids)/rows)).backward();weighted_mean+=float(part.detach())*(len(ids)/rows)
        accumulated_grad=torch.cat([p.grad.reshape(-1) for p in accumulated.parameters()])
        absolute_error=float((full_grad-accumulated_grad).abs().max());relative_error=float((full_grad-accumulated_grad).norm()/full_grad.norm())
        loss_error=abs(float(full_loss.detach())-weighted_mean)
        checks[f'zero_dropout_full_vs_micro2_loss_gradient_{arm}_{rows}']=loss_error<2e-6 and absolute_error<5e-6 and relative_error<1e-4
        gradient_agreement.append(dict(arm=arm,logical_rows=rows,loss_abs_error=loss_error,gradient_max_abs_error=absolute_error,gradient_relative_L2_error=relative_error))

partial_model,partial_opt,partial_rng,partial_cursor=fresh();guard_calls=[]
def interrupt_after_first_micro():
    guard_calls.append(1)
    if len(guard_calls)==3:raise TimeoutError('injected after partial backward, before logical optimizerstep')
try:
    run_updates(partial_model,partial_opt,partial_rng,partial_cursor,arm='paired_anchored',global_bank=global_bank,memory_bank=memory_bank,
        target_bank=targets,concept_ids=concepts,total_epochs=2,batch=4,micro_batch=2,maximum_updates=1,device='cpu',guard=interrupt_after_first_micro)
except TimeoutError:pass
checks['interrupted_microbatch_has_no_partial_Adam_or_cursor_step']=partial_cursor['logical_update_in_progress'] is True and partial_cursor['updates']==0 and not partial_opt.state
try:snapshot(partial_model,partial_opt,partial_rng,partial_cursor,arm='paired_anchored',spec_sha256='partial_fixture',device='cpu')
except AssertionError:checks['checkpoint_mid_accumulation_explicitly_rejected']=True
else:checks['checkpoint_mid_accumulation_explicitly_rejected']=False
try:restore(state,model,opt,shuffle,arm='donor_anchored',spec_sha256='CPU_FIXTURE_ONLY',count=9,batch=4,total_epochs=2,device='cpu',micro_batch=1)
except AssertionError:checks['restoring_different_microbatch_recipe_rejected']=True
else:checks['restoring_different_microbatch_recipe_rejected']=False
v2_model,v2_opt,v2_rng,v2_cursor=fresh()
v2_events=prior_v2.run_updates(v2_model,v2_opt,v2_rng,v2_cursor,arm='native_global',global_bank=global_bank,memory_bank=memory_bank,
    target_bank=targets,concept_ids=concepts,total_epochs=2,batch=4,micro_batch=2,maximum_updates=6,device='cpu')
v2_state=prior_v2.snapshot(v2_model,v2_opt,v2_rng,v2_cursor,arm='native_global',spec_sha256='API_EQUIVALENCE_FIXTURE',device='cpu')
none_model,none_opt,none_rng,none_cursor=fresh()
none_events=run_updates(none_model,none_opt,none_rng,none_cursor,arm='native_global',global_bank=global_bank,memory_bank=None,
    target_bank=targets,concept_ids=concepts,total_epochs=2,batch=4,micro_batch=2,maximum_updates=6,device='cpu')
none_state=snapshot(none_model,none_opt,none_rng,none_cursor,arm='native_global',spec_sha256='API_EQUIVALENCE_FIXTURE',device='cpu')
checks['native_None_v3_same_query_loss_events_as_v2_realbank']=none_events==v2_events
checks['native_None_v3_exact_model_Adam_as_v2_realbank']=all(torch.equal(v,none_state['model'][key]) for key,v in v2_state['model'].items()) and all(torch.equal(v,none_state['optimizer']['state'][key][field]) for key,item in v2_state['optimizer']['state'].items() for field,v in item.items())
checks['native_None_v3_exact_RNG_as_v2_realbank']=torch.equal(v2_state['torch_rng'],none_state['torch_rng']) and torch.equal(v2_state['shuffle_rng'],none_state['shuffle_rng'])
rejections=[]
for arm in ('paired_anchored','donor_anchored'):
    forbidden_model,forbidden_opt,forbidden_rng,forbidden_cursor=fresh()
    try:run_updates(forbidden_model,forbidden_opt,forbidden_rng,forbidden_cursor,arm=arm,global_bank=global_bank,memory_bank=None,
        target_bank=targets,concept_ids=concepts,total_epochs=2,batch=4,micro_batch=2,maximum_updates=1,device='cpu')
    except AssertionError:rejections.append(not forbidden_opt.state and forbidden_cursor['updates']==0)
    else:rejections.append(False)
checks['None_memory_rejected_before_any_local_arm_update']=all(rejections)
record=dict(status='PASS' if all(checks.values()) else 'FAIL',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,seconds=time.monotonic()-start,
    actual_author_reduced_VAR_VQ=True,random_model_only=True,real_EEG_or_images=False,GPU_jobs=0,formal_models=0,
    mathematical_training_budget_verified_fixture=dict(queries=9,epochs=2,batch=4,updates=6,query_exposures=18),
    logical_microbatch_contract_fixture=dict(queries=46,logical_batches=[16,16,14],micro_batches_per_step=[8,8,7],optimizer_steps=3),
    zero_dropout_full_micro_gradient_agreement=gradient_agreement,
    source_sha256=hashlib.sha256((L/'AR_anchored_training_core_DRAFT_20261029_v3.py').read_bytes()).hexdigest())
out=L/'audit/AR_anchored_training_DRAFT_20261029_v3';out.mkdir(exist_ok=True);path=out/'CPU_contract_actual_v1.json'
if path.exists():
    old=path.read_bytes();(out/('prior_'+hashlib.sha256(old).hexdigest()[:12]+'.json')).write_bytes(old)
path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(record));assert all(checks.values()),checks
