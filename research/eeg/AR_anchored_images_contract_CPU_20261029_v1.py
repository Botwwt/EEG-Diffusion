"""AR image provenance/admission contracts. Importing this module opens no GPU."""
import hashlib,json,math
from pathlib import Path

ARMS=('native_global','paired_anchored','donor_anchored')
HOST_UUID={'old':'GPU-0a99d971-a46c-6e81-f157-c33243c6d57a','new':'GPU-62a1aee8-4740-dc59-93f4-d50644db650d'}
POLICY_SHA='732c5de27b84b11ff6f65884456c0124abe55c26efae90c0cd4d5195fa19fce1'
PANEL_SHA='068fc441f1332f1d2b08e79a93a93cacdd498efc35b82010080e7f845e76ae1e'
EEG_SHA='c11bd68227feffe5efaebeac309e99090c5ab08cc8c44b87b7abe3882d4c2607'
TOKEN_SHA='e88a6ee59d4dba2d675d4760ea4e24a25214ecd3daa42624e6043437fd3b3205'
INITIAL='4917e94bd7b8f4d7f4fea0e294d76728c0c29b0f3eea0e75fb9222714764ca39'
LOADER_SHA='3f2abeca5dc1592b675991ae586f564382bf6a69c1c92dc917418d44408064db'
ANCHOR_SHA='7768c21b3dbd5cff03a7cb0b333c8531127f2d61a6bbcaa860de3ae28ce75fb0'
PREFIX_SHA='6af999d30a15e94121ba6468c07012388020e5a42eb368a2318b4504b5dbaa69'
MATH_SHA='c4f07adb170e8f4b343c7e82819f4b914aa5f7806fb9b3a72928ab0d949f19e9'
METRIC_SHA='bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'
DINO_SHA='08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def ref(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(item):
    assert isinstance(item,dict) and {'path','bytes','sha256'}<=set(item)
    p=Path(item['path']);assert p.is_absolute() and p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    return p
def write_once(path,value):
    path=Path(path)
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def seed_for(subject,stimulus):
    return int.from_bytes(hashlib.sha256(f'1729|EEG-anchored-AR|{subject}|{stimulus}|R4'.encode()).digest()[:8],'little')%(2**63-1)

def load_admission(path):
    p=Path(path);a=json.loads(p.read_bytes())
    if a.get('status')!='AR_ANCHORED_IMAGES_ADMITTED_v1':
        raise ValueError('NOT_ADMITTED_AR_IMAGE_DRAFT; no CUDA, images, features or scores opened')
    assert a['host_role'] in HOST_UUID and a['GPU_UUID']==HOST_UUID[a['host_role']]
    assert a['early_generation_gate']=='OWN_FIXED_FINAL_AND_SHARED_FIXED_THREE_ARM_TRAINING_RECIPE'
    assert a['early_generation_limit']==80 and a['inspection_and_effect_gate']=='ALL_THREE_FIXED_FITS_AND_ALL_THREE_STAGE_ARTIFACTS_AND_FEATURES'
    assert a['expansion_gate']=='LOCKED_ALL_THREE_80_DECISION_EXPAND400'
    assert a['policy']['sha256']==POLICY_SHA and a['panel']['sha256']==PANEL_SHA
    resources=a['resources'];assert resources['GPU_job_seconds']==900 and resources['body_hard_seconds']<=840
    assert resources['maximum_GPU_allocated_bytes']==20<<30 and resources['maximum_extra_CPU_bytes']==10<<30
    assert resources['maximum_queries_per_chunk']==20 and resources['unreserved_disk_buffer_bytes']>=2<<30
    assert resources['maximum_CPU_audit_seconds']<=300 and resources['maximum_CPU_score_seconds']<=300
    root=Path(a['output_root']);assert root.is_absolute() and root.name=='AR_anchored_images_v1' and root.resolve()==root and not root.is_symlink()
    assert set(a['training_admissions'])==set(a['fit_completion_paths'])==set(a['model_paths'])==set(ARMS)
    assert a['generation_cost']['native_scale_steps']==10 and a['generation_cost']['CFG_branches']==2 and a['generation_cost']['image_tokens_per_query']==680
    assert a['generation_cost']['UNet_forwards']==0 and a['generation_cost']['equivalent_to_HIVE_4_NFE'] is False
    assignment=a['generation_assignment'];assert set(assignment)=={'old','new'}
    assigned=assignment['old']+assignment['new'];assert len(assigned)==len(set(assigned))==3 and set(assigned)==set(ARMS)
    assert a['assigned_arms']==assignment[a['host_role']]
    predecessor=a['GPU_predecessor_controller'];assert type(predecessor['pid']) is int and predecessor['pid']>0 and type(predecessor['start_ticks']) is int and predecessor['start_ticks']>0
    return a,sha(p)

def validate_sources(a,own=None):
    names={}
    for item in a['source_pins']:
        path=checked(item);assert path.name not in names;names[path.name]=item
    expected={'EEG_generative_backend_loader_20261029_v1.py':LOADER_SHA,'EEG_anchored_attention_20261029_v1.py':ANCHOR_SHA,
              'eeg_multiscale_prefix_20261028_v1.py':PREFIX_SHA,'AR_anchored_image_screen_math_20261029_v1.py':MATH_SHA,
              'score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2.py':METRIC_SHA}
    for name,digest in expected.items():assert names[name]['sha256']==digest
    for path in [Path(__file__),*([] if own is None else [Path(own)])]:
        assert Path(names[path.name]['path']).resolve()==path.resolve() and names[path.name]['sha256']==sha(path)
    assert a['DINO_weight']['sha256']==DINO_SHA
    assert names['image_evaluation.py']['sha256']==a['image_evaluation_sha256']
    return names

def validate_shared_recipe(a):
    science=json.loads(checked(a['shared_science']).read_bytes())
    assert science['status']=='LOCKED_SHARED_THREE_ARM_AR_SCIENCE_BEFORE_FORMAL_FIT_v1'
    assert science['arms']==list(ARMS) and science['assigned_hosts']=={'new':['native_global'],'old':['paired_anchored','donor_anchored']}
    assert science['initial_generator_tensor_digest']==INITIAL and science['image_policy']['sha256']==POLICY_SHA
    assert science['panel']['sha256']==PANEL_SHA and science['image_binding']['sha256']=='6ae098556ca4aca24c751697b9f83d74b55da279de1e2a2073730754c77c504f'
    assert science['donor_training_only'] is True
    training=None;admission_hashes={}
    for arm in ARMS:
        path=checked(a['training_admissions'][arm]);value=json.loads(path.read_bytes())
        assert value['status']=='FORMAL_ANCHORED_AR_MICRO2_ADMITTED_v1' and arm in value['arms']
        assert value['shared_science']['sha256']==a['shared_science']['sha256'] and value['training']==science['training']
        assert value['training']['epochs']==10 and value['training']['updates_per_arm']==8270 and value['training']['seed']==1729
        assert value['training']['logical_batch']==16 and value['training']['micro_batch']==2 and value['training']['query_exposures_per_arm']==132300
        assert value['initial_generator_tensor_digest']==INITIAL and value['image_policy']['sha256']==POLICY_SHA
        assert value['data']['EEG_completion']['sha256']==EEG_SHA and value['data']['token_completion']['sha256']==TOKEN_SHA
        if training is None:training=value['training']
        else:assert value['training']==training
        admission_hashes[arm]=sha(path)
    return admission_hashes

def validate_fit(a,arm,*,model=False):
    assert arm in ARMS;admissions=validate_shared_recipe(a)
    path=Path(a['fit_completion_paths'][arm]);assert path.is_absolute() and path.is_file()
    done=json.loads(path.read_bytes());assert done['status']=='COMPLETE_FIXED_T_10_EPOCH_ANCHORED_AR_ARM' and done['arm']==arm
    assert done['admission']['sha256']==admissions[arm] and done['epochs']==10 and done['updates']==8270 and done['query_exposures']==132300
    assert done['new_research_images']==0 and done['old_ordinary_unchanged'] is True
    assert done['native_reads_local_memory_file']==(arm!='native_global')
    assert done['committed_log_chain'] and done['committed_log_chain'][0]['start_update']==1 and done['committed_log_chain'][-1]['end_update']==8270
    previous=0
    for event in done['committed_log_chain']:assert event['start_update']==previous+1 and event['end_update']>=event['start_update'];previous=event['end_update']
    if model:
        source=dict(done['final_model'],path=a['model_paths'][arm]);checked(source)
    return done,ref(path)

def validate_panel(a):
    policy=json.loads(checked(a['policy']).read_bytes());panel=json.loads(checked(a['panel']).read_bytes())
    assert panel['schema']=='AR_EEG_FIXED_DCHECK400_PANEL_v1' and panel['sampling']==policy['generation']
    assert policy['generation']['native_resolution']==[256,256] and policy['generation']['scales']==[1,2,3,4,5,6,8,10,13,16]
    assert policy['generation']['guidance']==1.5 and policy['generation']['top_k']==600 and policy['generation']['top_p']==0 and policy['generation']['more_smooth'] is False
    assert policy['generation']['one_image_per_query'] and not policy['generation']['best_of_N']
    rows=panel['queries'];assert len(rows)==len(panel['target_gallery'])==400 and rows==panel['target_gallery']
    groups={}
    for index,q in enumerate(rows):
        assert q['subject_id']=='sub-01' and q['partition']=='Dcheck' and q['repeat_protocol']=='R4'
        assert q['query_id']==f"sub-01/{q['stimulus_id']}/R4" and q['condition_index']==q['target_index']==index
        assert q['image_seed']==seed_for(q['subject_id'],q['stimulus_id']);groups[q['concept_id']]=groups.get(q['concept_id'],0)+1
    assert len(groups)==40 and set(groups.values())=={10} and len({q['query_id'] for q in rows})==400
    assert len({q['concept_id'] for q in rows[:80]})==8 and all(sum(x['concept_id']==q['concept_id'] for x in rows[:80])==10 for q in rows[:80])
    return policy,panel

def validate_expansion(a,stage):
    assert stage in (80,400)
    if stage==400:
        path=Path(a['stage80_decision_path']);assert path==Path(a['output_root'])/'screens/stage80/summary.json'
        decision=json.loads(path.read_bytes())
        assert decision['status']=='COMPLETE_FIXED_THREE_ARM_AR_IMAGE_SCREEN' and decision['stage']==80
        assert decision['decision']['action']=='EXPAND400' and decision['policy']['sha256']==POLICY_SHA and decision['panel']['sha256']==PANEL_SHA
        assert decision['image_screen_source_sha256']==MATH_SHA and decision['locked_metric_source_sha256']==METRIC_SHA
        assert set(decision['fit_closures'])==set(ARMS)
        for arm in ARMS:assert decision['fit_closures'][arm]['sha256']==validate_fit(a,arm)[1]['sha256']
        return ref(path)
    return None

def chunk_bounds(stage,lo,hi):
    assert stage in (80,400) and isinstance(lo,int) and isinstance(hi,int) and lo%20==0 and hi==lo+20
    assert (0<=lo<hi<=80) if stage==80 else (80<=lo<hi<=400)

def stripped_query(row):
    # Target paths/features, concept labels and gallery never enter the model call.
    return {key:row[key] for key in ('query_id','subject_id','stimulus_id','array_row','condition_index','image_seed')}

def image_paths(a,arm,index):
    assert arm in ARMS and 0<=index<400;root=Path(a['output_root'])/arm
    return root/'images'/f'q{index:05d}.png',root/'records'/f'q{index:05d}.json'

def validate_record(a,arm,index,row):
    image,record_path=image_paths(a,arm,index);record=json.loads(record_path.read_bytes())
    assert record['schema']=='AR_ANCHORED_SINGLE_FIXED_IMAGE_v1' and record['arm']==arm and record['query_index']==index
    assert record['query']==stripped_query(row) and record['sampling_seed']==row['image_seed']
    assert record['policy_sha256']==POLICY_SHA and record['panel_sha256']==PANEL_SHA
    assert record['inference_inputs']==('own_global_c' if arm=='native_global' else 'own_global_c_and_own_local_H')
    assert record['normal_inference_reads_query_image_or_label_or_gallery'] is False and record['best_of_N']==1
    assert checked(record['image'])==image and image.resolve().parent==(Path(a['output_root'])/arm/'images').resolve()
    assert record['resolution']==[256,256] and record['raw_pixels_dtype']=='uint8_RGB'
    return record,ref(record_path)
