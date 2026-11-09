"""Pure-stdlib production-admission/data/provenance checks. No GPU on import."""
import hashlib,json,math
from pathlib import Path

ROOT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/multiscale_EEG_20261028_v1')
ARM_ORDER={'old':['paired_anchored','donor_anchored'],'new':['native_global']}
GPU_UUIDS={'old':'GPU-0a99d971-a46c-6e81-f157-c33243c6d57a','new':'GPU-62a1aee8-4740-dc59-93f4-d50644db650d'}
CORE_SHA='69c0162afcd1b638191bb708c80a96def798a7c611ba1a3f5b857b3ee64f72b3'
LOADER_SHA='3f2abeca5dc1592b675991ae586f564382bf6a69c1c92dc917418d44408064db'
ANCHOR_SHA='7768c21b3dbd5cff03a7cb0b333c8531127f2d61a6bbcaa860de3ae28ce75fb0'
PREFIX_SHA='6af999d30a15e94121ba6468c07012388020e5a42eb368a2318b4504b5dbaa69'
CPU_TREE_MODULE_SHA='4e131212cfa0ecc23339873f6b0e262c6b964474947762b57fc61c090e8bdd7a'
AUTHOR_INVENTORY_SHA='c0d6b113c99f69a96036e95520fe64ccae7507bafd2674bfcb9b38024fe0f8e3'
VQ_SHA='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c'
VAR_SHA='4f6151aad91c94e03e224dd7358d8389fa05301c0c291279566998efb54b0ecb'
EEG_COMPLETION_SHA='c11bd68227feffe5efaebeac309e99090c5ab08cc8c44b87b7abe3882d4c2607'
TOKEN_COMPLETION_SHA='e88a6ee59d4dba2d675d4760ea4e24a25214ecd3daa42624e6043437fd3b3205'
IMAGE_POLICY_SHA='732c5de27b84b11ff6f65884456c0124abe55c26efae90c0cd4d5195fa19fce1'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(item):
    assert isinstance(item,dict) and {'path','sha256','bytes'}<=set(item)
    p=Path(item['path']);assert p.is_absolute() and p.is_file() and p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p)
    return p
def ref(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def load_admission(path,*,arm=None):
    p=Path(path);admission=json.loads(p.read_bytes())
    if admission.get('status')!='FORMAL_ANCHORED_AR_MICRO2_ADMITTED_v1':
        raise ValueError('NOT_ADMITTED_DRAFT_OR_UNKNOWN_STATUS; no CUDA, outputs or training opened')
    assert admission['host_role'] in ARM_ORDER and admission['arms']==ARM_ORDER[admission['host_role']]
    if arm is not None:assert arm in admission['arms']
    assert admission['GPU_UUID']==GPU_UUIDS[admission['host_role']]
    assert admission['training']==dict(epochs=10,logical_batch=16,micro_batch=2,updates_per_epoch=827,updates_per_arm=8270,
        queries=13230,concepts=1323,query_exposures_per_arm=132300,seed=1729,lr=1e-4,weight_decay=.01,betas=[.9,.999],eps=1e-8,
        gradient_clip=1.,condition_dropout=.1,final_epoch_only=True)
    resources=admission['resources']
    assert resources['maximum_job_seconds']==900 and resources['maximum_GPU_allocated_bytes']==20<<30 and resources['maximum_extra_CPU_bytes']==10<<30
    assert isinstance(resources['maximum_updates_per_job'],int) and 1<=resources['maximum_updates_per_job']<=8270
    assert 0<resources['training_soft_seconds']<resources['producer_hard_seconds']<=850
    assert 60<=resources['producer_hard_seconds']-resources['training_soft_seconds']
    assert resources['unreserved_disk_buffer_bytes']>=2<<30
    assert resources['maximum_resume_payload_bytes']>=310308096*12 and resources['maximum_final_model_bytes']>=310308096*4
    assert isinstance(resources['maximum_queue_seconds'],int) and 0<resources['maximum_queue_seconds']<=172800
    output=Path(admission['output_root']);assert output.is_absolute() and output.name=='AR_anchored_training_20261029_v1'
    assert output.parent==ROOT and output.resolve()==output and not output.is_symlink()
    assert len(admission['initial_generator_tensor_digest'])==64
    return admission,sha(p)

def validate_sources(admission,*,producer_path=None,controller_path=None):
    pins=admission['source_pins'];names={};repeated=set()
    for item in pins:
        path=checked(item)
        if path.name in names:repeated.add(path.name)
        else:names[path.name]=item
    expected={'AR_anchored_training_core_DRAFT_20261029_v3.py':CORE_SHA,
        'EEG_generative_backend_loader_20261029_v1.py':LOADER_SHA,
        'EEG_anchored_attention_20261029_v1.py':ANCHOR_SHA,
        'eeg_multiscale_prefix_20261028_v1.py':PREFIX_SHA,
        'eeg_alignment_training_core_20261028_v1.py':CPU_TREE_MODULE_SHA}
    for name,digest in expected.items():assert name not in repeated and names[name]['sha256']==digest
    for own in (producer_path,controller_path,Path(__file__)):
        if own is not None:
            own=Path(own).resolve();assert own.name not in repeated and names[own.name]['sha256']==sha(own) and Path(names[own.name]['path']).resolve()==own
    assert admission['author_inventory']['sha256']==AUTHOR_INVENTORY_SHA
    assert checked(admission['author_inventory'])==Path(admission['author_source'])/'source_inventory.json'
    return names

def validate_profiles(admission):
    strict=json.loads(checked(admission['strict_backend_profile']).read_bytes())
    assert strict['status']=='PASS_REAL_T_GENUINE_OOF_FULL_PRETRAINED_VQ_VAR_EEG_ENGINEERING'
    assert strict['checks'] and all(type(x) is bool and x for x in strict['checks'].values())
    assert strict['load_proof']['status']=='EXACT_AUTHOR_VQ_AND_EXPLICIT_CLASS_TO_EEG_VAR_TRANSFER'
    profile=json.loads(checked(admission['pretrained_micro2_profile']).read_bytes())
    assert profile['status']=='PASS_REAL_46_T_LOGICAL16_MICRO2_ALL3_ARMS_PRETRAINED_VAR_RESOURCE'
    assert profile['pretrained_VAR'] is True and profile['pretrained_VQ'] is True and profile['pretrained_VAR_loaded'] is True
    assert profile['logical_batch']==16 and profile['micro_batch']==2 and profile['logical_batch_rows']==[16,16,14]
    assert profile['arm_names']==['native_global','paired_anchored','donor_anchored'] and profile['optimizer_updates_per_arm']==3
    assert profile['all_temporary_updates_finite'] is True and profile['all_temporary_updates_reset'] is True and profile['all_initial_VAR_values_restored'] is True
    assert profile['CUDA_RNG_matched'] is True and profile['native_training_exercised_without_memory_argument'] is True
    assert profile['actual_core_source_sha256']==CORE_SHA and profile['public_VAR_sha256']==VAR_SHA and profile['public_VQ_sha256']==VQ_SHA
    assert profile['initial_generator_tensor_digest']==admission['initial_generator_tensor_digest']
    assert profile['peak_GPU_allocated']<20<<30 and profile['extra_CPU_peak']<10<<30
    assert profile['GPU_UUID']==admission['GPU_UUID']
    assert profile['resource_probe_version']>=4
    resume=profile['native_resume_serialization_probe']
    assert resume['status']=='ACTUAL_NATIVE_FULL_MODEL_ADAM_CURSOR_AND_RNG_SERIALIZE_RELOAD_RESTORE_PASSED'
    for key in ('model_state_exact','all_Adam_moments_and_steps_exact','cursor_exact','CPU_CUDA_shuffle_RNG_exact','CPU_snapshot_freed_before_reload'):
        assert resume[key] is True
    assert resume['payload_bytes']<=admission['resources']['maximum_resume_payload_bytes']
    proof=profile['strict_backend_transfer'];assert proof['status']=='EXACT_AUTHOR_VQ_AND_EXPLICIT_CLASS_TO_EEG_VAR_TRANSFER'
    assert proof['VQ']['all_transferred_tensors_bitwise_equal'] is True and proof['VAR']['all_transferred_tensors_bitwise_equal'] is True
    assert proof['VAR']['new_seeded_keys']==['embed_proj.bias','embed_proj.weight'] and proof['VAR']['unused_keys_with_expected_shapes']=={'class_emb.weight':[1001,1024]}
    assert checked(admission['public_VQ']) and admission['public_VQ']['sha256']==VQ_SHA
    assert checked(admission['public_VAR']) and admission['public_VAR']['sha256']==VAR_SHA
    return strict,profile

def validate_image_binding(admission):
    assert admission['image_policy']['sha256']==IMAGE_POLICY_SHA;checked(admission['image_policy'])
    binding=json.loads(checked(admission['image_binding']).read_bytes())
    assert binding['status']=='COMPLETE_FIXED_AR_DCHECK400_PANEL_AND_TARGET_GALLERY_BEFORE_FIT'
    assert binding['queries']==400 and binding['concepts']==40 and binding['target_gallery']==400
    assert binding['image_policy']['sha256']==IMAGE_POLICY_SHA
    assert binding['training_T_images_used_as_Dcheck_targets'] is False and binding['EEG_repeat_protocol']=='R4'
    assert binding['new_GPU_jobs']==0 and binding['new_research_images']==0 and binding['candidate_scores_read'] is False
    # Explicit rebased metadata refs may point to same-byte copies on NEW.
    for key in ('panel','target_cache'):
        assert admission['evaluation_metadata'][key]['sha256']==binding[key]['sha256']
        assert admission['evaluation_metadata'][key]['bytes']==binding[key]['bytes']
    panel=json.loads(checked(admission['evaluation_metadata']['panel']).read_bytes())
    assert panel['schema']=='AR_EEG_FIXED_DCHECK400_PANEL_v1' and panel['status']=='FIXED_BEFORE_AR_FIT_AND_RESEARCH_IMAGES'
    rows=panel['queries'];assert len(rows)==len(panel['target_gallery'])==400
    assert rows==panel['target_gallery'] and len({q['query_id'] for q in rows})==400
    for index,q in enumerate(rows):
        assert q['subject_id']=='sub-01' and q['repeat_protocol']=='R4' and q['partition']=='Dcheck'
        assert q['condition_index']==q['target_index']==index
        expected=int.from_bytes(hashlib.sha256(f"1729|EEG-anchored-AR|sub-01|{q['stimulus_id']}|R4".encode()).digest()[:8],'little')%(2**63-1)
        assert q['image_seed']==expected
    cache=json.loads(checked(admission['evaluation_metadata']['target_cache']).read_bytes())
    assert cache['status']=='EXISTING_LOCKED_TARGET_FEATURES_REUSED_WITH_COMPLETE_AR_EEG_IDENTITY_JOIN'
    assert cache['all400_original_archive_bytes_verified'] is True and cache['self_pair_DINO2AFC']==cache['self_pair_gallery_Top1']==1.
    return binding

def identity_join(eeg_rows,token_rows,*,expected_queries=13230,expected_concepts=1323):
    assert len(eeg_rows)==len(token_rows)==expected_queries
    assert len({q['stimulus_id'] for q in eeg_rows})==expected_queries and len({q['concept_id'] for q in eeg_rows})==expected_concepts
    concepts={}
    for index,(eeg,token) in enumerate(zip(eeg_rows,token_rows)):
        # The canonical EEG row schema has no subject field. Its pinned
        # sub-01 campaign supplies scope; do not invent per-row metadata.
        if 'subject' in eeg:assert eeg['subject']=='sub-01'
        if 'subject_id' in eeg:assert eeg['subject_id']=='sub-01'
        assert int(token['query_index'])==index
        assert (int(eeg['array_row']),eeg['stimulus_id'],eeg['concept_id'])==(int(token['array_row']),token['stimulus_id'],token['concept_id'])
        assert len(token['image_sha256'])==64;concepts.setdefault(eeg['concept_id'],0);concepts[eeg['concept_id']]+=1
    assert set(concepts.values())=={expected_queries//expected_concepts}
    return [q['concept_id'] for q in eeg_rows]

def load_training_banks(admission,arm):
    import numpy as np
    data=admission['data'];assert data['EEG_completion']['sha256']==EEG_COMPLETION_SHA and data['token_completion']['sha256']==TOKEN_COMPLETION_SHA
    eeg=json.loads(checked(data['EEG_completion']).read_bytes());tokens=json.loads(checked(data['token_completion']).read_bytes())
    assert eeg['status']=='COMPLETE_REAL_EEG_T13230_GENUINE_OOF_AND_DCHECK400_LOCAL_MEMORY' and eeg['all_T_rows_exactly_once_from_concept_excluded_encoder'] is True
    assert eeg['source_campaign_sha256']=='aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56'
    assert tokens['status']=='COMPLETE_T13230_FROZEN_STRICT_VQ_TOKEN_BANK_INDEPENDENT_CPU_JOIN_AUDITED'
    assert tokens['queries']==13230 and tokens['concepts']==1323 and tokens['token_width']==680 and tokens['indices_dtype']=='int16'
    assert tokens['source_images_T_only'] is True and tokens['VQ_weights_frozen'] is True and tokens['random_augmentation'] is False
    for actual,original in [('global_c',eeg['files']['T_OOF_c']),('EEG_identities',eeg['files']['identities']),('tokens',tokens['indices']),('token_identities',tokens['identities'])]:
        assert data[actual]['sha256']==original['sha256'] and data[actual]['bytes']==original['bytes']
    ids=json.loads(checked(data['EEG_identities']).read_bytes());token_ids=json.loads(checked(data['token_identities']).read_bytes())
    concepts=identity_join(ids['T'],token_ids)
    global_c=np.load(checked(data['global_c']),mmap_mode='r',allow_pickle=False)
    codes=np.load(checked(data['tokens']),mmap_mode='r',allow_pickle=False)
    assert global_c.shape==(13230,1024) and global_c.dtype==np.float32 and np.isfinite(global_c).all()
    assert codes.shape==(13230,680) and codes.dtype==np.int16 and codes.min()>=0 and codes.max()<4096
    local=None
    if arm!='native_global':
        assert data['local_memory']['sha256']==eeg['files']['T_OOF_memory']['sha256'] and data['local_memory']['bytes']==eeg['files']['T_OOF_memory']['bytes']
        local=np.load(checked(data['local_memory']),mmap_mode='r',allow_pickle=False)
        assert local.shape==(13230,63,1024) and local.dtype==np.float32
    return global_c,local,codes,concepts

def verify_protected_files(admission):
    protected=admission['protected_files'];assert protected
    assert any(x['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03' for x in protected)
    assert any(x['sha256']=='2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7' for x in protected)
    for item in protected:checked(item)
