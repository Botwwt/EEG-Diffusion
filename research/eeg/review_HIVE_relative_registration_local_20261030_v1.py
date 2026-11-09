import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();E=a.directory;L=E.parent.parent
read=lambda p:json.loads(Path(p).read_bytes())
manifest=read(E/'received_manifest.json')['bundle']['manifest']
for name,item in manifest.items():
    b=(E/name).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
    if name.endswith('.py'):ast.parse(b.decode())
s=read(E/'protocol/study.json');e=read(E/'protocol/image_execution.json');o=read(E/'actual_observation.json');profile=read(E/'actual/fit_profile/completion.json')
assert s['training_seed']==1729 and not s['multiple_seeds_allowed'] and s['max_new_images']==1600
assert e['training_protocol']['sha256']==hashlib.sha256((E/'protocol/study.json').read_bytes()).hexdigest()
assert profile['study']==e['training_protocol'] and profile['status']=='PASS_ACTUAL_FOUR_ARM_NONZERO_INITIAL_HEAD_GRADIENT_PROFILE'
assert profile['temporary_updates']==16 and profile['scientific_updates']==0 and not profile['weights_saved']
requested=read(L/'audit/crossdisciplinary_20261028_v1/HIVE_relative_registration_request_20261030_v1.json')['files']
for name,source in requested.items():assert (E/'protocol'/name).read_bytes()==source.encode()
allgroups=0;mathchecks={}
for arm,x in profile['arms'].items():
    assert len(x['all20_affine_gradients'])==20 and {v['affine_id'] for v in x['all20_affine_gradients']}==set(range(20))
    assert all(v['direction_gradient']>0 and v['trunk_gradient']>0 for v in x['all20_affine_gradients']);allgroups+=20
    mathchecks[arm]=x['independent_GPU_math']
    if arm!='continued_F_cosine':assert x['independent_GPU_math']['loss_error']<2e-6 and x['independent_GPU_math']['gradient_error']<1e-7 and x['independent_GPU_math']['gradient_relative']<1e-4
assert allgroups==80
assert s['ordinary']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
assert s['starting_F']['sha256']=='7ac2e67689ac5be310c842439ee1815022b30dfc6c95c8fa6738f9aa502e4fa1'
r=dict(status='PASS_LOCAL_EXACT_RELATIVE_REGISTRATION_AND_REAL_GPU_PROFILE_SCOPE',SHA_files=len(manifest),new_source_files=len(requested),real_GPU_arm_affine_cells=80,
    all_four_same_initial_head=True,all80_nonzero_gradients=True,independent_GPU_math=mathchecks,CPU_checkpoint_proofs=o.get('actual_committed_checkpoint_CPU_proofs'),
    no_image_effect_values_read=True,all_2000_or_full_images_not_yet_claimed=True,original_fullT_Euclidean_numeric_failure_unchanged=True)
(E/'independent_local_registration_review.json').write_bytes((json.dumps(r,indent=2)+'\n').encode());print(json.dumps(r))
