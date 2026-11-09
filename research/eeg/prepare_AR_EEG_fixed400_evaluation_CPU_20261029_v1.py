"""Bind the existing Dcheck400 identities/gallery to AR sampling before fit.

CPU-only preparation. Gallery data are evaluation assets; they are never
normal generator inputs. No candidate scores or image selection occur here.
"""
import argparse,datetime,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from AR_anchored_image_screen_math_20261029_v1 import locked,validate_math_source

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def checked(item):
    p=Path(item['path']);assert sha(p)==item['sha256']
    if 'bytes' in item:assert p.stat().st_size==item['bytes']
    return p
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def run(spec_path):
    spec=json.loads(spec_path.read_bytes());out=Path(spec['output']);assert not out.exists();out.mkdir();validate_math_source()
    for item in spec['source_pins']:checked(item)
    policy_path=checked(spec['image_policy']);policy=json.loads(policy_path.read_bytes());assert policy['generation']['native_resolution']==[256,256] and policy['generation']['top_p']==0
    memory=json.loads(checked(spec['memory_completion']).read_bytes());assert memory['source_campaign_sha256']=='aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56'
    identities=json.loads(checked(memory['files']['identities']).read_bytes());queries=identities['Dcheck'];assert len(queries)==400
    targets=json.loads(checked(spec['source_target_rows']).read_bytes());assert len(targets)==400
    feature_path=checked(spec['target_features']);features=np.load(feature_path,allow_pickle=False);assert features.shape==(400,768) and np.isfinite(features).all()
    assert not ({q['concept_id'] for q in queries}&{q['concept_id'] for q in identities['T']})
    archives={};rows=[];groups={}
    try:
        for index,(query,target) in enumerate(zip(queries,targets)):
            assert all(query[k]==target[k] for k in ('array_row','stimulus_id','concept_id')) and target['target_index']==index
            assert target['subject_id']=='sub-01' and target['partition']=='Dcheck'
            qid=f"sub-01/{query['stimulus_id']}/R4";assert target['query_id']==qid
            if 'query_id' in query:assert query['query_id']==qid
            archive=target['original_source_archive'];member=target['original_source_member']
            if archive not in archives:archives[archive]=zipfile.ZipFile(archive)
            data=archives[archive].read(member);assert len(data)==target['image']['bytes'] and hashlib.sha256(data).hexdigest()==target['image']['sha256']
            seed=int.from_bytes(hashlib.sha256(f"1729|EEG-anchored-AR|sub-01|{query['stimulus_id']}|R4".encode()).digest()[:8],'little')%(2**63-1)
            rows.append(dict(query_id=qid,subject_id='sub-01',stimulus_id=query['stimulus_id'],concept_id=query['concept_id'],array_row=query['array_row'],partition='Dcheck',repeat_protocol='R4',condition_index=index,target_index=index,image_seed=seed,target_image=dict(archive=archive,member=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),original_exposed_panel_HIVE_seed=target['image_seed']))
            groups[query['concept_id']]=groups.get(query['concept_id'],0)+1
    finally:
        for archive in archives.values():archive.close()
    assert len(groups)==40 and set(groups.values())=={10} and len({q['query_id'] for q in rows})==400
    assert len({q['concept_id'] for q in rows[:80]})==8 and all(sum(x['concept_id']==q['concept_id'] for x in rows[:80])==10 for q in rows[:80])
    self_check=locked.metric_arrays(features,features,rows,rows)
    assert np.allclose(self_check['target_cosine'],1,atol=2e-5) and np.all(self_check['DINO2AFC']==1) and np.all(self_check['gallery_top1']==1)
    write(out/'panel.json',dict(schema='AR_EEG_FIXED_DCHECK400_PANEL_v1',status='FIXED_BEFORE_AR_FIT_AND_RESEARCH_IMAGES',queries=rows,target_gallery=rows,stages={'80':dict(query_count=80,query_ids=[q['query_id'] for q in rows[:80]],target_gallery_count=400),'400':dict(query_count=400,query_ids=[q['query_id'] for q in rows],target_gallery_count=400)},sampling=policy['generation'],source_memory_completion=spec['memory_completion'],source_original_target_rows=spec['source_target_rows'],already_exposed_development_panel=True,independent_confirmation=False,normal_generator_accepts_target_images_or_labels=False))
    write(out/'target_cache.json',dict(status='EXISTING_LOCKED_TARGET_FEATURES_REUSED_WITH_COMPLETE_AR_EEG_IDENTITY_JOIN',features=spec['target_features'],source_target_rows=spec['source_target_rows'],source_D40_feature_closure=spec['source_D40_feature_closure'],panel=ref(out/'panel.json'),encoder=spec['DINO_binding'],all400_original_archive_bytes_verified=True,self_pair_DINO2AFC=1.,self_pair_gallery_Top1=1.,new_target_encodings=0))
    write(out/'completion.json',dict(status='COMPLETE_FIXED_AR_DCHECK400_PANEL_AND_TARGET_GALLERY_BEFORE_FIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),panel=ref(out/'panel.json'),target_cache=ref(out/'target_cache.json'),image_policy=spec['image_policy'],source_spec=ref(spec_path),queries=400,concepts=40,target_gallery=400,training_T_images_used_as_Dcheck_targets=False,EEG_repeat_protocol='R4',new_GPU_jobs=0,new_research_images=0,candidate_scores_read=False,training_execution_admitted_by_this_artifact=False))
    print(json.dumps(dict(panel=ref(out/'panel.json'),completion=ref(out/'completion.json'))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
