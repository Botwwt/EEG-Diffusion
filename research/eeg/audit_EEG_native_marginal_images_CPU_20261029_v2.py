"""Independent whole-panel artifact/seed/EEG source audit before effect reads."""
import argparse,datetime,hashlib,json,time
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def check(x):p=Path(x['path']);assert ref(p)==x;return p
def seed(stim,kind):return int.from_bytes(hashlib.sha256(f'1729|EEG-native-marginal-v2|{kind}|sub-01|{stim}|R4'.encode()).digest()[:8],'little')%(2**63-1)
def run(sp,stage):
    import numpy as np
    from PIL import Image
    s=json.loads(Path(sp).read_bytes());assert s['status']=='REGISTERED_FINITE_NATIVE_MARGINAL_IMAGE_WAITER_v2' and stage in (80,400)
    out=Path(s['queue_root'])/f'stage{stage}';queue=json.loads((out/'completion.json').read_bytes());assert queue['status']=='COMPLETE_NEW_ROUTE_FIXED_FINAL_STAGE_IMAGES_AND_DINO_FEATURES'
    image_spec=json.loads(check(queue['image_spec']).read_bytes());root=Path(image_spec['output_root']);panel=json.loads(check(image_spec['panel']).read_bytes());rows=panel['queries'];assert len(rows)==400
    gallery=panel['target_gallery'];assert rows==gallery
    groups={}
    for i,row in enumerate(rows):groups.setdefault(row['concept_id'],[]).append(i)
    assert len(groups)==40 and {len(g) for g in groups.values()}=={10}
    donor={i:g[(j+1)%10] for g in groups.values() for j,i in enumerate(g)}
    start=time.monotonic();arms=image_spec['arms'];parts={arm:[] for arm in arms};records={arm:[] for arm in arms};inodes=set();pixel_hashes={arm:{} for arm in arms};constants={arm:0 for arm in arms};feature_closures=[]
    for lo in range(0,stage,20):
        chunk=root/f'chunk_{lo:04d}_{lo+20:04d}';completion=chunk/'completion.json';c=json.loads(completion.read_bytes())
        assert c['status']=='COMPLETE_NATIVE_MARGINAL_FIXED_FINAL_TWO_ARM_IMAGE_FEATURE_CHUNK' and c['range']==[lo,lo+20] and c['arms']==arms and not c['effect_scores_read'] and c['query_target_images_read']==0
        own_spec=json.loads(check(c['spec']).read_bytes());assert own_spec['panel']==image_spec['panel'] and own_spec['fit_completion']==image_spec['fit_completion']
        fit=json.loads(check(c['fit_completion']).read_bytes());audit=json.loads(check(c['fit_CPU_audit']).read_bytes());assert fit['epoch']==10 and fit['EEG_updates']==2070 and audit['final_model']==fit['final_model']
        means=np.load(check(c['EEG_means']),allow_pickle=False);logs=np.load(check(c['EEG_logscales']),allow_pickle=False)
        assert means.shape==logs.shape==(400,1024) and np.isfinite(means).all() and np.isfinite(logs).all()
        for arm in arms:
            r=json.loads(check(c['records'][arm]).read_bytes());x=np.load(check(c['features'][arm]),allow_pickle=False)
            assert x.shape==(20,768) and x.dtype==np.float32 and np.isfinite(x).all() and np.allclose(np.linalg.norm(x,axis=1),1,atol=2e-5)
            assert len(r)==20
            for i,record in enumerate(r,lo):
                q=rows[i];assert record['arm']==arm and record['index']==i and all(record[key]==q[key] for key in ('query_id','stimulus_id','array_row'))
                assert record['latent_seed']==seed(q['stimulus_id'],'latent') and record['token_seed']==seed(q['stimulus_id'],'token')
                assert record['fit_model_sha256']==fit['final_model']['sha256'] and record['donor_intervention']==(arm=='donor')
                assert record['EEG_row']==(None if arm=='marginal' else donor[i] if arm=='donor' else i)
                assert record['generation']==dict(one_Gaussian_sample=True,one_image=True,best_of_N=1,guidance=0.,top_k=600,top_p=0.,resolution=[256,256],scales=[1,2,3,4,5,6,8,10,13,16])
                assert record['normal_inference_reads_query_image_or_label_or_gallery'] is False
                image=check(record['image']);st=image.stat();inode=(st.st_dev,st.st_ino);assert inode not in inodes;inodes.add(inode)
                with Image.open(image) as frame:
                    assert frame.format=='PNG' and frame.mode=='RGB' and frame.size==(256,256);pix=np.array(frame,copy=True)
                h=hashlib.sha256(pix.tobytes()).hexdigest();assert h==record['pixel_SHA256'];pixel_hashes[arm].setdefault(h,[]).append(i)
                constants[arm]+=int(bool((pix.max((0,1))==pix.min((0,1))).all()))
                latent=np.load(check(record['latent']),allow_pickle=False);assert latent.shape==(1,1024) and latent.dtype==np.float32 and np.isfinite(latent).all()
                if arm!='marginal':
                    source=record['EEG_row'];assert hashlib.sha256(means[source:source+1].tobytes()+logs[source:source+1].tobytes()).hexdigest()==record['EEG_parameters_sha256']
            parts[arm].append(x);records[arm].extend(r)
        feature_closures.append(ref(completion))
    artifacts={}
    for arm in arms:
        path=out/f'{arm}_features_stage{stage}.npy';assert not path.exists();np.save(path,np.concatenate(parts[arm]),allow_pickle=False)
        rp=out/f'{arm}_records_stage{stage}.json'
        with rp.open('x') as f:json.dump(records[arm],f,indent=2);f.write('\n')
        artifacts[arm]=dict(features=ref(path),records=ref(rp))
    result=dict(status='PASS_INDEPENDENT_NATIVE_MARGINAL_ALL_QUERY_ARTIFACT_SEED_EEG_ROW_AND_FEATURE_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,host=s['host'],arms=arms,expected_per_arm=stage,valid_per_arm=stage,invalid_queries=0,image_inodes_unique=len(inodes),
        spatial_constant_counts=constants,pixel_duplicate_groups={arm:[g for g in pixel_hashes[arm].values() if len(g)>1] for arm in arms},constants_or_duplicates_not_excluded=True,
        artifacts=artifacts,feature_chunk_closures=feature_closures,panel=image_spec['panel'],image_spec=queue['image_spec'],fit_completion=image_spec['fit_completion'],fit_CPU_audit=image_spec['fit_CPU_audit'],DINO_weight=image_spec['DINO_weight'],image_evaluation=image_spec['image_evaluation'],effect_scores_read=False,seconds=time.monotonic()-start)
    path=out/'independent_artifact_audit.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--stage',required=True,type=int);a=p.parse_args();print(json.dumps(run(a.spec,a.stage)))
