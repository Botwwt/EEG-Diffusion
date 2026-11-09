"""All-query pixel/identity audit. Corrupt/missing samples are never omitted."""
import argparse,datetime,hashlib,json,time,traceback
from pathlib import Path
import AR_anchored_images_contract_CPU_20261029_v1 as C

def audit(a,arm,stage,panel):
    import numpy as np
    from PIL import Image
    began=time.monotonic();records=[];errors=[];inodes=set();pixel_groups={};bindings={};quality=[]
    done,fit_ref=C.validate_fit(a,arm);C.validate_expansion(a,stage)
    for index,row in enumerate(panel['queries'][:stage]):
        try:
            assert time.monotonic()-began<a['resources']['maximum_CPU_audit_seconds']
            record,record_ref=C.validate_record(a,arm,index,row)
            assert record['fit_completion']['sha256']==fit_ref['sha256'] and record['model_sha256']==done['final_model']['sha256']
            binding_path=C.checked(record['generation_binding']);binding=json.loads(binding_path.read_bytes())
            assert binding['schema']=='AR_ANCHORED_GENERATION_CHUNK_BINDING_v1' and binding['arm']==arm and binding['range'][0]<=index<binding['range'][1]
            assert binding['policy']['sha256']==C.POLICY_SHA and binding['panel']['sha256']==C.PANEL_SHA
            assert binding['fit_completion']['sha256']==fit_ref['sha256'] and binding['model']['sha256']==done['final_model']['sha256']
            admitted,_=C.load_admission(C.checked(binding['admission']));C.validate_shared_recipe(admitted)
            assert admitted['policy']['sha256']==C.POLICY_SHA
            chunk_done=binding_path.parent/'completion.json';chunk=json.loads(chunk_done.read_bytes());assert chunk['status']=='COMPLETE_FIXED_AR_EEG_ONLY_IMAGE_CHUNK'
            assert record_ref['sha256'] in {x['sha256'] for x in chunk['records']}
            bindings[binding_path.name+':'+str(binding_path.parent)]=C.ref(binding_path)
            image,_=C.image_paths(a,arm,index)
            with Image.open(image) as frame:
                assert frame.format=='PNG' and frame.mode=='RGB' and frame.size==(256,256);pixels=np.array(frame,copy=True)
            assert pixels.dtype==np.uint8 and pixels.shape==(256,256,3) and np.isfinite(pixels).all()
            pixel_sha=hashlib.sha256(pixels.tobytes()).hexdigest();assert pixel_sha==record['raw_pixel_SHA256']
            stat=image.stat();inode=(stat.st_dev,stat.st_ino);assert inode not in inodes;inodes.add(inode)
            spatial_std=pixels.astype(np.float64).std(axis=(0,1));spatial_range=pixels.max(axis=(0,1)).astype(int)-pixels.min(axis=(0,1)).astype(int)
            quality.append(dict(query_index=index,query_id=row['query_id'],channel_spatial_std=spatial_std.tolist(),channel_spatial_range=spatial_range.tolist(),exact_spatial_constant=bool((spatial_range==0).all()),near_spatial_constant=bool(spatial_std.max()<1.)))
            pixel_groups.setdefault(pixel_sha,[]).append(index);records.append(dict(query_index=index,query_id=row['query_id'],record=record_ref,image=record['image'],raw_pixel_SHA256=pixel_sha,inode=list(inode)))
        except BaseException as error:errors.append(dict(query_index=index,query_id=row['query_id'],error=repr(error)))
    return dict(status='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT' if not errors else 'FAILED_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT_FULL_DENOMINATOR_RETAINED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,stage=stage,expected_queries=stage,valid_queries=len(records),invalid_queries=len(errors),all_query_ids=[x['query_id'] for x in panel['queries'][:stage]],errors=errors,records=records,fit_completion=fit_ref,
        generation_bindings=list(bindings.values()),spatial_quality=quality,exact_spatial_constant=sum(x['exact_spatial_constant'] for x in quality),near_spatial_constant=sum(x['near_spatial_constant'] for x in quality),pixel_duplicate_groups=[x for x in pixel_groups.values() if len(x)>1],
        constant_and_duplicate_counts_are_descriptive_not_sample_exclusions=True,object_correctness_or_perceptual_noninferiority_established=False,seconds=time.monotonic()-began)

def run(admission,arm,stage):
    a,_=C.load_admission(admission);C.validate_sources(a,Path(__file__));_,panel=C.validate_panel(a);assert stage in (80,400)
    target=Path(a['output_root'])/arm/'artifacts';target.mkdir(exist_ok=True);path=target/f'stage{stage}.json';assert not path.exists()
    result=audit(a,arm,stage,panel);C.write_once(path,result)
    assert result['status']=='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT',result['errors']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--stage',type=int,required=True);q=p.parse_args();run(q.admission,q.arm,q.stage)
