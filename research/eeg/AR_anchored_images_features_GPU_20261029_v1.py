"""Frozen DINOv3 feature chunks, with zero metric/target-gallery reads."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,hashlib,json,time,traceback
from pathlib import Path
import AR_anchored_images_contract_CPU_20261029_v1 as C
from AR_anchored_images_generate_GPU_20261029_v1 import peak_CPU,wrapper_receipt

def run(admission,arm,stage,lo,hi,job):
    a,_=C.load_admission(admission);C.validate_sources(a,Path(__file__));C.chunk_bounds(stage,lo,hi);_,panel=C.validate_panel(a);C.validate_fit(a,arm);C.validate_expansion(a,stage)
    root=Path(a['output_root'])/arm;artifact_path=root/'artifacts'/f'stage{stage}.json';artifact=json.loads(artifact_path.read_bytes())
    assert artifact['status']=='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT' and artifact['stage']==stage and artifact['valid_queries']==artifact['expected_queries']==stage and artifact['invalid_queries']==0
    directory=root/'feature_chunks';directory.mkdir(exist_ok=True);out=directory/f'{lo:05d}_{hi:05d}';assert not out.exists();out.mkdir();began=time.monotonic();base_CPU=peak_CPU();phase='FROZEN_DINO_FEATURES'
    try:
        wrapper_receipt(a,job)
        import numpy as np
        import torch
        import image_evaluation as E
        assert C.sha(Path(E.__file__))==a['image_evaluation_sha256'] and E.DINO_SHA256==C.DINO_SHA and E.DINO_WEIGHT==C.checked(a['DINO_weight'])
        torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        rows=[];paths=[]
        for index in range(lo,hi):
            record,source=C.validate_record(a,arm,index,panel['queries'][index]);assert source['sha256']==artifact['records'][index]['record']['sha256']
            rows.append(dict(query_index=index,query_id=panel['queries'][index]['query_id'],record=source,image=record['image'],raw_pixel_SHA256=record['raw_pixel_SHA256']));paths.append(C.checked(record['image']))
        model=E.dino_model();torch.cuda.reset_peak_memory_stats();features=E.dino_features(paths,model,batch_size=20,real_images=False).float().cpu().numpy()
        assert features.shape==(20,768) and features.dtype==np.float32 and np.isfinite(features).all() and np.all(np.linalg.norm(features,axis=1)>0)
        assert time.monotonic()-began<a['resources']['body_hard_seconds'] and torch.cuda.max_memory_allocated()<a['resources']['maximum_GPU_allocated_bytes'] and peak_CPU()-base_CPU<a['resources']['maximum_extra_CPU_bytes']
        for row in rows:C.checked(row['record']);C.checked(row['image'])
        np.save(out/'features.npy',features,allow_pickle=False);C.write_once(out/'rows.json',rows)
        C.write_once(out/'completion.json',dict(status='COMPLETE_FIXED_AR_FROZEN_DINO_FEATURE_CHUNK',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arm=arm,stage=stage,range=[lo,hi],queries=20,feature_dimension=768,
            features=C.ref(out/'features.npy'),rows=C.ref(out/'rows.json'),artifact_audit=C.ref(artifact_path),DINO_weight=a['DINO_weight'],image_evaluation_source=dict(path=str(Path(E.__file__)),sha256=C.sha(Path(E.__file__))),generated_image_resize_route='Historical generated CUDA bilinear256 align_cornersFalse antialiasTrue; normalize ImageNet; layer11 concatCLS/meanpatch L2',
            query_images_used_only_for_evaluation=True,target_features_or_scores_read=False,effect_metrics_computed=0,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=peak_CPU()-base_CPU,seconds=time.monotonic()-began))
    except BaseException as error:
        C.write_once(out/'failure.json',dict(status='AR_FIXED_FEATURE_FAILURE_PRESERVED_NO_QUERY_EXCLUSION',phase=phase,error=repr(error),traceback=traceback.format_exc(),range=[lo,hi],full_intended_denominator=stage,seconds=time.monotonic()-began));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--stage',type=int,required=True);p.add_argument('--lo',type=int,required=True);p.add_argument('--hi',type=int,required=True);p.add_argument('--job-name',required=True);q=p.parse_args();run(q.admission,q.arm,q.stage,q.lo,q.hi,q.job_name)
