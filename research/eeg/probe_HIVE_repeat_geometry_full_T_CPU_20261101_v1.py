"""Fixed all-T held-concept noise normalization and disjoint-repeat geometry."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,datetime,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import brainhive_repeat_geometry_CPU_20261101_v1 as M

def run(protocol):
    started=time.monotonic();s=json.loads(Path(protocol).read_bytes());out=Path(s['output_root']);assert not out.exists();out.mkdir()
    def ref(p):
        p=Path(p)
        with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        return dict(path=str(p),bytes=p.stat().st_size,sha256=sha)
    def check(pin):assert ref(pin['path'])==pin;return Path(pin['path'])
    import sys
    sys.path.insert(0,s['repository_root']);from brainhive_functional_training_contract_v2 import resource_headroom
    def rss():return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')))*1024
    base=rss()
    def guard():
        assert time.monotonic()-started<240 and rss()-base<2<<30
        assert resource_headroom()['effective_headroom_bytes']>=10<<30
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<8<<20
        assert not any(k in sys.modules for k in ('torch','tensorflow','jax','cupy'))
    try:
        assert s['status']=='REGISTERED_FULL_T_REPEAT_GEOMETRY_DIAGNOSTIC_ONLY'
        for pin in s['source_pins']:check(pin)
        rows=[json.loads(line) for line in check(s['index']).read_text().splitlines()]
        assert len(rows)==13230 and [r['row'] for r in rows]==list(range(13230))
        assert all(r['partition']=='T' and r['subject_id']=='sub-01' and r['fold'] in (0,1) for r in rows)
        concepts=sorted({r['concept_id'] for r in rows});assert len(concepts)==1323
        groups=[sorted([r['row'] for r in rows if r['concept_id']==c],key=lambda i:rows[i]['stimulus_id']) for c in concepts]
        assert all(len(group)==10 and len({rows[i]['fold'] for i in group})==1 for group in groups)
        folds=np.array([rows[group[0]]['fold'] for group in groups],dtype=np.int16)
        trials=np.load(check(s['trials']),mmap_mode='r',allow_pickle=False);assert trials.shape==(13230,4,17,250) and trials.dtype==np.float32
        products=np.empty((1323,6,45),dtype=np.float64);variances=np.empty((2,4250));normalizers=[]
        for held in (0,1):
            fit_indices=np.array([r['row'] for r in rows if r['fold']!=held]);held_groups=np.flatnonzero(folds==held)
            assert len(fit_indices)+len(held_groups)*10==13230
            total=np.zeros(4250,dtype=np.float64)
            for lo in range(0,len(fit_indices),64):
                x=np.asarray(trials[fit_indices[lo:lo+64]],dtype=np.float64).reshape(-1,4,4250)
                assert np.isfinite(x).all();x-=x.mean(axis=1,keepdims=True);total+=(x*x).sum(axis=(0,1));guard()
            variance=total/(len(fit_indices)*3);floor=float(variance.mean())*1e-8;assert floor>0
            variance=np.maximum(variance,floor);variances[held]=variance
            normalizers.append(dict(held_fold=held,fit_stimuli=len(fit_indices),held_concepts=len(held_groups),
                noise_variance_floor=floor,clamped_coordinates=int((total/(len(fit_indices)*3)<floor).sum())))
            for k in held_groups:
                x=np.asarray(trials[groups[k]],dtype=np.float64).reshape(10,4,4250)
                products[k]=M.pair_products(x,variance)
                if k%16==0:guard()
        result,statistics,bootstrap=M.fixed_readout(products);guard()
        np.save(out/'all1323_crossrepeat_products.npy',products,allow_pickle=False)
        np.save(out/'all1323_concept_statistics.npy',statistics,allow_pickle=False)
        np.save(out/'two_heldfold_noise_variances.npy',variances,allow_pickle=False)
        if bootstrap is not None:np.save(out/'all1000_shared_bootstraps.npy',bootstrap,allow_pickle=False)
        index=dict(concepts=concepts,concept_rows=groups,folds=folds.tolist(),all_pair_order='lexicographic combinations of10 stimulus_id-sorted rows',
                   repeat_edge_order=M.REPEAT_EDGES,disjoint_partitions=M.PARTITIONS)
        (out/'full_index.json').write_text(json.dumps(index,indent=2)+'\n')
        original_stat=Path(s['trials']['path']).stat();del trials,x;gc.collect()
        # The active fullfit consumes raw OOF conditions, not this historical R4
        # diagnostic input. Advise its clean pages away, keeping every byte.
        assert ref(s['trials']['path'])==s['trials']
        current=Path(s['trials']['path']).stat();assert (original_stat.st_ino,original_stat.st_size,original_stat.st_mtime_ns)==(current.st_ino,current.st_size,current.st_mtime_ns)
        fd=os.open(s['trials']['path'],os.O_RDONLY|os.O_NOFOLLOW)
        try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
        finally:os.close(fd)
        payloads=[ref(p) for p in sorted(out.iterdir()) if p.is_file()]
        result.update(protocol=ref(protocol),subject='sub-01',stimuli=13230,concept_pairs=59535,electrodes=17,timepoints=250,normalizers=normalizers,
            stimulus_pairs_selected=False,time_or_channel_scan=False,normalizer_fit_concepts_exclude_evaluated_concepts=True,
            within_concept_centering=True,same_concept_control='single fixed cyclic relabeling of10 stimulus identities',
            all6_product_negative_fraction=float((products<0).mean()),six_product_mean=float(products.mean()),
            physical_repetitions_verified=True,independent_session_metadata_available=False,
            noise_independence_not_established=True,Gaussian_assumption_required=False,conditional_unbiasedness_claim_only=True,
            inputs_after_SHA_unchanged=True,own_input_clean_page_advice_only=True,files_deleted=0,
            new_models_or_updates_or_images=0,new_GPU_jobs=0,ordinary_weights_read=False,R80_payloads_read=False,
            seconds=time.monotonic()-started,extra_CPU_peak_bytes=rss()-base,files=payloads,NEW_contacted=False,novelty_claimed=False)
        guard();(out/'completion.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    except BaseException as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True),indent=2)+'\n');raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);run(parser.parse_args().protocol)
