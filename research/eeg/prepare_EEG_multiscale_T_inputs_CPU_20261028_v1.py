"""Complete T, true role-excluded EEG normalization and explicit 200Hz export."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,datetime,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from eeg_labram_interface_20261028_v1 import CacheNativeEEGTransform

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def checked(item):
    path=Path(item['path']);assert sha(path)==item['sha256'],str(path)
    if 'bytes' in item:assert path.stat().st_size==item['bytes']
    return path
def ref(path):return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def run(spec_path):
    start=time.monotonic();spec=json.loads(spec_path.read_bytes());out=Path(spec['output'])
    assert not out.exists() and shutil.disk_usage(out.parent).free>12<<30
    for item in spec['source_pins']:checked(item)
    out.mkdir();phase='INPUT_CONTRACT'
    try:
        def budget():
            peak=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
            assert peak<10<<30 and time.monotonic()-start<900
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<4<<30
        metadata=json.loads(checked(spec['source_metadata']).read_bytes())
        names=tuple(metadata['metadata']['ch_names']['values']);assert len(names)==63 and len(set(x.upper() for x in names))==63
        times=np.asarray(metadata['metadata']['times']['values'])
        # Adopt the already audited released-loader convention, without
        # applying an additional crop to the 250-point EEG array.
        assert times.shape==(300,) and np.allclose(times[50:],np.arange(250)/250,atol=1e-12,rtol=0)
        split=json.loads(checked(spec['split']).read_bytes())
        cpu=json.loads(checked(spec['spatial_CPU']).read_bytes());files=cpu['files']
        raw=np.load(checked(files['T_mean63']),allow_pickle=False,mmap_mode='r');assert raw.shape==(13230,63,250) and raw.dtype==np.float32
        identities=json.loads(checked(files['queries']).read_bytes())['T']
        rows=[q['array_row'] for q in identities];assert rows==split['image_rows']['T'] and len(set(rows))==13230
        assert len({q['concept_id'] for q in identities})==1323
        image_rows={int(r['array_row']):r for r in csv.DictReader(checked(spec['image_manifest']).open()) if r['dataset_split']=='train'}
        for q in identities:
            assert q['concept_id']==image_rows[q['array_row']]['concept_id'] and q['stimulus_id']==image_rows[q['array_row']]['stimulus_id']
        folds=np.array([split['oof_fold_by_concept'][q['concept_id']] for q in identities],dtype=np.int64)
        assert set(folds)=={0,1} and set(rows).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
        write(out/'T_identities.json',dict(subject='sub-01',T=identities,folds=folds.tolist(),image_manifest=spec['image_manifest']))
        records={}
        for role in ('full','oof0','oof1'):
            phase='ROLE_'+role
            selected=np.arange(13230) if role=='full' else np.flatnonzero(folds!=int(role[-1]))
            assert selected.size and (role=='full' or (folds[selected]!=int(role[-1])).all())
            sums=np.zeros(63,np.float64);squares=np.zeros(63,np.float64)
            for lo in range(0,len(selected),64):
                x=np.asarray(raw[selected[lo:lo+64]],dtype=np.float64);assert np.isfinite(x).all()
                sums+=x.sum((0,2));squares+=(x*x).sum((0,2));budget()
            count=len(selected)*250;center=sums/count;variance=squares/count-center*center
            assert (variance>1e-12).all();scale=np.sqrt(variance)
            stats=dict(role=role,center=center.tolist(),scale=scale.tolist(),channel_names=list(names),training_positions=selected.tolist(),training_array_rows=[rows[i] for i in selected],heldout_fold=None if role=='full' else int(role[-1]),fit_only_allowed_role_T=True)
            write(out/(role+'_normalization.json'),stats)
            transform=CacheNativeEEGTransform(names,center,scale)
            cache=out/(role+'_T_EEG200.npy');result=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(13230,63,1,200))
            for lo in range(0,13230,64):
                result[lo:lo+64]=transform(raw[lo:lo+64],channel_names=names);budget()
            result.flush();del result
            records[role]=dict(EEG=ref(cache),normalization=ref(out/(role+'_normalization.json')),training_queries=len(selected),heldout_queries=13230-len(selected))
        phase='TRAINING_VISUAL_TARGET_EXPORT'
        import torch
        assert not torch.cuda.is_initialized()
        target=torch.load(checked(spec['CLIP_H_targets']),map_location='cpu',weights_only=True,mmap=True)
        assert target.shape==(16540,1024) and target.dtype==torch.float32
        y=target[rows].numpy();assert np.isfinite(y).all()
        np.save(out/'T_raw_CLIP_H.npy',y,allow_pickle=False)
        np.save(out/'T_original_array_rows.npy',np.array(rows,dtype=np.int64),allow_pickle=False)
        budget()
        value=dict(status='COMPLETE_SUB01_T13230_TRUE_ROLE_NORMALIZATION_AND_200HZ_EEG_INPUTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),roles=records,identities=ref(out/'T_identities.json'),visual_targets=ref(out/'T_raw_CLIP_H.npy'),original_array_rows=ref(out/'T_original_array_rows.npy'),subject='sub-01',T_stimuli=13230,T_concepts=1323,physical_repeats=4,preprocessing='Released 250-point cache unchanged before T-role centering/scaling; scipy resample_poly up4/down5, Kaiser5, zero boundary padding; no additional EEG crop',time_metadata_convention='Released stale 300-point metadata sliced times[50:] only; not independent raw event replay',physical_microvolt_or_author_unwhitened_units_claimed=False,normalization_fits=3,heldout_concept_EEG_used_to_fit_OOF_normalization=False,Dfit_Dcheck_V_used_for_fit=False,formal_all10_training_ready=False,optimizer_updates=0,GPU_jobs=0,new_images=0,numpy_version=np.__version__,scipy_version=__import__('scipy').__version__,torch_version=torch.__version__,seconds=time.monotonic()-start)
        assert not torch.cuda.is_initialized();write(out/'completion.json',value);print(json.dumps(value))
    except BaseException as error:
        import traceback
        write(out/'failure.json',dict(status='EEG_INPUT_PREPARATION_FAILURE_PRESERVED',phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);run(parser.parse_args().spec)
