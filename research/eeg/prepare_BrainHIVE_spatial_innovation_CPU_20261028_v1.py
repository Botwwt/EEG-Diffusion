"""D40 complete-T physical63 means and repeat-only separable noise geometry."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import argparse,datetime,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
import brainhive_spatial_innovation_20261028_v1 as M

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_bytes())
def checked(x):
    p=Path(x['path']);assert sha(p)==x['sha256'];return p
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def memory():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def run(sp):
    began=time.monotonic();base=memory();spec=read(sp);B=Path(spec['output']);assert not B.exists();B.mkdir(parents=True)
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    def budget():assert time.monotonic()-began<850 and memory()-base<10<<30
    for x in spec['source_pins']:checked(x)
    rawpath=checked(spec['physical_R4']);source=np.load(rawpath,allow_pickle=True);raw=source['preprocessed_eeg_data']
    assert raw.shape==(16540,4,63,250) and raw.dtype==np.float32;budget()
    split=read(checked(spec['split']));ids=split['image_rows']['T'];assert len(ids)==13230
    index=[json.loads(x) for x in checked(spec['T_index']).read_text().splitlines()]
    assert [q['array_row'] for q in index]==ids and len({q['concept_id'] for q in index})==1323
    old_trials=np.load(checked(spec['T17_trials']),mmap_mode='r',allow_pickle=False)
    bindings=read(checked(spec['original_bindings']));old_mean=torch.load(checked(dict(path=bindings['subjects']['sub-01']['train_path'],sha256=bindings['subjects']['sub-01']['train_sha256'])),weights_only=True,map_location='cpu',mmap=True)
    means=np.lib.format.open_memmap(B/'T_mean63.npy',mode='w+',dtype=np.float32,shape=(13230,63,250))
    Cs=np.zeros((63,63),np.float64);Ct=np.zeros((250,250),np.float64);mu=np.zeros((63,250),np.float64)
    for lo in range(0,13230,64):
        budget();rows=ids[lo:lo+64];x=raw[rows];assert np.isfinite(x).all() and np.array_equal(x[:,:,46:],old_trials[lo:lo+len(x)])
        value=x.mean(1,dtype=np.float32);assert np.array_equal(value,old_mean[rows].numpy())
        means[lo:lo+len(x)]=value;mu+=value.sum(0,dtype=np.float64)/13230
        d=(x.astype(np.float64)-x.mean(1,dtype=np.float64)[:,None])/math.sqrt(12)
        spatial=d.transpose(0,1,3,2).reshape(-1,63);temporal=d.reshape(-1,250)
        Cs+=spatial.T@spatial/(13230*250);Ct+=temporal.T@temporal/(13230*63)
    means.flush();del means,old_trials;budget()
    panel=read(checked(spec['panel']));qs=panel['Dcheck'];assert len(qs)==400
    d63=raw[[q['array_row'] for q in qs]].mean(1,dtype=np.float32)
    assert np.array_equal(d63,old_mean[[q['array_row'] for q in qs]].numpy())
    oldcheck=np.load(checked(spec['Dcheck17']),allow_pickle=False);assert np.array_equal(d63[:,46:],oldcheck)
    np.save(B/'Dcheck_mean63.npy',d63,allow_pickle=False);del d63,raw,source,old_mean
    parent=torch.load(checked(spec['parent']),map_location='cpu',weights_only=True)['model']
    stats=M.geometry(torch.from_numpy(Cs),torch.from_numpy(Ct),parent['proj_in.weight'])
    stats['mean']=torch.from_numpy(mu)
    for name,value in dict(spatial_covariance=torch.from_numpy(Cs),temporal_covariance=torch.from_numpy(Ct),**stats).items():
        np.save(B/(name+'.npy'),value.numpy(),allow_pickle=False)
    groups={}
    for i,q in enumerate(index):groups.setdefault(q['concept_id'],[]).append(i)
    donor=np.empty(13230,np.int64)
    for group in groups.values():
        assert len(group)==10
        for i,j in zip(group,group[1:]+group[:1]):donor[i]=j
    assert (donor!=np.arange(13230)).all();np.save(B/'same_concept_training_donor.npy',donor,allow_pickle=False)
    write(B/'queries.json',dict(T=index,Dcheck=qs))
    files={p.stem:ref(p) for p in B.iterdir() if p.is_file()}
    v=dict(status='D40_COMPLETE_T_TRUE63_ELECTRODES_REPEAT_NOISE_INPUTS_VERIFIED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(sp),files=files,
      T_stimuli=13230,T_concepts=1323,subjects=1,physical_repeats=4,added_real_electrodes=46,
      all_T63_means_match_original_published_mean=True,all_T17_repeat_values_match_original=True,all400_Dcheck17_means_match_original=True,
      original_hidden_working_noise_energy=float(stats['original_hidden_working_noise_energy']),B_Frobenius_radius=float(stats['weight_Frobenius_radius']),
      shrinkage=.1,working_spatiotemporal_covariance_is_separable_assumption=True,repeat_independence_not_established=True,
      Dfit_Dcheck_V_or_visual_targets_used_for_noise_geometry=False,whole_source_pickle_contains_other_training_rows=True,
      original_formal_D_resumed=False,ordinary_modified=False,new_GPU_jobs=0,new_fits=0,seconds=time.monotonic()-began,peak_extra_CPU_bytes=memory()-base)
    budget();assert not torch.cuda.is_initialized();write(B/'completion.json',v);print(json.dumps(v),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);run(p.parse_args().spec)
