"""Independent D31 statistics/input-algebra audit; NumPy + standard library only.

No producer function imports, Torch imports, model forward, GPU, fitting,
images or model/image-admission files. Old scalar temperatures are extracted
from pinned Torch ZIP files by a non-executing restricted tensor descriptor.
The stored cosine scores are audited statistically, not independently
recomputed through the encoder. All original source/data files are read-only.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'2')
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import signal as signal_control
import struct
import sys
import tempfile
import time
import traceback
import zipfile
import numpy as np

N, DIM, CONCEPTS = 13230, 4250, 1323
MODELS=('original25','native50','serial_reversible','parallel_matched')
METHODS=('native_condition','frozen_old_ordinary')
METRICS=('correct_probability','correct_minus_nine_wrong_mean_cosine','same_concept_Top1','same_concept_2AFC')
PRODUCER_SHA='8c9b09e1037ad45cdf34b67d1d4c5bbd8fc64f96a0e4e6615a9abf3dfc2d425d'
TRIAL_SHA='b804a105822bfab5eeb853b6942d25109eaf88f3f8a1eb5b503b444a4dc0d393'
INDEX_SHA='a12c2758f8555223e08022e5b4a195da37ad3dac37cb6e2806aaf17bac014e67'
TARGET_SHA='363e8e74514ca8d1bcbfb29afc2023ed4a5c3a6c4f49bf8b1134e7ae7a090336'
NATIVE_SHA='bb6e38350282af7b37ece04689adcec7a5f87dccf68de5968058433faece0597'


def need(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def ref(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def checked(item):
    path=Path(item['path']);need(path.is_file() and sha(path)==item['sha256'],'SHA differs: '+str(path))
    if 'bytes' in item:need(path.stat().st_size==item['bytes'],'Byte count differs: '+str(path))
    return path


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def peak_rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux VmHWM is required')


class Limits:
    def __init__(self,seconds,output):
        self.start=time.monotonic();self.seconds=seconds;self.output=output;self.initial=peak_rss();self.peak=self.initial;self.stop=False
        signal_control.signal(signal_control.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal_control.signal(signal_control.SIGINT,lambda *_:setattr(self,'stop',True))

    def check(self,stage):
        need(not self.stop and time.monotonic()-self.start<self.seconds,'CPU time/stop cap: '+stage)
        self.peak=max(self.peak,peak_rss());need(self.peak-self.initial<=10<<30,'Additional CPU cap')
        need(not any(name=='torch' or name.startswith('torch.') for name in sys.modules),'Torch was imported')
        need(sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())<=64<<20,'Audit output cap')


class FloatStorageDescriptor:
    """Marker only: no Torch module or real storage object is instantiated."""
    pass


class StorageDescriptor:
    def __init__(self,key,location,count):self.key,self.location,self.count=str(key),str(location),int(count)


class TensorDescriptor:
    def __init__(self,storage,offset,size,stride,*ignored):
        need(isinstance(storage,StorageDescriptor),'Unexpected persistent storage')
        self.storage,self.offset,self.size,self.stride=storage,int(offset),tuple(size),tuple(stride)


class DescriptorUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','OrderedDict'):OrderedDict,
                 ('torch','FloatStorage'):FloatStorageDescriptor,
                 ('torch._utils','_rebuild_tensor_v2'):TensorDescriptor,
                 ('torch._utils','_rebuild_tensor'):TensorDescriptor}
        need((module,name) in allowed,'Disallowed checkpoint global: '+module+'.'+name)
        return allowed[(module,name)]

    def persistent_load(self,identity):
        need(isinstance(identity,tuple) and len(identity)==5 and identity[0]=='storage','Unexpected persistent ID')
        _,kind,key,location,count=identity
        need(kind is FloatStorageDescriptor and location=='cpu' and int(count)>=0,'Expected original CPU FP32 storage')
        return StorageDescriptor(key,location,count)


def old_temperature(path):
    """Read ONLY the original scalar payload bytes; model tensors remain descriptors."""
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();need(len(names)==len(set(names)),'Duplicate Torch ZIP entries')
        pickles=[name for name in names if name.endswith('/data.pkl')]
        need(len(pickles)==1,'Expected one Torch ZIP data.pkl')
        prefix=pickles[0][:-len('data.pkl')]
        payload=DescriptorUnpickler(io.BytesIO(archive.read(pickles[0]))).load()
        need(isinstance(payload,dict) and payload['epochs']==25 and payload['seed']==1729,'Wrong original checkpoint metadata')
        tensor=payload['logit_scale'];need(isinstance(tensor,TensorDescriptor),'Missing original scalar tensor')
        need(tensor.size==tensor.stride==() and tensor.offset==0 and tensor.storage.count==1,'Expected standalone scalar logit_scale')
        scalar_name=prefix+'data/'+tensor.storage.key
        scalar_bytes=archive.read(scalar_name);need(len(scalar_bytes)==4,'Expected one FP32 scalar')
        byteorder=(archive.read(prefix+'byteorder').decode() if prefix+'byteorder' in names else 'little')
        need(byteorder in ('little','big'),'Unknown checkpoint byte order')
        logit=struct.unpack('<f' if byteorder=='little' else '>f',scalar_bytes)[0]
        kappa=math.exp(logit);need(math.isfinite(kappa) and 0<kappa<1e6,'Invalid original temperature')
    return dict(kappa=kappa,logit_scale=logit,checkpoint=ref(path),
        scalar_storage_entry=scalar_name,scalar_storage_sha256=hashlib.sha256(scalar_bytes).hexdigest(),
        dtype='float32',byteorder=byteorder,model_tensors_materialized=False,
        original_registration_sha256=payload['registration_sha256'])


def rebuild_statistics(scores,labels,kappas):
    """Independent direct KLs and explicit wrong-candidate mask.

    scores [B,method,model,9,10]. Output all finite-view metrics and KLs.
    """
    a=np.asarray(scores,np.float64);b=len(a)
    logits=a*np.asarray(kappas,np.float64)[:,None,None,None,None]
    centered=logits-logits.max(-1,keepdims=True)
    logp=centered-np.log(np.exp(centered).sum(-1,keepdims=True));p=np.exp(logp)
    truth=np.asarray(labels,np.int64)[:,None,None,None,None]
    correct=np.take_along_axis(a,np.broadcast_to(truth,a.shape[:-1]+(1,)),axis=-1)[...,0]
    correct_p=np.take_along_axis(p,np.broadcast_to(truth,p.shape[:-1]+(1,)),axis=-1)[...,0]
    wrong=(np.arange(10)[None,None,None,None,:]!=truth)
    margin=correct-(a*wrong).sum(-1)/9.
    hits=(a==a.max(-1,keepdims=True));top1=(correct==a.max(-1))/hits.sum(-1)
    afc=(((correct[...,None]>a)&wrong).sum(-1)+.5*((correct[...,None]==a)&wrong).sum(-1))/9.
    vm=np.stack((correct_p,margin,top1,afc),axis=-1)
    kl_forward=(p[:,:,:,:1]*(logp[:,:,:,:1]-logp[:,:,:,1:])).sum(-1)
    kl_reverse=(p[:,:,:,1:]*(logp[:,:,:,1:]-logp[:,:,:,:1])).sum(-1)
    repeats=((kl_forward+kl_reverse)/2.).reshape(a.shape[:3]+(2,4))
    K=(kl_forward+kl_reverse).reshape(a.shape[:3]+(2,4)).sum(-1)/8.
    return vm,K,repeats


def algebra_from_R4(raw,permutation):
    """Independent loop over physical deletions, plus direct-average challenge."""
    x=np.asarray(raw);need(x.shape[1:]==(4,17,250) and x.dtype==np.float32 and np.isfinite(x).all(),'Invalid physical R4 block')
    mean=x.mean(axis=1,dtype=np.float32).reshape(len(x),DIM)
    raw64=x.reshape(len(x),4,DIM).astype(np.float64);mean64=mean.astype(np.float64)
    offsets=[];real=[];control=[];direct=[]
    for repeat in range(4):
        offset=(mean64-raw64[:,repeat])/3.
        offsets.append(offset)
        real.append((mean64+offset).astype(np.float32))
        control.append((mean64+offset[:,permutation]).astype(np.float32))
        direct.append(raw64[:,[j for j in range(4) if j!=repeat]].sum(1)/3.)
    delta=np.stack(offsets,axis=1);declared=delta@delta.transpose(0,2,1)
    moved=delta[:,:,permutation];moved_gram=moved@moved.transpose(0,2,1)
    gram_norm=np.linalg.norm(declared,axis=(1,2));denom=np.maximum(gram_norm,1e-20)
    declared_error=np.linalg.norm(declared-moved_gram,axis=(1,2))/denom
    declared_spectrum=np.linalg.eigh(declared)[0];moved_spectrum=np.linalg.eigh(moved_gram)[0]
    spectrum_error=np.linalg.norm(declared_spectrum-moved_spectrum,axis=-1)/denom
    actual=np.stack((np.stack(real,axis=1),np.stack(control,axis=1)),axis=1).astype(np.float64)-mean64[:,None,None,:]
    actual_gram=actual@actual.transpose(0,1,3,2)
    scale=gram_norm+np.square(mean64).sum(-1)*1e-10+1e-20
    actual_error=np.linalg.norm(actual_gram[:,0]-actual_gram[:,1],axis=(1,2))/scale
    actual_spectrum=np.linalg.eigh(actual_gram)[0]
    actual_spectrum_error=np.linalg.norm(actual_spectrum[:,0]-actual_spectrum[:,1],axis=-1)/scale
    direct=np.stack(direct,axis=1)
    direct_error=np.linalg.norm(np.stack(real,axis=1).astype(np.float64)-direct,axis=-1)/np.maximum(np.linalg.norm(direct,axis=-1),1e-20)
    proof=np.stack((declared_error,actual_error,np.linalg.norm(delta.sum(1),axis=-1),direct_error.max(-1),spectrum_error,actual_spectrum_error),axis=-1)
    need(declared_error.max()<1e-12 and spectrum_error.max()<1e-12,'Declared common-P Gram/spectrum failure')
    need(actual_error.max()<3e-5 and actual_spectrum_error.max()<3e-5 and direct_error.max()<3e-6,'Actual FP32 view algebra failure')
    return dict(input_declared_repeat_Gram=declared,input_actual_repeat_Gram=actual_gram,
        input_actual_repeat_Gram_spectrum=actual_spectrum,
        perturbation_norms=np.sqrt(np.maximum(np.diagonal(actual_gram,axis1=-2,axis2=-1),0.)),input_proof_values=proof)


def compare(actual,expected,name,proofs,absolute=2e-12,relative=2e-12):
    a=np.asarray(actual);e=np.asarray(expected)
    need(a.shape==e.shape and np.isfinite(a).all() and np.isfinite(e).all(),'Invalid array '+name)
    error=np.abs(a-e)
    need(bool(np.all(error<=absolute+relative*np.abs(e))),'Numerical mismatch '+name+' max='+str(float(error.max())))
    entry=proofs.setdefault(name,dict(maximum_absolute_error=0.,absolute_tolerance=absolute,relative_tolerance=relative))
    entry['maximum_absolute_error']=max(entry['maximum_absolute_error'],float(error.max(initial=0.)))


def compare_gram_spectrum(actual,expected,gram,name,proofs):
    a,e,g=np.asarray(actual),np.asarray(expected),np.asarray(gram)
    need(a.shape==e.shape and np.isfinite(a).all() and np.isfinite(e).all(),'Invalid geometry array '+name)
    scale=np.linalg.norm(g,axis=(-2,-1))
    while scale.ndim<a.ndim:scale=scale[...,None]
    error=np.abs(a-e);bound=5e-11+5e-12*scale
    need(bool(np.all(error<=bound)),'Gram-scale numerical mismatch '+name)
    entry=proofs.setdefault(name,dict(maximum_absolute_error=0.,absolute_tolerance_floor=5e-11,
        relative_Frobenius_Gram_tolerance=5e-12,near_zero_eigenvalues_not_used_as_error_scale=True))
    entry['maximum_absolute_error']=max(entry['maximum_absolute_error'],float(error.max(initial=0.)))


def self_check():
    labels=np.array([0,1]);prob=np.array([.55]+[.05]*9)
    scores=np.zeros((2,1,1,9,10));scores[:,:,:,1]=np.log(prob)/4.
    vm,K,repeats=rebuild_statistics(scores,labels,np.array([4.,4.]))
    need(np.allclose(K[...,0],.45*math.log(11)/8.,atol=1e-14) and np.all(K[...,1]==0.),'Known-probability KL fixture')
    need(np.allclose(vm[0,0,0,1],[.55,math.log(11)/4.,1.,1.],atol=1e-14),'Correct-label fixture')
    need(np.allclose(vm[1,0,0,1],[.05,-math.log(11)/36.,0.,4./9.],atol=1e-14),'Wrong-label/tie fixture')
    need(np.allclose(vm[:,:,:,0,2],.1) and np.allclose(vm[:,:,:,0,3],.5),'Fractional uniform ties')
    rng=np.random.default_rng(1729);raw=rng.normal(size=(2,4,17,250)).astype(np.float32)
    proof=algebra_from_R4(raw,rng.permutation(DIM));need(np.isfinite(proof['input_proof_values']).all(),'Physical algebra fixture')
    # Hand-written standard pickle opcodes and four scalar bytes, not Torch.
    def unicode(value):
        data=value.encode();return b'X'+struct.pack('<I',len(data))+data
    synthetic=b'\x80\x02}('+unicode('epochs')+b'K\x19'+unicode('seed')+b'M\xc1\x06'
    synthetic+=unicode('registration_sha256')+unicode('a'*64)+unicode('logit_scale')
    synthetic+=b'ctorch._utils\n_rebuild_tensor_v2\n(('+unicode('storage')+b'ctorch\nFloatStorage\n'
    synthetic+=unicode('0')+unicode('cpu')+b'K\x01tQK\x00))\x89ccollections\nOrderedDict\n)RtRu.'
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'synthetic_scalar_only.pt'
        with zipfile.ZipFile(path,'w') as archive:
            archive.writestr('fixture/data.pkl',synthetic)
            archive.writestr('fixture/data/0',struct.pack('<f',2.6592))
            archive.writestr('fixture/byteorder','little')
        extracted=old_temperature(path)
        need(extracted['kappa']==math.exp(struct.unpack('<f',struct.pack('<f',2.6592))[0]),'Independent scalar ZIP parser fixture')
        need(extracted['original_registration_sha256']=='a'*64,'Scalar registration fixture')
    need(not any(name=='torch' or name.startswith('torch.') for name in sys.modules),'Self-check imported Torch')
    return dict(known_probability_one_eighth_symmetric_KL=True,correct_and_wrong_label_axes=True,
                uniform_fractional_ties=True,physical_mean3_and_permutation_Gram_spectrum=True,
                restricted_Torch_ZIP_scalar_without_Torch=True)


def audit(args):
    output=Path(args.output).absolute();need(not output.exists() and not output.is_symlink(),'Audit output must be new')
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir();limits=Limits(args.max_seconds,output)
    audit_source=ref(__file__)
    try:
        fixture=self_check();completion_ref=ref(args.completion);done=read(checked(completion_ref))
        need(done['status']=='D31_ALL13230_TRUE_OOF_FINITE_LEAVEONE_DECISIONS_COMPLETE','D31 completion missing')
        need(done['source']['sha256']==PRODUCER_SHA,'Unexpected sealed producer source')
        for key in ('source','spec','D30_completion','output_arrays'):checked(done[key])
        need(done['T_queries']==N and done['T_concepts']==CONCEPTS and done['encoder_true_concept_OOF'] is True,'Incomplete T/OOF')
        for key in ('original_ordinary_checkpoint_changed','model_weights_changed','model_or_image_admission_written',
                    'automatic_following_fit_or_images','Dfit_Dcheck_V_or_official_values_read','formal_D_resumed'):
            need(done[key] is False,'Unexpected scientific state '+key)
        need(done['new_training_updates']==done['new_images']==0,'Unexpected fitting/images')
        spec=read(done['spec']['path']);need(spec['status']=='D31_LEAVEONE_DECISION_DIAGNOSTIC_LOCKED_BEFORE_GPU' and spec['seed']==1729,'Wrong spec')
        need(spec['D30_completion']==done['D30_completion'],'Wrong D30 reference')
        source_refs=list(spec['source_pins'])+[completion_ref,done['source'],done['spec'],done['output_arrays']]
        for item in source_refs:checked(item)
        d30_done=read(checked(spec['D30_completion']));source_refs.append(spec['D30_completion'])
        need(d30_done['status']=='D30_ALL9_MATCHED_ADDITIONAL25_EPOCH_T_FITS_AND_TRUE_OOF_COMPLETE','Wrong D30 training state')
        d30=read(checked(d30_done['spec']));source_refs += [d30_done[k] for k in ('spec','source','model_source','role_rows','model_bundle','OOF_T_native')]+list(d30['source_pins'])
        for item in source_refs:checked(item)
        split_ref,manifest_ref=spec['split'],spec['image_manifest'];split=read(checked(split_ref));source_refs += [split_ref,manifest_ref]
        with checked(manifest_ref).open(encoding='utf-8',newline='') as stream:
            manifest=[row for row in csv.DictReader(stream) if row['dataset_split']=='train']
        by_row={int(row['array_row']):row for row in manifest};rows=split['image_rows']['T']
        need(len(manifest)==len(by_row)==16540 and len(rows)==len(set(rows))==N,'Split/manifest coverage')
        need(not set(rows)&(set(split['image_rows']['D'])|set(split['image_rows']['V'])),'T overlaps D/V')
        need({by_row[row]['concept_id'] for row in rows}==set(split['train_concepts']['T']),'Real concept split schema')
        refs={Path(item['path']).name:item for item in d30_done['actual_T_inputs']}
        for name,expected in [('trials.npy',TRIAL_SHA),('index.jsonl',INDEX_SHA),('targets.npy',TARGET_SHA)]:
            need(refs[name]['sha256']==expected,'Wrong physical source '+name);checked(refs[name]);source_refs.append(refs[name])
        physical=read(checked(d30['physical_R4_input_contract']));source_refs.append(d30['physical_R4_input_contract'])
        need(physical['status']=='PHYSICAL_R4_T_INPUTS_READY' and physical['export_shape']==[N,4,17,250] and physical['mean_replay_exact'] is True,'Wrong physical R4 contract')
        for name in refs:need(physical['files'][name]['sha256']==refs[name]['sha256'],'Physical contract ref mismatch')
        checked(d30['native_input_contract']);need(d30['native_input_contract']['sha256']==NATIVE_SHA,'Native source contract changed')
        native=read(d30['native_input_contract']['path']);source_refs.append(d30['native_input_contract'])
        subject=native['subjects'][0];need(subject['subject']=='sub-01','Wrong subject')
        native_sources={item['role']:item for item in subject['sources']}
        index=[json.loads(line) for line in Path(refs['index.jsonl']['path']).read_text().splitlines()]
        need(len(index)==N,'Physical index size')
        groups=OrderedDict();folds=np.empty(N,np.int8);labels=np.empty(N,np.int8);candidates=np.empty((N,10),np.int64)
        for i,row in enumerate(index):
            metadata=by_row[rows[i]]
            need(row['row']==i and row['array_row']==rows[i] and row['partition']=='T' and row['subject_id']=='sub-01','Index row route')
            need(row['concept_id']==metadata['concept_id'] and row['stimulus_id']==metadata['stimulus_id'],'Stimulus identity')
            fold=split['oof_fold_by_concept'][row['concept_id']];need(row['fold']==fold and fold in (0,1),'Index fold route')
            folds[i]=fold;groups.setdefault(row['concept_id'],[]).append(i)
        need(len(groups)==CONCEPTS and set(map(len,groups.values()))=={10},'1323 complete ten-stimulus concepts')
        for selected in groups.values():
            need(len(set(folds[selected]))==1,'Concept split crosses folds')
            for label,position in enumerate(selected):labels[position]=label;candidates[position]=selected
        role_rows=read(d30_done['role_rows']['path']);temperature_proofs=[];kappas=np.empty(N,np.float64);coverage=np.zeros(N,np.uint8)
        need({proof['role'] for proof in done['role_proofs']}=={'oof0','oof1'},'Missing held-role proof')
        for role in ('oof0','oof1'):
            selected=np.flatnonzero(folds==int(role[-1]));fit=np.flatnonzero(folds!=int(role[-1]));source=native_sources[role]
            need(role_rows[role]['held_compact_rows']==selected.tolist() and role_rows[role]['fit_compact_rows']==fit.tolist(),'D30 FIT/held route')
            need(role_rows[role]['held_array_rows']==[rows[i] for i in selected] and role_rows[role]['fit_array_rows']==[rows[i] for i in fit],'D30 physical array routes')
            routes={entry['array_row']:entry['cache_row'] for entry in source['groups']['component_train']}
            need(set(routes)=={rows[i] for i in selected} and len(set(routes.values()))==len(selected),'Native held cache route')
            cp_ref=d30['roles'][role]['checkpoint'];cache_ref=d30['roles'][role]['condition_cache']
            need(cp_ref['sha256']==source['checkpoint_sha256'] and cache_ref['sha256']==source['condition_cache_sha256'],'Native checkpoint/cache pin')
            checked(cp_ref);checked(cache_ref);source_refs += [cp_ref,cache_ref]
            temperature=old_temperature(cp_ref['path']);kappas[selected]=temperature['kappa'];coverage[selected]+=1
            reg_ref=dict(path=source['parent_registration_path'],sha256=source['parent_registration_sha256'])
            registration=read(checked(reg_ref));source_refs.append(reg_ref)
            need(registration['role']==role and registration['epochs']==25 and registration['training_seed']==1729,'Original registration identity')
            need(set(registration['training_array_rows'])=={rows[i] for i in fit} and set(registration['excluded_oof_rows'])=={rows[i] for i in selected},'Original encoder concept exclusion')
            need(temperature['original_registration_sha256']==reg_ref['sha256'],'Original scalar checkpoint registration binding')
            proof=next(item for item in done['role_proofs'] if item['role']==role)
            need(proof['held']==len(selected) and proof['FIT']==len(fit) and proof['only_held_EEG_values_forwarded'] is True,'Producer OOF routing proof')
            need(math.isclose(proof['fixed_original25_kappa'],temperature['kappa'],rel_tol=2e-13,abs_tol=2e-13),'Original25 κ mismatch')
            temperature_proofs.append(dict(role=role,held_queries=len(selected),FIT_queries=len(fit),temperature=temperature,
                parent_registration=reg_ref,all_fit_and_held_concepts_disjoint=True))
        need(bool((coverage==1).all()),'T routing coverage incomplete')
        methods=METHODS if spec['include_frozen_ordinary'] else METHODS[:1];nm=len(methods)
        need(done['axes']['method']==list(methods) and done['axes']['model']==list(MODELS) and done['axes']['view_metric']==list(METRICS),'Producer axes differ')
        if nm==2:
            checked(spec['ordinary_freeze']);freeze=read(spec['ordinary_freeze']['path']);source_refs.append(spec['ordinary_freeze'])
            need(freeze['checkpoint']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03','Original ordinary identity')
            need(freeze['selected_strength']==1. and freeze['radius_rho_before_strength']==.25,'Original ordinary controls')
            source_refs += [freeze['checkpoint'],freeze['source']]
            for item in (freeze['checkpoint'],freeze['source']):checked(item)
        limits.check('COMPLETE_SOURCE_AND_ROUTING_CHECKS')
        with np.load(done['output_arrays']['path'],allow_pickle=False) as archive:saved={name:archive[name] for name in archive.files}
        required={'cosines','view_metrics','per_query_K','per_repeat_symmetric_KL','input_declared_repeat_Gram','input_actual_repeat_Gram',
            'input_actual_repeat_Gram_spectrum','perturbation_norms','input_proof_values','mean4_cache_relative_errors','kappas','folds','labels',
            'candidate_compact_rows','permutation','per_concept','serial_minus_native50','bootstrap_means','bootstrap_serial_minus_native50',
            'bootstrap_true_minus_control_interaction'}
        need(set(saved)==required,'Unexpected original NPZ arrays')
        need(saved['cosines'].shape==(N,nm,4,9,10) and saved['cosines'].dtype==np.float64,'Cosine shape/dtype')
        for name,value in [('folds',folds),('labels',labels),('candidate_compact_rows',candidates)]:need(np.array_equal(saved[name],value),'Reconstructed '+name+' differs')
        permutation=np.random.default_rng(1729).permutation(DIM);need(np.array_equal(saved['permutation'],permutation),'Fixed same-P seed differs')
        checks={};compare(kappas,saved['kappas'],'independently_extracted_original25_temperatures',checks,2e-12,2e-13)
        vm=np.empty((N,nm,4,9,4));K=np.empty((N,nm,4,2));repeat_K=np.empty((N,nm,4,2,4))
        for lo in range(0,N,256):
            stop=min(lo+256,N);v,k,repeated=rebuild_statistics(saved['cosines'][lo:stop],labels[lo:stop],kappas[lo:stop])
            vm[lo:stop]=v;K[lo:stop]=k;repeat_K[lo:stop]=repeated
            compare(v,saved['view_metrics'][lo:stop],'all_view_probability_margin_Top1_2AFC',checks)
            compare(k,saved['per_query_K'][lo:stop],'all_query_finite_K',checks)
            compare(repeated,saved['per_repeat_symmetric_KL'][lo:stop],'all_repeat_symmetric_KL',checks)
            limits.check('COMPLETE_FINITE_PROBABILITY_STATISTICS')
        trials=np.load(refs['trials.npy']['path'],mmap_mode='r',allow_pickle=False)
        need(trials.shape==(N,4,17,250) and trials.dtype==np.float32,'Physical R4 shape')
        input_proofs=np.empty((N,6));input_error_per_query=np.zeros(N)
        for lo in range(0,N,128):
            stop=min(lo+128,N);algebra=algebra_from_R4(np.asarray(trials[lo:stop]),permutation)
            for key,value in algebra.items():
                if key=='input_declared_repeat_Gram':
                    compare_gram_spectrum(value,saved[key][lo:stop],saved[key][lo:stop],key,checks)
                elif key in ('input_actual_repeat_Gram','input_actual_repeat_Gram_spectrum'):
                    compare_gram_spectrum(value,saved[key][lo:stop],saved['input_actual_repeat_Gram'][lo:stop],key,checks)
                else:compare(value,saved[key][lo:stop],key,checks,5e-11,5e-12)
                difference=np.abs(value-saved[key][lo:stop]).reshape(stop-lo,-1).max(-1)
                input_error_per_query[lo:stop]=np.maximum(input_error_per_query[lo:stop],difference)
            input_proofs[lo:stop]=algebra['input_proof_values'];limits.check('FULL_PHYSICAL_R4_INPUT_ALGEBRA')
        need(saved['mean4_cache_relative_errors'].shape==(N,4) and np.isfinite(saved['mean4_cache_relative_errors']).all() and saved['mean4_cache_relative_errors'].min()>=0 and saved['mean4_cache_relative_errors'].max()<3e-5,'Invalid producer forward-parity evidence')
        compare(saved['mean4_cache_relative_errors'].max(0),done['mean4_replay_max_relative_error_by_model'],'saved_forward_parity_aggregation_only',checks)
        query=np.concatenate((K,vm[:,:,:,0],vm[:,:,:,1:5].mean(3),vm[:,:,:,5:9].mean(3)),axis=-1)
        concept=np.asarray([query[positions].mean(0) for positions in groups.values()])
        paired=np.asarray([row[:,2,:]-row[:,1,:] for row in concept])
        interaction=paired[:,:,0]-paired[:,:,1]
        compare(concept,saved['per_concept'],'all_concept_metrics',checks)
        compare(paired,saved['serial_minus_native50'],'all_concept_serial_minus_native50',checks)
        draws=np.empty((1000,nm,4,14));paired_draws=np.empty((1000,nm,14));inter_draws=np.empty((1000,nm))
        rng=np.random.default_rng(1729)
        for bi in range(1000):
            positions=rng.integers(0,CONCEPTS,size=CONCEPTS)
            draws[bi]=np.mean(concept[positions],axis=0)
            paired_draws[bi]=np.mean(paired[positions],axis=0)
            inter_draws[bi]=np.mean(interaction[positions],axis=0)
            if bi%100==0:limits.check('ALL1000_SHARED_CONCEPT_BOOTSTRAPS')
        for key,value in [('bootstrap_means',draws),('bootstrap_serial_minus_native50',paired_draws),('bootstrap_true_minus_control_interaction',inter_draws)]:compare(value,saved[key],key,checks)
        def interval(mean,bootstrap):return dict(mean=np.asarray(mean).tolist(),descriptive95_lower=np.quantile(bootstrap,.025,axis=0).tolist(),descriptive95_upper=np.quantile(bootstrap,.975,axis=0).tolist())
        def interval_compare(recomputed,original,label):
            for key,value in recomputed.items():compare(value,original[key],label+'_'+key,checks)
        all_models=interval(concept.mean(0),draws);paired_result=interval(paired.mean(0),paired_draws);interaction_result=interval(interaction.mean(0),inter_draws)
        primary=interval(paired[:,0,0].mean(),paired_draws[:,0,0])
        for computed,original,label in [(all_models,done['all_models'],'all_model_summary'),(paired_result,done['serial_minus_native50'],'paired_summary'),
            (interaction_result,done['serial_minus_native50_true_minus_control_K'],'orientation_interaction'),(primary,done['primary_native_serial_minus_native50_real_K'],'primary')]:interval_compare(computed,original,label)
        increased=bool(primary['mean']>0 and primary['descriptive95_lower']>0)
        need(increased is done['primary_K_increase_mean_and_lower95_positive'],'Primary directional arithmetic differs')
        for column,key in [(0,'repeat_Gram_max_declared_relative_error'),(1,'repeat_Gram_max_actual_FP32_scaled_relative_error'),
            (2,'maximum_delta_sum_L2'),(3,'maximum_direct_physical_mean3_relative_error'),
            (4,'repeat_spectrum_max_declared_relative_error'),(5,'repeat_spectrum_max_actual_FP32_scaled_relative_error')]:
            compare(input_proofs[:,column].max(),done[key],'input_max_'+key,checks,5e-11,5e-12)
        compact=output/'independent_statistics_and_input_algebra.npz'
        with compact.open('xb') as stream:np.savez_compressed(stream,per_query_K=K,view_metrics=vm,
            input_proof_values=input_proofs,input_array_replay_max_error_per_query=input_error_per_query,
            independently_extracted_kappas=kappas,per_concept=concept,serial_minus_native50=paired,
            bootstrap_means=draws,bootstrap_serial_minus_native50=paired_draws,
            bootstrap_true_minus_control_interaction=inter_draws,folds=folds,labels=labels)
        limits.check('COMPACT_AUDIT_OUTPUT_SAVED')
        # Include the audit's own exact source pin and every original producer
        # spec/bank/data/checkpoint pin before claiming successful closure.
        source=audit_source;unique={str(Path(item['path']).resolve()):item for item in source_refs}
        for item in unique.values():checked(item);limits.check('FINAL_SOURCE_SHA_RECHECK')
        need(sha(__file__)==source['sha256'],'Audit source changed')
        report=dict(status='D31_ALL13230_STATS_AND_PHYSICAL_INPUT_ALGEBRA_CPU_INDEPENDENTLY_VERIFIED',
            UTC=datetime.now(timezone.utc).isoformat(),source=source,completion=completion_ref,
            T_queries=N,T_concepts=CONCEPTS,methods=list(methods),models=list(MODELS),
            original25_temperature_extraction=temperature_proofs,all_mean4_and_repeat_metrics_verified=True,
            full_physical_R4_mean3_control_Gram_spectrum_recomputed=True,
            exact_label_candidate_and_concept_OOF_routes_reconstructed=True,
            independent_arithmetic='Direct two-KL formula; explicit wrong-candidate masks; independently looped physical mean3 and matrix-product Gram; original checkpoint scalar ZIP extraction without Torch',
            mathematical_fixture=fixture,checks=checks,all_models=all_models,serial_minus_native50=paired_result,
            serial_minus_native50_true_minus_control_K=interaction_result,
            primary_native_serial_minus_native50_real_K=primary,
            primary_K_increase_mean_and_lower95_positive=increased,
            all1000_shared_concept_bootstraps_recomputed=True,
            old_ordinary_previously_fitted_on_all_T=True,whole_ordinary_pipeline_independent_OOF=False,
            encoder_true_concept_OOF=True,all_source_SHA256_reverified=True,
            Torch_imported=False,GPU_initialized=False,new_training_updates=0,new_images=0,
            Dfit_Dcheck_V_or_official_values_read=False,formal_D_resumed=False,
            model_or_image_admission_written=False,no_model_or_winner_selected=True,
            verification_limits=['Cosines are the pinned producer scores; no independent noisy-view encoder forward was run.',
                'Mean4 forward-parity errors were checked for coverage/bounds/aggregation only, not independently remeasured.',
                'FP32 physical views have recorded nonzero zero-sum roundoff; no Gaussian/iid or independent R4 augmentation claim.',
                'Greater finite leave-one sensitivity does not prove a causal failure mechanism or regularization benefit.',
                'Repeatedly used T is development data; old ordinary used all T; no independent confirmation or novelty claim.'],
            decision='INDEPENDENT_STATISTICS_AND_INPUT_ALGEBRA_ONLY_NO_MODEL_OR_IMAGE_ADMISSION',
            compact_results=ref(compact),seconds=time.monotonic()-limits.start,
            peak_extra_CPU_bytes=limits.peak-limits.initial,maximum_extra_CPU_bytes=10<<30,maximum_output_bytes=64<<20)
        write_new(output/'summary.json',report);limits.check('FINAL_CPU_AUDIT_CLOSURE')
        print(json.dumps(dict(status=report['status'],primary=primary,seconds=report['seconds'])),flush=True)
        return report
    except BaseException as exc:
        write_new(output/'failure.json',dict(status='FAILED_D31_CPU_AUDIT_PRESERVED',error=repr(exc),traceback=traceback.format_exc(),
            seconds=time.monotonic()-limits.start,new_training_updates=0,new_images=0,model_or_image_admission_written=False))
        if (output/'summary.json').exists():os.replace(output/'summary.json',output/'failed_summary_preserved.json')
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--completion');parser.add_argument('--output',required=True)
    parser.add_argument('--max-seconds',type=float,default=240.);parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        need(not Path(args.output).exists(),'Preserve old self-test output')
        result=dict(status='D31_INDEPENDENT_CPU_STATS_AND_ALGEBRA_FIXTURES_PASSED',checks=self_check(),source=ref(__file__),
                    Torch_imported=False,GPU_initialized=False,new_training_updates=0,new_images=0)
        write_new(args.output,result);print(json.dumps(result),flush=True)
    else:
        need(args.completion and 30<=args.max_seconds<=240,'Finite CPU max-seconds must be in30..240')
        audit(args)


if __name__=='__main__':main()
