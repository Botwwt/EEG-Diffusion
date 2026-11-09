"""T-only native-F working prior and exact training-only EEG KL contexts.

This prepares a future generator engineering probe. It does not fit a semantic
teacher or EEG model and establishes no heldout likelihood/image improvement.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args();p=json.loads(args.protocol.read_bytes())
sys.path[:0]=[p['source_root'],p['repository_root']]
import brainhive_paired_study_contract_20261025_v1 as C
C.setup_runtime(dict(runtime_manifest=p['runtime_manifest']))
import numpy as np
import torch
from scipy.special import logsumexp
from brainhive_semantic_teacher_Torch_20261029_v1 import F_native_coordinates,native_F_condition
from brainhive_marginal_mixture_fit_CPU_20261029_v1 import fit_tied_vmf_mixture,fit_conditional_tilt,conditional_parameters
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_log_normalizer_array
torch.set_num_threads(2);start=time.monotonic();out=Path(p['output_root'])
def deadline(signum,frame):raise TimeoutError('Fixed native-F CPU preparation deadline')
signal.signal(signal.SIGALRM,deadline);signal.alarm(900)
def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()
def write(name,value):
    path=out/name;pending=path.with_name(path.name+'.pending');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)
def guard():
    assert not torch.cuda.is_initialized() and time.monotonic()-start<895
    assert sum(path.stat().st_size for path in out.iterdir() if path.is_file())<256*1024**2
def log_evidence(evidence,model):
    norm2=np.sum(evidence*evidence,axis=1)
    lengths=np.sqrt(np.maximum(0,model['concentrations']**2+norm2[:,None]+2*model['concentrations']*(evidence@model['means'].T)))
    return logsumexp(np.log(model['weights'])+vmf_log_normalizer_array(1023,model['concentrations'])-vmf_log_normalizer_array(1023,lengths),axis=1)
try:
    for name,digest in p['sources_sha256'].items():assert sha(Path(p['source_root'])/name)==digest
    for ref in p['input_pins']:assert sha(ref['path'])==ref['sha256']
    geometry=np.load(p['F_geometry']['path'],allow_pickle=False);weight=torch.from_numpy(geometry['weight']);bias=torch.from_numpy(geometry['bias']);radius=float(np.median(geometry['native_pre_affine_radius']))
    assert np.array_equal(geometry['weight'],np.ones(1024)) and np.any(geometry['bias']!=0)
    split=json.loads(Path(p['split_manifest']['path']).read_bytes());rows=split['image_rows']['T'];assert len(rows)==13230;position={row:i for i,row in enumerate(rows)}
    contract=json.loads(Path(p['input_contract']['path']).read_bytes());assert contract['status']=='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    target=torch.load(contract['target_cache_path'],map_location='cpu',mmap=True,weights_only=True)[rows].double();assert sha(contract['target_cache_path'])==contract['target_cache_sha256']
    with torch.no_grad():
        coordinates=F_native_coordinates(target,weight,bias);radii=coordinates.norm(dim=-1);unit=coordinates/radii[:,None]
        exact=native_F_condition(unit,weight,bias,radii);constant_radius=native_F_condition(unit,weight,bias,radius)
    roundtrip=float((exact-target).abs().max());constant_error=float((constant_radius-target).abs().max());assert roundtrip<1e-6 and constant_error<1e-5
    visual=unit.numpy();assert visual.shape==(13230,1023)
    progress=dict(phase='native_F_visual_working_prior',seconds=time.monotonic()-start);write('progress.json',progress)
    model=fit_tied_vmf_mixture(visual,components=32,iterations=30);guard()
    np.savez_compressed(out/'native_F_working_prior.npz',weights=model['weights'],means=model['means'],concentrations=model['concentrations'],F_weight=geometry['weight'],F_bias=geometry['bias'],reference_pre_affine_radius=np.array(radius))
    index=[json.loads(line) for line in Path(p['T_index']['path']).read_text().splitlines()];assert len(index)==13230 and [row['array_row'] for row in index]==rows
    groups={}
    for i,record in enumerate(index):groups.setdefault(record['concept_id'],[]).append(i)
    assert len(groups)==1323 and all(len(group)==10 for group in groups.values());donor=np.arange(13230)
    for group in groups.values():donor[group]=np.roll(group,-1)
    positions=np.array([0,1889,3779,5669,7559,9449,11339,13229])
    B=np.zeros((13230,1023),dtype=np.float64);mean_logZ=np.zeros(13230,dtype=np.float64);alpha=[];subjects=[];profiles=[];T8_EEG=[];T8_logZ=[];T8_raw=[];T8_sources=[]
    for subject in contract['subjects']:
        raw=torch.empty(13230,1024,dtype=torch.float64);coverage=torch.zeros(13230,dtype=torch.int16);source_rows=[];source_refs={}
        for source in subject['sources']:
            if source['role'] not in ('oof0','oof1'):continue
            assert sha(source['condition_cache_path'])==source['condition_cache_sha256'] and sha(source['checkpoint_path'])==source['checkpoint_sha256']
            cache=torch.load(source['condition_cache_path'],map_location='cpu',mmap=True,weights_only=True);assert cache['model_sha256']==source['checkpoint_sha256']
            records=source['groups']['component_train'];dst=np.array([position[r['array_row']] for r in records]);src=np.array([r['cache_row'] for r in records])
            assert np.array_equal(cache['array_rows'][src].numpy(),np.array([r['array_row'] for r in records]));raw[dst]=cache['condition'][src].double();coverage[dst]+=1
            for record in records:source_refs[record['array_row']]=dict(role=source['role'],checkpoint_path=source['checkpoint_path'],checkpoint_SHA256=source['checkpoint_sha256'])
            source_rows.append(dict(role=source['role'],condition_SHA256=source['condition_cache_sha256'],checkpoint_SHA256=source['checkpoint_sha256'],queries=len(records)))
        assert torch.all(coverage==1)
        with torch.no_grad():E=F_native_coordinates(raw,weight,bias);E=E/E.norm(dim=-1,keepdim=True)
        E=E.numpy();correct=np.einsum('ij,ij->i',E,visual)
        a,fit=fit_conditional_tilt(model['weights'],model['concentrations'],E@model['means'].T,correct,1023,max_scale=4096.)
        unused,unused2,z=conditional_parameters(model['weights'],model['concentrations'],E@model['means'].T,a,1023)
        B+=a*E/10;mean_logZ+=z/10;alpha.append(a);subjects.append(subject['subject']);T8_EEG.append(E[positions]);T8_logZ.append(z[positions]);T8_raw.append(raw[positions].numpy());T8_sources.append([source_refs[rows[int(i)]] for i in positions])
        profiles.append(dict(subject=subject['subject'],working_prior_strength=a,MLE=fit,upstream_sources=source_rows,upstream_real_OOF_queries=13230))
        guard();write('progress.json',dict(phase='OOF_EEG_teacher_contexts',completed_subjects=len(subjects),expected_subjects=10,seconds=time.monotonic()-start))
    zbar=log_evidence(B,model);jensen=mean_logZ-zbar;assert np.min(jensen)>-1e-8
    np.savez_compressed(out/'full_T_training_teacher_contexts.npz',array_rows=np.array(rows),evidence=B,barycenter_log_evidence=zbar,KL_Jensen_constant=jensen,sameconcept_donor_index=donor,subject_strengths=np.array(alpha))
    wrong=B[donor[positions]];wrong_z=log_evidence(wrong,model)
    np.savez_compressed(out/'fixed_T8_profile_inputs.npz',positions=positions,array_rows=np.array(rows)[positions],visual_latent_mean=visual[positions],correct_evidence=B[positions],wrong_sameconcept_evidence=wrong,barycenter_log_evidence=zbar[positions],wrong_barycenter_log_evidence=wrong_z,KL_Jensen_constant=jensen[positions],individual_EEG_unit=np.stack(T8_EEG),individual_log_evidence=np.stack(T8_logZ),individual_raw_condition=np.stack(T8_raw),subject_strengths=np.array(alpha))
    metadata=[dict(position=int(i),array_row=rows[int(i)],concept_id=index[int(i)]['concept_id'],stimulus_id=index[int(i)]['stimulus_id'],image_zip=index[int(i)]['image_zip'],image_member=index[int(i)]['image_member'],image_sha256=index[int(i)]['image_sha256'],subject01_source=T8_sources[0][j]) for j,i in enumerate(positions)]
    write('fixed_T8_profile_metadata.json',metadata);write('native_prior_EM_and_subject_strengths.json',dict(EM_history=model['history'],subjects=profiles))
    names=['native_F_working_prior.npz','full_T_training_teacher_contexts.npz','fixed_T8_profile_inputs.npz','fixed_T8_profile_metadata.json','native_prior_EM_and_subject_strengths.json']
    result=dict(status='COMPLETE_NATIVE_F_FULL_T_WORKING_PRIOR_AND_TRAINING_ONLY_EEG_CONTEXTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),T_images=13230,subjects=10,upstream_OOF_conditions=132300,native_F_affine_weight_exactly_one=True,native_F_bias_nonzero=True,intrinsic_dimension=1023,reference_pre_affine_radius=radius,per_row_radius_roundtrip_maxabs=roundtrip,fixed_T_radius_F_reconstruction_maxabs=constant_error,new_visual_density_models=1,density_EM_steps=30,new_EEG_scalar_MLEs=10,new_EEG_models=0,new_EEG_optimizer_updates=0,actual_semantic_teacher_fitted=False,new_images=0,new_GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),NEW_contacted=False,working_prior_empirical_calibration_established=False,full_chain_OOF_not_claimed=True,T_only_reference_radius_no_query_image_at_inference=True,teacher_KL_barycenter_Jensen_min=float(jensen.min()),seconds=time.monotonic()-start,protocol_SHA256=sha(args.protocol),outputs=[dict(name=name,bytes=(out/name).stat().st_size,SHA256=sha(out/name)) for name in names])
    guard();write('completion.json',result);signal.alarm(0);print(json.dumps(result))
except BaseException as error:
    import traceback
    write('failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start,CUDA_initialized=torch.cuda.is_initialized()));raise
