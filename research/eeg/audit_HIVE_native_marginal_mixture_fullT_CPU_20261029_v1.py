"""Concept-heldout visual mixture plus true-OOF EEG evidence, CPU only."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True);args=parser.parse_args()
protocol=json.loads(args.protocol.read_bytes())
assert protocol['components']==32 and protocol['EM_iterations']==30 and protocol['scalar_tilt_max']==4096.
sys.path[:0]=[protocol['source_root'],protocol['host_source_root']]
import brainhive_paired_study_contract_20261025_v1 as C
C.setup_runtime(dict(runtime_manifest=dict(path=protocol['runtime_manifest'],sha256=protocol['required_sha256'][protocol['runtime_manifest']])) )
import numpy as np
import torch
from scipy.special import logsumexp
torch.set_num_threads(2)
from brainhive_spherical_predictive_density_CPU_20261029_v1 import vmf_log_normalizer_array
from brainhive_marginal_mixture_fit_CPU_20261029_v1 import fit_tied_vmf_mixture,conditional_parameters,fit_conditional_tilt,component_coverage

root=Path(protocol['output_root']);start=time.monotonic()
def deadline(signum,frame):raise TimeoutError('Fixed fullT CPU diagnostic deadline')
signal.signal(signal.SIGALRM,deadline);signal.alarm(protocol['max_seconds'])
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(name,value):
    path=root/name;pending=path.with_name(path.name+'.pending');pending.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');pending.replace(path)
def guard():
    assert not torch.cuda.is_initialized() and time.monotonic()-start<protocol['max_seconds']-5
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available>4*1024**3
    assert sum(path.stat().st_size for path in root.iterdir() if path.is_file())<128*1024**2
def progress(**extra):write('progress.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,CUDA_initialized=torch.cuda.is_initialized(),**extra))
def bootstrap(values):
    assert values.shape==(10,1323,10)
    cells=values.mean(axis=2);rng=np.random.default_rng(1729)
    estimates=np.array([cells[np.ix_(rng.integers(10,size=10),rng.integers(1323,size=1323))].mean() for unused in range(1000)])
    return dict(mean=float(values.mean()),descriptive_subject_concept_95=np.quantile(estimates,[.025,.975]).tolist(),per_subject=cells.mean(axis=1).tolist())

try:
    assert not (root/'completion.json').exists()
    for path,digest in protocol['required_sha256'].items():assert sha(path)==digest,path
    for name,digest in protocol['sources_sha256'].items():assert sha(Path(protocol['source_root'])/name)==digest
    contract=json.loads(Path(protocol['input_contract']).read_bytes());split=json.loads(Path(protocol['split_manifest']).read_bytes())
    assert contract['status']=='ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    rows=split['image_rows']['T'];position={row:i for i,row in enumerate(rows)};assert len(rows)==13230
    entries=[json.loads(line) for line in Path(protocol['T_index']).read_text().splitlines()]
    assert len(entries)==13230 and {entry['array_row'] for entry in entries}==set(rows)
    byrow={entry['array_row']:entry for entry in entries};assert all(entry['partition']=='T' for entry in entries)
    concepts=sorted({byrow[row]['concept_id'] for row in rows});assert len(concepts)==1323
    groups={concept:sorted(row for row in rows if byrow[row]['concept_id']==concept) for concept in concepts};assert all(len(group)==10 for group in groups.values())
    fold={concept:int(hashlib.sha256(('HIVE_directional_density_v1|'+str(concept)).encode()).hexdigest()[:8],16)%2 for concept in concepts}
    density_fold=np.array([fold[byrow[row]['concept_id']] for row in rows])
    order=np.array([position[row] for concept in concepts for row in groups[concept]])
    target_all=torch.load(contract['target_cache_path'],map_location='cpu',mmap=True,weights_only=True);assert sha(contract['target_cache_path'])==contract['target_cache_sha256']
    visual=target_all[rows].numpy().astype(np.float64);visual/=np.linalg.norm(visual,axis=1,keepdims=True)
    assert visual.shape==(13230,1024)
    role_rows={s['role']:[record['array_row'] for record in s['groups']['component_train']] for s in contract['subjects'][0]['sources'] if s['role'].startswith('oof')}
    assert set(role_rows)=={'oof0','oof1'} and set(role_rows['oof0']).isdisjoint(role_rows['oof1'])
    models=[];reports=[];lookup={};F_component_cosines=np.full((13230,32),np.nan);model_id=np.full(13230,-1,dtype=np.int16)
    for role in ('oof0','oof1'):
        role_positions=np.array([position[row] for row in role_rows[role]])
        for heldout in (0,1):
            guard();fit=role_positions[density_fold[role_positions]!=heldout];check=role_positions[density_fold[role_positions]==heldout]
            model=fit_tied_vmf_mixture(visual[fit],components=32,iterations=30)
            identifier=len(models);lookup[(role,heldout)]=identifier;models.append(model)
            F_component_cosines[check]=visual[check]@model['means'].T;model_id[check]=identifier
            reports.append(dict(model_id=identifier,role=role,heldout_density_fold=heldout,fit_queries=len(fit),check_queries=len(check),components=32,tied_kappa=float(model['concentrations'][0]),history=model['history'],fit_array_rows_SHA256=hashlib.sha256(np.array(rows,dtype=np.int64)[fit].tobytes()).hexdigest(),check_array_rows_SHA256=hashlib.sha256(np.array(rows,dtype=np.int64)[check].tobytes()).hexdigest()))
            progress(phase='visual_marginal_EM',completed_models=len(models),expected_models=4)
    assert np.all(model_id>=0) and np.isfinite(F_component_cosines).all()
    metrics=np.full((10,13230,12),np.nan);EEG_component_cosines=np.full((10,13230,32),np.nan);coverage=np.zeros((10,13230),dtype=np.int8);source_reports=[]
    for subject_index,subject in enumerate(contract['subjects']):
        for source in subject['sources']:
            if source['role'] not in ('oof0','oof1'):continue
            guard();assert sha(source['condition_cache_path'])==source['condition_cache_sha256'];assert sha(source['checkpoint_path'])==source['checkpoint_sha256']
            cache=torch.load(source['condition_cache_path'],map_location='cpu',mmap=True,weights_only=True)
            assert cache['model_sha256']==source['checkpoint_sha256']
            records=source['groups']['component_train'];source_rows=[record['array_row'] for record in records]
            assert source_rows==role_rows[source['role']]
            cache_rows=np.array([record['cache_row'] for record in records]);dest=np.array([position[row] for row in source_rows])
            assert np.array_equal(cache['array_rows'][cache_rows].numpy(),source_rows)
            EEG=cache['condition'][cache_rows].numpy().astype(np.float64);EEG/=np.linalg.norm(EEG,axis=1,keepdims=True)
            native_cosine=np.einsum('ij,ij->i',EEG,visual[dest])
            concept_indices=collections.defaultdict(list)
            for i,row in enumerate(source_rows):concept_indices[byrow[row]['concept_id']].append(i)
            donor=np.arange(len(source_rows))
            for indices in concept_indices.values():
                ordered=sorted(indices,key=lambda i:source_rows[i]);assert len(ordered)==10;donor[ordered]=np.roll(ordered,-1)
            donor_cosine=np.einsum('ij,ij->i',EEG[donor],visual[dest])
            record=dict(subject=subject['subject'],role=source['role'],condition_SHA256=source['condition_cache_sha256'],checkpoint_SHA256=source['checkpoint_sha256'],query_count=len(dest),folds=[])
            for heldout in (0,1):
                guard();fit=density_fold[dest]!=heldout;check=~fit;identifier=lookup[(source['role'],heldout)];model=models[identifier]
                E_cos=EEG@model['means'].T
                alpha,fit_report=fit_conditional_tilt(model['weights'],model['concentrations'],E_cos[fit],native_cosine[fit],1024,max_scale=4096.)
                unused,unused2,log_z=conditional_parameters(model['weights'],model['concentrations'],E_cos,alpha,1024)
                F_cos=F_component_cosines[dest[check]]
                logits=np.log(model['weights'])+vmf_log_normalizer_array(1024,model['concentrations'])+model['concentrations']*F_cos
                marginal=logsumexp(logits,axis=1)
                conditional=marginal+alpha*native_cosine[check]-log_z[check]
                donor_score=marginal+alpha*donor_cosine[check]-log_z[donor][check]
                PIT,quantile_record,details=component_coverage(model['weights'],model['concentrations'],E_cos[check],F_cos,native_cosine[check],alpha,1024)
                entropy=-np.sum(details['posterior_weights']*np.log(np.maximum(details['posterior_weights'],1e-300)),axis=1)
                responsibility_entropy=-np.sum(details['responsibilities']*np.log(np.maximum(details['responsibilities'],1e-300)),axis=1)
                metrics[subject_index,dest[check]]=np.stack((marginal,conditional,conditional-marginal,conditional-donor_score,PIT,np.zeros(check.sum()),np.zeros(check.sum()),native_cosine[check],donor_cosine[check],entropy,responsibility_entropy,np.full(check.sum(),alpha)),axis=1)
                EEG_component_cosines[subject_index,dest[check]]=E_cos[check]
                coverage[subject_index,dest[check]]+=1
                record['folds'].append(dict(heldout_density_fold=heldout,model_id=identifier,fit_queries=int(fit.sum()),check_queries=int(check.sum()),EEG_tilt_scale=alpha,MLE=fit_report,component_PIT_quantiles=quantile_record,mean_component_PIT_95_coverage=float(PIT.mean())))
            source_reports.append(record);progress(phase='EEG_conditional_tilt',completed_sources=len(source_reports),expected_sources=20)
    assert np.all(coverage==1) and np.isfinite(metrics).all() and np.isfinite(EEG_component_cosines).all()
    rectangle=metrics[:,order].reshape(10,1323,10,12)
    baseline=np.load(protocol['single_component_baseline_npz'],allow_pickle=False)
    assert np.array_equal(baseline['array_rows'],np.array(rows)[order])
    rectangle[...,5]=rectangle[...,1]-baseline['values'][...,0]
    rectangle[...,6]=rectangle[...,0]-baseline['values'][...,1]
    summaries={name:bootstrap(rectangle[...,index]) for name,index in [('conditional_minus_matching_marginal_nats',2),('correct_minus_sameconcept_donor_nats',3),('latent_component_PIT_nominal95_coverage',4),('conditional_mixture_minus_single_vMF_nats',5),('marginal_mixture_minus_single_vMF_nats',6)]}
    score_support=all(summaries[name]['descriptive_subject_concept_95'][0]>0 for name in ('conditional_minus_matching_marginal_nats','correct_minus_sameconcept_donor_nats','marginal_mixture_minus_single_vMF_nats'))
    lower,upper=summaries['latent_component_PIT_nominal95_coverage']['descriptive_subject_concept_95'];calibrated=bool(lower<=.95<=upper)
    np.savez_compressed(root/'all132300_T_predictions.npz',values=rectangle,array_rows=np.array(rows)[order],concept_ids=np.array([str(c) for c in concepts]),columns=np.array(['mixture_marginal_logscore','mixture_conditional_logscore','conditional_minus_matching_marginal','correct_minus_donor','component_PIT_coverage95','conditional_minus_single_vMF','marginal_minus_single_vMF','native_EEG_target_cosine','donor_EEG_target_cosine','conditional_component_entropy','component_responsibility_entropy','EEG_tilt_scale']))
    np.savez_compressed(root/'all132300_sufficient_statistics.npz',EEG_component_cosines=EEG_component_cosines[:,order].reshape(10,1323,10,32),visual_component_cosines=F_component_cosines[order].reshape(1323,10,32),model_id=model_id[order].reshape(1323,10),weights=np.stack([m['weights'] for m in models]),means=np.stack([m['means'] for m in models]),concentrations=np.stack([m['concentrations'] for m in models]))
    write('visual_mixture_EM_and_scope.json',reports);write('all20_conditional_source_reports.json',source_reports)
    import resource
    result=dict(status='COMPLETE_TRUE_OOF_FULL10_T_FIXED32_VMF_MIXTURE_DENSITY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),queries=132300,subjects=10,concepts=1323,T_stimuli=13230,visual_density_models=4,visual_EM_M_steps=120,EEG_scalar_MLE_fits=40,new_EEG_models=0,new_EEG_optimizer_updates=0,new_images=0,new_GPU_calls=0,CUDA_initialized=torch.cuda.is_initialized(),NEW_contacted=False,summaries=summaries,nested_EEG_logscore_and_marginal_structure_support=bool(score_support),latent_component_PIT_calibration_supported=calibrated,observed_space_95_credible_region_not_claimed=True,generation_or_full_teacher_fit_admitted=False,action='RECORD_FIXED_MODEL_DENSITY_EVIDENCE_NO_AUTOMATIC_GPU',seconds=time.monotonic()-start,peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,protocol_SHA256=sha(args.protocol),outputs=[dict(name=name,bytes=(root/name).stat().st_size,SHA256=sha(root/name)) for name in ('all132300_T_predictions.npz','all132300_sufficient_statistics.npz','visual_mixture_EM_and_scope.json','all20_conditional_source_reports.json')])
    guard();write('completion.json',result);signal.alarm(0);print(json.dumps(result))
except BaseException as error:
    import traceback
    write('failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),seconds=time.monotonic()-start,CUDA_initialized=torch.cuda.is_initialized()));raise
