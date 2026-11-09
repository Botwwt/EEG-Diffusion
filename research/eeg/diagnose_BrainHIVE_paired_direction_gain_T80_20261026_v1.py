"""Supplementary frozen T-only direction/amplitude measurements, not a fit.

The original four-arm study and its pass gates are not modified. The ordinary
import is standard-library only; CPU preflight permits an explicitly pending
upstream queue. Real execution requires its completed controller to have exited.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import stat
import sys

import brainhive_paired_study_contract_20261025_v1 as C

SCHEMA='BrainHIVE_paired_direction_gain_T80_20261026_v1'
REGISTERED='REGISTERED_SUPPLEMENTARY_T_ONLY_FROZEN_DIRECTION_GAIN_PROBE'
COMPLETE='COMPLETE_FIXED_PAIRED_DIRECTION_GAIN_T80'
UPSTREAM_COMPLETE='ALL_FOUR_PAIRED_EFFECT_FULL_T_FULL_V_LIFECYCLES_AND_FIXED_COMPARISON_COMPLETE'
EPSILON=1e-6
STUDENT_ENERGY_FLOOR=1e-12
SOURCE_NAMES=(
    Path(__file__).name,'check_BrainHIVE_paired_direction_gain_CPU_20261026_v1.py',
    'brainhive_paired_study_contract_20261025_v1.py',
    'brainhive_paired_effect_integration_20261025_v1.py',
    'diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2.py',
    'brainhive_bounded_component_v1.py','brainhive_component_generator_v1.py',
    'brainhive_native_models_v1.py','brainhive_response_rank_v1.py',
    'brainhive_functional_training_contract_v2.py',
    'run_BrainHIVE_frozen_function_low_memory_20261025_v3.py',
)


def safe_bytes_used(path):
    total=0
    for p in Path(path).rglob('*'):
        try:entry=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(entry.st_mode):total+=entry.st_size
    return total


# This separate process can inspect the live study during CPU-only preflight.
# No pinned scientific module on disk, or running GPU process, is modified.
C.bytes_used=safe_bytes_used


def validate_addon(document):
    assert document['schema']==SCHEMA and document['status']==REGISTERED
    assert document['seed']==1729 and document['batch_size']==2
    assert document['maximum_seconds']==7200
    assert document['maximum_output_bytes']==512<<20
    assert document['maximum_extra_CPU_bytes']==10<<30
    assert document['maximum_GPU_bytes']==20<<30
    assert document['minimum_CPU_headroom_bytes']>=10<<30
    assert Path(document['output_directory'])==C.DATA/'scientific_basis_probe_20261026_v1'
    assert document['gpu_job_name']==Path(document['gpu_job_name']).name
    assert set(SOURCE_NAMES)<=set(document['source_sha256'])
    for name,digest in document['source_sha256'].items():
        assert Path(name).name==name and C.sha(C.ROOT/name)==digest,name
    study=C.bound(document['original_study_registration']);C.validate(study)
    assert study['registration_path']==document['original_study_registration']['path']
    assert document['query_selection']==study['functional_queries']
    assert document['resource_authorization']==study['resource_authorization']
    C.bound(document['resource_authorization'])
    upstream=document['formal_queue_completion']
    assert upstream['expected_status']==UPSTREAM_COMPLETE
    assert Path(upstream['state_path']).is_relative_to(C.ROOT/'audit')
    return study


def status_prefix(path,expected):
    """Verify the producer's first status field without parsing metric fields."""
    with Path(path).open('rb') as stream:
        assert stream.readline(4).strip()==b'{'
        line=stream.readline(512).strip()
    assert line==(json.dumps('status')+': '+json.dumps(expected)+',').encode()


def upstream_state(study,document,allow_pending=False):
    import queue_BrainHIVE_decisive_T_analysis_20261025_v2 as Q
    state=C.read(document['formal_queue_completion']['state_path'])
    assert state['registration']==document['original_study_registration']
    assert state['study_core_sha256']==C.core(study)
    ready=state['status']==UPSTREAM_COMPLETE
    if not ready:
        assert allow_pending and state['status']=='PAIRED_EFFECT_FOUR_ARM_LIFECYCLES_RUNNING'
        return dict(ready=False,status=state['status'],controller_pid=state['pid'],controller_live=Q.live(state['pid']))
    assert not Q.live(state['pid']) and state.get('child_pid') is None and not state.get('unresolved_live_child')
    completed=state['completed']
    for arm in C.ARMS:
        assert {v['stage'] for v in completed if v.get('arm')==arm}=={'train','postfit','generate','audit','features','score'}
    comparison=[v for v in completed if v['stage']=='compare'];assert len(comparison)==1
    reference=comparison[0]['completion'];assert Path(reference['path'])==C.DATA/'comparison/summary.json'
    assert C.sha(reference['path'])==reference['sha256']
    status_prefix(reference['path'],'FOUR_FULL16600_PAIRED_EFFECT_COMPARISONS_COMPLETE')
    return dict(ready=True,status=state['status'],controller_pid=state['pid'],controller_live=False,
        comparison_completion=reference,comparison_metric_fields_parsed=False,
        functional_UNKNOWN_is_not_an_upstream_failure=True)


def choose_probe_pairs(rows,allowed,subjects,training_partner):
    """Keep original T80 identities; exclude both directions of every train edge."""
    assert len(rows)==80 and len(set(allowed))==len(allowed) and len(subjects)==10
    assert len(training_partner)==len(subjects)*len(allowed)
    assert sorted(training_partner)==list(range(len(training_partner)))
    positions={value:i for i,value in enumerate(allowed)}
    subject_positions={value:i for i,value in enumerate(subjects)}
    assert len(subject_positions)==10
    global_indices=[];groups=collections.defaultdict(list)
    for i,row in enumerate(rows):
        assert row['array_row'] in positions and row['subject'] in subject_positions
        global_indices.append(subject_positions[row['subject']]*len(allowed)+positions[row['array_row']])
        groups[(row['subject'],row['source_role'])].append(i)
    assert len(set(global_indices))==80 and len(groups)==20 and all(len(v)==4 for v in groups.values())
    assert set(groups)=={(subject,role) for subject in subjects for role in ('oof0','oof1')}
    def overlap(i,j):
        gi,gj=global_indices[i],global_indices[j]
        return training_partner[gi]==gj,training_partner[gj]==gi
    def validate(mapping,exclude):
        assert len(mapping)==80 and sorted(mapping)==list(range(80))
        for i,j in enumerate(mapping):
            a,b=rows[i],rows[j]
            assert a['subject']==b['subject'] and a['source_role']==b['source_role']
            assert a['concept_id']!=b['concept_id'] and a['array_row']!=b['array_row']
            if exclude:assert not any(overlap(i,j))
    original=[row['negative_query_index'] for row in rows];validate(original,False)
    original_forward=sum(overlap(i,j)[0] for i,j in enumerate(original))
    original_reverse=sum(overlap(i,j)[1] for i,j in enumerate(original))
    chosen=list(original);shifts=[]
    if original_forward or original_reverse:
        chosen=[-1]*80
        for (subject,role),indices in sorted(groups.items()):
            ordered=sorted(indices,key=lambda i:hashlib.sha256(
                ('1729|supplementary_pair_order|'+role+'|'+str(rows[i]['array_row'])).encode()).digest())
            for shift in range(1,len(ordered)):
                candidate={i:ordered[(k+shift)%len(ordered)] for k,i in enumerate(ordered)}
                if all(not any(overlap(i,j)) and rows[i]['concept_id']!=rows[j]['concept_id'] for i,j in candidate.items()):
                    for i,j in candidate.items():chosen[i]=j
                    shifts.append(dict(subject=subject,source_role=role,hash_order=ordered,selected_shift=shift));break
            else:raise AssertionError('No identity-only shift excludes training edges for '+subject+' '+role)
    validate(chosen,True)
    pairs=[dict(query_index=i,paired_query_index=j,global_training_query_index=global_indices[i],
        paired_global_training_query_index=global_indices[j],subject=rows[i]['subject'],source_role=rows[i]['source_role'],
        array_row=rows[i]['array_row'],paired_array_row=rows[j]['array_row'],concept_id=rows[i]['concept_id'],
        paired_concept_id=rows[j]['concept_id'],training_forward_edge=False,training_reverse_edge=False)
        for i,j in enumerate(chosen)]
    return dict(original_permutation=original,probe_permutation=chosen,original_forward_training_edge_overlaps=original_forward,
        original_reverse_training_edge_overlaps=original_reverse,replaced_mapping=bool(shifts),group_shifts=shifts,pairs=pairs,
        final_forward_training_edge_overlaps=0,final_reverse_training_edge_overlaps=0,
        selection_used_only_identities=True,selection_used_features_or_scores=False,
        queries=80,subjects=10,unique_stimuli=len({r['array_row'] for r in rows}))


def prepare_rows(study,document):
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    protocol=C.bound(study['input_bindings']['split']);contract=C.bound(study['input_bindings']['input_contract'])
    manifest=study['input_bindings']['image_manifest'];assert C.sha(manifest['path'])==manifest['sha256']==protocol['image_manifest_sha256']
    allowed=protocol['image_rows']['T'];assert len(allowed)==13230;allowed_set=set(allowed)
    with Path(manifest['path']).open(encoding='utf-8',newline='') as stream:
        metadata={int(r['array_row']):r for r in csv.DictReader(stream) if r['dataset_split']=='train' and int(r['array_row']) in allowed_set}
    rows=D.choose_rows(protocol,metadata,contract)
    assert rows==C.bound(document['query_selection'])['rows']
    plan,plan_sha=C.prepare_plan(study)
    pairing=choose_probe_pairs(rows,allowed,[r['subject'] for r in contract['subjects']],plan['partner'])
    assert pairing['unique_stimuli']==8
    return rows,pairing,plan_sha


def inspect_finals(study,require_all=True):
    result={}
    for arm in C.ARMS:
        root=C.DATA/'arms'/arm/'train';path=root/'completion.json'
        if not path.exists():
            assert not require_all,'Missing full final '+arm
            continue
        done=C.read(path)
        assert done['status']=='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE' and done['arm']==arm
        assert done['study_core_sha256']==C.core(study) and done['updates']==24807
        assert done['primary_exposures']==done['paired_exposures']==396900 and done['parameters']==525568
        assert Path(done['checkpoint']['path'])==root/'component_final.pt'
        assert C.sha(done['checkpoint']['path'])==done['checkpoint']['sha256']
        calibration=C.bound(done['calibration']);provenance=C.bound(done['provenance'])
        assert calibration['kind']=='Dfit_only_global_strength_calibration' and calibration['selected_strength'] in (0,.25,.5,.75,1)
        assert provenance['study_core_sha256']==C.core(study) and provenance['arm']==arm
        result[arm]=dict(completion=C.ref(path),checkpoint=done['checkpoint'],calibration=done['calibration'],
            provenance=done['provenance'],selected_strength=calibration['selected_strength'])
    return result


def preflight(document,allow_upstream_pending=False):
    study=validate_addon(document);C.require_admission(study)
    upstream=upstream_state(study,document,allow_pending=allow_upstream_pending)
    rows,pairing,plan_sha=prepare_rows(study,document)
    finals=inspect_finals(study,require_all=upstream['ready'])
    assert not Path(document['output_directory']).exists(),'Preserve previous probe output, no automatic replay'
    return dict(status='SUPPLEMENTARY_T_ONLY_DIRECTION_GAIN_CPU_PREFLIGHT_PASSED',upstream=upstream,
        study_core_sha256=C.core(study),pair_plan_sha256=plan_sha,query_selection=document['query_selection'],
        pairing=pairing,available_finals=finals,queries=len(rows),unique_stimuli=8,subjects=10,
        GPU_used=False,new_fit=False,original_formal_gates_changed=False)


def response_measures(si,sj,tp,tn,torch):
    """All means are per response coordinate; these are basic algebra, not theory."""
    assert si.ndim==4 and si.shape==sj.shape==tp.shape==tn.shape
    si,sj,tp,tn=[v.float() for v in (si,sj,tp,tn)];axes=(1,2,3)
    finite=torch.stack([torch.isfinite(v).all(axes) for v in (si,sj,tp,tn)]).all(0)
    ds,dt=si-sj,tp-tn
    es,et=ds.square().mean(axes),dt.square().mean(axes);dot=(ds*dt).mean(axes)
    scale=et+EPSILON;teacher_valid=finite&(et>EPSILON);direction_valid=teacher_valid&(es>STUDENT_ENERGY_FLOOR)
    root_product=es.clamp_min(0).sqrt()*et.clamp_min(0).sqrt()
    mse=(ds-dt).square().mean(axes)
    amplitude_error=(es.clamp_min(0).sqrt()-et.clamp_min(0).sqrt()).square()/scale
    angular_error=2*(root_product-dot)/scale
    ei,ej=si-tp,sj-tn
    common=(.5*(ei+ej)).square().mean(axes)/scale
    contrast=.25*mse/scale;absolute=.5*(ei.square().mean(axes)+ej.square().mean(axes))/scale
    if bool(finite.any()):
        assert torch.allclose((amplitude_error+angular_error)[finite],(mse/scale)[finite],rtol=2e-5,atol=2e-5)
        assert torch.allclose(absolute[finite],(common+contrast)[finite],rtol=2e-5,atol=2e-5)
    values=dict(teacher_difference_energy=et,student_difference_energy=es,difference_dot=dot,raw_difference_MSE=mse,
        cosine=(dot/root_product.clamp_min(1e-30)).clamp(-1,1),
        projection_gain=dot/es.clamp_min(1e-30),student_along_teacher_gain=dot/et.clamp_min(1e-30),
        amplitude_ratio=(es/et.clamp_min(1e-30)).clamp_min(0).sqrt(),
        normalized_difference_MSE=mse/scale,amplitude_error=amplitude_error,angular_error=angular_error,
        pair_gap=2*dot/scale,common=common,contrast=contrast,absolute=absolute)
    masks={k:(finite if k in ('teacher_difference_energy','student_difference_energy','difference_dot','raw_difference_MSE')
        else direction_valid if k in ('cosine','projection_gain') else teacher_valid) for k in values}
    return values,masks,dict(finite_inputs=finite,teacher_valid=teacher_valid,direction_valid=direction_valid,
        teacher_degenerate=finite&(et<=EPSILON),student_degenerate=finite&(es<=STUDENT_ENERGY_FLOOR))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--allow-upstream-pending',action='store_true')
    args=parser.parse_args();assert not args.allow_upstream_pending or args.preflight_only
    document=C.read(args.registration);checked=preflight(document,allow_upstream_pending=args.allow_upstream_pending)
    if args.preflight_only:
        print(json.dumps(checked));return
    study=validate_addon(document);assert checked['upstream']['ready'];C.validate(study,all_inputs=True)
    C.setup_runtime(study)
    import torch
    import gzip
    import signal
    import time
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native,denoise
    from brainhive_response_rank_v1 import NativeIPResponseCapture,ip_response
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    D.torch=torch;torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    wrapper=D.verify_wrapper(document);budget=D.Budget(document);output=Path(document['output_directory'])
    output.mkdir(exist_ok=False);collector=None;started=time.monotonic()
    def stop(signum,frame):raise InterruptedError('Supplementary probe received signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        C.atomic(output/'preflight.json',checked);C.atomic(output/'launch.json',dict(registration=C.ref(args.registration),wrapper=wrapper))
        recipe_path=C.ROOT/'configs/functional_full_lifecycle_20261022_v1/O.json';recipe=C.read(recipe_path)
        assert recipe['input_contract_sha256']==study['input_bindings']['input_contract']['sha256']
        assert recipe['latent_receipt']==study['input_bindings']['latent_receipt']['path']
        assert C.sha(recipe['latent_receipt'])==study['input_bindings']['latent_receipt']['sha256']
        rows,inputs,provenance=D.selected_inputs(recipe,budget)
        assert rows==C.bound(document['query_selection'])['rows']
        pairing=checked['pairing'];permutation=torch.tensor(pairing['probe_permutation'],device='cuda')
        C.atomic(output/'queries_and_pairing.json',dict(original_rows=rows,pairing=pairing,input_provenance=provenance))
        finals=inspect_finals(study);heads={};head_hashes={}
        for arm in C.ARMS:
            item=finals[arm];state=torch.load(item['checkpoint']['path'],map_location='cpu',weights_only=True)
            assert state['arm']==arm and state['study_core_sha256']==C.core(study) and state['updates']==24807
            assert state['seen_queries']==396900 and state['seed']==1729 and state['passes']==3
            assert state['provenance_sha256']==item['provenance']['sha256'] and state['pair_plan_sha256']==checked['pair_plan_sha256']
            assert state['calibration']['selected_strength']==item['selected_strength']
            head=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
            head.load_state_dict(state['model'],strict=True);heads[arm]=head;head_hashes[arm]=D.module_digest(head)
            del state
        loader=output/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
        try:pipe,adapter,projector,prompts,host=load_native(training=False)
        finally:audit.restore()
        proof=audit.complete();budget.check();collector=NativeIPResponseCapture(pipe.unet)
        modules={name:module for name,module in pipe.components.items() if isinstance(module,torch.nn.Module)}
        modules.update(A=adapter,P=projector)
        assert len(collector.layers)==70
        assert all(not p.requires_grad for module in [*modules.values(),*heads.values()] for p in module.parameters())
        host_before={name:D.module_digest(module) for name,module in modules.items()}
        from diffusers import DDPMScheduler
        ddpm=DDPMScheduler.from_config(pipe.scheduler.config);pipe.scheduler.set_timesteps(4,device='cuda')
        timesteps=pipe.scheduler.timesteps.clone();assert timesteps.tolist()==[999.,749.,499.,249.]
        radii=('training','deployment');bank={}
        with torch.no_grad():
            parent=D.encode_tokens(adapter,projector,inputs['condition']);teachers=D.encode_tokens(adapter,projector,inputs['target'])
            for arm,head in heads.items():
                for radius in radii:
                    head.radius=.25*(1 if radius=='training' else finals[arm]['selected_strength'])
                    changed,_=head(inputs['condition'],inputs['weight'],inputs['bias'],gate_mode='fixed')
                    bank[(arm,radius)]=D.encode_tokens(adapter,projector,changed)
        cells={};counts=collections.Counter();record_count=0
        def measure(domain,ti,indices,queries,stream):
            nonlocal record_count
            with torch.no_grad():
                for layer,(attention,processor) in collector.layers.items():
                    q=queries[layer];assert not q.requires_grad
                    tp=ip_response(q,teachers[indices],processor,attention.heads)
                    tn=ip_response(q,teachers[permutation[indices]],processor,attention.heads)
                    for arm in C.ARMS:
                        for radius in radii:
                            tokens=bank[(arm,radius)]
                            si=ip_response(q,tokens[indices],processor,attention.heads)
                            sj=ip_response(q,tokens[permutation[indices]],processor,attention.heads)
                            metrics,masks,flags=response_measures(si,sj,tp,tn,torch)
                            for offset,index in enumerate(indices.tolist()):
                                key=(domain,arm,radius,index);cell=cells.setdefault(key,dict(n=0,metrics={}))
                                cell['n']+=1;record=dict(domain=domain,arm=arm,radius=radius,query_index=index,
                                    paired_query_index=pairing['probe_permutation'][index],timestep_index=ti,layer=layer)
                                for flag,mask in flags.items():
                                    good=bool(mask[offset]);record[flag]=good;counts[flag]+=int(good)
                                for metric,values in metrics.items():
                                    value=float(values[offset]);defined=bool(masks[metric][offset]) and math.isfinite(value)
                                    record[metric]=value if defined else None
                                    entry=cell['metrics'].setdefault(metric,dict(sum=0.,defined=0))
                                    if defined:entry['sum']+=value;entry['defined']+=1
                                budget.append(stream,record);record_count+=1
        raw=output/'all_query_layer_direction_gain.jsonl.gz'
        with gzip.open(raw,'xb',compresslevel=3) as stream:
            for start in range(0,80,2):
                budget.check();indices=torch.arange(start,start+2,device='cuda')
                with torch.no_grad():
                    trajectory=D.native_parent_states(pipe,prompts,parent[indices],[r['trajectory_seed'] for r in rows[start:start+2]],collector)
                    for ti,timestep in enumerate(timesteps):
                        generator=torch.Generator(device='cuda').manual_seed(D.stable_seed('DDPM_noise',start,ti))
                        noise=torch.randn(inputs['latent'][indices].shape,device='cuda',generator=generator)
                        noisy=ddpm.add_noise(inputs['latent'][indices],noise,timestep.long().expand(2))
                        with collector.capture():denoise(pipe,prompts,noisy,timestep,parent[indices])
                        queries=dict(collector.queries);collector.queries.clear()
                        measure('DDPM_real_T',ti,indices,queries,stream)
                        measure('native_parent_trajectory',ti,indices,trajectory[ti]['queries'],stream)
                    del trajectory,queries
                print(json.dumps(dict(stage='SUPPLEMENTARY_DIRECTION_GAIN_T80',queries=start+2,expected=80)),flush=True)
        assert record_count==2*4*2*80*4*70 and len(cells)==2*4*2*80
        assert all(cell['n']==280 for cell in cells.values())
        query_rows=[];aggregate={}
        for (domain,arm,radius,index),cell in sorted(cells.items()):
            result=dict(domain=domain,arm=arm,radius=radius,query_index=index,records=280,metrics={})
            for name,entry in cell['metrics'].items():
                result['metrics'][name]=dict(masked_full_denominator_mean=entry['sum']/280,
                    defined_only_mean=entry['sum']/entry['defined'] if entry['defined'] else None,
                    defined_records=entry['defined'],undefined_records=280-entry['defined'])
                value=aggregate.setdefault(domain,{}).setdefault(radius,{}).setdefault(arm,{}).setdefault(name,dict(sum=0.,defined=0,total=0))
                value['sum']+=entry['sum'];value['defined']+=entry['defined'];value['total']+=280
            query_rows.append(result)
        for domains in aggregate.values():
            for arms in domains.values():
                for metrics in arms.values():
                    for name,value in list(metrics.items()):
                        metrics[name]=dict(masked_full_denominator_mean=value['sum']/value['total'],
                            defined_only_mean=value['sum']/value['defined'] if value['defined'] else None,
                            defined_records=value['defined'],undefined_records=value['total']-value['defined'],full_denominator=value['total'])
        C.atomic(output/'per_query_summary.json',dict(rows=query_rows))
        assert head_hashes=={a:D.module_digest(h) for a,h in heads.items()}
        assert host_before=={name:D.module_digest(module) for name,module in modules.items()}
        assert all(not p.requires_grad and p.grad is None for module in [*modules.values(),*heads.values()] for p in module.parameters())
        for item in finals.values():assert C.sha(item['checkpoint']['path'])==item['checkpoint']['sha256']
        budget.check();assert safe_bytes_used(output)<=document['maximum_output_bytes']-(1<<20)
        C.atomic(output/'completion.json',dict(status=COMPLETE,study_core_sha256=C.core(study),registration=C.ref(args.registration),
            queries=80,subjects=10,unique_stimuli=8,shared_stimuli_are_not_80_independent_stimuli=True,
            state_domains=2,timesteps=4,layers=70,radii=list(radii),raw_query_layer_records=record_count,missing_states=0,
            finite_response_records=counts['finite_inputs'],nonfinite_response_records=record_count-counts['finite_inputs'],
            teacher_degenerate_records=counts['teacher_degenerate'],student_degenerate_records=counts['student_degenerate'],
            direction_defined_records=counts['direction_valid'],direction_undefined_records=record_count-counts['direction_valid'],
            teacher_energy_floor=EPSILON,student_direction_energy_floor=STUDENT_ENERGY_FLOOR,
            undefined_values_are_null_not_zero=True,full_denominator_preserved=True,aggregate=aggregate,
            pairing=C.ref(output/'queries_and_pairing.json'),raw_states=C.ref(raw),per_query_summary=C.ref(output/'per_query_summary.json'),
            trained_finals=finals,head_hashes=head_hashes,frozen_host_hashes=host_before,loader_proof=proof,host=host,
            selected_input_recipe=C.ref(recipe_path),
            state_definitions=dict(DDPM_real_T='Forward-noised allowed T latent, parent EEG reference Q; image information is present in the state',
                native_parent_trajectory='Original four-step parent sampling with EEG and fixed noise only; teacher/target not used in state updates'),
            metric_definitions=dict(projection_gain='dot/student_difference_energy: best scalar multiplying student to fit teacher; undefined at or below declared student direction energy floor',
                student_along_teacher_gain='dot/teacher_difference_energy: student amplitude along teacher; zero for zero student with valid teacher',
                amplitude_error='(sqrt(student_energy)-sqrt(teacher_energy))^2/(teacher_energy+1e-6)',
                angular_error='2*(sqrt(student_energy)*sqrt(teacher_energy)-dot)/(teacher_energy+1e-6)',
                normalized_difference_MSE='amplitude_error + angular_error; not multiplied by 1/4',
                masked_full_denominator_mean='Undefined observations contribute zero only for full-denominator accounting; inspect counts and defined-only mean together'),
            original_formal_gates_changed=False,new_pass_gate=False,interpretation='DESCRIPTIVE_T_ONLY_MEASUREMENTS_NO_NEW_METHOD_OR_CAUSAL_SUCCESS_DECISION',
            optimizer_created=False,parameter_updates=0,inference_images_generated=0,V_scores_read=False,
            postfit_T_not_independent_confirmation=True,peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
            peak_GPU_bytes=budget.peak_GPU_bytes,uncompressed_record_bytes=budget.uncompressed_bytes,seconds=time.monotonic()-started))
        print(COMPLETE,flush=True)
    except BaseException as error:
        if not (output/'failure.json').exists():C.atomic(output/'failure.json',dict(status='SUPPLEMENTARY_PROBE_FAILED_PRESERVED',error=repr(error),automatic_retry=False))
        raise
    finally:
        if collector is not None:collector.close()


if __name__=='__main__':main()
