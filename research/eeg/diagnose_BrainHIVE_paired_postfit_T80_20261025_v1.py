"""Fixed postfit T80 same-Q paired effects; no optimizer, images or V scores."""
import argparse
import gzip
import json
from pathlib import Path
import time
import brainhive_paired_study_contract_20261025_v1 as C


def response_measures(si,sj,tp,tn,torch):
    axes=(1,2,3);si,sj,tp,tn=[v.float() for v in (si,sj,tp,tn)]
    separation=(tp-tn).square().mean(axes);valid=separation>1e-6
    scale=separation+1e-6
    ei,ej=si-tp,sj-tn
    common=(.5*(ei+ej)).square().mean(axes)/scale
    contrast=.25*((si-sj)-(tp-tn)).square().mean(axes)/scale
    absolute=.5*(ei.square().mean(axes)+ej.square().mean(axes))/scale
    assert torch.allclose(absolute,common+contrast,rtol=2e-5,atol=2e-5)
    return dict(pair_gap=2*((si-sj)*(tp-tn)).mean(axes)/scale,
        common=common,contrast=contrast,absolute=absolute),valid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registration',type=Path,required=True)
    args=parser.parse_args();config=C.read(args.registration);core=C.validate(config,all_inputs=True);C.require_admission(config)
    C.setup_runtime(config)
    import torch
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native,denoise
    from brainhive_response_rank_v1 import NativeIPResponseCapture,ip_response
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    D.torch=torch;torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    output=C.DATA/'functional_T80';output.mkdir(exist_ok=False)
    wrapper,budget=C.verify_wrapper(config,config['functional_job_name'],7200,output)
    # Numerical/input helpers are reused, never the old v2 trained heads or
    # its scientific decision/admission. The original T80 selection is exact.
    recipe=C.read(C.ROOT/'configs/functional_full_lifecycle_20261022_v1/O.json')
    rows,inputs,provenance=D.selected_inputs(recipe,budget)
    old=C.bound(config['functional_queries'])['rows']
    assert rows==old
    heads={};calibration={};files={}
    for arm in C.ARMS:
        done=C.read(C.DATA/'arms'/arm/'train/completion.json')
        assert done['status']=='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE' and done['study_core_sha256']==core and done['updates']==24807
        assert done['primary_exposures']==done['paired_exposures']==396900
        payload=torch.load(done['checkpoint']['path'],map_location='cpu',weights_only=True)
        assert C.sha(done['checkpoint']['path'])==done['checkpoint']['sha256']
        files[arm]=done['checkpoint'];head=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
        head.load_state_dict(payload['model'],strict=True);heads[arm]=head;calibration[arm]=payload['calibration']['selected_strength']
    hashes_before={a:D.module_digest(h) for a,h in heads.items()}
    loader=output/'loader';loader.mkdir();audit=ConvertedWeightAudit(torch,loader);audit.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:audit.restore()
    proof=audit.complete();collector=NativeIPResponseCapture(pipe.unet)
    from diffusers import DDPMScheduler
    ddpm=DDPMScheduler.from_config(pipe.scheduler.config);pipe.scheduler.set_timesteps(4,device='cuda')
    timesteps=pipe.scheduler.timesteps.clone();assert timesteps.tolist()==[999.,749.,499.,249.]
    permutation=torch.tensor([r['negative_query_index'] for r in rows],device='cuda')
    bank={};radii=('training','deployment')
    with torch.no_grad():
        parent=D.encode_tokens(adapter,projector,inputs['condition'])
        teachers=D.encode_tokens(adapter,projector,inputs['target'])
        for arm,head in heads.items():
            for radius in radii:
                head.radius=.25*(1. if radius=='training' else calibration[arm])
                value,_=head(inputs['condition'],inputs['weight'],inputs['bias'],gate_mode='fixed')
                bank[(arm,radius)]=D.encode_tokens(adapter,projector,value)
    sums={};valid_count=0;total_count=0
    def measure(domain,ti,indices,queries,stream):
        nonlocal valid_count,total_count
        with torch.no_grad():
            for layer,(attention,processor) in collector.layers.items():
                q=queries[layer]
                tp=ip_response(q,teachers[indices],processor,attention.heads)
                tn=ip_response(q,teachers[permutation[indices]],processor,attention.heads)
                for arm in C.ARMS:
                    for radius in radii:
                        tokens=bank[(arm,radius)]
                        si=ip_response(q,tokens[indices],processor,attention.heads)
                        sj=ip_response(q,tokens[permutation[indices]],processor,attention.heads)
                        metrics,valid=response_measures(si,sj,tp,tn,torch)
                        for offset,index in enumerate(indices.tolist()):
                            good=bool(valid[offset]);key=(domain,arm,radius,index)
                            cell=sums.setdefault(key,dict(n=0,valid=0,pair_gap=0.,common=0.,contrast=0.,absolute=0.))
                            cell['n']+=1;cell['valid']+=int(good)
                            record=dict(domain=domain,arm=arm,radius=radius,query_index=index,timestep_index=ti,layer=layer,valid=good)
                            for metric,value in metrics.items():
                                number=float(value[offset]);assert torch.isfinite(value[offset])
                                if good:cell[metric]+=number
                                record[metric]=number if good else None
                            stream.write((json.dumps(record)+'\n').encode());valid_count+=int(good);total_count+=1
    try:
        with gzip.open(output/'all_query_layer_states.jsonl.gz','xb',compresslevel=3) as stream:
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
                print(json.dumps(dict(stage='FIXED_POSTFIT_T80',queries=start+2,expected=80)),flush=True)
        assert len(sums)==2*4*2*80 and all(v['n']==280 for v in sums.values())
        minimum=min(v['valid']/280 for v in sums.values())
        aggregate={}
        for domain in ('DDPM_real_T','native_parent_trajectory'):
            aggregate[domain]={}
            for radius in radii:
                aggregate[domain][radius]={arm:{metric:sum(sums[(domain,arm,radius,i)][metric]/280 for i in range(80))/80
                    for metric in ('pair_gap','common','contrast','absolute')} for arm in C.ARMS}
        differences={domain:{radius:{other:aggregate[domain][radius]['D']['pair_gap']-aggregate[domain][radius][other]['pair_gap']
            for other in ('O','A','U')} for radius in radii} for domain in aggregate}
        all_positive=all(v>0 for d in differences.values() for r in d.values() for v in r.values())
        result='UNKNOWN' if minimum<.9 else ('SUPPORTED' if all_positive else 'NOT_SUPPORTED')
        assert {a:D.module_digest(h) for a,h in heads.items()}==hashes_before
        for item in files.values():assert C.sha(item['path'])==item['sha256']
        budget.check()
        C.atomic(output/'queries.json',dict(rows=rows,input_provenance=provenance))
        C.atomic(output/'summary.json',dict(status='COMPLETE_FIXED_POSTFIT_PAIRED_T80',study_core_sha256=core,
            queries=80,unique_stimuli=8,subjects=10,state_domains=2,timesteps=4,layers=70,
            raw_query_layer_records=total_count,missing_states=0,minimum_valid_fraction=minimum,
            aggregate=aggregate,D_pair_gap_minus_controls=differences,
            paired_projection_increment=dict(result=result,rule='D greater than O/A/U at both training and deployed radius, in both domains; directional, not significance'),
            inference_images_generated=0,optimizer_created=False,parameter_updates=0,V_scores_read=False,
            postfit_T_not_independent_confirmation=True,teacher_noise_UNKNOWN_gate_unchanged=True,
            original_helper_source_sha256=C.sha(Path(D.__file__)),loader_proof=proof,host=host,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes))
        print('COMPLETE_FIXED_POSTFIT_PAIRED_T80',flush=True)
    finally:collector.close()


if __name__=='__main__':main()
