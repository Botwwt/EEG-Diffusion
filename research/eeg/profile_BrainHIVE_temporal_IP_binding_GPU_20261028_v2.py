"""Fixed T engineering images for D37. No model fitting or efficacy selection."""
import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parent
P=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
W=Path('/root/brainhive_reproduction_20261020_v1')
START=time.monotonic()


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))


def check(v):
    p=Path(v['path']);assert sha(p)==v['sha256']
    if 'bytes' in v:assert p.stat().st_size==v['bytes']
    return p


def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run(spec_path):
    spec=read(spec_path);assert spec['schema']=='D37_REAL_UNET_TEMPORAL_IP_ENGINEERING_V1'
    for item in spec['source_pins']:check(item)
    output=Path(spec['output']);assert not output.exists();output.mkdir(parents=True)
    cpu=read(check(spec['CPU_completion']))
    assert cpu['status']=='D37_REAL_T_AND_ALL70_NATIVE_IP_CPU_ENGINEERING_COMPLETE'
    assert cpu['additional_learned_parameters']==0
    import brainhive_paired_study_contract_20261025_v1 as C
    manifest=ROOT/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json'
    C.setup_runtime(dict(runtime_manifest=dict(path=str(manifest),sha256=sha(manifest))))
    import torch
    import numpy as np
    from PIL import Image
    from brainhive_component_generator_v1 import load_native
    from brainhive_native_models_v1 import prior_models
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_temporal_conditional_attention_20261028_v1 import temporal_tangents,frozen_prior_temporal_tokens
    from brainhive_temporal_IP_binding_20261028_v2 import TemporalContext,bind_temporal_IP,sample_temporal_observed as sample_observed
    from verify_HIVE_second_host_native_GPU_20261028_v1 import wrapper,memory
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import save_image
    wrapper_proof=wrapper(spec['GPU_job_name'])
    baseline_memory=memory()['VmRSS'];torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.is_initialized()
    provenance=W/'audit/functional_T_pixel_replay_v2/input_provenance.json'
    assert sha(provenance)=='b94ba05eba3f99a73d06e02ac0b0cb403f5e6f0d404405b68350e1ee9a793b09'
    source=read(provenance);binding=source['evidence']['binding'];origin=binding['source']
    assert binding['query_id']=='sub-01/moose_05s.jpg/R4' and binding['query']['array_row']==9024
    for pk,sk in [('condition_cache_path','condition_cache_sha256'),('checkpoint_path','checkpoint_sha256')]:
        assert sha(origin[pk])==origin[sk]
    cache=torch.load(origin['condition_cache_path'],map_location='cpu',weights_only=True,mmap=True)
    pos=binding['cache_row'];assert int(cache['array_rows'][pos])==9024
    c=cache['condition'][pos:pos+1].clone();assert c.shape==(1,1024)
    assert hashlib.sha256(c.numpy().tobytes()).hexdigest()==source['condition_sha256']
    state=torch.load(origin['checkpoint_path'],map_location='cpu',weights_only=True)['model']
    cpu_spec=read(check(cpu['spec']));d33=read(check(cpu_spec['D33_spec']))
    index=[json.loads(x) for x in check(d33['T_index']).read_text().splitlines()]
    selected=[row for row in index if row['array_row']==9024]
    assert len(selected)==1 and selected[0]['partition']=='T'
    entry=selected[0];trials=np.load(check(d33['T_trials']),mmap_mode='r',allow_pickle=False)
    x=torch.tensor(np.asarray(trials[entry['row']:entry['row']+1]).mean(axis=1,dtype=np.float32).reshape(1,4250))
    c_manual,temporal,_=temporal_tangents(x,state)
    difference=float((c_manual-c).abs().max());assert difference<2e-4
    frozen=read(check(cpu_spec['ordinary_frozen_spec']))
    trained=torch.load(check(frozen['checkpoint']),map_location='cpu',weights_only=True)
    assert trained['calibration']['selected_strength']==1
    ordinary_CPU=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
    ordinary_CPU.load_state_dict(trained['model'],strict=True)
    models=prior_models('cpu',include_attention=False)
    gamma=state['block.layers.0.norm.weight'][None];beta=state['block.layers.0.norm.bias'][None]
    def prior(value):return models['proj'](models['fusion_encoder'].adapter(ordinary_CPU(value,gamma,beta,gate_mode='fixed')[0]))
    _,detail,bound=frozen_prior_temporal_tokens(c,temporal.detach(),prior)
    assert torch.isfinite(detail).all() and detail.abs().max()>0
    detail=detail.detach();token_geometry=dict(detail_RMS=float(detail.square().mean().sqrt()),
        detail_sum_max=float(detail.sum(2).abs().max()),time_bin_edges=[0,50,100,150,200,250],
        native_condition_vs_cache_max_error=difference,native_checkpoint=origin['checkpoint_path'],
        native_checkpoint_sha256=origin['checkpoint_sha256'],EEG_source=entry)
    del models,ordinary_CPU,prior,cache,x,trials;gc.collect()
    ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    ordinary.load_state_dict(trained['model'],strict=True)
    c=c.cuda();gamma=gamma.cuda();beta=beta.cuda();detail=detail.cuda()
    with torch.no_grad():corrected=ordinary(c,gamma,beta,gate_mode='fixed')[0]
    (output/'converted_weights').mkdir();observer=ConvertedWeightAudit(torch,output/'converted_weights');observer.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:observer.restore()
    loading=observer.complete()
    frozen_before={name:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                   for name,t in pipe.unet.state_dict().items()}
    params=sum(p.numel() for p in pipe.unet.parameters());context=TemporalContext();records=[]
    reference=read(check(spec['native_engineering_completion']))
    old=next(row for row in reference['cases'] if row['case']=='frozen_ordinary')
    assert sha(old['generated_path'])==old['generated_sha256']
    with Image.open(old['generated_path']) as image:reference_pixels=np.asarray(image.convert('RGB')).copy()
    def budget():
        assert time.monotonic()-START<850
        assert memory()['VmHWM']-baseline_memory<10<<30
        assert torch.cuda.max_memory_allocated()<20<<30
        assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<64<<20
    with torch.no_grad():tokens=projector(adapter(corrected.half()))
    original=bind_temporal_IP(pipe.unet,context)
    cases=[('disabled',None,'conditional',False),('zero_conditional',torch.zeros_like(detail),'conditional',True),
           ('zero_flat',torch.zeros_like(detail),'flat_residual',True),('conditional',detail,'conditional',True),
           ('flat_residual',detail,'flat_residual',True)]
    try:
        with torch.inference_mode():
            for name,value,mode,enabled in cases:
                context.configure(value,mode,enabled);budget();case_start=time.monotonic()
                image,proof=sample_observed(pipe,prompts,tokens,binding['image_seed'],torch)
                pixels=np.asarray(image.convert('RGB'));saved=save_image(output/'images'/(name+'.webp'),image)
                row=dict(case=name,query_id=binding['query_id'],image_seed=binding['image_seed'],mode=mode,
                    enabled=enabled,**proof,**saved,seconds=time.monotonic()-case_start,
                    baseline_pixel_max_difference=int(np.abs(pixels.astype(np.int16)-reference_pixels.astype(np.int16)).max()),
                    active_IP_layers=len(context.calls),total_active_IP_calls=sum(context.calls.values()),
                    actual_layer_diagnostics=context.records)
                assert row['UNet_NFE']==4 and row['initial_UNet_input_sha256']==old['initial_UNet_input_sha256']
                assert row['ancestral_noise_observations']==old['ancestral_noise_observations']
                assert not enabled or len(context.calls)==70 and set(context.calls.values())=={4}
                if name in ['disabled','zero_conditional','zero_flat']:
                    assert row['baseline_pixel_max_difference']==0
                if name=='conditional':
                    assert max(v['native_marginal_max_error'] for v in context.records.values())<2e-6
                    assert max(v['delta_RMS'] for v in context.records.values())>0
                records.append(row);write(output/(name+'_receipt.json'),row);budget()
    finally:pipe.unet.set_attn_processor(dict(original))
    assert sum(p.numel() for p in pipe.unet.parameters())==params
    for name,t in pipe.unet.state_dict().items():
        assert hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()==frozen_before[name],name
    for item in spec['source_pins']:check(item)
    assert sha(check(frozen['checkpoint']))==frozen['checkpoint']['sha256']
    value=dict(status='D37_ACTUAL_UNET_BINDING_AND_THREE_PIXEL_FALLBACKS_PASSED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),CPU_completion=spec['CPU_completion'],
        wrapper=wrapper_proof,host=host,converted_weights=loading,geometry=token_geometry,cases=records,
        additional_learned_parameters=0,extra_UNet_calls_per_image=0,UNet_NFE_per_image=4,
        extra_attention_projections_and_CPU_JVP_cost_present=True,ordinary_and_all_UNet_bytes_unchanged=True,
        actual_engineering_images=len(records),new_research_images=0,SGD_updates=0,
        no_Dfit_Dcheck_V_or_official_image_values_read=True,no_image_gain_or_novelty_established=True,
        seconds=time.monotonic()-START,peak_extra_CPU_bytes=memory()['VmHWM']-baseline_memory,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    budget();write(output/'completion.json',value)
    print(json.dumps({key:value[key] for key in ['status','seconds','actual_engineering_images','peak_extra_CPU_bytes','peak_GPU_allocated_bytes']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    try:run(a.spec)
    except BaseException as error:
        out=Path(read(a.spec)['output'])
        if out.is_dir():write(out/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-START))
        raise
