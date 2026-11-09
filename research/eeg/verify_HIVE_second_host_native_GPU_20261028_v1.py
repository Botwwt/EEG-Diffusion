"""Four engineering images on the additional host; no fitting or efficacy read.

Replays the historical fixed T query through native / zero / disabled paths,
then separately exercises the immutable trained ordinary at strength one.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parent
PERSISTENT=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
WORK=Path('/root/brainhive_reproduction_20261020_v1')
START=time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def memory():
    return {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/self/status').read_text().splitlines()
            if line.startswith(('VmRSS:','VmHWM:'))}


BASE=memory()['VmRSS']


def wrapper(name):
    parent=Path('/proc')/str(os.getppid());argv=(parent/'cmdline').read_bytes().split(b'\0')
    assert str(ROOT/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py').encode() in argv
    assert argv[argv.index(b'--name')+1].decode()==name
    lock=ROOT/'audit/GPU_JOB.lock';identity=lock.stat()
    descriptors=[]
    for descriptor in (parent/'fd').iterdir():
        try:v=descriptor.stat();descriptors.append((v.st_dev,v.st_ino))
        except FileNotFoundError:pass
    assert (identity.st_dev,identity.st_ino) in descriptors
    with lock.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(stream,fcntl.LOCK_UN);raise AssertionError('Wrapper must own exclusive GPU lock')
    path=ROOT/'audit'/(name+'_job.json')
    for _ in range(100):
        if path.exists():break
        time.sleep(.05)
    job=read(path)
    assert job['status']=='RUNNING' and job['pid']==os.getpid() and job['wrapper_pid']==os.getppid()
    assert job['maximum_seconds']==900 and job['GPU_UUID']=='GPU-62a1aee8-4740-dc59-93f4-d50644db650d'
    assert job['authorization']['sha256']=='d167cdd82209cbd8bf61bc5747f01489dcf750fc445567e6703ffcfa77c831b4'
    return job


def run(spec_path,output):
    spec=read(spec_path)
    assert spec['schema']=='ADDITIONAL_HOST_NATIVE_HIVE_ENGINEERING_V1'
    for name,digest in spec['source_sha256'].items():assert Path(name).name==name and sha(ROOT/name)==digest,name
    assert output==Path(spec['output']) and not output.exists();output.mkdir(parents=True)
    installed=read(spec['installation_completion'])
    assert installed['status']=='ADDITIONAL_HOST_NATIVE_HIVE_RUNTIME_INSTALLED_AND_STRICT_CPU_PRIORS_VERIFIED'
    assert installed['transfer_plan_sha256']=='1d845a56007089dd5683cf50e7ecb353ed512816570226a29af671b4472fa8ba'
    for item in installed['per_file_SHA']:assert sha(item['path'])==item['sha256'],item['path']
    original=WORK/'audit/functional_T_pixel_replay_v2'
    provenance=original/'input_provenance.json'
    assert sha(provenance)=='b94ba05eba3f99a73d06e02ac0b0cb403f5e6f0d404405b68350e1ee9a793b09'
    prior=read(provenance);binding=prior['evidence']['binding'];source=binding['source']
    assert binding['query_id']=='sub-01/moose_05s.jpg/R4' and binding['query']['array_row']==9024
    assert binding['image_seed']==5533825044790129053
    for path_key,sha_key in [('condition_cache_path','condition_cache_sha256'),('checkpoint_path','checkpoint_sha256')]:
        assert sha(source[path_key])==source[sha_key]
    old_summary=read(original/'summary.json')
    assert old_summary['status']=='FUNCTIONAL_T_PIXEL_REPLAY_V2_EXACT_THREE_PATHS_PASSED'
    old_parent=next(x for x in old_summary['cases'] if x['case']=='parent')
    assert sha(old_parent['generated_path'])==old_parent['generated_sha256']=='adef964d4a122b8f73caf58b47f8adb6977b87d3c2a0439c068fa66ff4eaaffd'
    import brainhive_paired_study_contract_20261025_v1 as C
    manifest=ROOT/'audit/diagnostic_memory_recovery_20261025_v3/isolated_runtime_manifest.json'
    C.setup_runtime(dict(runtime_manifest=dict(path=str(manifest),sha256=sha(manifest))))
    import torch
    import numpy as np
    from PIL import Image
    from brainhive_component_generator_v1 import load_native
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 import sample_observed,save_image
    wrapper_proof=wrapper(spec['GPU_job_name'])
    torch.set_num_threads(2);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not torch.cuda.is_initialized()
    exported=torch.load(source['condition_cache_path'],map_location='cpu',weights_only=True,mmap=True)
    position=binding['cache_row'];assert int(exported['array_rows'][position])==9024
    assert exported['model_sha256']==source['checkpoint_sha256']
    condition=exported['condition'][position:position+1].clone()
    assert condition.shape==(1,1024) and torch.isfinite(condition).all()
    assert hashlib.sha256(condition.contiguous().numpy().tobytes()).hexdigest()==prior['condition_sha256']
    state=torch.load(source['checkpoint_path'],weights_only=True,map_location='cpu')['model']
    weight=state['block.layers.0.norm.weight'][None].cuda();bias=state['block.layers.0.norm.bias'][None].cuda()
    condition=condition.cuda()
    frozen=read(ROOT/'configs/ordinary_frozen_base_20261028_v1.json')
    assert sha(frozen['checkpoint']['path'])==frozen['checkpoint']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
    assert sha(frozen['source']['path'])==frozen['source']['sha256']=='2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7'
    trained=torch.load(frozen['checkpoint']['path'],weights_only=True,map_location='cpu')
    assert trained['calibration']['selected_strength']==frozen['selected_strength']==1
    ordinary=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    ordinary.load_state_dict(trained['model'],strict=True)
    zero=BoundedConditionAdapter(predict_risk=False).cuda().eval().requires_grad_(False)
    assert sum(p.numel() for p in ordinary.parameters())==525568
    (output/'converted_weights').mkdir()
    observer=ConvertedWeightAudit(torch,output/'converted_weights');observer.install()
    try:pipe,adapter,projector,prompts,host=load_native(training=False)
    finally:observer.restore()
    loading=observer.complete()
    def budget():
        assert time.monotonic()-START<850
        assert memory()['VmHWM']-BASE<=10<<30
        assert torch.cuda.max_memory_allocated()<=20<<30
        assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file())<32<<20
    budget();records=[]
    with Image.open(old_parent['generated_path']) as image:reference=np.asarray(image.convert('RGB')).copy()
    with torch.inference_mode():
        corrected=ordinary(condition,weight,bias,gate_mode='fixed')[0]
        assert not torch.equal(corrected,condition),'Disabled test must use an otherwise active trained head'
        conditions=[('native',condition),('zero_head',zero(condition,weight,bias,gate_mode='fixed')[0]),
                    ('explicitly_disabled',ordinary(condition,weight,bias,enabled=False)[0]),('frozen_ordinary',corrected)]
        for name,value in conditions:
            budget();native=value.half();tokens=projector(adapter(native))
            assert tokens.shape==(1,4,2048) and torch.isfinite(tokens).all()
            image,proof=sample_observed(pipe,prompts,tokens,binding['image_seed'],torch)
            pixels=np.asarray(image.convert('RGB'));saved=save_image(output/'images'/(name+'.webp'),image)
            row=dict(case=name,query_id=binding['query_id'],image_seed=binding['image_seed'],
                     condition_sha256=hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest(),
                     native_A_input_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                     IP_token_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                     **proof,**saved)
            if name!='frozen_ordinary':
                row['historical_RGB_max_absolute_difference']=int(np.abs(pixels.astype(np.int16)-reference.astype(np.int16)).max())
                row['historical_native_pixel_exact']=row['historical_RGB_max_absolute_difference']==0
                row['historical_A_token_noise_exact']=all(row[key]==old_parent[key] for key in ('native_A_input_sha256','IP_token_sha256','initial_UNet_input_sha256'))
                assert torch.equal(value,condition)
            records.append(row);write(output/(name+'_receipt.json'),row);budget()
    all_exact=all(x.get('historical_native_pixel_exact',True) and x.get('historical_A_token_noise_exact',True) for x in records)
    assert all(x['initial_UNet_input_sha256']==old_parent['initial_UNet_input_sha256'] for x in records)
    for name,digest in spec['source_sha256'].items():assert sha(ROOT/name)==digest
    assert sha(frozen['checkpoint']['path'])==frozen['checkpoint']['sha256']
    value=dict(status='ADDITIONAL_HOST_HIVE_NATIVE_PIXEL_AND_FROZEN_ORDINARY_ENGINEERING_PASSED' if all_exact else 'ADDITIONAL_HOST_HIVE_HISTORICAL_PIXEL_DRIFT_PRESERVED',
               UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec_sha256=sha(spec_path),
               installation_completion_sha256=sha(spec['installation_completion']),wrapper=wrapper_proof,
               exact_meta_load=loading,host=host,cases=records,historical_three_paths_all_exact=all_exact,
               actual_engineering_images=4,actual_UNet_NFE=16,actual_ancestral_noises_observed=16,
               new_training_updates=0,new_research_images=0,Dfit_Dcheck_V_or_official_targets_read=False,
               no_image_gain_claim=True,ordinary_weights_and_strength_preserved=True,
               peak_extra_CPU_bytes=memory()['VmHWM']-BASE,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
               seconds=time.monotonic()-START)
    write(output/'completion.json',value)
    print(json.dumps({k:value[k] for k in ['status','actual_engineering_images','historical_three_paths_all_exact','seconds','peak_extra_CPU_bytes','peak_GPU_allocated_bytes']}),flush=True)
    assert all_exact,'Historical pixel drift retained; do not admit efficacy based on engineering mismatch'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);a=p.parse_args()
    spec=read(a.spec);output=Path(spec['output'])
    assert output.is_relative_to(PERSISTENT/'native_HIVE_runtime_20261028_v1')
    try:run(a.spec,output)
    except BaseException as error:
        if output.is_dir():write(output/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-START,new_training_updates=0))
        raise


if __name__=='__main__':main()
