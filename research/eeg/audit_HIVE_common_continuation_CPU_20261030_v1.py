"""Actual manual-SGD state, parameter-fraction and full independent forwards."""
import argparse,hashlib,os,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from brainhive_common_continuation_contract_CPU_20261030_v1 import ARMS,FRACTIONS,atomic,checked,read,ref,spec,image_decision
def fit(s,path):
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import torch,numpy as np
    from scipy.special import erf
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_data_v1 import load_training
    from train_BrainHIVE_bounded_component_v1 import seed
    torch.set_num_threads(2);assert not torch.cuda.is_initialized();started=time.monotonic();root=Path(s['output_root']);out=root/'CPU_audit';out.mkdir(exist_ok=False)
    data=load_training('cpu');assert data['input_contract_sha256']==s['input_contract']['sha256'];original=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model']
    orders=[torch.randperm(132300,generator=torch.Generator().manual_seed(seed('pass'+str(i)))) for i in range(3)];hashes=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders]
    regenerated=torch.zeros(3,132300,dtype=torch.bool)
    for i,x in enumerate(orders):assert torch.unique(x).numel()==132300;regenerated[i,x]=True
    assert hashes==read(checked(s['starting_F_provenance']))['orders_sha256'];profile=read(root/'fit_profile/completion.json')
    d=read(checked(s['D09_completion']));di=read(checked(d['inputs']));queries=read(checked(di['panel']))['Dcheck'];raw=np.load(checked(d['check_conditions']),allow_pickle=False)
    cache=torch.load(checked(di['native_cache']),weights_only=True,map_location='cpu',mmap=True);parent=torch.load(checked(di['native_checkpoint']),weights_only=True,map_location='cpu')['model'];positions={int(v):i for i,v in enumerate(cache['array_rows'])}
    native=cache['condition'][[positions[q['array_row']] for q in queries]].numpy().astype(np.float64);assert np.array_equal(native.astype(np.float32),raw[:,0]);wg=parent['block.layers.0.norm.weight'].numpy().astype(np.float64)[None,:];bg=parent['block.layers.0.norm.bias'].numpy().astype(np.float64)[None,:]
    def independent(c,w,b,p):
        u=(c-b)/w;r2=np.sum(u*u,axis=1,keepdims=True);x=u@p['trunk.0.weight'].T+p['trunk.0.bias'];x=.5*x*(1+erf(x/np.sqrt(2)));d=x@p['direction.weight'].T+p['direction.bias'];d-=d.mean(1,keepdims=True);d-=u*np.sum(d*u,axis=1,keepdims=True)/r2
        delta=.25*np.sqrt(r2)*d/np.sqrt(r2+np.sum(d*d,axis=1,keepdims=True));scale=np.sqrt(r2)/np.linalg.norm(u+delta,axis=1,keepdims=True);return c+w*((scale-1)*u+scale*delta)
    reports={}
    for arm in ARMS:
        done=read(root/arm/'completion.json');assert done['study']==ref(path) and done['updates']==24807 and done['query_exposures']==396900
        state=torch.load(checked(done['checkpoint']),weights_only=False,map_location='cpu');rawlast=torch.load(checked(done['raw_last']),weights_only=True,map_location='cpu');final=torch.load(checked(done['component']),weights_only=True,map_location='cpu');cal=read(checked(done['calibration']));prov=read(checked(done['provenance']))
        assert state['arm']==rawlast['arm']==final['arm']==arm and state['seed']==rawlast['seed']==final['seed']==1729
        assert state['updates']==rawlast['updates']==final['updates']==24807 and state['seen']==rawlast['seen_queries']==final['seen_queries']==396900
        assert torch.equal(state['coverage'],regenerated) and state['coverage'].dtype==torch.bool
        assert state['SGD']['kind']=='MANUAL_NORMALIZED_SGD' and state['SGD']['momentum']==0 and state['SGD']['iterations']==24807 and state['SGD']['parameter_step']==s['fixed_parameter_step']
        assert 0<=state['SGD']['nonzero_steps']==done['nonzero_steps']<=24807 and 'optimizer' not in state
        assert state['orders_sha256']==prov['orders_sha256']==hashes and state['torch_rng'].dtype==torch.uint8 and all(x.dtype==torch.uint8 for x in state['cuda_rng'])
        assert prov['initial_checkpoint']==final['initial_checkpoint']==s['ordinary'] and prov['initial_tensor_sha256']==profile['initial_tensor_sha256'] and final['radius']==.25
        assert final['raw_last']==done['raw_last'] and cal==final['calibration'] and cal['fractions']==list(FRACTIONS) and cal['radius']==.25 and cal['parameter_interpolation_not_radius_scaling']
        assert final['total_historical_updates']==49614 and final['total_historical_exposures']==793800
        j=min(range(5),key=lambda i:(cal['Dfit_token_errors'][i],i));fraction=FRACTIONS[j];assert fraction==cal['selected_parameter_fraction']==done['selected_parameter_fraction']
        for k,v in state['model'].items():
            assert torch.equal(v,rawlast['model'][k]) and torch.isfinite(v).all()
            want=original[k].clone() if fraction==0 else v.clone() if fraction==1 else original[k]+(v-original[k])*fraction
            assert torch.equal(want,final['model'][k]) and torch.isfinite(want).all()
        log=[__import__('json').loads(x) for x in (root/arm/'training.jsonl').read_text().splitlines()];assert log[-1]['updates']==24807 and log[-1]['seen_queries']==396900
        assert all(x['updates']<y['updates'] for x,y in zip(log,log[1:]));assert log[-1]['nonzero_steps']==done['nonzero_steps']
        identity=all(torch.equal(final['model'][k],original[k]) for k in original);assert identity==done['selected_parameters_exact_original_ordinary']
        m=BoundedConditionAdapter().eval().requires_grad_(False);m.load_state_dict(final['model'],strict=True);assert m.radius==.25;p={k:v.numpy().astype(np.float64) for k,v in final['model'].items()};maximum=scaled=error2=actual2=0.
        with torch.inference_mode():
            for begin in range(0,132300,512):
                sl=slice(begin,begin+512);c=data['condition'][sl];aid=data['affine_ids'][sl];w=data['weight'][aid];b=data['bias'][aid];actual=m(c,w,b)[0].numpy();ideal=independent(c.numpy().astype(np.float64),w.numpy().astype(np.float64),b.numpy().astype(np.float64),p)
                err=actual.astype(np.float64)-ideal;maximum=max(maximum,float(abs(err).max()));scaled=max(scaled,float((abs(err)/(1+abs(ideal))).max()));error2+=float((err*err).sum());actual2+=float((actual.astype(np.float64)**2).sum())
        relative=(error2/actual2)**.5;assert scaled<2e-6 and relative<1e-5
        expected=independent(native,wg,bg,p);dest=out/f'expected400_{arm}.npy';np.save(dest,expected,allow_pickle=False)
        reports[arm]=dict(component=done['component'],raw_last=done['raw_last'],full_T_forward_queries=132300,additional_updates=24807,primary_query_exposures=396900,selected_parameter_fraction=fraction,
            selected_parameters_exact_original_ordinary=identity,maximum_absolute_error=maximum,scaled_max_error=scaled,relative_L2_error=relative,expected400=ref(dest),
            SGD_state_coverage_orders_RNG_and_last_log_verified=True,parameter_interpolation_bit_exact=True,calibration_AP_aggregates_not_independently_reexecuted=True)
    checked(s['ordinary']);checked(s['starting_F']);assert not torch.cuda.is_initialized()
    atomic(out/'fit_completion.json',dict(status='PASS_ALL4_FULL_OOF_STATE_AND_INDEPENDENT_HEAD_EQUATIONS',study=ref(path),arms=reports,full_T_CPU_forwards=529200,coverage_exposures=1587600,
        primary_selected_parameters_exact_original_ordinary=reports['joint_same']['selected_parameters_exact_original_ordinary'],original_Euclidean_fullT_numeric_failure_unchanged=True,
        forward_numeric_limits=dict(scaled_max=2e-6,relative_L2=1e-5),future_trajectory_bit_equivalence_not_claimed=True,seconds=time.monotonic()-started))
def images(s,path):
    import numpy as np
    from PIL import Image
    start=time.monotonic();root=Path(s['output_root']);im=root/'images';e=read(Path(path).parent/'image_execution.json')
    d=read(checked(s['D09_completion']));di=read(checked(d['inputs']));queries=read(checked(di['panel']))['Dcheck']
    oldrows=[json.loads(x) for x in checked(d['check_manifest']).read_text().splitlines()];base={(r['query_id'],r['arm']):r for r in oldrows}
    oldinodes={(Path(r['generated_path']).stat().st_dev,Path(r['generated_path']).stat().st_ino) for r in oldrows}
    stages=[n for n in (80,400) if (im/f'score{n}.json').exists()];n=max(stages);paths=set();inodes=set();count=0
    for arm in ARMS:
        bank=np.load(im/f'conditions400_{arm}.npy',allow_pickle=False);expected=np.load(root/'CPU_audit'/f'expected400_{arm}.npy',allow_pickle=False)
        assert bank.shape==expected.shape==(400,1024);err=bank.astype(np.float64)-expected;assert np.max(np.abs(err)/(1+np.abs(expected)))<2e-6
        for i in range(n):
            r=read(im/'records'/f'{i:03d}_{arm}.json');q=queries[i];native=base[(q['query_id'],'parent')]
            assert all(r[k]==v for k,v in q.items()) and r['arm']==arm and r['execution']==ref(Path(path).parent/'image_execution.json')
            assert r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280 and not r['normal_inference_target_access']
            for key in ('initial_UNet_input_sha256','initial_noise_latents_sha256','ancestral_noise_observations'):assert r[key]==native[key]
            assert hashlib.sha256(bank[i:i+1].tobytes()).hexdigest()==r['condition_sha256']
            assert hashlib.sha256(bank[i:i+1].astype(np.float16).tobytes()).hexdigest()==r['condition_FP16_sha256']
            p=Path(r['generated_path']);assert p.resolve().is_relative_to(im.resolve()) and str(p.resolve()) not in paths;paths.add(str(p.resolve()))
            inode=(p.stat().st_dev,p.stat().st_ino);assert inode not in inodes|oldinodes;inodes.add(inode);assert ref(p)['sha256']==r['generated_sha256']
            with Image.open(p) as picture:assert picture.size==(512,512) and hashlib.sha256(np.asarray(picture.convert('RGB')).tobytes()).hexdigest()==r['generated_pixel_sha256']
            count+=1
    target=np.load(checked(d['check_target_features']),allow_pickle=False).astype(np.float64);target/=np.linalg.norm(target,axis=1,keepdims=True)
    old=np.load(checked(d['check_generated_features']),allow_pickle=False);starting=np.load(checked(e['starting_F_features400']),allow_pickle=False);errors=[];raw_cells=0
    def measure(features,n):
        f=features.astype(np.float64);f/=np.linalg.norm(f,axis=1,keepdims=True);sims=f@target.T;result={k:np.empty(n) for k in ('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')}
        for i in range(n):
            row=sims[i];truth=row[i];local=[j for j,q in enumerate(queries) if q['concept_id']==queries[i]['concept_id']];assert len(local)==10
            result['target_cosine'][i]=truth;result['DINO2AFC'][i]=(sum(row<truth)+.5*(sum(row==truth)-1))/399;result['gallery_top1'][i]=int(np.argmax(row)==i)
            near=row[local];result['within_concept10_top1'][i]=int(local[int(np.argmax(near))]==i);result['within_concept10_two_afc'][i]=(sum(near<truth)+.5*(sum(near==truth)-1))/9
        return result
    for n in stages:
        score=read(im/f'score{n}.json');metrics={a:measure(old[:n,j],n) for j,a in enumerate(('parent','ordinary','risk'))};metrics['starting_F']=measure(starting[:n],n)
        metrics.update({a:measure(np.load(im/f'features{n}_{a}.npy',allow_pickle=False),n) for a in ARMS});raw=np.load(im/f'all_query_values{n}.npz',allow_pickle=False)
        groups=[[i for i in range(n) if queries[i]['concept_id']==c] for c in sorted({q['concept_id'] for q in queries[:n]})];ng=len(groups);draw=np.random.default_rng(1729).integers(0,ng,size=(1000,ng));weights=np.asarray([np.bincount(row,minlength=ng) for row in draw])/ng
        for a,fields in metrics.items():
            for key,x in fields.items():
                errors.append(abs(x.mean()-score['means'][a][key]));errors.append(float(np.max(np.abs(x-raw[a+'__'+key]))));raw_cells+=len(x)
        for a in ARMS:
            for b in metrics:
                if a==b:continue
                for key in metrics[a]:
                    delta=metrics[a][key]-metrics[b][key];cm=np.asarray([delta[g].mean() for g in groups]);interval=np.quantile(weights@cm,[.025,.975]);v=score['contrasts'][a][b][key]
                    errors.append(abs(delta.mean()-v['difference']));errors.extend(np.abs(interval-v['descriptive_concept_bootstrap95']).tolist());errors.append(float(np.max(np.abs(delta-raw[a+'_minus_'+b+'__'+key]))));raw_cells+=len(delta)
        assert score['decision']==image_decision(score['contrasts'],n)
    assert max(errors)<1e-12 and count==4*max(stages)
    atomic(root/'CPU_audit/images_completion.json',dict(status='PASS_ALL_ACTUAL_IMAGES_AND_INDEPENDENT_RANK_BOOTSTRAP_RAW_CELLS',stages=stages,images=count,unique_paths=len(paths),unique_inodes=len(inodes),raw_metric_and_contrast_cells=raw_cells,
        statistic_checks=len(errors),maximum_error=max(errors),all_denominators_retained=True,independent_visual_reencoding=False,independent_generation_reexecution=False,seconds=time.monotonic()-start))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',choices=['fit','images'],required=True);a=p.parse_args();s=spec(a.study)
    if a.mode=='fit':fit(s,a.study)
    else:images(s,a.study)
