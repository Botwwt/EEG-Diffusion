"""Independent full training, inference, identity and image arithmetic checks."""
import argparse,datetime,hashlib,json,os,time
from pathlib import Path
from brainhive_paired_semantic_study_contract_CPU_20261030_v1 import ARMS,IMAGE_ARMS,atomic,checked,read,ref,scan_checkpoints,image_decision

def checksum(x):return hashlib.sha256(x.tobytes()).hexdigest()
def tree_finite(x,torch):
    if isinstance(x,torch.Tensor):assert x.device.type=='cpu' and torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():tree_finite(v,torch)
    elif isinstance(x,(tuple,list)):
        for v in x:tree_finite(v,torch)
def loss_values(g,v,arm,np):
    g=g.astype('float64');v=v.astype('float64');g/=np.linalg.norm(g,axis=-1,keepdims=True);v/=np.linalg.norm(v,axis=-1,keepdims=True)
    if arm=='joint_swapped_targets':v=v[:,::-1]
    effect=((g[:,0]-g[:,1])*(v[:,0]-v[:,1])).sum(-1);absolute=1-(g*v).sum(-1).mean(-1);pairing=.07*np.logaddexp(0,-effect/.07)
    return dict(loss=absolute+(0 if arm=='joint_absolute_only' else pairing),absolute=absolute,pairing=pairing,effect=effect)
def same(a,b,torch):
    assert set(a)==set(b)
    return all(torch.equal(a[n],b[n]) for n in a)

def fit(path,s,torch,np):
    root=Path(s['output_root']);cache=read(checked(s['cache_completion']));pairs=read(checked(cache['pairs']));order=np.load(checked(s['training_order']),allow_pickle=False)
    targets=np.load(checked(cache['target_features']),allow_pickle=False,mmap_mode='r');original=torch.load(checked(s['parent']),map_location='cpu',weights_only=True)['model']
    import brainhive_native_models_v1 as N
    initial_P={n:p.half().float() for n,p in N.prior_models('cpu')['proj'].state_dict().items()};affine=('block.layers.0.norm.weight','block.layers.0.norm.bias');noise_reference=[];results={};maximum=0.
    for arm in ARMS:
        folder=root/arm;completion=read(folder/'completion.json');assert completion['updates']==6615 and completion['registration']==ref(path)
        committed=scan_checkpoints(folder,path,arm,order,pairs,all_state_hashes=True);assert committed[-1]['end_update']==6615
        noises=[]
        for c in committed:
            p=Path(c['directory']);rows=[json.loads(x) for x in (p/'training_log.jsonl').read_text().splitlines()];g=np.load(p/'generated_features.npy',allow_pickle=False);assert g.shape==(len(rows),2,768) and np.isfinite(g).all()
            v=targets[np.array([r['T_indices'] for r in rows])];values=loss_values(g,v,arm,np)
            for k,value in values.items():
                error=float(np.max(np.abs(value-np.array([r[k] for r in rows]))));maximum=max(maximum,error);assert error<2e-6
            for row in rows:
                seed=int(hashlib.sha256(f"1729|{row['pair_index']}|fixed_pair_semantic_v1".encode()).hexdigest()[:16],16)%(2**63-1);assert row['seed']==seed
                assert row['gradients']['EEG']['updated']==(arm!='P_only_correct') and row['gradients']['P']['updated']==(arm!='EEG_only_correct')
                noises.append((row['pair_index'],row['proof']['initial'],row['proof']['ancestral']))
            state=torch.load(p/'state.pt',map_location='cpu',weights_only=False);tree_finite(state,torch)
            assert np.array_equal(state['order'],order) and state['updates']==c['end_update'] and state['log_sha256']==c['log_sha256'] and state['arm']==arm
            assert state['CPU_RNG'].numel()>0 and len(state['CUDA_RNG'])==1 and state['CUDA_RNG'][0].numel()>0
            assert all(torch.equal(state['EEG'][k],original[k]) for k in affine)
            assert set(state['EEG'])==set(original) and set(state['P'])==set(initial_P)
            for key in ('EEG','P'):
                expected=original if key=='EEG' else initial_P
                assert all(state[key][k].shape==v.shape and state[key][k].dtype==v.dtype for k,v in expected.items())
            if arm=='P_only_correct':assert same(state['EEG'],original,torch)
            if arm=='EEG_only_correct':assert same(state['P'],initial_P,torch)
            assert len(state['optimizer']['param_groups'])==2
            for group in state['optimizer']['param_groups']:assert group['lr']==1e-5 and group['betas']==(.9,.999) and group['weight_decay']==.01
            del state
        final=torch.load(checked(completion['model']),map_location='cpu',weights_only=False);last=torch.load(Path(committed[-1]['directory'])/'state.pt',map_location='cpu',weights_only=False)
        assert final['registration']==ref(path) and final['updates']==6615 and final['seed']==1729 and final['original_ordinary']==s['ordinary']
        assert same(final['EEG'],last['EEG'],torch) and same(final['P'],last['P'],torch)
        assert same(final['EEG'],original,torch)==(arm=='P_only_correct') and same(final['P'],initial_P,torch)==(arm=='EEG_only_correct')
        if noise_reference:assert noises==noise_reference
        else:noise_reference=noises
        results[arm]=dict(model=completion['model'],immutable_checkpoints=len(committed),updates=6615,unique_T_pairs=6615,EEG_exposures=13230,all_losses_replayed=True)
        del final,last
    checked(s['ordinary']);checked(s['parent']);assert not torch.cuda.is_initialized()
    value=dict(status='COMPLETE_INDEPENDENT_ALL5_FULL_T_FIT_CPU_AUDIT',all5_fixed_finals_verified=True,registration=ref(path),arms=results,maximum_loss_replay_error=maximum,all_arm_noise_streams_identical=True,future_GPU_bit_exactness_NOT_claimed=True)
    atomic(root/'independent_fit_CPU_audit.json',value);return value

def manual_EEG(x,w,torch,F):
    w={n:v.double() for n,v in w.items()};x=x.double().reshape(1,4250);z=F.linear(x,w['proj_in.weight'],w['proj_in.bias'])
    z=z+F.linear(F.gelu(F.linear(z,w['block.layers.0.ff.w1.weight'])),w['block.layers.0.ff.w2.weight'])
    return F.layer_norm(z,(1024,),w['block.layers.0.norm.weight'],w['block.layers.0.norm.bias'],1e-6)
def manual_ordinary(c,w,gamma,beta,torch,F):
    w={n:v.double() for n,v in w.items()};u=(c-beta)/gamma;h=F.gelu(F.linear(u,w['trunk.0.weight'],w['trunk.0.bias']));d=F.linear(h,w['direction.weight'],w['direction.bias']);d=d-d.mean(-1,keepdim=True)
    r2=(u*u).sum(-1,keepdim=True);d=d-(d*u).sum(-1,keepdim=True)/r2*u;delta=.25*r2.sqrt()*d/(r2+(d*d).sum(-1,keepdim=True)).sqrt();scale=r2.sqrt()/((u+delta).square().sum(-1,keepdim=True)).sqrt()
    return c+gamma*((scale-1)*u+scale*delta)
def manual_tokens(c,A,P,torch,F):
    # Each native FP16 operation rounds its output, including residual addition.
    h=F.linear(c.half().double(),A['ff.w1.weight'].half().double()).half()
    h=F.gelu(h.double()).half();h=F.linear(h.double(),A['ff.w2.weight'].half().double()).half()
    h=(h+c.half()).half();z=F.linear(h.double(),P['proj.weight'].half().double(),P['proj.bias'].half().double()).half().reshape(1,4,2048)
    return F.layer_norm(z.double(),(2048,),P['norm.weight'].half().double(),P['norm.bias'].half().double(),1e-5).half()
def tokens(path,s,torch,np):
    import torch.nn.functional as F
    import brainhive_native_models_v1 as N
    e=read(path);root=Path(e['output_root']);export=read(root/'export_completion.json');assert export['execution']==ref(path)
    certificate=read(e['fit_CPU_audit_path']);assert certificate['registration']==e['training_protocol'] and certificate['all5_fixed_finals_verified']
    d=read(checked(e['D09_completion']));inputs=read(checked(d['inputs']));q=read(checked(inputs['panel']))['Dcheck'];x=np.load(checked(export['EEG']),allow_pickle=False)
    raw=torch.load(checked(e['raw_EEG']),weights_only=True,map_location='cpu',mmap=True);channels=read(checked(s['input_bindings']))['selected_indices'];source=raw[[int(v['array_row']) for v in q]][:,channels,:].numpy();assert np.array_equal(x,source)
    initial=torch.load(checked(s['parent']),weights_only=True,map_location='cpu')['model'];O=torch.load(checked(s['ordinary']),weights_only=True,map_location='cpu')['model'];prior=N.prior_models('cpu');A=prior['fusion_encoder'].adapter.state_dict();initialP=prior['proj'].state_dict();result={}
    for arm in IMAGE_ARMS:
        model=export['models'][arm];w=initial;P=initialP
        if arm in ARMS:
            assert model['source']==certificate['arms'][arm]['model'];v=torch.load(checked(model['source']),map_location='cpu',weights_only=False);w=v['EEG'];P=v['P']
        else:assert model['source']==s['parent']
        actual=np.load(checked(model['raw']),allow_pickle=False);corrected=np.load(checked(model['conditions']),allow_pickle=False);tk=np.load(checked(model['tokens']),allow_pickle=False);assert actual.shape==corrected.shape==(400,1024) and tk.shape==(400,4,2048) and tk.dtype==np.float16
        errors=[0.,0.,0.]
        with torch.inference_mode():
            for i in range(400):
                c=manual_EEG(torch.from_numpy(x[i:i+1].copy()),w,torch,F);gamma=w['block.layers.0.norm.weight'].double()[None];beta=w['block.layers.0.norm.bias'].double()[None]
                co=c if arm=='parent' else manual_ordinary(c,O,gamma,beta,torch,F)
                for k,expected,observed in [(0,c.numpy(),actual[i:i+1]),(1,co.numpy(),corrected[i:i+1])]:
                    error=float(np.max(np.abs(expected-observed)))/(1+float(np.max(np.abs(expected))));errors[k]=max(errors[k],error);assert error<3e-5
                # Isolate native A/P precision from the separately checked E/O.
                expected=manual_tokens(torch.from_numpy(corrected[i:i+1].copy()),A,P,torch,F).float().numpy();observed=tk[i:i+1].astype(np.float32)
                scaled=float(np.max(np.abs(expected-observed)))/(1+float(np.max(np.abs(expected))));relative=float(np.linalg.norm(expected-observed))/max(float(np.linalg.norm(expected)),1e-12)
                errors[2]=max(errors[2],scaled);assert scaled<.02 and relative<.005
        result[arm]=dict(source=model['source'],all400_passed=True,maximum_scaled_EEG_O_token_errors=errors)
    assert not torch.cuda.is_initialized();value=dict(status='COMPLETE_INDEPENDENT_ALL7_EEG_ORDINARY_A_P_CPU_AUDIT',all7_inference_paths_passed=True,execution=ref(path),export_completion=ref(root/'export_completion.json'),raw_EEG_bit_exact=True,arms=result,not_bit_exact_GPU_claim=True)
    atomic(e['tokens_CPU_audit_path'],value);return value

def metric_values(features,targets,queries,gallery,np):
    f=features.astype(float);v=targets.astype(float);f/=np.linalg.norm(f,axis=1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True);rows={x['stimulus_id']:i for i,x in enumerate(gallery)};out=[]
    for feature,q in zip(f,queries):
        scores=v@feature;j=rows[q['stimulus_id']];other=np.delete(scores,j);correct=scores[j];ids=[i for i,x in enumerate(gallery) if x['concept_id']==q['concept_id']];within=np.array([scores[i] for i in ids if i!=j]);assert len(within)==9
        out.append([correct,float(((other<correct)+.5*(other==correct)).mean()),float(int(np.argmax(scores))==j),float(ids[int(np.argmax(scores[ids]))]==j),float(((within<correct)+.5*(within==correct)).mean())])
    return np.array(out)
def images(path,s,stage,np):
    from PIL import Image
    e=read(path);root=Path(e['output_root']);score=read(root/f'score{stage}.json');d=read(checked(e['D09_completion']));i=read(checked(d['inputs']));q=read(checked(i['panel']))['Dcheck'];target=np.load(checked(d['check_target_features']),allow_pickle=False)
    arrays=np.load(root/f'all_query_values{stage}.npz',allow_pickle=False);export=read(root/'export_completion.json');keys=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc');features={};inodes=set();maximum=0.;pixels=[];spatial_constants=[]
    for arm in IMAGE_ARMS:
        bank=np.load(checked(export['models'][arm]['tokens']),allow_pickle=False);f=np.load(root/f'features{stage}_{arm}.npy',allow_pickle=False);assert f.shape==(stage,768) and np.isfinite(f).all();features[arm]=metric_values(f,target,q[:stage],q,np)
        for j,k in enumerate(keys):maximum=max(maximum,float(np.max(np.abs(features[arm][:,j]-arrays[arm+'__'+k]))));assert maximum<2e-12
        for n,query in enumerate(q[:stage]):
            r=read(root/'records'/f'{n:03d}_{arm}.json');assert all(r[k]==v for k,v in query.items()) and r['arm']==arm and r['query_index']==n and r['execution']==ref(path)
            assert r['source_model']==export['models'][arm]['source'] and r['tokens_sha256']==checksum(bank[n:n+1]) and not r['normal_inference_target_access'] and r['one_image_per_query']
            p=Path(r['generated_path']);assert p.resolve().is_relative_to((root/'pixels').resolve()) and not p.is_symlink();assert ref(p)['sha256']==r['generated_sha256'];stat=p.stat();assert (stat.st_dev,stat.st_ino) not in inodes;inodes.add((stat.st_dev,stat.st_ino))
            with Image.open(p) as im:a=np.asarray(im.convert('RGB'));assert a.shape==(512,512,3) and checksum(a)==r['generated_pixel_sha256']
            pixels.append(r['generated_pixel_sha256']);spatial_constants.append(bool(np.max(a.reshape(-1,3).std(0))<1e-8))
            assert r['UNet_NFE']==4 and r['native_IP_K_calls']==r['native_IP_V_calls']==280 and len(r['ancestral_noise_observations'])==4
            base=read(root/'records'/f'{n:03d}_parent.json')
            for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):assert r[k]==base[k]
    concepts=sorted({x['concept_id'] for x in q[:stage]});groups=[[i for i,x in enumerate(q[:stage]) if x['concept_id']==c] for c in concepts];draws=np.random.default_rng(1729).integers(0,len(groups),(1000,len(groups)));contrasts={}
    for a in ARMS:
        contrasts[a]={}
        for b in IMAGE_ARMS:
            if a==b:continue
            delta=features[a]-features[b];per=np.stack([delta[g].mean(0) for g in groups]);ci=np.quantile(per[draws].mean(1),[.025,.975],axis=0)
            contrasts[a][b]={k:dict(difference=float(delta[:,j].mean()),descriptive_concept_bootstrap95=ci[:,j].tolist()) for j,k in enumerate(keys)}
            for j,k in enumerate(keys):
                original=score['contrasts'][a][b][k];assert abs(original['difference']-delta[:,j].mean())<2e-12 and np.max(np.abs(ci[:,j]-original['descriptive_concept_bootstrap95']))<2e-12
                assert np.max(np.abs(delta[:,j]-arrays[a+'_minus_'+b+'__'+k]))<2e-12
    decision=image_decision(contrasts,stage);assert decision==score['decision']
    value=dict(status='COMPLETE_INDEPENDENT_ALL7_IMAGE_IDENTITY_NOISE_RANK_AND_30_COMPARISON_AUDIT',stage=stage,images=7*stage,execution=ref(path),score=ref(root/f'score{stage}.json'),all_images_verified=True,maximum_metric_error=maximum,pixel_duplicate_count=len(pixels)-len(set(pixels)),spatial_constant_count=sum(spatial_constants),all_queries_retained=True,decision=decision)
    atomic(root/f'independent_images{stage}_CPU_audit.json',value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--mode',choices=['fit','tokens','images'],required=True);p.add_argument('--stage',type=int,choices=[80,400],default=80);a=p.parse_args();start=time.monotonic()
    if a.mode=='fit':s=read(a.protocol)
    else:s=read(checked(read(a.protocol)['training_protocol']))
    assert not os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=''
    import sys;sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C;C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import numpy as np
    if a.mode=='images':value=images(a.protocol,s,a.stage,np)
    else:
        import torch;torch.set_num_threads(2)
        value=fit(a.protocol,s,torch,np) if a.mode=='fit' else tokens(a.protocol,s,torch,np)
    assert time.monotonic()-start<1800;print(json.dumps(dict(**value,program_seconds=time.monotonic()-start)))
