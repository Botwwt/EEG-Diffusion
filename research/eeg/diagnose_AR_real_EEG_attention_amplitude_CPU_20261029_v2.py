"""CPU-only finite-amplitude diagnostic at one committed trained snapshot.

Uses eight predetermined T concepts, actual OOF EEG and the first generated
scale's conditional path through all16 actual author blocks. No query image,
target tokens, development scores, optimizer step or GPU context is used.
Float64 is intentional diagnostic arithmetic, not a GPU/pixel replay claim.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import argparse,datetime,gc,hashlib,json,math,sys,time,traceback,types
from functools import partial
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def reference(path):
    p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def peak():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def run(spec_path):
    spec=json.loads(Path(spec_path).read_bytes());out=Path(spec['output_root']);assert not out.exists();out.mkdir();started=time.monotonic();base=peak();phase='INPUTS_AND_COMMITTED_SNAPSHOT';source_file=None
    try:
        assert spec['status']=='REGISTERED_CPU_ONLY_REAL_EEG_FIXED_AMPLITUDE_DIAGNOSTIC_v1' and spec['arm']=='paired_anchored'
        assert spec['amplitudes']==[0.,.0125,.025,.25,.5,1.] and spec['T_concepts']==8 and spec['maximum_seconds']==240
        assert spec['maximum_extra_CPU_bytes']==10<<30
        code=Path(spec['training_code']);sys.path.insert(0,str(code))
        import AR_anchored_training_contract_CPU_20261029_v1 as C
        admission,admission_sha=C.load_admission(code/'formal_admission_OLD_v1.json',arm='paired_anchored');C.validate_sources(admission);C.verify_protected_files(admission)
        a=admission['data'];identities=json.loads(C.checked(a['EEG_identities']).read_bytes())['T'];groups={}
        for i,row in enumerate(identities):groups.setdefault(row['concept_id'],[]).append(i)
        concepts=sorted(groups,key=lambda c:hashlib.sha256(('1729|AR-fixed-amplitude-T|'+c).encode()).digest())[:8]
        indices=[min(groups[c],key=lambda i:(int(identities[i]['array_row']),identities[i]['stimulus_id'])) for c in concepts]
        assert len(set(indices))==8 and len(groups)==1323
        selected=[dict(query_index=i,**identities[i]) for i in indices]
        root=Path(admission['output_root'])/'paired_anchored';manifest=json.loads((root/'resume/manifest.json').read_bytes());descriptor=spec['registered_committed_snapshot']
        assert manifest['status'] in ('READY','PREPARING_CHECKPOINT') and descriptor['arm']=='paired_anchored' and descriptor['spec_sha256']==admission_sha
        assert descriptor['slot'] in ('slot_A','slot_B')
        payload=root/'resume'/descriptor['slot']/'state.pt';source_file=payload.open('rb')
        assert hashlib.file_digest(source_file,'sha256').hexdigest()==descriptor['state_sha256'] and os.fstat(source_file.fileno()).st_size==descriptor['state_bytes']
        # Pin the immutable inode by its opened fd even if the rolling slot is
        # later retired. No checkpoint copy or persistent model is created.
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(2);torch.manual_seed(1729)
        assert not torch.cuda.is_initialized()
        state=torch.load('/proc/self/fd/'+str(source_file.fileno()),map_location='cpu',weights_only=True,mmap=True)
        assert state['arm']=='paired_anchored' and state['spec_sha256']==admission_sha and state['cursor']['updates']==descriptor['updates']
        assert state['cursor']['logical_update_in_progress'] is False
        weights=state['model'];global_bank=np.load(C.checked(a['global_c']),mmap_mode='r',allow_pickle=False);memory_bank=np.load(C.checked(a['local_memory']),mmap_mode='r',allow_pickle=False)
        c=torch.from_numpy(np.array(global_bank[indices],copy=True)).double();H=torch.from_numpy(np.array(memory_bank[indices],copy=True)).double();delta=H-c[:,None]
        assert c.shape==(8,1024) and H.shape==(8,63,1024) and torch.isfinite(H).all()
        source=Path(admission['author_source']);inventory=json.loads((source/'source_inventory.json').read_bytes())
        for item in inventory['files']:assert sha(source/item['path'])==item['sha256']
        sys.path[:0]=[admission['vendor'],str(source)]
        package=types.ModuleType('models');package.__path__=[str(source/'models')];assert 'models' not in sys.modules;sys.modules['models']=package
        from models.basic_var import AdaLNSelfAttn
        from EEG_anchored_attention_20261029_v1 import EEGAnchoredAttentionDecoder
        decoder=torch.nn.Module();decoder.embed_proj=torch.nn.Linear(1024,1024).double()
        decoder.embed_proj.load_state_dict({k:weights['decoder.embed_proj.'+k].double() for k in ('weight','bias')},strict=True)
        condition=decoder.embed_proj(c).detach();conditioning=condition
        hidden=(condition[:,None]+weights['decoder.pos_start'].double()+weights['decoder.lvl_embed.weight'][0].double()+weights['decoder.pos_1LC'][:,:1].double()).detach()
        decoder.blocks=torch.nn.ModuleList();rows=[];amplitudes=spec['amplitudes'];zero_checks=0;decomposition_error=0.;production_formula_error=0.
        def guard():
            assert time.monotonic()-started<240 and peak()-base<10<<30 and not torch.cuda.is_initialized()
        def rms(x):return x.square().mean(tuple(range(1,x.ndim))).sqrt()
        phase='ALL16_REAL_BLOCKS_FIXED_CONDITIONAL_FIRST_SCALE_Q'
        with torch.inference_mode():
            for layer in range(16):
                guard();block=AdaLNSelfAttn(block_idx=layer,last_drop_p=0.,embed_dim=1024,cond_dim=1024,shared_aln=False,norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),num_heads=16,mlp_ratio=4.,drop=0.,attn_drop=0.,drop_path=0.,attn_l2_norm=True,flash_if_available=False,fused_if_available=False).double().eval()
                prefix=f'decoder.blocks.{layer}.';part={k[len(prefix):]:v.double() for k,v in weights.items() if k.startswith(prefix)}
                assert part;block.load_state_dict(part,strict=True)
                for key,value in block.state_dict().items():assert torch.equal(value,part[key])
                decoder.blocks=torch.nn.ModuleList([block]);model=EEGAnchoredAttentionDecoder(decoder).eval();attention=block.attn
                gamma1,_,scale1,_,shift1,_=model._native_condition_parts(block,conditioning)
                image_input=block.ln_wo_grad(hidden).mul(scale1.add(1)).add(shift1)
                qkv=F.linear(image_input,attention.mat_qkv.weight,torch.cat((attention.q_bias,attention.zero_k_bias,attention.v_bias)))
                query=qkv.view(8,1,3,16,64)[:,:,0].transpose(1,2)
                query=F.normalize(query,dim=-1)*attention.scale_mul_1H11.clamp_max(attention.max_scale_mul).exp()
                curves={};actual=None;native=block.attn(image_input,attn_bias=None)
                for amount in amplitudes:
                    memory=c[:,None]+amount*delta
                    embeddings=model._memory_embeddings(c,memory,condition);k,v=model._prepare_layer(block,conditioning,embeddings)
                    probability=((query@k.transpose(-1,-2))*attention.scale).softmax(-1)
                    response=F.linear((probability@v).transpose(1,2).reshape(8,1,1024),attention.proj.weight,None)
                    uniform=F.linear(v.mean(2).reshape(8,1,1024),attention.proj.weight,None)
                    routed=F.linear(((probability-1/63)@v).transpose(1,2).reshape(8,1,1024),attention.proj.weight,None)
                    decomposition_error=max(decomposition_error,float((response-uniform-routed).abs().max()))
                    error=float((response-model._innovation(image_input,attention,(k,v))).abs().max());production_formula_error=max(production_formula_error,error)
                    if amount==0.:
                        assert bool((response==0).all());zero_checks+=8
                    curves[str(amount)]=rms(response).tolist()
                    if amount==1.:
                        actual=(k,v);actual_response=response;actual_uniform=uniform;actual_routed=routed
                        entropy=-(probability*probability.clamp_min(1e-300).log()).sum(-1).mean((1,2))
                        max_weight=probability.max(-1).values.mean((1,2))
                # Closed-form first derivative for the cached small residual
                # mean. At exactly centered perturbations this is zero.
                a0=condition;z=a0-a0.mean(-1,keepdim=True);inv=(z.square().mean(-1,keepdim=True)+1e-6).rsqrt();u=z*inv
                de=F.linear(delta.mean(1),decoder.embed_proj.weight,None)
                ln_derivative=(de-de.mean(-1,keepdim=True)-u*(u*de).mean(-1,keepdim=True))*inv
                dv=F.linear(ln_derivative*(scale1[:,0]+1),attention.mat_qkv.weight[2048:],None)
                first=F.linear(dv,attention.proj.weight,None)
                for i,row in enumerate(selected):
                    curve={key:value[i] for key,value in curves.items()}
                    order=None if curve['0.0125']<1e-14 else math.log2(curve['0.025']/curve['0.0125'])
                    rows.append(dict(layer=layer,query=row,response_RMS_by_fixed_amplitude=curve,small_amplitude_log2_halving_order=order,
                        finite_1_vs_half_norm_ratio=None if curve['0.5']<1e-14 else curve['1.0']/curve['0.5'],
                        raw_delta_RMS=float(delta[i].square().mean().sqrt()),raw_global_RMS=float(c[i].square().mean().sqrt()),cached_delta_mean_RMS=float(delta[i].mean(0).square().mean().sqrt()),
                        cached_residual_mean_first_derivative_RMS=float(first[i].square().mean().sqrt()),
                        attention_entropy_mean_heads=float(entropy[i]),effective_memory_tokens=float(entropy[i].exp()),mean_head_maximum_weight=float(max_weight[i]),
                        query_independent_channel_mean_term_RMS=float(rms(actual_uniform)[i]),query_weighted_deviation_term_RMS=float(rms(actual_routed)[i]),
                        native_image_attention_RMS=float(rms(native)[i]),innovation_after_native_gamma_RMS=float(rms(actual_response*gamma1)[i]),
                        native_attention_after_same_gamma_RMS=float(rms(native*gamma1)[i]),hidden_RMS=float(rms(hidden)[i])))
                # Exact conditional first-scale path of this trained model,
                # using its own EEG only and the actual local memory at alpha1.
                hidden=model._block(block,hidden,conditioning,None,actual).detach()
                del model,block,part;gc.collect();guard()
        assert len(rows)==128 and zero_checks==128 and decomposition_error<1e-7 and production_formula_error<1e-7
        C.verify_protected_files(admission);guard()
        result=dict(status='COMPLETE_REAL_OOF_EEG_T8_ALL16_LAYER_FIRST_SCALE_FINITE_AMPLITUDE_CPU_DIAGNOSTIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=reference(spec_path),
            fixed_committed_checkpoint=descriptor,checkpoint_update=descriptor['updates'],checkpoint_incomplete_training=True,checkpoint_selection='First current committed snapshot when explicitly invoked; no performance selection',
            queries=selected,all_layers=16,layer_query_units=128,precision='float64 CPU diagnostic; not BF16 GPU replay',amplitudes=amplitudes,rows=rows,zero_anchor_units_exact=zero_checks,
            independent_attention_formula_max_error=production_formula_error,exact_mean_plus_routed_decomposition_max_error=decomposition_error,
            EEG_global_source=a['global_c'],EEG_memory_source=a['local_memory'],no_query_image_or_target_token_used=True,development_scores_read=False,
            trained_weights_changed=False,training_queues_changed=False,GPU_jobs=0,CUDA_context_created=False,research_images=0,peak_extra_CPU_bytes=peak()-base,seconds=time.monotonic()-started,
            interpretation='T-only one first-scale state, frozen intermediate weights, fixed Q within each layer. Channel-mean term may depend on stimulus and is not a background constant. Finite-response shape alone does not establish EEG discrimination or image benefit.')
        write(out/'completion.json',result);print(json.dumps(dict(completion=reference(out/'completion.json'),seconds=result['seconds'],checkpoint_update=result['checkpoint_update'])))
    except BaseException as error:
        write(out/'failure.json',dict(status='CPU_REAL_EEG_FINITE_AMPLITUDE_DIAGNOSTIC_FAILURE_PRESERVED',phase=phase,committed_snapshot=locals().get('descriptor'),error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started,GPU_jobs=0,training_queues_changed=False));raise
    finally:
        if source_file is not None:source_file.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);q=p.parse_args();run(q.spec)
