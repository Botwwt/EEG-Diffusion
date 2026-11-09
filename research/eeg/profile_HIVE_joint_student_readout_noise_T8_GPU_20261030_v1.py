"""Fixed T8 follow-on: temporary joint EEG/scalar updates, readout, noise probe.

Requires the previous actual-G teacher profile to finish successfully. Uses its
24 fitted engineering teachers, never a speculative or substituted q table.
No formal fitting, heldout efficacy scoring, research PNGs or selected winner.
"""
import argparse,datetime,gc,hashlib,io,json,os,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


def read(path):return json.loads(Path(path).read_bytes())
def ref(path):
    path=Path(path)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest)
def checked(item):
    actual=ref(item['path']);assert actual['sha256']==item['sha256']
    if 'bytes' in item:assert actual['bytes']==item['bytes']
    return Path(item['path'])
def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def seed(domain,row):return int(hashlib.sha256((domain+str(row)).encode()).hexdigest()[:8],16)


def main(protocol_path,binding_path,binding_SHA):
    s=read(protocol_path);assert s['host']=='old' and s['status']=='REGISTERED_JOINT_STUDENT_READOUT_NOISE_T8_ENGINEERING'
    assert 'run_gpu_job_v2.py' in Path('/proc',str(os.getppid()),'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==s['GPU_UUID']
    for item in s['source_pins']+s['payloads']:checked(item)
    checked(dict(path=binding_path,sha256=binding_SHA));bound=read(binding_path)
    for item in bound['input_pins']:checked(item)
    assert bound['upstream_protocol']==s['upstream_teacher_protocol']
    out=Path(s['output_root']);out.mkdir();phase='IMPORTS';started=time.monotonic()
    def memory():return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmHWM:')))*1024
    baseline=memory()
    sys.path.insert(0,s['repository_root']);import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']));sys.path.insert(0,str(Path(protocol_path).parent))
    import numpy as np
    import torch
    from PIL import Image
    import brainhive_native_models_v1 as N
    from brainhive_frozen_semantic_runtime_GPU_20261030_v1 import FrozenSemanticRuntime,module_digest
    from brainhive_joint_student_engineering_Torch_20261030_v1 import make_optimizer,student_step,capture_engineering_state,restore_engineering_state,assert_tree_equal,posterior_conditions
    from brainhive_joint_student_followon_contract_CPU_20261030_v1 import dependencies_ready,validate_teacher_arrays
    from brainhive_spherical_predictive_density_CPU_20261029_v1 import sample_power_spherical
    from brainhive_semantic_teacher_Torch_20261029_v1 import native_F_condition
    def live(pid,ticks):
        path=Path('/proc',str(pid),'stat')
        try:fields=path.read_text().rsplit(')',1)[1].split()
        except FileNotFoundError:return False
        return fields[0] not in ('Z','X') and int(fields[19])==ticks
    assert dependencies_ready(s['dependencies'],live,lambda path:read(path) if Path(path).is_file() else None)['ready']
    torch.set_num_threads(2);torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(min(1.,s['maximum_GPU_bytes']/torch.cuda.get_device_properties(0).total_memory))
    def budget():
        assert time.monotonic()-started<800 and memory()-baseline<s['maximum_extra_CPU_bytes']
        assert torch.cuda.max_memory_allocated()<s['maximum_GPU_bytes'] and shutil.disk_usage(out).free>3<<30
        assert sum(path.stat().st_size for path in out.rglob('*') if path.is_file())<s['output_cap_bytes']
    try:
        phase='BIND_ACTUAL_G_TEACHERS'
        teachers=dict(np.load(bound['teacher_parameters']['path'],allow_pickle=False));teacher_check=validate_teacher_arrays(teachers,s['teacher_arms'])
        prior=dict(np.load(s['prior']['path'],allow_pickle=False));inputs=np.load(s['T8_inputs']['path'],allow_pickle=False);metadata=read(s['T8_metadata']['path'])
        assert [row['array_row'] for row in metadata]==inputs['array_rows'].tolist() and inputs['positions'].tolist()==s['T_positions']
        binding=read(s['actual_input_bindings']['path']);assert binding['selected_indices']==list(range(46,63))
        raw=torch.load(s['EEG']['path'],map_location='cpu',weights_only=True,mmap=True)
        eeg=raw[torch.from_numpy(inputs['array_rows'])][:,binding['selected_indices'],:].clone().cuda()
        original=torch.load(s['parent']['path'],map_location='cpu',weights_only=True,mmap=True)
        weight=torch.as_tensor(prior['F_weight'],device='cuda');bias=torch.as_tensor(prior['F_bias'],device='cuda');radius=float(prior['reference_pre_affine_radius'])
        def fresh():
            model=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval();model.load_state_dict(original['model'],strict=True)
            model.block.layers[0].norm.requires_grad_(False)
            alpha=torch.nn.Parameter(torch.tensor(np.log(float(inputs['subject_strengths'][0])),dtype=torch.float32,device='cuda'))
            return model,alpha,make_optimizer(model,alpha,s['optimizer']['network_learning_rate'],s['optimizer']['log_scale_learning_rate'])
        phase='FROZEN_NATIVE_HOST';runtime=FrozenSemanticRuntime(s,out);budget()
        reference=N.classes()['BrainMLP'](4250,1024,num_layers=1,dropout=.1).cuda().eval().requires_grad_(False);reference.load_state_dict(original['model'],strict=True)
        reference_hash=module_digest(reference)
        with torch.no_grad():reference_conditions=reference(eeg)
        gamma=reference.block.layers[0].norm.weight;beta=reference.block.layers[0].norm.bias
        condition_seeds=[seed('HIVE_conditional_latent_T_',row['array_row']) for row in metadata]
        noise_seeds=[[seed('HIVE_semantic_teacher_T_',row['array_row']),seed('HIVE_semantic_teacher_held_noise1_T_',row['array_row']),seed('HIVE_semantic_teacher_held_noise2_T_',row['array_row'])] for row in metadata]
        assert len(set(value for row in noise_seeds for value in row))==24
        targets=[];noise_states=[];parity=[]
        for j,item in enumerate(metadata):
            phase='ORIGINAL_NATIVE_AND_ORDINARY_PARITY_'+str(j);budget()
            with zipfile.ZipFile(item['image_zip']) as archive:image_bytes=archive.read(item['image_member'])
            assert hashlib.sha256(image_bytes).hexdigest()==item['image_sha256']
            image=torch.from_numpy(np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB')).copy()).permute(2,0,1)[None].cuda().float()/255.
            targets.append(runtime.target_feature(image));del image
            c=reference_conditions[j:j+1]
            with torch.no_grad():ordinary=runtime.ordinary(c,gamma[None].expand_as(c),beta[None].expand_as(c))[0]
            for name,value in (('native',c),('ordinary',ordinary)):
                unused,actual,state=runtime.render(value,noise_seeds[j][0]);manual,manual_state=runtime.manual_latent(value,noise_seeds[j][0])
                assert torch.equal(actual,manual) and torch.equal(state,manual_state)
                parity.append(dict(position=j,path=name,all4step_latent_and_noise_state_exact=True))
                if name=='native':noise_states.append(state)
        target=torch.cat(targets);del targets
        arms=[];working_models={};arrays={'target_features':target.cpu().numpy()};all_noise_records=[]
        for arm_index,arm in enumerate(s['teacher_arms']):
            phase='TEMPORARY_JOINT_STUDENT_'+arm;torch.manual_seed(1729);model,log_scale,optimizer=fresh()
            before_affine={name:value.detach().clone() for name,value in model.block.layers[0].norm.named_parameters()}
            teacher=torch.as_tensor(np.concatenate([teachers[f'{j}_{arm}_moment'] for j in range(8)]),device='cuda').detach()
            steps=[student_step(model,log_scale,optimizer,eeg,teacher,prior,weight,bias) for unused in range(4)]
            assert len(optimizer.state)==5 and all(int(state['step'])==4 for state in optimizer.state.values())
            assert all(torch.equal(before_affine[name],value.detach()) for name,value in model.block.layers[0].norm.named_parameters())
            snapshot=capture_engineering_state(model,log_scale,optimizer,4,s['parent']['sha256'])
            roundtrip=False
            if arm_index==0:
                torch.save(snapshot,out/'ENGINEERING_ONLY_correct_state_roundtrip.pt')
                loaded=torch.load(out/'ENGINEERING_ONLY_correct_state_roundtrip.pt',map_location='cpu',weights_only=True)
                second,second_scale,second_optimizer=fresh();restore_engineering_state(loaded,second,second_scale,second_optimizer)
                restored=capture_engineering_state(second,second_scale,second_optimizer,4,s['parent']['sha256']);assert_tree_equal(snapshot,restored);roundtrip=True
                model.eval();second.eval()
                with torch.no_grad():forward_error=float((model(eeg)-second(eeg)).abs().max())
                assert forward_error<3e-5
                del second,second_scale,second_optimizer,loaded,restored
            working_models[arm]=dict(role='ENGINEERING_ONLY_REJECT_IN_FORMAL_TRAINING_OR_CONDITION_BANKS',source_checkpoint_SHA256=s['parent']['sha256'],model=snapshot['model'],log_scale=snapshot['log_scale'],temporary_updates=4)
            model.eval()
            with torch.no_grad():raw_condition=model(eeg)
            prediction,samples,sample_records=posterior_conditions(raw_condition,log_scale,prior,weight,bias,radius,condition_seeds)
            replay,replay_samples,replay_records=posterior_conditions(raw_condition,log_scale,prior,weight,bias,radius,condition_seeds)
            assert torch.equal(prediction,replay) and torch.equal(samples,replay_samples) and sample_records==replay_records
            predictive_records=[]
            for j in range(8):
                phase='ACTUAL_PREDICTIVE_DRAW_'+arm+'_'+str(j);budget()
                CPU_rng=torch.get_rng_state().clone();CUDA_rng=torch.cuda.get_rng_state().clone()
                feature,z,after=runtime.render(prediction[j:j+1],noise_seeds[j][0])
                assert torch.equal(CPU_rng,torch.get_rng_state()) and torch.equal(CUDA_rng,torch.cuda.get_rng_state())
                assert torch.equal(after,noise_states[j]),'Generator random stream must not depend on the EEG conditional draw'
                manual,manual_state=runtime.manual_latent(prediction[j:j+1],noise_seeds[j][0]);assert torch.equal(z,manual) and torch.equal(after,manual_state)
                predictive_records.append(dict(position=j,condition=sample_records[j],normal_readout_teacher_table_access=False,production_UNet_calls=4,diagnostic_manual_replay_extra_UNet_calls=4,latent_SHA256=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest(),T_training_critic_cosine=float((feature*target[j:j+1]).sum())))
            # Same chosen q sample is held fixed across three generator-noise
            # streams. These T-only readings are not heldout EEG efficacy.
            phase='FIXED_Q_CROSS_NOISE_DIAGNOSTIC_'+arm
            for j in range(8):
                q_seed=seed('HIVE_q_noise_probe_'+arm+'_T_',metadata[j]['array_row'])
                for version in ('initial','optimized'):
                    mean=inputs['visual_latent_mean'][j] if version=='initial' else teachers[f'{j}_{arm}_mean']
                    concentration=100000. if version=='initial' else float(teachers[f'{j}_{arm}_concentration'])
                    v=sample_power_spherical(mean,concentration,1,np.random.default_rng(q_seed))
                    c=native_F_condition(torch.as_tensor(v,device='cuda'),weight,bias,radius).float()
                    features=[]
                    for noise_index,noise in enumerate(noise_seeds[j]):
                        feature,unused,unused_state=runtime.render(c,noise);features.append(feature.cpu().numpy()[0]);budget()
                    cosines=[float(np.dot(value,target[j].cpu().numpy())) for value in features]
                    record=dict(arm=arm,position=j,version=version,q_seed=q_seed,one_q_sample_fixed_across_generator_noises=True,noise_seeds=noise_seeds[j],noise0_used_in_previous_q_optimization=True,noise1_and2_not_used_in_previous_q_optimization=True,T_only_CLIP_cosines=cosines,not_EEG_generalization_or_method_effect=True)
                    all_noise_records.append(record);arrays[f'{arm}_{j}_{version}_q']=v;arrays[f'{arm}_{j}_{version}_features']=np.stack(features)
            arrays[arm+'_raw_condition']=raw_condition.detach().cpu().numpy();arrays[arm+'_native_condition']=prediction.cpu().numpy();arrays[arm+'_latent_sample']=samples.cpu().numpy();arrays[arm+'_log_scale']=log_scale.detach().cpu().numpy()
            record=dict(arm=arm,steps=steps,temporary_EEG_optimizer_updates=4,formal_EEG_optimizer_updates=0,source_affine_unchanged=True,state_roundtrip_tested_for_correct_arm=roundtrip,restored_GPU_forward_maxabs=forward_error if roundtrip else None,predictive_engineering=predictive_records)
            arms.append(record);write(out/(arm+'_engineering.json'),record)
            del model,log_scale,optimizer,snapshot,teacher,raw_condition,prediction,samples,replay,replay_samples;gc.collect();torch.cuda.empty_cache();budget()
        torch.save(working_models,out/'ENGINEERING_ONLY_all3_working_models.pt')
        np.savez_compressed(out/'all3_predictive_and_noise_engineering.npz',**arrays)
        assert module_digest(reference)==reference_hash
        frozen=runtime.frozen_audit()
        for item in s['payloads']+bound['input_pins']:checked(item)
        result=dict(status='COMPLETE_JOINT_STUDENT_READOUT_NOISE_T8_ENGINEERING',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=ref(protocol_path),binding=ref(binding_path),teacher_source_validation=teacher_check,parity=parity,arms=arms,T_only_fixed_q_noise_diagnostics=all_noise_records,temporary_EEG_optimizer_updates=12,formal_EEG_optimizer_updates=0,new_scientific_models=0,new_saved_research_images=0,source_ordinary_and_parent_unchanged=True,all_frozen_host_weights_unchanged=True,private_condition_RNG_and_generator_noise_separate=True,full_T_fit_or_image_efficacy_admitted=False,training_critic_not_independent_metric=True,full_chain_OOF=False,future_optimizer_trajectory_equality_not_claimed=True,working_prior_calibration_not_established=True,peak_GPU_allocated=torch.cuda.max_memory_allocated(),extra_CPU_peak=memory()-baseline,seconds=time.monotonic()-started,frozen=frozen,loader=runtime.loading,host=runtime.host,NEW_contacted=False)
        write(out/'completion.json',result);budget()
    except BaseException as error:
        write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--binding',required=True);parser.add_argument('--binding-sha',required=True);args=parser.parse_args();main(args.protocol,args.binding,args.binding_sha)
