"""Full Dcheck frozen-UNet response to real versus simulated EEG conditions.

Uses actual paired images only for this labelled diagnostic. No adapter fitting,
V/test access, image reconstruction claim, or target-at-inference deployment.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
import os
import time
import zipfile

from profile_standard_generator import load_pipeline, torch, ROOT
from diffusers import DDPMScheduler
from PIL import Image
import numpy as np
from native_error_channel import NativeErrorChannel,stream_seed
from error_channel_validation_metrics import energy_score
from paired_analysis import paired_summary


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    assert not Path(path).exists(), 'Preserve earlier scientific results'
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ['plan','cache-config','validation-output','output']:
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan_path = Path(args.plan); plan = load(plan_path)
    assert plan['version'] == 'idea_first_full_Dcheck_UNet_response_v1'
    assert plan['seed'] == 1729 and plan['draws_per_condition_arm'] == 4
    assert plan['Dcheck_queries'] == 8300 and plan['latent_batch'] == 16
    assert plan['timesteps'] == [124,374,624,874]
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    cpath = Path(args.cache_config); config = load(cpath)
    root = Path(config['condition_cache_output_root'])
    complete = load(root/'completion.json')
    assert complete['status'] == 'COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS'
    assert complete['config_sha256'] == sha(cpath) and complete['subjects'] == 10
    assert complete['T_OOF_queries'] == 132300 and complete['D_queries_per_teacher'] == 16500
    assert config['condition_chain'] == 'REF_REG'
    validation = Path(args.validation_output); summary = load(validation/'summary.json')
    assert summary['subjects'] == 10 and summary['Dcheck_queries'] == 8300
    assert summary['input_provenance_sha256'] == sha(validation/'input_provenance.json')
    provenance = load(validation/'input_provenance.json')
    assert provenance['cache_config_sha256'] == sha(cpath)
    assert provenance['cache_completion_sha256'] == sha(root/'completion.json')
    gate = summary['gate']
    assert gate['native_energy_CI_upper_below_zero_vs_Iso_and_Moment']
    assert gate['projected_energy_CI_upper_below_zero_vs_Iso_and_Moment']
    assert sha(ROOT/'configs/idea_first_error_validation_plan_v1.json') == summary['plan_sha256']
    numerical_plan_version = load(ROOT/'configs/idea_first_error_validation_plan_v1.json')['version']
    parent = Path('/proc')/str(os.getppid())/'cmdline'
    assert parent.exists() and str(ROOT/'run_gpu_job.py').encode() in parent.read_bytes()
    out = Path(args.output); assert not out.exists(); out.mkdir(parents=True)
    protocol = load(ROOT/'configs/split_manifest.json')
    assert sha(ROOT/'configs/split_manifest.json') == config['split_sha256']
    bank = torch.load(root/'C_I_D.pt',map_location='cpu',weights_only=True)
    check = [i for i,concept in enumerate(bank['concept_ids']) if concept in protocol['d_check_concepts']]
    assert len(check) == 830 and bank['array_rows'][check].unique().numel() == 830
    metadata = {int(x['array_row']):x for x in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if x['dataset_split']=='train'}
    report_by_path = {x['path']:x for x in complete['reports']}
    pipe, infos, receipt, prompts = load_pipeline()
    prompt, negative, pool, negative_pool = prompts
    pipe.set_progress_bar_config(disable=True)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type == 'epsilon'
    time_ids = pipe._get_add_time_ids((512,512),(0,0),(512,512),dtype=prompt.dtype,
                                    text_encoder_projection_dim=pool.shape[-1]).cuda()
    archives = {}; latent_bank = []; image_identity = []
    with torch.inference_mode():
        for start in range(0,len(check),4):
            pixels = []
            for position in check[start:start+4]:
                row = int(bank['array_rows'][position]); item = metadata[row]
                assert item['stimulus_id'] == bank['stimulus_ids'][position]
                if item['image_zip'] not in archives:
                    archives[item['image_zip']] = zipfile.ZipFile(item['image_zip'])
                raw = archives[item['image_zip']].read(item['image_member'])
                assert hashlib.sha256(raw).hexdigest() == item['image_sha256'] == bank['image_sha256'][position]
                with Image.open(io.BytesIO(raw)) as image:
                    a = np.array(image.convert('RGB').resize((512,512),Image.Resampling.BICUBIC),copy=True)
                pixels.append(torch.from_numpy(a).permute(2,0,1).float()/127.5-1)
                image_identity.append(dict(array_row=row,stimulus_id=item['stimulus_id'],image_sha256=item['image_sha256']))
            latent_bank.append((pipe.vae.encode(torch.stack(pixels).cuda()).latent_dist.mode()*pipe.vae.config.scaling_factor).cpu())
    for archive in archives.values():archive.close()
    latent_bank = torch.cat(latent_bank)
    write_new(out/'input_provenance.json',dict(plan_sha256=sha(plan_path),cache_config_sha256=sha(cpath),
        numerical_validation_summary_sha256=sha(validation/'summary.json'),D_image_identities=image_identity,
        generator_revision=receipt['revision'],raw_images_used_for_diagnostic_only=True,
        V_or_official_test_accessed=False))
    calls = dict(UNet_forward_calls=0,UNet_sample_evaluations=0)
    def count(module,inputs):
        calls['UNet_forward_calls'] += 1; calls['UNet_sample_evaluations'] += len(inputs[0])
    hook = pipe.unet.register_forward_pre_hook(count)

    @torch.inference_mode()
    def response(z,t,condition,unconditional=False):
        n = len(z)
        text = negative if unconditional else prompt
        pooled = negative_pool if unconditional else pool
        value = pipe.unet(z.to(torch.bfloat16),t,encoder_hidden_states=text.repeat(n,1,1),
            added_cond_kwargs=dict(text_embeds=pooled.repeat(n,1),time_ids=time_ids.repeat(n,1),
                image_embeds=[condition.cuda().to(torch.bfloat16)[:,None,:]]),return_dict=False)[0]
        assert torch.isfinite(value).all()
        return value.float().flatten(1).cpu()

    results = []
    for subject in protocol['subjects']:
        path = root/subject/'D_full.pt'; proof = report_by_path[str(path)]
        assert sha(path) == proof['sha256']
        full = torch.load(path,map_location='cpu',weights_only=True)
        assert full['query_ids'] == [f'{subject}/{stim}/R4' for stim in bank['stimulus_ids']]
        assert torch.equal(full['array_rows'],bank['array_rows'])
        assert full['C_E'].shape == bank['C_I'].shape and torch.isfinite(full['C_E']).all()
        assert full['C_I_bank_sha256'] == sha(root/'C_I_D.pt') and full['config_sha256'] == sha(cpath)
        fitted = torch.load(validation/subject/'error_channel.pt',map_location='cpu',weights_only=True)
        assert fitted['actual_scientific_fit'] and fitted['plan_sha256'] == summary['plan_sha256']
        assert fitted['T_source_sha256'] == report_by_path[str(root/subject/'T_oof.pt')]['sha256']
        for name,value in fitted['D_source_sha256'].items():
            assert value == report_by_path[str(root/subject/(name+'.pt'))]['sha256']
        channel = NativeErrorChannel(fitted['state'])
        neighbors = torch.load(validation/subject/'D_nearest_other_T_concepts.pt',map_location='cpu',weights_only=True)
        for start in range(0,830,plan['latent_batch']):
            stop = min(start+plan['latent_batch'],830);positions = check[start:stop]
            ids = [bank['stimulus_ids'][i] for i in positions]
            concepts = [bank['concept_ids'][i] for i in positions]
            ci = bank['C_I'][positions]; real = full['C_E'][positions]
            target = latent_bank[start:stop].cuda(); b = len(positions)
            seeds = [stream_seed(1729,plan['version'],subject,s,'diagnostic-noise') for s in ids]
            epsilon = torch.cat([torch.randn((1,4,64,64),device='cuda',dtype=torch.float32,
                generator=torch.Generator(device='cuda').manual_seed(seed)) for seed in seeds])
            t = torch.tensor([plan['timesteps'][stream_seed(1729,subject,s,'response-timestep')%4] for s in ids],device='cuda')
            z = scheduler.add_noise(target,epsilon,t)
            unconditional = response(z,t,torch.zeros_like(ci),unconditional=True)
            def guided(condition):
                return unconditional + 5.*(response(z,t,condition)-unconditional)
            observed = guided(real)
            oracle = guided(ci)
            arm_scores = {'Clean':(oracle-observed).norm(dim=-1)}
            for arm in ['Iso','Moment','Conditional']:
                draws = []
                for draw in range(4):
                    # Same allocated first four numerical validation draws, not
                    # new seeds selected from generator response outcomes.
                    streams = [stream_seed(1729,numerical_plan_version,subject,s,'R4','error-draw',draw) for s in ids]
                    simulated,_ = channel.sample(arm,ci,concepts,streams,neighbors[positions])
                    draws.append(guided(simulated))
                arm_scores[arm] = energy_score(torch.stack(draws,1),observed)
            for i,stimulus in enumerate(ids):
                results.append(dict(query_id=f'{subject}/{stimulus}/R4',subject=subject,stimulus_id=stimulus,
                    concept_id=concepts[i],array_row=int(bank['array_rows'][positions[i]]),timestep=int(t[i]),
                    noise_seed=seeds[i],real_condition_norm=float(real[i].norm()),
                    **{arm+'_response_energy':float(value[i]) for arm,value in arm_scores.items()},
                    real_noise_prediction_MSE=float((observed[i]-epsilon[i].flatten().cpu()).square().mean()),
                    oracle_noise_prediction_MSE=float((oracle[i]-epsilon[i].flatten().cpu()).square().mean())))
            parts=out/'per_query_parts'/subject;parts.mkdir(parents=True,exist_ok=True)
            write_new(parts/f'{start:04d}.json',dict(rows=results[-b:],plan_sha256=sha(plan_path),
                real_condition_sha256=proof['sha256'],fitted_channel_sha256=sha(validation/subject/'error_channel.pt')))
            (out/'progress.json').write_text(json.dumps(dict(status='RUNNING',subject=subject,
                subject_queries_completed=stop,total_query_rows=len(results),seconds=time.monotonic()-started,**calls))+'\n')
        print(json.dumps(dict(stage='full_Dcheck_frozen_UNet',subject=subject,queries_completed=len(results),seconds=time.monotonic()-started)),flush=True)
    hook.remove()
    assert len(results) == 8300 and len({x['query_id'] for x in results}) == 8300
    assert calls['UNet_sample_evaluations'] == 8300*15
    with (out/'per_query.csv').open('x',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(results[0]),lineterminator='\n');writer.writeheader();writer.writerows(results)
    comparisons = {}
    for control in ['Clean','Iso','Moment']:
        delta = np.array([[x['Conditional_response_energy']-x[control+'_response_energy']] for x in results])
        comparisons[control] = paired_summary(delta,[x['subject'] for x in results],[x['concept_id'] for x in results],1000,1729)
    promising = all(comparisons[c]['CI95'][0][1] < 0 for c in ['Iso','Moment'])
    report = dict(status='FROZEN_UNET_RESPONSE_PROMISING_REQUIRES_MATCHED_ACTUAL_IMAGES' if promising else 'NO_CONDITIONAL_INCREMENT_UNDER_REGISTERED_UNET_RESPONSE_GATE',
        comparisons=comparisons,gate_passed=promising,queries=8300,subjects=10,concepts=83,
        simulated_draws_per_arm=4,timesteps=plan['timesteps'],one_fixed_timestep_and_noise_per_query=True,
        CFG=5.,IP_scale=1.,shared_unconditional_zero_native_condition=True,
        timestep_counts={str(t):sum(x['timestep']==t for x in results) for t in plan['timesteps']},
        per_query_sha256=sha(out/'per_query.csv'),input_provenance_sha256=sha(out/'input_provenance.json'),
        plan_sha256=sha(plan_path),seconds=time.monotonic()-started,**calls,
        images_generated=0,adapter_training_started=False,EEG_reconstruction_gain_established=False,
        limits=['Target-image noising is privileged diagnostic supervision, never formal inference.',
                'Response fidelity is not proof that finite interface training will improve image content.',
                'One noise/timestep and four condition draws per Dcheck query; no seed robustness or trajectory-wide equality claim.'])
    write_new(out/'summary.json',report);print(json.dumps(report),flush=True)


if __name__ == '__main__':main()
