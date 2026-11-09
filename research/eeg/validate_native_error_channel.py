"""Full T-OOF/Dfit/Dcheck CPU error-channel validation, not image evaluation.

Requires real completed OOF models and cache receipts. Never starts training a
teacher, generates images, accesses V/test conditions, or unlocks final data.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,'/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.embeddings import ImageProjection
from safetensors import safe_open

from native_error_channel import NativeErrorChannel,stream_seed,apply_calibration
from error_channel_validation_metrics import (energy_score,distribution_summary,
    classifier_features,fit_two_sample,confusion_js)
from locked_metrics import fixed_gallery_metrics
from paired_analysis import paired_summary

ROOT=Path(__file__).resolve().parent
ARMS=['Clean','Iso','Moment','Conditional']
ASSET=Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write_new(path,value):
    assert not path.exists(),('Preserve existing result',str(path))
    path.write_text(json.dumps(value,indent=2)+'\n')


def save_new(path,value):
    assert not path.exists(),('Preserve existing tensor',str(path))
    torch.save(value,path)


class Validator:
    def __init__(self,args):
        self.started=time.monotonic();self.plan_path=Path(args.plan)
        self.plan=json.loads(self.plan_path.read_text());p=self.plan
        assert p['version']=='native_error_channel_full_Dcheck_v1'
        assert p['seed']==1729 and p['repeat_protocol']=='R4' and p['simulation_draws']==16
        assert (p['pca_components'],p['neighbors'],p['shrinkage'])==(32,32,.1)
        assert p['bootstrap_draws']==1000
        assert p['image_metrics_unchanged'] and p['formal_images_generated_by_this_runner']==0
        assert set(p['source_sha256'])=={'native_error_channel.py','error_channel_validation_metrics.py',
            'validate_native_error_channel.py','build_native_oof_cache.py','locked_metrics.py','paired_analysis.py'}
        for name,value in p['source_sha256'].items():assert sha(ROOT/name)==value,('Source changed after plan lock',name)
        assert sha(ROOT/'audit/image_manifest.csv')==p['image_manifest_sha256']
        assert sha(ROOT/'audit/query_manifest_R4.csv')==p['query_manifest_sha256']
        self.images={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open())
                     if r['dataset_split']=='train'}
        self.cache_config=Path(args.cache_config);self.config=json.loads(self.cache_config.read_text());c=self.config
        self.root=Path(c['condition_cache_output_root'])
        completion=json.loads((self.root/'completion.json').read_text())
        assert completion['status']=='COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS'
        assert completion['config_sha256']==sha(self.cache_config)
        assert completion['subjects']==10 and completion['T_OOF_queries']==132300 and completion['D_queries_per_teacher']==16500
        assert completion['D_teachers']==['oof0','oof1','full'] and completion['partitions']==['T','D']
        assert completion['all_T_queries_have_genuinely_heldout_teacher'] and not completion['V_or_official_test_conditions_exported']
        assert completion['seed']==1729 and completion['number_of_training_seeds']==1
        assert c['cache_builder_sha256']==sha(ROOT/'build_native_oof_cache.py')
        assert c['coordinate_space']=='raw_pooled_CLIP_H_1024_before_native_IP_projection'
        self.chain=c['condition_chain'];assert self.chain in ['REF_REG','REF_PRIOR_V']
        assert completion['condition_chain']==self.chain
        selection=ROOT/'audit/reference_control_selection.json'
        assert sha(selection)==c['reference_selection_sha256']
        self.selection=json.loads(selection.read_text());assert self.selection['selected_reference']==self.chain
        assert sha(c['oof_training_config_path'])==c['oof_training_config_sha256']
        training=json.loads(Path(c['oof_training_config_path']).read_text())
        assert training['condition_chain']==self.chain and training['output_root']==c['output_root']
        self.protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
        assert sha(ROOT/'configs/split_manifest.json')==c['split_sha256']
        self.subjects=self.protocol['subjects'];assert self.subjects==[f'sub-{i:02}' for i in range(1,11)]
        assert sha(c['visual_features_path'])==c['visual_features_sha256']
        self.visual=torch.load(c['visual_features_path'],weights_only=True,map_location='cpu')
        assert self.visual.shape==(16540,1024) and torch.isfinite(self.visual).all()
        assert len(completion['reports'])==40
        self.receipts={}
        for report in completion['reports']:
            path=Path(report['path']);assert path.is_relative_to(self.root)
            assert sha(path)==report['sha256'] and report['config_sha256']==sha(self.cache_config)
            key=(path.parent.name,path.stem);assert key not in self.receipts
            self.receipts[key]=report
        self.banks={}
        for role,n in [('T',13230),('D',1650)]:
            path=self.root/f'C_I_{role}.pt';bank=torch.load(path,weights_only=True,map_location='cpu')
            rows=self.protocol['image_rows'][role]
            assert len(rows)==n and bank['partition']==role
            assert torch.equal(bank['array_rows'],torch.tensor(rows)) and torch.equal(bank['C_I'],self.visual[rows])
            assert bank['visual_source_sha256']==c['visual_features_sha256']
            assert bank['coordinate_space']==c['coordinate_space']
            assert len(bank['stimulus_ids'])==len(set(bank['stimulus_ids']))==n
            assert bank['stimulus_ids']==[self.images[i]['stimulus_id'] for i in rows]
            assert bank['concept_ids']==[self.images[i]['concept_id'] for i in rows]
            assert bank['image_sha256']==[self.images[i]['image_sha256'] for i in rows]
            self.banks[role]=bank|dict(path=str(path),sha256=sha(path))
        fit=set(self.protocol['d_fit_concepts']);check=set(self.protocol['d_check_concepts'])
        assert len(fit)==82 and len(check)==83 and fit.isdisjoint(check)
        assert fit|check==set(self.banks['D']['concept_ids'])
        self.fit=torch.tensor([x in fit for x in self.banks['D']['concept_ids']])
        self.check=~self.fit;assert int(self.fit.sum())==820 and int(self.check.sum())==830
        assert self.banks['D']['calibration_roles']==['fit' if a else 'check' for a in self.fit.tolist()]
        self.oof_proofs={}
        for fold in [0,1]:
            role=f'oof{fold}';base=Path(c['output_root'])/role
            lock=json.loads((base/'OOF_data_roles.json').read_text())
            assert lock['role']==role and lock['config_sha256']==c['oof_training_config_sha256']
            actual=set(lock['training_array_rows']);held=set(lock['heldout_T_array_rows'])
            all_t=set(self.protocol['image_rows']['T'])
            expected={row for row,concept in zip(self.banks['T']['array_rows'].tolist(),self.banks['T']['concept_ids'])
                      if self.protocol['oof_fold_by_concept'][concept]==fold}
            assert held==expected and actual|held==all_t and actual.isdisjoint(held)
            assert actual.isdisjoint(self.protocol['image_rows']['D']+self.protocol['image_rows']['V'])
            self.oof_proofs[role]={}
            for subject in self.subjects:
                proof=json.loads((base/subject/'completion.json').read_text())
                assert proof['status']=='COMPLETED' and proof['role']==role and proof['subject']==subject
                assert proof['training_seed']==1729 and proof['split_sha256']==c['split_sha256']
                e=proof['encoder'];assert e['unique_training_rows']==len(actual) and e['seen_samples']==e['epochs']*len(actual)
                assert sha(e['checkpoint'])==e['checkpoint_sha256'] and sha(proof['condition_cache'])==proof['condition_cache_sha256']
                prior=proof['prior']
                if self.chain=='REF_REG':assert prior is None
                else:
                    assert prior['prediction_type']=='v_prediction' and prior['epochs']==400
                    assert prior['unique_training_rows']==len(actual) and prior['seen_samples']==400*len(actual)
                    assert sha(prior['checkpoint'])==prior['checkpoint_sha256']
                self.oof_proofs[role][subject]=dict(encoder_sha256=e['checkpoint_sha256'],
                    prior_sha256=prior['checkpoint_sha256'] if prior else None)
        # Weight Dfit OOF distributions by their actual T deployment proportions.
        roles=[self.protocol['oof_fold_by_concept'][c] for c in self.banks['T']['concept_ids']]
        self.fold_weights=[roles.count(f)/len(roles) for f in [0,1]]
        self.output=Path(args.output)
        assert not self.output.exists(),'Use a new run directory; never overwrite a partial or completed validation'
        self.output.mkdir(parents=True)
        write_new(self.output/'input_provenance.json',dict(plan_path=str(self.plan_path),plan_sha256=sha(self.plan_path),
            cache_config_path=str(self.cache_config),cache_config_sha256=sha(self.cache_config),
            cache_completion_sha256=sha(self.root/'completion.json'),condition_chain=self.chain,
            bank_sha256={k:v['sha256'] for k,v in self.banks.items()},fold_weights=self.fold_weights,
            partitions=['T_OOF','Dfit','Dcheck'],V_or_official_benchmark_loaded=False))
        assert sha(ASSET)==p['native_projector_asset_sha256']
        self.projector=ImageProjection(image_embed_dim=1024,cross_attention_dim=2048,num_image_text_embeds=4).eval().requires_grad_(False)
        with safe_open(str(ASSET),framework='pt',device='cpu') as file:
            weights={new:file.get_tensor(old).float() for new,old in [
                ('image_embeds.weight','image_proj.proj.weight'),('image_embeds.bias','image_proj.proj.bias'),
                ('norm.weight','image_proj.norm.weight'),('norm.bias','image_proj.norm.bias')]}
        self.projector.load_state_dict(weights,strict=True)
        assert sum(x.numel() for x in self.projector.parameters())==8400896
        self.scalar_rows=[];self.descriptions=[];self.features={arm:dict(real=[],sim=[]) for arm in ARMS}

    def load(self,subject,name,role):
        path=self.root/subject/f'{name}.pt';receipt=self.receipts[(subject,name)]
        assert sha(path)==receipt['sha256']
        data=torch.load(path,weights_only=True,map_location='cpu');bank=self.banks[role]
        assert data['subject']==subject and data['partition']==role and data['repeat_protocol']=='R4'
        assert data['condition_chain']==self.chain and data['config_sha256']==sha(self.cache_config)
        assert data['coordinate_space']==self.config['coordinate_space'] and data['C_I_bank_sha256']==bank['sha256']
        assert torch.equal(data['array_rows'],bank['array_rows']) and torch.equal(data['C_I_positions'],torch.arange(len(bank['C_I'])))
        assert data['concept_ids']==bank['concept_ids'] and data['stimulus_ids']==bank['stimulus_ids']
        expected=[f'{subject}/{stim}/R4' for stim in bank['stimulus_ids']]
        assert data['query_ids']==expected and len(set(expected))==len(expected)
        seeds=[int(hashlib.sha256(f'1729|{subject}|{stim}|R4|{self.protocol["version"]}'.encode()).hexdigest()[:16],16)%(2**63-1)
               for stim in bank['stimulus_ids']]
        assert data['image_seeds']==seeds
        assert data['prior_seeds']==[int(hashlib.sha256(f'{s}|prior'.encode()).hexdigest()[:16],16)%(2**63-1) for s in seeds]
        if role=='T':
            roles=[f'oof{self.protocol["oof_fold_by_concept"][c]}' for c in bank['concept_ids']]
            assert data['teacher_roles']==roles and set(roles)=={'oof0','oof1'}
            assert data['checkpoint_proofs']=={r:self.oof_proofs[r][subject] for r in ['oof0','oof1']}
        elif name=='D_full':
            assert data['teacher_roles']==['D_full']*1650
            selected=self.selection['frozen_reference_checkpoints'][subject]
            assert data['checkpoint_proofs']=={'full':selected}
            assert sha(selected['encoder_checkpoint'])==selected['encoder_sha256']
            if self.chain=='REF_PRIOR_V':assert sha(selected['prior_checkpoint'])==selected['prior_sha256']
        else:
            teacher=name[2:];assert teacher in ['oof0','oof1']
            assert data['teacher_roles']==[name]*1650
            assert data['checkpoint_proofs']=={teacher:self.oof_proofs[teacher][subject]}
        ce=data['C_E'];assert ce.shape==bank['C_I'].shape and ce.dtype==torch.float32 and torch.isfinite(ce).all()
        return data

    def project(self,x):
        with torch.inference_mode():
            return torch.cat([self.projector(x[i:i+128]).flatten(1) for i in range(0,len(x),128)])

    def subject(self,subject,subject_index):
        t=self.load(subject,'T_oof','T');d=self.load(subject,'D_full','D')
        oof=[self.load(subject,f'D_oof{k}','D')['C_E'] for k in [0,1]]
        ci=self.banks['D']['C_I'];ce=d['C_E'];concepts=d['concept_ids'];ids=d['stimulus_ids']
        model=NativeErrorChannel.fit(self.banks['T']['C_I'],t['C_E'],t['concept_ids'],
            ci[self.fit],torch.stack(oof)[:,self.fit],ce[self.fit],self.fold_weights,
            pca_components=self.plan['pca_components'],neighbors=self.plan['neighbors'],shrinkage=self.plan['shrinkage'])
        folder=self.output/subject;folder.mkdir()
        save_new(folder/'error_channel.pt',dict(state=model.state,
            fit_query_ids=[q for q,use in zip(d['query_ids'],self.fit) if use],
            T_source_sha256=self.receipts[(subject,'T_oof')]['sha256'],
            D_source_sha256={n:self.receipts[(subject,n)]['sha256'] for n in ['D_oof0','D_oof1','D_full']},
            plan_sha256=sha(self.plan_path),actual_scientific_fit=True))
        neighbors=model.nearest(ci,concepts);save_new(folder/'D_nearest_other_T_concepts.pt',neighbors)
        # Semantic-distance strata are set from Dfit distances, never Dcheck.
        pca=model.state['pca'];coordinates=(ci-pca['mean'])@pca['basis']
        nearest_distance=(coordinates-model.state['coordinates'][neighbors[:,0]]).norm(dim=-1)
        bins=torch.quantile(nearest_distance[self.fit],torch.tensor([.25,.5,.75]))
        groups=torch.bucketize(nearest_distance,bins,right=False)
        full_error=(ce-ci)[self.check]
        oof_error=torch.stack(oof)[:,self.check]-ci[self.check][None]
        def weighted_error_moments(value):
            value=value.double();w=torch.tensor(self.fold_weights,dtype=torch.float64)
            mu=(value.mean(1)*w[:,None]).sum(0)
            trace=(((value-mu).square().sum(-1)).mean(1)*w).sum()
            return mu,trace
        real_mu=full_error.double().mean(0);real_trace=(full_error.double()-real_mu).square().sum(-1).mean()
        raw_mu,raw_trace=weighted_error_moments(oof_error)
        corrected=apply_calibration(oof_error,model.state['calibration'])
        corrected_mu,corrected_trace=weighted_error_moments(corrected)
        calibration_drift=dict(Dcheck_queries=830,
            raw_OOF_vs_full_bias_error_norm=float((raw_mu-real_mu).norm()),
            calibrated_OOF_vs_full_bias_error_norm=float((corrected_mu-real_mu).norm()),
            raw_OOF_vs_full_trace_ratio=float(raw_trace/real_trace),
            calibrated_OOF_vs_full_trace_ratio=float(corrected_trace/real_trace),
            Dfit_semantic_distance_quartiles=bins.tolist())
        real_projected=self.project(ce)
        rank_real=fixed_gallery_metrics(ce,ci,ids,ids)
        real_top=(F.normalize(ce,dim=-1)@F.normalize(ci,dim=-1).T).argmax(1)
        real_x=classifier_features(ci,ce,model.state['pca'],subject_index)
        for arm in ARMS:
            simulated=[];source_indices=[]
            for draw in range(self.plan['simulation_draws']):
                seeds=[stream_seed(self.plan['seed'],self.plan['version'],subject,s,'R4','error-draw',draw) for s in ids]
                value,metadata=model.sample(arm,ci,concepts,seeds,neighbors)
                simulated.append(value)
                if arm=='Conditional':source_indices.append(metadata['source_indices'])
            sim=torch.stack(simulated,dim=1);first=sim[:,0]
            native=[];projected=[]
            for start in range(0,len(ci),32):
                selected=sim[start:start+32]
                native.append(energy_score(selected,ce[start:start+32]))
                tokens=self.project(selected.flatten(0,1)).view(len(selected),self.plan['simulation_draws'],-1)
                projected.append(energy_score(tokens,real_projected[start:start+32]))
            native=torch.cat(native);projected=torch.cat(projected)
            first_tokens=self.project(first)
            rank=fixed_gallery_metrics(first,ci,ids,ids)
            top=(F.normalize(first,dim=-1)@F.normalize(ci,dim=-1).T).argmax(1)
            cosine=F.cosine_similarity(first,ci);real_cos=F.cosine_similarity(ce,ci)
            for i,query_id in enumerate(d['query_ids']):
                self.scalar_rows.append(dict(query_id=query_id,subject=subject,stimulus_id=ids[i],concept_id=concepts[i],
                    role='Dfit' if self.fit[i] else 'Dcheck',arm=arm,native_energy_score=float(native[i]),
                    semantic_distance_group=int(groups[i]),
                    projected_energy_score=float(projected[i]),sim_condition_norm=float(first[i].norm()),
                    real_condition_norm=float(ce[i].norm()),sim_target_cosine=float(cosine[i]),real_target_cosine=float(real_cos[i]),
                    sim_native_2AFC=float(rank['two_afc'][i]),real_native_2AFC=float(rank_real['two_afc'][i]),
                    sim_native_Top1=int(rank['top1'][i]),real_native_Top1=int(rank_real['top1'][i]),
                    sim_native_Top5=int(rank['top5'][i]),real_native_Top5=int(rank_real['top5'][i]),
                    projected_response_L2_firstdraw=float((first_tokens[i]-real_projected[i]).norm())))
            description=dict(subject=subject,arm=arm,role='Dcheck',queries=830,
                native_error_distribution=distribution_summary((ce-ci)[self.check],(first-ci)[self.check]),
                condition_distribution=distribution_summary(ce[self.check],first[self.check]),
                wrong_concept_pattern=confusion_js(real_top[self.check],top[self.check],
                    [c for c,use in zip(concepts,self.check) if use],concepts),
                native_energy_score=float(native[self.check].mean()),projected_energy_score=float(projected[self.check].mean()),
                Dcheck_calibration_drift=calibration_drift,
                semantic_distance_groups={str(group):dict(queries=int((self.check&(groups==group)).sum()),
                    native_energy_score=float(native[self.check&(groups==group)].mean())
                        if (self.check&(groups==group)).any() else None,
                    real_condition_target_cosine=float(real_cos[self.check&(groups==group)].mean())
                        if (self.check&(groups==group)).any() else None,
                    simulated_condition_target_cosine=float(cosine[self.check&(groups==group)].mean())
                        if (self.check&(groups==group)).any() else None)
                    for group in range(4)},
                calibration=model.state['calibration']|dict(oof_bias_norm=float(model.state['calibration']['oof_bias'].norm()),
                    full_bias_norm=float(model.state['calibration']['full_bias'].norm())))
            description['calibration'].pop('oof_bias');description['calibration'].pop('full_bias')
            self.descriptions.append(description)
            sample=dict(C_simulated_first_draw=first,query_ids=d['query_ids'],source_residual_indices=source_indices,
                simulation_draws=self.plan['simulation_draws'],plan_sha256=sha(self.plan_path),
                all_draws_reproducible_from_saved_model_and_registered_query_stream=True)
            save_new(folder/f'{arm}_D_validation_samples.pt',sample)
            self.features[arm]['real'].append(real_x)
            self.features[arm]['sim'].append(classifier_features(ci,first,model.state['pca'],subject_index))
            print(json.dumps(dict(stage='full_D_error_validation',subject=subject,arm=arm,
                Dcheck_native_energy_score=description['native_energy_score'],seconds=time.monotonic()-self.started)),flush=True)
        (self.output/'progress.json').write_text(json.dumps(dict(status='RUNNING',subject_completed=subject,
            seconds=time.monotonic()-self.started))+'\n')

    def run(self):
        for index,subject in enumerate(self.subjects):self.subject(subject,index)
        assert len(self.scalar_rows)==66000 and len(self.descriptions)==40
        with (self.output/'per_query.csv').open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(self.scalar_rows[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(self.scalar_rows)
        results={}
        for arm in ARMS:
            rows=[r for r in self.scalar_rows if r['arm']==arm and r['role']=='Dcheck']
            assert len(rows)==8300 and len({r['query_id'] for r in rows})==8300
            results[arm]=dict(queries=8300,subjects=10,concepts=83,
                native_energy_score=float(np.mean([r['native_energy_score'] for r in rows])),
                projected_energy_score=float(np.mean([r['projected_energy_score'] for r in rows])))
            real=torch.cat(self.features[arm]['real']);sim=torch.cat(self.features[arm]['sim'])
            mask=self.fit.repeat(10);assert int(mask.sum())==8200
            classifiers,states=fit_two_sample(real[mask],sim[mask],real[~mask],sim[~mask],self.plan['classifier'])
            results[arm]['two_sample_classifiers']=classifiers
            save_new(self.output/f'{arm}_classifier_states.pt',states)
            print(json.dumps(dict(stage='Dfit_classifiers_evaluated_on_Dcheck',arm=arm,results=classifiers)),flush=True)
        by_arm={arm:{r['query_id']:r for r in self.scalar_rows if r['role']=='Dcheck' and r['arm']==arm} for arm in ARMS}
        keys=sorted(by_arm['Conditional']);subjects=[by_arm['Conditional'][k]['subject'] for k in keys]
        concepts=[by_arm['Conditional'][k]['concept_id'] for k in keys];comparisons={}
        for control in ['Clean','Iso','Moment']:
            assert by_arm[control].keys()==by_arm['Conditional'].keys()
            fields=['native_energy_score','projected_energy_score']
            delta=np.array([[by_arm['Conditional'][k][m]-by_arm[control][k][m] for m in fields] for k in keys])
            stats=paired_summary(delta,subjects,concepts,self.plan['bootstrap_draws'],self.plan['seed'])
            comparisons[control]=dict(direction='Conditional minus control; lower energy score is better',
                metrics={m:dict(paired_delta=stats['mean'][i],CI95_subject_concept=stats['CI95'][i]) for i,m in enumerate(fields)},
                per_subject={s:dict(zip(fields,v)) for s,v in stats['per_subject'].items()},
                queries=len(keys),concepts=83,subjects=10)
        native_better=all(comparisons[c]['metrics']['native_energy_score']['CI95_subject_concept'][1]<0 for c in ['Iso','Moment'])
        projected_better=all(comparisons[c]['metrics']['projected_energy_score']['CI95_subject_concept'][1]<0 for c in ['Iso','Moment'])
        status=('CPU_CONDITIONAL_FIDELITY_PROMISING_REQUIRES_FULL_DENOISER_RESPONSE_AND_BUDGET' if native_better and projected_better
                else 'NO_VALIDATED_CONDITIONAL_INCREMENT_UNDER_LOCKED_CPU_GATE')
        assert not torch.cuda.is_initialized()
        report=dict(status=status,condition_chain=self.chain,seed=1729,training_seeds=1,subjects=10,
            T_OOF_queries_fitted=132300,Dfit_queries=8200,Dcheck_queries=8300,simulation_draws_per_query=self.plan['simulation_draws'],
            results=results,comparisons=comparisons,per_subject_distributions=self.descriptions,
            gate=dict(native_energy_CI_upper_below_zero_vs_Iso_and_Moment=native_better,
                projected_energy_CI_upper_below_zero_vs_Iso_and_Moment=projected_better,
                full_frozen_UNet_response_checked=False,S4_may_start=False,full_matched_adapter_budget_passed=False),
            plan_sha256=sha(self.plan_path),input_provenance_sha256=sha(self.output/'input_provenance.json'),
            per_query_path=str(self.output/'per_query.csv'),per_query_sha256=sha(self.output/'per_query.csv'),
            GPU_initialized=False,images_generated=0,V_or_official_test_accessed=False,
            seconds=time.monotonic()-self.started,
            limits=['CPU native/projector fidelity is not full UNet response or image improvement.',
                'Projection uses original FP32 weights on CPU; deployed image UNet is BF16.',
                'A two-sample classifier near chance does not establish equal distributions or small KL.',
                'One actual deployed CE per query; prior sampling error, when present, is included, not pure physiology.',
                'Dcheck intervals describe subjects/concepts, not training-seed variance or independent final confirmation.',
                'A failed v1 gate does not reject all possible error channels; revisions require diagnosed cause and prospective version.'])
        write_new(self.output/'summary.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['per_subject_distributions','results']}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for arg in ['plan','cache-config','output']:parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    Validator(args).run()


if __name__=='__main__':main()
