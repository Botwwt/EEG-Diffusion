"""Idea-first authority revision; identical cache identity and scientific checks.
The direct chain is fixed prospectively, not selected by partial V scores.
Original builder is retained without changes.
"""
"""Export complete concept-OOF T and paired D conditions from the real chain.

No template h(E) or surrogate interface: use actual frozen encoder caches and,
when selected, the exact deployed v-prior sampler from the protected generator.
C_I is stored once per partition in a shared explicitly indexed tensor bank.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import time

from generate_reference_images import (torch, ROOT, DiffusionPriorUNet,
    sample_prior, query_seed, prior_seed, sha, tensor_sha)


def write_json(path,value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def save_new(path,value):
    assert not path.exists(),('Never overwrite a cache',str(path))
    tmp=path.with_name(path.name+'.tmp');torch.save(value,tmp);os.replace(tmp,path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_role(proof,role,lock,chain):
    assert proof['status']=='COMPLETED' and proof['role']==role
    assert proof['encoder']['unique_training_rows']==len(lock['training_array_rows'])
    assert proof['encoder']['seen_samples']==proof['encoder']['epochs']*len(lock['training_array_rows'])
    assert sha(proof['encoder']['checkpoint'])==proof['encoder']['checkpoint_sha256']
    assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
    if chain=='REF_REG':assert proof['prior'] is None
    else:
        p=proof['prior'];assert p['prediction_type']=='v_prediction' and p['parameters']==9675648 and p['epochs']==400
        assert p['unique_training_rows']==len(lock['training_array_rows']) and p['seen_samples']==400*len(lock['training_array_rows'])
        assert sha(p['checkpoint'])==p['checkpoint_sha256']


class CacheBuilder:
    def __init__(self,path):
        self.started=time.monotonic();self.config_path=Path(path);self.config=json.loads(self.config_path.read_text())
        c=self.config
        assert c['seed']==1729 and c['cache_builder_sha256']==sha(__file__)
        assert c['condition_chain'] in ['REF_REG','REF_PRIOR_V']
        assert c['inference_batch']==4 and c['coordinate_space']=='raw_pooled_CLIP_H_1024_before_native_IP_projection'
        training_config=Path(c['oof_training_config_path'])
        assert sha(training_config)==c['oof_training_config_sha256']
        training=json.loads(training_config.read_text())
        for field in ['seed','condition_chain','output_root','split_sha256','visual_features_path','visual_features_sha256']:
            assert c[field]==training[field],('Cache/training contract mismatch',field)
        selection=ROOT/'audit/idea_first_fixed_chain_v1.json';assert sha(selection)==c['reference_selection_sha256']
        self.selection=json.loads(selection.read_text());assert self.selection['selected_reference']==c['condition_chain']
        gate=ROOT/'audit/idea_first_prerequisite_budget_gate_v1.json';assert sha(gate)==c['budget_gate_sha256']
        assert json.loads(gate.read_text())['full_two_fold_training_and_condition_sampling_funded']
        interface=json.loads((ROOT/'configs/image_evaluation_resolved_batchsafe_v3.json').read_text())
        assert sha(ROOT/'generate_reference_images.py')==interface['source_sha256']['generate_reference_images.py']
        assert sha(ROOT/'audit/query_manifest_R4.csv')==interface['query_manifest_sha256']
        self.protocol=json.loads((ROOT/'configs/split_manifest.json').read_text())
        assert sha(ROOT/'configs/split_manifest.json')==c['split_sha256']
        assert sha(c['visual_features_path'])==c['visual_features_sha256']
        self.visual=torch.load(c['visual_features_path'],weights_only=True,map_location='cpu')
        assert self.visual.shape==(16540,1024) and self.visual.dtype==torch.float32 and torch.isfinite(self.visual).all()
        self.images={int(r['array_row']):r for r in csv.DictReader((ROOT/'audit/image_manifest.csv').open()) if r['dataset_split']=='train'}
        self.queries={};expected_roles={i:role for role in ['T','D'] for i in self.protocol['image_rows'][role]}
        with (ROOT/'audit/query_manifest_R4.csv').open() as file:
            for row in csv.DictReader(file):
                if row['role'] in ['T','D']:
                    key=(row['subject_id'],int(row['array_row']));assert key not in self.queries
                    assert row['role']==expected_roles[key[1]]
                    assert row['image_sha256']==self.images[key[1]]['image_sha256']
                    assert row['stimulus_id']==self.images[key[1]]['stimulus_id'] and row['concept_id']==self.images[key[1]]['concept_id']
                    assert row['query_id']==f'{row["subject_id"]}/{row["stimulus_id"]}/R4'
                    assert int(row['generation_seed'])==query_seed(row['subject_id'],row['stimulus_id'],'R4',self.protocol['version'])
                    self.queries[key]=row
        assert len(self.queries)==148800
        self.out=Path(c['condition_cache_output_root']);self.out.mkdir(parents=True,exist_ok=True)
        assert not (self.out/'completion.json').exists(),'Preserve a completed OOF condition experiment'
        assert shutil.disk_usage(self.out).free>c['minimum_free_bytes_before_cache'],'Insufficient verified storage reserve'
        self.counts=dict(prior_forward_calls=0,prior_sample_evaluations=0,loaded_cached_queries=0,new_computed_conditions=0)
        self.banks={};self.role_locks={};self.role_roots={}
        t=set(self.protocol['image_rows']['T']);d=set(self.protocol['image_rows']['D'])
        for fold in [0,1]:
            role=f'oof{fold}';root=Path(c['output_root'])/role;complete=json.loads((root/'completion.json').read_text())
            assert complete['status']=='COMPLETED' and len(complete['subjects'])==10
            assert {a['subject'] for a in complete['subjects']}==set(self.protocol['subjects'])
            lock=json.loads((root/'OOF_data_roles.json').read_text());assert lock['role']==role and lock['config_sha256']==c['oof_training_config_sha256']
            assert lock['condition_chain']==c['condition_chain']
            train=set(lock['training_array_rows']);held=set(lock['heldout_T_array_rows'])
            assert train.isdisjoint(held) and train|held==t and train.isdisjoint(d)
            expected={i for i in t if self.protocol['oof_fold_by_concept'][self.images[i]['concept_id']]==fold}
            assert held==expected and set(lock['training_concepts']).isdisjoint(lock['heldout_concepts'])
            self.role_locks[role]=lock;self.role_roots[role]=root
        self.make_banks()

    def make_banks(self):
        for role in ['T','D']:
            rows=self.protocol['image_rows'][role]
            metadata=dict(partition=role,array_rows=torch.tensor(rows),
                stimulus_ids=[self.images[i]['stimulus_id'] for i in rows],concept_ids=[self.images[i]['concept_id'] for i in rows],
                image_sha256=[self.images[i]['image_sha256'] for i in rows],C_I=self.visual[rows].contiguous(),
                coordinate_space=self.config['coordinate_space'],visual_source_sha256=self.config['visual_features_sha256'])
            if role=='D':
                fit=set(self.protocol['d_fit_concepts']);check=set(self.protocol['d_check_concepts'])
                assert fit.isdisjoint(check) and fit|check==set(metadata['concept_ids'])
                metadata['calibration_roles']=['fit' if c in fit else 'check' for c in metadata['concept_ids']]
            else:metadata['heldout_folds']=[self.protocol['oof_fold_by_concept'][c] for c in metadata['concept_ids']]
            path=self.out/f'C_I_{role}.pt'
            if path.exists():
                saved=torch.load(path,weights_only=True,map_location='cpu')
                assert saved.keys()==metadata.keys()
                for key,value in metadata.items():
                    assert torch.equal(saved[key],value) if isinstance(value,torch.Tensor) else saved[key]==value
            else:save_new(path,metadata)
            self.banks[role]=dict(path=str(path),sha256=sha(path),array_rows=rows)

    def budget(self):
        if time.monotonic()-self.started>self.config['condition_cache_max_seconds']-45:
            write_json(self.out/'progress.json',dict(status='BUDGET_STOPPED_RESUMABLE',counts=self.counts,seconds=time.monotonic()-self.started))
            raise RuntimeError('Registered cache time reached; keep exact-seed completed parts')

    def load_encoder_cache(self,proof):
        assert sha(proof['condition_cache'])==proof['condition_cache_sha256']
        cache=torch.load(proof['condition_cache'],weights_only=True,map_location='cpu')
        assert torch.equal(cache['array_rows'],torch.arange(16540))
        assert cache['encoder_sha256']==proof['encoder']['checkpoint_sha256']
        assert cache['native_visual_sha256']==self.config['visual_features_sha256']
        assert cache['training_role']==proof['role']
        assert cache['condition'].dtype==torch.float32 and torch.isfinite(cache['condition']).all();return cache['condition']

    def teacher(self,proof):
        if self.config['condition_chain']=='REF_REG':return None
        p=proof['prior'];assert p['prediction_type']=='v_prediction' and sha(p['checkpoint'])==p['checkpoint_sha256']
        model=DiffusionPriorUNet(cond_dim=1024,dropout=.1).cuda().eval().requires_grad_(False)
        model.load_state_dict(torch.load(p['checkpoint'],weights_only=True,map_location='cpu'),strict=True)
        def count(module,inputs):
            self.counts['prior_forward_calls']+=1;self.counts['prior_sample_evaluations']+=len(inputs[0])
        model.register_forward_pre_hook(count);return model

    def collect(self,subject,partition,role,proof,raw,rows):
        """Every existing part is reused only under identical query/model hashes."""
        bank=self.banks[partition];queries=[self.queries[(subject,i)] for i in rows]
        identity=dict(config_sha256=sha(self.config_path),subject=subject,partition=partition,teacher_role=role,
            encoder_sha256=proof['encoder']['checkpoint_sha256'],prior_sha256=proof['prior']['checkpoint_sha256'] if proof['prior'] else None,
            C_I_bank_sha256=bank['sha256'],query_ids=[q['query_id'] for q in queries],repeat_protocol='R4',
            inference_batch=4,coordinate_space=self.config['coordinate_space'])
        model=None;parts=[];values=[]
        for start in range(0,len(rows),128):
            self.budget();selected=rows[start:start+128];q=queries[start:start+128]
            path=self.out/'parts'/subject/f'{partition}_{role}'/f'{start:08d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            expected=fingerprint(identity|dict(start=start,query_ids=[a['query_id'] for a in q]))
            if path.exists():
                saved=torch.load(path,weights_only=True,map_location='cpu');assert saved['fingerprint']==expected
                x=saved['C_E'];self.counts['loaded_cached_queries']+=len(x)
            else:
                if model is None and self.config['condition_chain']=='REF_PRIOR_V':model=self.teacher(proof)
                x=[]
                for k in range(0,len(selected),4):
                    indices=selected[k:k+4];batch=q[k:k+4];c=raw[indices]
                    if model is None:y=c.clone()
                    else:
                        before=self.counts['prior_sample_evaluations']
                        with torch.inference_mode():y=sample_prior(model,c.cuda(),[int(a['generation_seed']) for a in batch],prediction_type='v_prediction').cpu()
                        assert self.counts['prior_sample_evaluations']-before==100*len(batch)
                    x.append(y.float().contiguous())
                x=torch.cat(x);save_new(path,dict(fingerprint=expected,query_ids=[a['query_id'] for a in q],C_E=x))
                self.counts['new_computed_conditions']+=len(x)
            assert x.shape==(len(selected),1024) and torch.isfinite(x).all()
            parts.append(dict(path=str(path),sha256=sha(path),queries=len(selected)));values.append(x)
        del model
        if self.config['condition_chain']=='REF_PRIOR_V':torch.cuda.empty_cache()
        return torch.cat(values),parts,identity

    def export(self,subject,partition,ce,roles,proofs,parts):
        rows=self.banks[partition]['array_rows'];queries=[self.queries[(subject,i)] for i in rows]
        name='T_oof' if partition=='T' else roles[0]
        folder=self.out/subject;folder.mkdir(exist_ok=True);path=folder/f'{name}.pt';receipt=folder/f'{name}_completion.json'
        data=dict(subject=subject,partition=partition,repeat_protocol='R4',array_rows=torch.tensor(rows),
            query_ids=[q['query_id'] for q in queries],stimulus_ids=[q['stimulus_id'] for q in queries],
            concept_ids=[q['concept_id'] for q in queries],teacher_roles=roles,
            image_seeds=[int(q['generation_seed']) for q in queries],prior_seeds=[prior_seed(int(q['generation_seed'])) for q in queries],
            C_E=ce,C_I_bank_path=self.banks[partition]['path'],C_I_bank_sha256=self.banks[partition]['sha256'],
            C_I_positions=torch.arange(len(rows)),condition_chain=self.config['condition_chain'],
            coordinate_space=self.config['coordinate_space'],checkpoint_proofs=proofs,config_sha256=sha(self.config_path))
        assert len(ce)==len(rows)==len(roles) and torch.isfinite(ce).all()
        assert len(set(data['query_ids']))==len(rows)
        if path.exists():
            saved=torch.load(path,weights_only=True,map_location='cpu')
            assert saved['config_sha256']==data['config_sha256'] and saved['query_ids']==data['query_ids'] and torch.equal(saved['C_E'],ce)
        else:save_new(path,data)
        checked=torch.load(path,weights_only=True,map_location='cpu');assert torch.equal(checked['C_E'],ce)
        completion=dict(status='COMPLETED',path=str(path),sha256=sha(path),queries=len(rows),config_sha256=sha(self.config_path),
            condition_chain=self.config['condition_chain'],C_I_bank=self.banks[partition],parts=parts,
            C_E_norm_mean=float(ce.norm(dim=-1).mean()),source_part_data_preserved_in_verified_merged_tensor=True)
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['sha256']==completion['sha256'] and old['config_sha256']==completion['config_sha256']
        else:write_json(receipt,completion)
        # These are redundant cache shards, not experimental checkpoints/images.
        # Exact merged values and each input shard SHA were verified above.
        for p in parts:
            target=Path(p['path']).resolve();assert target.is_relative_to((self.out/'parts').resolve())
            assert sha(target)==p['sha256'];target.unlink()
        print(json.dumps(dict(subject=subject,export=name,queries=len(rows),C_E_norm_mean=completion['C_E_norm_mean'])),flush=True)
        return completion

    def existing_export(self,subject,partition,name,roles,proofs):
        folder=self.out/subject;receipt=folder/f'{name}_completion.json'
        if not receipt.exists():return None
        report=json.loads(receipt.read_text());path=folder/f'{name}.pt'
        assert report['status']=='COMPLETED' and sha(path)==report['sha256']
        assert report['config_sha256']==sha(self.config_path)
        data=torch.load(path,weights_only=True,map_location='cpu');rows=self.banks[partition]['array_rows']
        assert data['config_sha256']==sha(self.config_path) and data['subject']==subject and data['partition']==partition
        assert data['query_ids']==[self.queries[(subject,i)]['query_id'] for i in rows]
        assert torch.equal(data['array_rows'],torch.tensor(rows)) and data['teacher_roles']==roles
        assert data['checkpoint_proofs']==proofs and data['C_I_bank_sha256']==self.banks[partition]['sha256']
        assert data['C_E'].shape==(len(rows),1024) and data['C_E'].dtype==torch.float32 and torch.isfinite(data['C_E']).all()
        self.counts['loaded_cached_queries']+=len(rows)
        return report

    def run(self):
        all_reports=[]
        for subject in self.protocol['subjects']:
            reports=[];t=self.protocol['image_rows']['T'];d=self.protocol['image_rows']['D']
            roles=[f'oof{self.protocol["oof_fold_by_concept"][self.images[i]["concept_id"]]}' for i in t]
            proof_map={};proofs={};t_parts=[]
            for fold in [0,1]:
                role=f'oof{fold}';root=self.role_roots[role];proof=json.loads((root/subject/'completion.json').read_text())
                assert proof['subject']==subject and proof['training_seed']==1729 and proof['split_sha256']==self.config['split_sha256']
                verify_role(proof,role,self.role_locks[role],self.config['condition_chain']);proofs[role]=proof
                proof_map[role]=dict(encoder_sha256=proof['encoder']['checkpoint_sha256'],prior_sha256=proof['prior']['checkpoint_sha256'] if proof['prior'] else None)
            old_t=self.existing_export(subject,'T','T_oof',roles,proof_map)
            result=torch.empty((len(t),1024)) if old_t is None else None
            valid=torch.zeros(len(t),dtype=torch.bool)
            for fold in [0,1]:
                role=f'oof{fold}';proof=proofs[role]
                old_d=self.existing_export(subject,'D',f'D_{role}',[f'D_{role}']*len(d),{role:proof_map[role]})
                if old_t is None or old_d is None:raw=self.load_encoder_cache(proof)
                if old_t is None:
                    held=set(self.role_locks[role]['heldout_T_array_rows'])
                    chosen=[(k,i) for k,i in enumerate(t) if i in held]
                    positions=[k for k,_ in chosen];rows=[i for _,i in chosen]
                    x,parts,_=self.collect(subject,'T',role,proof,raw,rows)
                    assert not valid[positions].any();result[positions]=x;valid[positions]=True;t_parts+=parts
                if old_d is None:
                    d_ce,d_parts,_=self.collect(subject,'D',role,proof,raw,d)
                    old_d=self.export(subject,'D',d_ce,[f'D_{role}']*len(d),{role:proof_map[role]},d_parts)
                reports.append(old_d)
            if old_t is None:
                assert valid.all() and set(roles)=={'oof0','oof1'}
                old_t=self.export(subject,'T',result,roles,proof_map,t_parts)
            reports.append(old_t)
            full=json.loads((Path(self.config['full_reference_root'])/subject/'completion.json').read_text())
            assert full['subject']==subject and full['role']=='full'
            selected=self.selection['frozen_reference_checkpoints'][subject]
            assert selected['encoder_sha256']==full['encoder']['checkpoint_sha256']
            if self.config['condition_chain']=='REF_PRIOR_V':
                full=full|dict(prior=dict(prediction_type='v_prediction',checkpoint=selected['prior_checkpoint'],checkpoint_sha256=selected['prior_sha256']))
            else:full=full|dict(prior=None)
            old_d=self.existing_export(subject,'D','D_full',['D_full']*len(d),{'full':selected})
            if old_d is None:
                raw=self.load_encoder_cache(full);x,parts,_=self.collect(subject,'D','full',full,raw,d)
                old_d=self.export(subject,'D',x,['D_full']*len(d),{'full':selected},parts)
            reports.append(old_d)
            all_reports.extend(reports)
            write_json(self.out/'progress.json',dict(status='RUNNING',subjects_completed=subject,counts=self.counts,seconds=time.monotonic()-self.started))
        assert len(all_reports)==40
        assert sum(r['queries'] for r in all_reports)==132300+3*16500
        completion=dict(status='COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS',subjects=10,
            T_OOF_queries=132300,D_queries_per_teacher=16500,D_teachers=['oof0','oof1','full'],
            condition_chain=self.config['condition_chain'],all_T_queries_have_genuinely_heldout_teacher=True,
            number_of_training_seeds=1,seed=1729,prior_samples_per_query_per_teacher=1 if self.config['condition_chain']=='REF_PRIOR_V' else 0,
            counts_this_attempt=self.counts,partitions=['T','D'],V_or_official_test_conditions_exported=False,
            config_sha256=sha(self.config_path),reports=all_reports,seconds=time.monotonic()-self.started,
            limits=['Each T query has exactly one concept-held-out teacher; the two D fold predictions are model roles, not independent EEG repetitions.',
                'Prior errors include fixed stochastic-prior sampling and decoding error; not pure physiological uncertainty.',
                'This is a condition cache, not a trained error channel, adapter result or image gain.'])
        write_json(self.out/'completion.json',completion);print(json.dumps({k:v for k,v in completion.items() if k!='reports'}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    builder=CacheBuilder(args.config)
    if builder.config['condition_chain']=='REF_PRIOR_V':
        cmd=Path('/proc')/str(os.getppid())/'cmdline'
        assert cmd.exists() and str(ROOT/'run_gpu_job.py').encode() in cmd.read_bytes(),'Use the registered GPU wrapper for sampled conditions'
    builder.run()


if __name__=='__main__':main()
