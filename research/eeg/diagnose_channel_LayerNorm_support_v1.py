"""Read-only support check of completed q-v2; no new fitting or simulator version.

ATMS ends in learned affine LayerNorm. Inverting its learned affine transform
must give near-zero coordinate mean for genuine full-teacher outputs. Record
this invariant on all existing Dcheck rows before proposing another model.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from direct_mean_error_channel_v2 import predict, sample, tensor_sha
from native_error_channel import stream_seed

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    torch.set_num_threads(2)
    os.nice(10)
    started = time.monotonic()
    config = json.loads((ROOT/'configs/idea_first_native_cache_v1.json').read_text())
    protocol_path = ROOT/'configs/split_manifest.json'
    assert sha(protocol_path) == config['split_sha256']
    protocol = json.loads(protocol_path.read_text())
    plan_path = ROOT/'configs/direct_mean_channel_exploratory_v2.json'
    plan = json.loads(plan_path.read_text())
    output = Path(plan['output'])
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'COMPLETED_FULL10_CPU_EXPLORATORY_DIAGNOSIS'
    assert summary['plan_sha256'] == sha(plan_path)
    proof_path = ROOT/'audit/direct_mean_channel_v2_LayerNorm_support_v1.json'
    assert not proof_path.exists()
    cache = Path(config['condition_cache_output_root'])
    tbank = torch.load(cache/'C_I_T.pt', weights_only=True, map_location='cpu')
    dbank = torch.load(cache/'C_I_D.pt', weights_only=True, map_location='cpu')
    check = torch.tensor([c in set(protocol['d_check_concepts']) for c in dbank['concept_ids']])
    folds = torch.tensor([protocol['oof_fold_by_concept'][c] for c in tbank['concept_ids']])
    with (output/'per_query.csv').open(newline='') as file:
        original = {r['query_id']:r for r in csv.DictReader(file)}
    assert len(original) == 8300 and sha(output/'per_query.csv') == summary['per_query_sha256']
    records, subjects = [], []
    for subject in protocol['subjects']:
        model_path = output/subject/'mean_models_and_calibration.pt'
        model = torch.load(model_path, weights_only=True, map_location='cpu')
        expected = next(x for x in summary['per_subject'] if x['subject']==subject)
        assert sha(model_path) == expected['model_sha256']
        tpath, dpath = cache/subject/'T_oof.pt', cache/subject/'D_full.pt'
        assert sha(tpath) == model['T_source_sha256']
        t = torch.load(tpath, weights_only=True, map_location='cpu')
        d = torch.load(dpath, weights_only=True, map_location='cpu')
        means = torch.empty_like(t['C_E'])
        for fold in [0,1]:
            mask = folds==fold
            means[mask] = predict(model['models'][f'holdout{fold}'], tbank['C_I'][mask])
        residual = t['C_E']-means
        assert tensor_sha(residual) == model['residual_sha256']
        neighbors = torch.load(Path(plan['v1_completed_validation'])/subject/'D_nearest_other_T_concepts.pt', weights_only=True, map_location='cpu')
        seeds = [stream_seed(1729,'native_error_channel_full_Dcheck_v1',subject,stim,'R4','error-draw',0) for stim in d['stimulus_ids']]
        simulated, _ = sample(predict(model['models']['full'], dbank['C_I']),residual,neighbors,seeds,model['calibration'])
        completion_path = Path(config['full_reference_root'])/subject/'completion.json'
        complete = json.loads(completion_path.read_text())
        checkpoint = complete['encoder']['checkpoint']
        assert sha(checkpoint) == complete['encoder']['checkpoint_sha256']
        weights = torch.load(checkpoint, weights_only=True, map_location='cpu')
        gamma, beta = weights['proj_eeg.2.weight'], weights['proj_eeg.2.bias']
        assert gamma.shape == beta.shape == (1024,) and gamma.abs().min() > 1e-6
        values = {}
        for arm, conditions in [('real',d['C_E']),('qv2',simulated)]:
            z = (conditions.double()-beta.double())/gamma.double()
            values[arm] = dict(inverse_affine_coordinate_mean_abs=z.mean(-1).abs(),
                inverse_affine_second_moment_defect=(z.square().mean(-1)-1).abs())
        for i, query in enumerate(d['query_ids']):
            if not check[i]:
                continue
            row = original[query]
            assert abs(float(simulated[i].norm())-float(row['sim_norm'])) < 1e-5
            assert abs(float(d['C_E'][i].norm())-float(row['real_norm'])) < 1e-5
            records.append(dict(query_id=query,subject=subject,stimulus_id=d['stimulus_ids'][i],
                **{arm+'_'+name:float(value[i]) for arm, measurements in values.items() for name,value in measurements.items()}))
        subjects.append(dict(subject=subject,D_full_cache_sha256=sha(dpath),
            encoder_sha256=sha(checkpoint), completion_sha256=sha(completion_path),
            gamma_min_absolute=float(gamma.abs().min()),gamma_max_absolute=float(gamma.abs().max()),
            model_sha256=sha(model_path)))
    assert len(records) == len({r['query_id'] for r in records}) == 8300
    csv_path = output/'LayerNorm_support_per_query_v1.csv'
    with csv_path.open('x',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    statistics = {}
    for field in list(records[0])[3:]:
        value = torch.tensor([r[field] for r in records],dtype=torch.float64)
        statistics[field] = dict(mean=float(value.mean()),q50=float(value.median()),q95=float(torch.quantile(value,.95)),max=float(value.max()),fraction_above_fixed1e_minus4=float((value>1e-4).double().mean()))
    result = dict(status='COMPLETED_READ_ONLY_ALL8300_LAYER_NORM_SUPPORT_DIAGNOSIS',
        queries=8300,subjects=subjects,statistics=statistics,source_sha256=sha(__file__),
        qv2_summary_sha256=sha(output/'summary.json'),per_query_sha256=sha(csv_path),
        numerical_tolerance=1e-4,tolerance_chosen_before_this_readout=True,
        fits_performed=0,simulator_versions_added=0,GPU_used=False,images_generated=0,
        Dcheck_already_exposed=True,scientific_goal_complete=False,
        interpretation_limit='A support mismatch can explain easy discrimination; it does not quantify this classifier attribution or the cause of image failure. No condition is projected or changed here.',
        seconds=time.monotonic()-started)
    assert not torch.cuda.is_initialized()
    proof_path.write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='qv2_read_only_LayerNorm_support_v1',seconds=result['seconds'],receipt=str(proof_path),receipt_sha256=sha(proof_path),GPU_used=False))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='subjects'}))


if __name__ == '__main__':
    main()
