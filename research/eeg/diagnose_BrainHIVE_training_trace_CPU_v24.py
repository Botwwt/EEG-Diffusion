"""Immutable T-only runtime trace; no model change or image-success inference."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parent
WORK=Path('/root/brainhive_reproduction_20261020_v1')


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    source=WORK/'components/bounded_v1/risk/training.jsonl'
    raw=source.read_bytes();last=raw.rfind(b'\n');assert last>=0
    complete=raw[:last+1]
    events=[json.loads(line) for line in complete.splitlines()]
    assert events and all(x['stage']=='BRAINHIVE_REAL_OOF_IMAGE_TRAINING' and x['arm']=='risk' for x in events)
    # A restart can replay an earlier checkpoint. Do not silently deduplicate or
    # splice such branches into one curve; that would need a separate diagnosis.
    assert all(b['updates']>a['updates'] and b['seen_queries']>a['seen_queries'] for a,b in zip(events,events[1:]))
    intervals=[];previous_sum=0.;previous_seen=0
    for event in events:
        for key in ['cumulative_mean_loss','last_loss','epsilon_loss','token_loss','risk_auxiliary_loss',
                    'gradient_norm','correction_RMS','mean_gate','seconds']:
            assert math.isfinite(event[key]),(key,event['updates'])
        summed=event['epsilon_loss']+.1*event['token_loss']+.1*event['risk_auxiliary_loss']
        assert abs(summed-event['last_loss'])<1e-6
        cumulative_sum=event['cumulative_mean_loss']*event['seen_queries']
        mean=(cumulative_sum-previous_sum)/(event['seen_queries']-previous_seen)
        assert mean>=0
        intervals.append(dict(updates=event['updates'],seen_queries=event['seen_queries'],
            interval_queries=event['seen_queries']-previous_seen,exact_interval_total_mean=mean))
        previous_sum=cumulative_sum;previous_seen=event['seen_queries']
    steady=[(b['seconds']-a['seconds'])/(b['updates']-a['updates']) for a,b in zip(events,events[1:])]
    seconds_per_update=statistics.median(steady)
    out=ROOT/'audit/BrainHIVE_training_trace_v24.json';snapshot=out.with_suffix('.jsonl')
    assert not out.exists() and not snapshot.exists()
    snapshot.write_bytes(complete)
    progress=events[-1]
    receipt=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        status='ACTUAL_T_TRAINING_TRACE_OBJECTIVE_AND_CUMULATIVE_INTERVAL_ARITHMETIC_VERIFIED',
        source=str(source),source_prefix_bytes=len(complete),source_prefix_sha256=digest(complete),
        immutable_log_snapshot=str(snapshot),ignored_only_incomplete_tail_bytes=len(raw)-len(complete),
        snapshot_sha256=digest(snapshot.read_bytes()),entry_source_sha256=digest(Path(__file__).read_bytes()),
        observed_updates=progress['updates'],expected_updates=progress['expected_updates'],
        observed_queries=progress['seen_queries'],expected_query_exposures=progress['expected_queries'],
        logged_event_count=len(events),sampled_last_batches_do_not_cover_every_update=True,
        interval_means_use_all_queries_between_cumulative_totals=True,
        median_observed_seconds_per_update=seconds_per_update,
        fixed_schedule_training_only_remaining_hours_linear_estimate=(progress['expected_updates']-progress['updates'])*seconds_per_update/3600,
        estimate_excludes_calibration_generation_and_evaluation=True,
        sampled_nonfinite_loss_or_gradient_count=0,
        sampled_gradient_norm_is_before_clipping=True,
        source_checks_finite_loss_and_clipped_gradient_each_update=True,
        first_cumulative_mean_loss=events[0]['cumulative_mean_loss'],
        last_cumulative_mean_loss=progress['cumulative_mean_loss'],
        sampled_last_loss_range=[min(x['last_loss'] for x in events),max(x['last_loss'] for x in events)],
        events=events,intervals=intervals,
        changes_to_model_or_schedule_or_selection=0,new_GPU_jobs=0,new_images=0,
        V_or_official_test_data_or_scores_read=False,
        interpretations=['Random samples, noise and timesteps vary between batches; a nonmonotone sampled loss is not by itself a failure diagnosis.',
            'These are current BrainHIVE training logs, not a retrospective explanation of old C2 logs.',
            'Loss arithmetic and finite gradients do not establish image fidelity, generalization, or component gain.',
            'The mean training gate is uncalibrated conditional prediction risk, not physiological EEG SNR.',
            'Raw correction RMS is not the standardized radius constraint and must not be labeled as a0.25 RMS bound.'],
        component_image_gain_established=False,scientific_goal_complete=False)
    out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['events','intervals','interpretations']}),flush=True)


if __name__=='__main__':main()
