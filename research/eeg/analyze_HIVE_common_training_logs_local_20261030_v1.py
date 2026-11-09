"""All saved logs, exact pass totals and snapshot-only gradient interpretation."""
import argparse,datetime,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();E=a.directory
read=lambda p:json.loads(Path(p).read_bytes());s=read(E/'protocol/study.json');reports={}
for arm in s['arms']:
    folder=E/'actual'/arm;events=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()];done=read(folder/'completion.json');cal=read(folder/'calibration.json')
    ends=[x for x in events if x['updates'] in (8269,16538,24807)];assert len(ends)==3 and [x['seen_queries'] for x in ends]==[132300,264600,396900]
    passes=[];previousR=previousA=0.
    for i,x in enumerate(ends):
        totalR=x['cumulative_mean_relative']*x['seen_queries'];totalA=x['cumulative_mean_AP']*x['seen_queries']
        passes.append(dict(pass_number=i+1,trajectory_average_relative=(totalR-previousR)/132300,trajectory_average_AP_scaled_point1=(totalA-previousA)/132300));previousR=totalR;previousA=totalA
    cos=np.asarray([x['gradient_cosine'] for x in events]);rr=np.asarray([x['relative_derivative'] for x in events]);aa=np.asarray([x['AP_derivative'] for x in events]);sizes=np.asarray([12 if x['updates']%8269==0 else 16 for x in events])
    assert len(events)==len(set(x['updates'] for x in events)) and all(x['direction_norm']<=1+1e-6 for x in events)
    reports[arm]=dict(training_seconds=done['training_seconds'],updates=done['updates'],exposures=done['query_exposures'],per_pass_trajectory_averages=passes,
        logged_snapshot_events=len(events),current_batch_queries_in_logged_snapshots=int(sizes.sum()),snapshot_gradient_cosine_mean=float(cos.mean()),snapshot_gradient_cosine_min=float(cos.min()),snapshot_negative_cosines=int((cos<0).sum()),
        snapshot_positive_relative_derivatives=int((rr>0).sum()),snapshot_positive_AP_derivatives=int((aa>0).sum()),
        gradient_scope='Only logged current minibatch snapshots, NOT all24807 gradients or validation or paired image causes',calibration=cal,
        pass_loss_scope='All actual primary exposures via exact cumulative totals along each training trajectory, NOT frozen-epoch evaluation; losses across armteacher-mismatch not directly comparable')
r=dict(status='COMPLETE_ALL4_TRAINING_LOG_TRAJECTORY_AND_FIXED_CALIBRATION_ANALYSIS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=reports,
    new_fit=False,new_images=False,no_winner_reselection=True,no_gradient_conflict_or_overfit_causal_conclusion=True)
(E/'training_log_analysis.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(r))
