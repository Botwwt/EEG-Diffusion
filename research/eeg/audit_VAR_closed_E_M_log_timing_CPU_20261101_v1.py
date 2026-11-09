"""Independent scalar Decimal aggregation of every real closed E/M log row."""
from decimal import Decimal,localcontext
import hashlib
import json
from pathlib import Path
import time

L=Path(__file__).resolve().parent;ROOT=L/'audit/VAR_closed_E_M_log_timing_20261101_v1'
def read(path):return json.loads(Path(path).read_bytes())
def close(a,b,tol=Decimal('1e-12')):
    error=abs(Decimal(str(a))-Decimal(str(b)));assert error<tol,(a,b,error);return float(error)
def audit():
    started=time.perf_counter();scope=read(ROOT/'analysis_scope.json');analysis=read(ROOT/'all_epoch_analysis.json')
    for pin in scope['source_pins'].values():
        p=Path(pin['path']);raw=p.read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
    comparisons=0;maximum=0.;total=0;rows=[]
    with localcontext() as ctx:
        ctx.prec=50
        for arm in ('coupled','visual'):
            events=read(scope['source_pins'][arm+'/fit_events.json']['path']);total+=len(events)
            for epoch in list(range(1,11))+[None]:
                selected=[item for item in events if epoch is None or item['epoch']==epoch]
                value=analysis['results'][arm]['all' if epoch is None else 'epochs']
                if epoch is not None:value=value[str(epoch)]
                n=sum(item['queries'] for item in selected);assert n==value['queries']
                columns=['EEG_KL_nats_per_token','preclip_EEG_gradient','backbone_gradient','ELBO_nats_per_token','NLL_nats_per_query','KL_nats_per_query']
                scalar={name:sum(Decimal(str(item[name]))*item['queries'] for item in selected)/n for name in columns}
                clipped_queries=sum(item['queries'] for item in selected if item['preclip_EEG_gradient']>1)
                clipped_batches=sum(item['preclip_EEG_gradient']>1 for item in selected)
                scalar['clip_active_query_fraction']=Decimal(clipped_queries)/n
                scalar['clip_active_batch_fraction']=Decimal(clipped_batches)/len(selected)
                clip=Decimal(0);energy=Decimal(0);error=Decimal(0);drifts=[]
                for item in selected:
                    g=Decimal(str(item['preclip_EEG_gradient']));b=Decimal(str(item['backbone_gradient']));count=item['queries']
                    clip+=min(Decimal(1),Decimal(1)/max(g,Decimal('1e-30')))*count
                    energy+=(b*b/(g*g) if g else Decimal(0))*count
                    drift=Decimal(str(item['EEG_KL_nats_per_token']))-Decimal(str(item['KL_nats_per_query']))/5
                    drifts.append((drift,count))
                    error=max(error,abs(Decimal(str(item['ELBO_nats_per_token']))-(Decimal(str(item['NLL_nats_per_query']))+Decimal(str(item['KL_nats_per_query'])))/5))
                scalar['nominal_gradient_clip_multiplier_query_mean']=clip/n
                scalar['backbone_preclip_gradient_energy_fraction_query_mean']=energy/n
                scalar['maximum_logged_ELBO_mean_identity_rounding_error']=error
                gradients=sorted(item['preclip_EEG_gradient'] for item in selected);middle=len(gradients)//2
                scalar['preclip_gradient_median']=Decimal(str(gradients[middle])) if len(gradients)%2 else (Decimal(str(gradients[middle-1]))+Decimal(str(gradients[middle])))/2
                if arm=='coupled':
                    scalar['last_E_step_KL_change_per_token_query_mean']=sum(x*count for x,count in drifts)/n
                    scalar['last_E_step_KL_increased_query_fraction']=Decimal(sum(count for x,count in drifts if x>0))/n
                    scalar['last_E_step_KL_increased_batch_fraction']=Decimal(sum(x>0 for x,count in drifts))/len(drifts)
                    sort=sorted(x for x,count in drifts);middle=len(sort)//2
                    scalar['last_E_step_KL_change_per_token_median']=sort[middle] if len(sort)%2 else (sort[middle-1]+sort[middle])/2
                for key,x in scalar.items():maximum=max(maximum,close(x,value[key]));comparisons+=1
                assert value['batches']==len(selected) and error<Decimal('5e-6')
                rows.append(dict(arm=arm,epoch=epoch,records=len(selected),queries=n))
    assert total==4140 and len(rows)==22
    result=dict(status='PASS_INDEPENDENT_ALL4140_LOG_ROWS_AND22_GROUP_SCALAR_DECIMAL_AUDIT',
        all_log_records=total,groups=len(rows),quantities_compared=comparisons,maximum_decimal_vs_NumPy_error=maximum,
        full_group_coverage=rows,seconds=time.perf_counter()-started,
        data_are_training_logs_not_independent_validation=True,no_optimizer_or_causal_explanation_proved=True,
        no_EEG_model_or_GPU_loaded=True,new_images=0,new_training_updates=0)
    (ROOT/'independent_scalar_audit.json').open('x',encoding='utf-8').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','all_log_records','groups','quantities_compared','maximum_decimal_vs_NumPy_error')}))
if __name__=='__main__':audit()
