"""Independent scalar/first-real-trace audit; never executes remote source."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def audit(directory):
    report_path = directory/'actual_all256_tangent_diagnostics.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    sample = directory/'first_native_trace_and_actual_projection_weights.npz'
    pin = report['first_case_npz']
    assert sample.stat().st_size == pin['bytes']
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == pin['sha256']
    per, cases = report['per_trace'], report['per_common_token_case']
    assert len(per) == 256 and len(cases) == 16
    identities = {(r['arm'],r['query_index'],r['step'],r['layer']) for r in per}
    assert len(identities) == 256
    for case in cases:
        rows = [r for r in per if (r['arm'],r['query_index']) == (case['arm'],case['query_index'])]
        assert len(rows) == case['traces'] == 16
        assert {(r['step'],r['layer']) for r in rows} == {(s,l) for s in range(4) for l in [0,23,46,69]}
    identity_error = 0.
    for row in per+cases:
        a,b,c = [row[k] for k in ['key_squared_norm','value_squared_norm','key_value_dot']]
        assert a >= 0 and b >= 0 and abs(c) <= np.sqrt(a*b)+1e-12
        identity_error = max(identity_error, abs(row['shared_squared_norm']-(a+b+2*c)))
        assert abs(row['shared_along_negative_value_derivative']+(b+c)) < 1e-12
        assert abs(row['separated_value_along_negative_value_derivative']+b) < 1e-12
        assert abs(row['negative_shared_gradient_derivative']+row['shared_squared_norm']) < 1e-12
        assert abs(row['key_value_cosine']-c/np.sqrt(a*b)) < 1e-12
    summary_error = 0.
    for arm in [0,1]:
        for label, values in [('trace',per),('common_token_16trace',cases)]:
            values=[r for r in values if r['arm']==arm]
            original=report['arm_summary'][str(arm)][label]
            assert original['count']==len(values)
            assert original['negative_key_value_dot_count']==sum(r['key_value_dot']<0 for r in values)
            assert original['shared_negative_value_is_ascent_count']==sum(r['shared_along_negative_value_derivative']>0 for r in values)
            cosines=[r['key_value_cosine'] for r in values]
            rebuilt=dict(key_value_cosine_mean=np.mean(cosines),key_value_cosine_min=min(cosines),key_value_cosine_max=max(cosines))
            for name in ['key_squared_norm','value_squared_norm','key_value_dot']:
                rebuilt[name+'_mean']=np.mean([r[name] for r in values])
            summary_error=max(summary_error,max(abs(original[k]-v) for k,v in rebuilt.items()))
    z=np.load(sample,allow_pickle=False)
    q,k,v,target=[z[n] for n in ['q','k','v','target']]
    wk,wv=z['wk'].astype(np.float64),z['wv'].astype(np.float64)
    h_count,n,d=k.shape
    dk,dv=np.zeros_like(k),np.zeros_like(v)
    for h in range(h_count):
        for s in range(q.shape[1]):
            logits=k[h]@q[h,s]/np.sqrt(d)
            p=np.exp(logits-logits.max());p/=p.sum()
            output=p@v[h]
            go=2*(output-target[h,s])/target.size
            dv[h]+=np.outer(p,go)
            # Explicit 4x4 softmax Jacobian, independent of server vectorization.
            jacobian=np.diag(p)-np.outer(p,p)
            dl_dscore=jacobian@(v[h]@go)
            dk[h]+=np.outer(dl_dscore,q[h,s])/np.sqrt(d)
    zk=dk.transpose(1,0,2).reshape(n,-1)@wk
    zv=dv.transpose(1,0,2).reshape(n,-1)@wv
    gradient_error=max(float(np.max(np.abs(zk-z['key_gradient']))),float(np.max(np.abs(zv-z['value_gradient']))))
    assert gradient_error < 1e-12
    def loss(x,y):
        total=0.
        for h in range(h_count):
            for s in range(q.shape[1]):
                logits=x[h]@q[h,s]/np.sqrt(d)
                p=np.exp(logits-logits.max());p/=p.sum()
                total+=np.sum((p@y[h]-target[h,s])**2)
        return total/target.size
    direction=np.random.default_rng(1729).normal(size=zk.shape)
    direction/=np.linalg.norm(direction)
    project=lambda w:(direction@w.T).reshape(n,h_count,d).transpose(1,0,2)
    kd,vd=project(wk),project(wv)
    step=1e-5;fd_errors={}
    for mode in ['key','value','shared']:
        ka=kd if mode!='value' else np.zeros_like(k)
        va=vd if mode!='key' else np.zeros_like(v)
        fd=(loss(k+step*ka,v+step*va)-loss(k-step*ka,v-step*va))/(2*step)
        grad=zk if mode=='key' else zv if mode=='value' else zk+zv
        fd_errors[mode]=abs(fd-float(np.sum(grad*direction)))
    assert max(fd_errors.values()) < 1e-7
    mean_key_error=float(np.max(np.abs(zk.sum(0))))
    assert mean_key_error < 1e-12
    result=dict(status='PASS_INDEPENDENT_LOCAL_SCALARS_AND_FIRST_NATIVE_TRACE',
        all256_trace_identities_checked=True,all16_case_identities_checked=True,
        all272_reported_norm_and_descent_identities_checked=True,
        all_arm_summary_error_max=float(summary_error),norm_decomposition_error_max=float(identity_error),
        first_actual_native_trace_explicit_Jacobian_token_gradient_error=gradient_error,
        first_actual_native_trace_finite_difference_errors=fd_errors,
        first_actual_native_trace_key_gradient_token_sum_error=mean_key_error,
        first_trace_finite_difference_step=step,first_trace_finite_difference_tolerance=1e-7,
        other255_full_projection_gradients_NOT_recomputed_locally=True,
        per_stimulus_aggregate_gradients_NOT_recomputed_locally=True,
        no_parameter_or_final_image_gradient_verification=True,
        actual_report_SHA256=hashlib.sha256(report_path.read_bytes()).hexdigest())
    (directory/'independent_local_numeric_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True,type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
