"""Independent QR and support-bound verification of actual 400x4 outputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();d=np.load(a.directory/'all400_geometry_inputs_and_outputs.npz',allow_pickle=False)
cn,co,g,b,m=[d[k].astype(np.float64) for k in ('native','ordinary','gamma','beta','population')];un=(cn-b)/g;uo=(co-b)/g;cap=1/np.sqrt(1+.25**2);reports={}
for key in d.files:
    if not key.endswith('_projected'):continue
    arm=key[:-10];out=d[key].astype(np.float64);u=(out-b)/g;v=uo+(d[arm+'_relative'].astype(np.float64)-d['initial_F'].astype(np.float64))/g
    errors=[];optimal=[]
    for i in range(400):
        # Different basis construction from production's analytic Gram-Schmidt.
        Q=np.linalg.qr(np.stack([np.ones(1024),g[i]*m],axis=1),mode='reduced')[0]
        fixed=Q@(Q.T@uo[i]);null=lambda z:z-Q@(Q.T@z)
        rr=np.linalg.norm(uo[i]);R=np.sqrt(rr**2-fixed@fixed);h=null(un[i]);hn=np.linalg.norm(h);axis=h/hn;w=null(v[i]);wn=np.linalg.norm(w)
        tau=(rr*np.linalg.norm(un[i])*cap-un[i]@fixed)/(R*hn)
        z=w/wn
        if z@axis<tau:
            orth=w-(w@axis)*axis;z=tau*axis+np.sqrt(1-tau*tau)*orth/np.linalg.norm(orth)
            upper=tau*(w@axis)+np.sqrt(1-tau*tau)*np.linalg.norm(orth)
        else:upper=wn
        expected=fixed+R*z;expected_c=b[i]+g[i]*expected
        errors.append(np.max(np.abs(out[i]-expected_c)/(1+np.abs(expected_c))))
        actual_z=null(u[i])/R;optimal.append(abs(w@actual_z-upper)/(1+abs(upper)))
    norm=np.max(abs(np.linalg.norm(u,axis=1)/np.linalg.norm(uo,axis=1)-1));mean=np.max(abs(np.mean(u-uo,axis=1)));score=np.max(abs((out-co)@m));cos=np.min(np.sum(u*un,axis=1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(un,axis=1)))
    assert max(errors)<2e-6 and norm<2e-6 and mean<2e-6 and score<2e-6 and cos>=cap-2e-6 and max(optimal)<2e-6
    reports[arm]=dict(queries=400,max_independent_QR_output_scaled_error=max(errors),max_support_bound_scaled_error=max(optimal),norm_relative_error=norm,mean_error=mean,linear_score_error=score,minimum_native_cosine=cos)
assert len(reports)==4
result=dict(status='PASS_INDEPENDENT_LOCAL_QR_AND_SUPPORT_BOUND_ALL1600_REAL_CONDITIONS',reports=reports,queries_per_arm=400,arms=4,no_generator_response_or_image_claim=True)
(a.directory/'independent_local_geometry.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
