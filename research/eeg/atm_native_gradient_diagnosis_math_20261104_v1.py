"""Classical gradient diagnostics; never an EEG method or causal guarantee."""
import numpy as np
GROUPS=('temperature','EEG_transformer','EEG_CNN','input_projection','residual_projection','output_affine','unused_native_subject_linear')
def group(name):
    if name=='logit_scale':return GROUPS[0]
    for prefix,target in [('encoder.',GROUPS[1]),('enc_eeg.',GROUPS[2]),('proj_eeg.0.',GROUPS[3]),('proj_eeg.1.',GROUPS[4]),('proj_eeg.2.',GROUPS[5]),('subject_wise_linear.',GROUPS[6])]:
        if name.startswith(prefix):return target
    raise AssertionError(name)
def stats(c,f):
    c=np.asarray(c,np.float64);f=np.asarray(f,np.float64);a=float(c@c);b=float(f@f);dot=float(c@f);den=np.sqrt(a*b)
    return dict(coarse_squared_norm=a,fine_squared_norm=b,dot=dot,cosine=float(dot/den) if den else None,negative_dot=dot<0,
                coarse_directional_derivative_under_global_descent=-a-dot,coarse_directional_derivative_under_extra_fine_descent=-a-2*dot,
                fine_directional_derivative_under_global_descent=-dot-b,fine_directional_derivative_under_extra_fine_descent=-dot-2*b)
def grouped(c,f,layout):
    out={}
    for name in GROUPS:
        parts=[r for r in layout if r['group']==name];assert parts
        a=np.concatenate([c[r['start']:r['stop']] for r in parts]);b=np.concatenate([f[r['start']:r['stop']] for r in parts]);out[name]=stats(a,b)
    out['all_parameters']=stats(c,f);return out
def tradeoff_without_initial_gradient_conflict():
    x=np.array([1.,0.]);gc=np.array([1.,0.]);gf=np.array([1.,10.]);after=x-.1*gf
    coarse=lambda z:float(.5*(z[0]**2+100*z[1]**2))
    fine=lambda z:float(.5*(z[0]**2+(z[1]+10)**2))
    assert gc@gf>0 and coarse(after)>coarse(x) and fine(after)<fine(x)
    return dict(EXPLICIT_SYNTHETIC_NOT_EEG=True,initial_coarse_fine_gradient_dot=float(gc@gf),coarse_before=coarse(x),coarse_after=coarse(after),fine_before=fine(x),fine_after=fine(after),
                reason='Finite curvature permits a loss tradeoff despite initially aligned gradients; different trained states are weaker evidence still.')
