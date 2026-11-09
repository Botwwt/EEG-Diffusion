"""Newseed identity and original Dfit calibration; no effects used for selection."""
import math

GRID=[0.,.25,.5,.75,1.]
def validate_replicate_state(state,spec):
    assert spec['training_seed'] in (2027,3407) and state['seed']==spec['training_seed']
    assert state['arm']=='ordinary' and state['updates']==24807 and state['seen_queries']==396900 and state['passes']==3
    assert state['protocol_sha256']==spec['training_protocol']['sha256']
    c=state['calibration'];assert c['kind']=='Dfit_only_global_strength_calibration'
    assert c['strengths']==GRID and len(c['Dfit_token_errors'])==5 and all(math.isfinite(x) for x in c['Dfit_token_errors'])
    selected=GRID[min(range(5),key=lambda i:(c['Dfit_token_errors'][i],i))]
    assert c['selected_strength']==spec['ordinary_strength']==selected and c['selected_without_V_or_test_images'] is True
    return selected

def bounded_numpy64(condition,gamma,beta,weights,strength):
    """Independent matrix/erf implementation, radius=.25*actual strength."""
    import numpy as np
    assert strength in GRID
    c=np.asarray(condition,dtype=np.float64);g=np.asarray(gamma,dtype=np.float64);b=np.asarray(beta,dtype=np.float64)
    assert c.ndim==2 and np.isfinite(c).all() and np.isfinite(g).all() and np.all(np.abs(g)>1e-12)
    if strength==0:return c.copy()
    w={k:np.asarray(v,dtype=np.float64) for k,v in weights.items()}
    u=(c-b)/g;a=u@w['trunk.0.weight'].T+w['trunk.0.bias'];h=.5*a*(1+np.vectorize(math.erf,otypes=[np.float64])(a/np.sqrt(2)))
    d=h@w['direction.weight'].T+w['direction.bias'];d-=d.mean(1,keepdims=True);r2=np.square(u).sum(1,keepdims=True);assert np.all(r2>1e-20)
    d-=np.sum(d*u,axis=1,keepdims=True)/r2*u;delta=(.25*strength)*np.sqrt(r2)*d/np.sqrt(r2+np.square(d).sum(1,keepdims=True));scale=np.sqrt(r2)/np.linalg.norm(u+delta,axis=1,keepdims=True)
    return c+g*((scale-1)*u+scale*delta)
