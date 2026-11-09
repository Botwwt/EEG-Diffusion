"""Independent NumPy checks of finite captured FP32 operations, not FP64 parity.

Linear stages use a dimension-derived roundoff bound, not a tuned tolerance.
The CUDA erf allowance is its documented empirical 2-ULP value; no universal
library-error theorem is claimed. Final residual operations must be bit exact.
"""
import numpy as np
from scipy.special import erf

UNIT32=2.**-24
UNIT64=2.**-53


def bit_equal(a,b):
    return a.dtype==b.dtype==np.float32 and np.array_equal(a.view(np.uint32),b.view(np.uint32))


def gamma(n,unit):
    assert n*unit<1
    return n*unit/(1-n*unit)


def _linear(x,actual,weight,bias):
    x=x.astype(np.float64);w=weight.astype(np.float64);b=bias.astype(np.float64)
    n=x.shape[1];expected=x@w.T+b;absolute_sum=np.abs(x)@np.abs(w).T+np.abs(b)
    # At most 2n+2 scalar multiply/add roundings on a path (also covers FMA
    # and reassociated accumulation); include rounding in the FP64 reference.
    bound=(gamma(2*n+2,UNIT32)+gamma(2*n+2,UNIT64))*absolute_sum
    error=np.abs(actual.astype(np.float64)-expected)
    assert np.all(error<=bound),dict(stage='linear',bad=int((error>bound).sum()),max_error=float(error.max()),max_bound=float(bound.max()))
    ratio=np.divide(error,bound,out=np.zeros_like(error),where=bound!=0)
    return dict(elements=int(error.size),max_absolute_error=float(error.max()),max_bound_ratio=float(ratio.max()),roundings_on_path=2*n+2)


def _GELU(z,actual):
    assert z.dtype==actual.dtype==np.float32
    # PyTorch2.3 exact-branch source uses float32 sqrt(1/2), erf, then
    # x*.5*(1+erf). Check the rounded operations, rather than erf at exact x/sqrt2.
    argument=np.multiply(z,np.float32(np.sqrt(.5)),dtype=np.float32)
    rounded=erf(argument.astype(np.float64)).astype(np.float32)
    lower=rounded.copy();upper=rounded.copy()
    for unused in range(2):
        lower=np.nextafter(lower,np.float32(-np.inf));upper=np.nextafter(upper,np.float32(np.inf))
    half=np.multiply(z,np.float32(.5),dtype=np.float32)
    a=np.multiply(half,np.add(np.float32(1),lower,dtype=np.float32),dtype=np.float32)
    b=np.multiply(half,np.add(np.float32(1),upper,dtype=np.float32),dtype=np.float32)
    lo=np.minimum(a,b);hi=np.maximum(a,b)
    assert np.all((actual>=lo)&(actual<=hi)),dict(stage='GELU',bad=int(((actual<lo)|(actual>hi)).sum()))
    ideal=.5*z.astype(np.float64)*(1+erf(z.astype(np.float64)/np.sqrt(2)))
    return dict(elements=int(z.size),empirical_erf_ULPs=2,all_captured_values_within_rounded_interval=True,
        max_absolute_difference_from_exact_formula=float(np.abs(actual.astype(np.float64)-ideal).max()),universal_erf_error_bound_claimed=False)


def verify(trace,weights,c,g,b,radius):
    assert 0<=radius<=.25 and radius in (0.,.0625,.125,.1875,.25)
    assert c.dtype==g.dtype==b.dtype==np.float32 and c.shape==g.shape==b.shape
    assert c.ndim==2 and c.shape[1]==1024 and np.all(np.abs(g)>1e-6)
    for name,shape in [('u',c.shape),('z',(len(c),256)),('h',(len(c),256)),('raw',c.shape),('output',c.shape)]:
        assert trace[name].dtype==np.float32 and trace[name].shape==shape and np.isfinite(trace[name]).all()
    u=np.divide(np.subtract(c,b,dtype=np.float32),g,dtype=np.float32)
    assert bit_equal(u,trace['u']),'Actual affine coordinate calculation differs from independent FP32 operations'
    l1=_linear(u,trace['z'],weights['trunk.0.weight'],weights['trunk.0.bias'])
    gelu=_GELU(trace['z'],trace['h'])
    l2=_linear(trace['h'],trace['raw'],weights['direction.weight'],weights['direction.bias'])
    delta=np.multiply(np.float32(radius),trace['raw'],dtype=np.float32)
    correction=np.multiply(g,delta,dtype=np.float32)
    expected=np.add(c,correction,dtype=np.float32)
    assert bit_equal(expected,trace['output']),'Actual residual formula differs from independent FP32 operations'
    return dict(queries=len(c),affine_FP32_bit_exact=True,residual_FP32_bit_exact=True,linear1=l1,GELU=gelu,linear2=l2)


def merge(records):
    assert records
    result=dict(queries=sum(x['queries'] for x in records),batches=len(records),affine_FP32_bit_exact=all(x['affine_FP32_bit_exact'] for x in records),residual_FP32_bit_exact=all(x['residual_FP32_bit_exact'] for x in records))
    for name in ('linear1','linear2'):
        result[name]=dict(elements=sum(x[name]['elements'] for x in records),max_absolute_error=max(x[name]['max_absolute_error'] for x in records),max_bound_ratio=max(x[name]['max_bound_ratio'] for x in records),roundings_on_path=records[0][name]['roundings_on_path'])
    result['GELU']=dict(elements=sum(x['GELU']['elements'] for x in records),all_captured_values_within_rounded_interval=True,empirical_erf_ULPs=2,universal_erf_error_bound_claimed=False)
    return result
