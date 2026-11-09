"""Counterexamples and KKT checks for an untrained post-D31 update proposal.

These are mathematical checks, not evidence of EEG or image improvement.
The Euclidean halfspace projection is classical constrained optimization.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize


def project_step(step,rows):
    step=np.asarray(step,np.float64);A=np.asarray(rows,np.float64)
    norms=np.linalg.norm(A,axis=1);A=A[norms>1e-14]/norms[norms>1e-14,None]
    assert len(A)<=2
    candidates=[]
    if not len(A):return step.copy(),np.empty(0)
    for mask in range(1<<len(A)):
        active=[i for i in range(len(A)) if mask&(1<<i)]
        if active:
            B=A[active];lam=np.linalg.pinv(B@B.T,rcond=1e-13)@(B@step)
            if np.any(lam<-1e-9):continue
            candidate=step-B.T@lam
            if np.linalg.norm(B@candidate)>1e-8*max(1,np.linalg.norm(step)):continue
            dual=np.zeros(len(A));dual[active]=lam
        else:candidate=step.copy();dual=np.zeros(len(A))
        if np.max(A@candidate)<=1e-8*max(1,np.linalg.norm(step)):
            candidates.append((float(np.square(candidate-step).sum()),candidate,dual))
    assert candidates,'Zero is always feasible; projection solver missed a valid active set'
    _,candidate,dual=min(candidates,key=lambda x:x[0])
    return candidate,dual


def main():
    start=time.perf_counter();rng=np.random.default_rng(1729)
    maximum_oracle_error=0.;checks=[]
    examples=[(np.array([1.,2.]),np.array([[1.,0.],[1.,1.]])),
              (np.array([1.,2.]),np.array([[1.,0.],[-1.,0.]])),
              (np.array([1.,2.]),np.array([[1.,0.],[2.,0.]])),
              (np.array([1.,2.]),np.zeros((2,2)))]
    examples += [(rng.normal(size=7),rng.normal(size=(2,7))) for _ in range(100)]
    for step,rows in examples:
        answer,dual=project_step(step,rows)
        norms=np.linalg.norm(rows,axis=1);A=rows[norms>1e-14]/norms[norms>1e-14,None]
        assert not len(A) or np.max(A@answer)<1e-7
        if len(A):
            assert np.linalg.norm(answer-step+A.T@dual)<1e-7
            assert np.max(np.abs(dual*(A@answer)))<1e-7 and np.min(dual)>-1e-9
        oracle=minimize(lambda z:.5*np.square(z-step).sum(),np.zeros_like(step),
                        jac=lambda z:z-step,constraints=[dict(type='ineq',fun=lambda z:-rows@z,jac=lambda z:-rows)],
                        method='SLSQP',options=dict(ftol=1e-12,maxiter=500))
        assert oracle.success,oracle.message
        error=float(np.linalg.norm(answer-oracle.x));maximum_oracle_error=max(maximum_oracle_error,error)
        assert error<1e-6
    checks.append('Classical two-halfspace projection: KKT and104 independent SLSQP comparisons, including duplicate/opposed/zero constraints')
    # K(theta)=theta[0]+theta[1]^2. The projected tangent step has zero
    # first derivative but every nonzero tested finite step increases K.
    proposed=np.array([1.,1.]);gradient_K=np.array([1.,0.]);gradient_M=np.array([0.,1.])
    projected,_=project_step(proposed,np.stack([gradient_K,-gradient_M]))
    assert np.array_equal(projected,np.array([0.,1.]))
    fractions=np.array([1.,.5,.25,.125,.0625,.03125])
    finite_increases=[float((alpha*projected)[0]+(alpha*projected)[1]**2) for alpha in fractions]
    assert all(v>0 for v in finite_increases) and gradient_K@projected==0
    checks.append('First-order protected update can increase the exact nonlinear K for every finite backtracking fraction')
    # A Euclidean tangent gradient does not remain tangent after Adam-like
    # coordinate preconditioning: the actual parameter step must be inspected.
    gradient_K=np.array([1.,1.]);tangent_step=np.array([1.,-1.]);preconditioner=np.diag([10.,1.])
    assert gradient_K@tangent_step==0 and gradient_K@(preconditioner@tangent_step)==9.
    corrected,_=project_step(preconditioner@tangent_step,gradient_K[None])
    assert abs(float(gradient_K@corrected))<1e-12
    checks.append('Projecting a gradient before anisotropic preconditioning does not certify the actual Adam-like update')
    changes=np.array([.2,-.1]);assert changes.mean()>0 and changes.min()<0
    checks.append('Preserving average correct margin does not preserve each query margin or ranking')
    logits=np.array([[2.,-2.],[-2.,2.]])
    def softmax(x):
        e=np.exp(x-x.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)
    p=softmax(logits);q=softmax(logits[::-1]);sym=.5*np.sum((p-q)*(np.log(p)-np.log(q)),axis=-1)
    collapsed=softmax(np.zeros_like(logits));assert sym.mean()>0 and np.array_equal(collapsed,collapsed[::-1])
    checks.append('Constant predictions attain zero consistency divergence without stimulus discrimination')
    path=Path(__file__);out=path.parent/'audit/crossdisciplinary_20261028_v1/protected_update_math_checks_v1.json'
    value=dict(status='POST_D31_UNTRAINED_PROTECTED_UPDATE_LOGIC_CHECKED',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               checks=checks,oracle_cases=len(examples),maximum_SLSQP_solution_L2_error=maximum_oracle_error,
               nonlinear_counterexample_projected_step=projected.tolist(),finite_fractions=fractions.tolist(),nonlinear_K_increases=finite_increases,
               preconditioned_linear_constraint_violation=9.,seconds=time.perf_counter()-start,
               actual_EEG_read=False,new_model_or_fit=False,new_GPU_jobs=0,new_images=0,
               original_projection_algorithm_claimed=False,finite_stability_guarantee_established=False,
               decision='DO_NOT_PROMOTE_CLASSICAL_GRADIENT_PROJECTION_OR_FIRST_ORDER_CONSTRAINT_TO_A_NOVEL_CERTIFIED_EEG_METHOD')
    with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))

if __name__=='__main__':main()
