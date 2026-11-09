"""Check the local quadratic solution on an attainable functional subspace.

This synthetic algebra check does not compute the huge native parameter
Jacobian or claim that AdamW solves this local surrogate exactly.
"""
import argparse,datetime,json
from pathlib import Path
import numpy as np

def run(output):
    g=np.random.default_rng(1729);D=8;tau=.25
    # A rank-deficient parameter-to-unit Jacobian is intentional. A full
    # parameter normal-equation matrix need not be invertible.
    H=g.normal(size=(D,5))@g.normal(size=(5,12));J=g.normal(size=(7,D));r=g.normal(size=7)
    Z=g.normal(size=(D,D));Q=Z@Z.T+.3*np.eye(D);Q/=np.trace(Q)/D
    U,s,_=np.linalg.svd(H,full_matrices=False);U=U[:,s>1e-10];assert U.shape==(D,5)
    a=np.linalg.solve(U.T@J.T@J@U+U.T@Q@U/(2*D*tau*tau),-U.T@J.T@r)
    constrained=U@a
    A=np.concatenate([J@H,np.linalg.cholesky(Q).T@H/np.sqrt(2*D*tau*tau)])
    b=np.concatenate([-r,np.zeros(D)])
    theta=np.linalg.lstsq(A,b,rcond=None)[0];direct=H@theta
    unconstrained=np.linalg.solve(J.T@J+Q/(2*D*tau*tau),-J.T@r)
    projected_gradient=U.T@(2*J.T@(r+J@constrained)+Q@constrained/(D*tau*tau))
    error=float(np.max(np.abs(constrained-direct)));normal=float(np.max(np.abs(projected_gradient)))
    infeasible=float(np.linalg.norm(unconstrained-U@(U.T@unconstrained)))
    assert error<1e-12 and normal<1e-12 and infeasible>1e-3
    v=dict(status='D39_ATTAINABLE_SUBSPACE_LOCAL_QUADRATIC_ALGEBRA_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      functional_dimensions=D,parameter_dimensions=12,rank=5,constrained_vs_augmented_least_squares_max_error=error,
      projected_normal_equation_max_error=normal,unconstrained_solution_infeasible_component_norm=infeasible,
      native_parameter_Jacobian_computed=False,neural_AdamW_exact_solution_claimed=False,new_GPU_jobs=0,new_training_updates=0)
    with Path(output).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps(v))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
