"""Exact nested dependency and Power-Spherical artifact contracts, no CUDA."""
import numpy as np


def dependencies_ready(dependencies,is_live,read_optional):
    waiting=[]
    for item in dependencies:
        if read_optional(item['failure']) is not None:raise RuntimeError('Exact upstream study failed; no retry or restart')
        if is_live(item['PID'],item['start_ticks']):
            waiting.append(dict(PID=item['PID'],start_ticks=item['start_ticks']));continue
        complete=read_optional(item['completion'])
        if complete is None or complete.get('status')!=item['success_status']:raise RuntimeError('Missing exact handle without proven upstream success')
        value=complete
        for part in item['completion_protocol_field'].split('.'):
            if not isinstance(value,dict) or part not in value:raise RuntimeError('Missing exact upstream protocol link')
            value=value[part]
        if value!=item['protocol']:raise RuntimeError('Upstream success belongs to a different protocol')
    return dict(ready=not waiting,waiting=waiting)


def validate_teacher_arrays(data,arms):
    expected={f'{j}_{arm}_{name}' for j in range(8) for arm in arms for name in ('mean','concentration','moment')}
    if set(data)!=expected:raise ValueError('Incomplete fixed8x3 teacher artifact')
    errors=[]
    for j in range(8):
        for arm in arms:
            key=f'{j}_{arm}';mean=np.asarray(data[key+'_mean']);kappa=np.asarray(data[key+'_concentration']);moment=np.asarray(data[key+'_moment'])
            if mean.shape!=(1023,) or kappa.shape!=() or moment.shape!=(1,1023):raise ValueError('Teacher coordinate shape is wrong')
            if not np.isfinite(mean).all() or not np.isfinite(kappa) or not np.isfinite(moment).all() or kappa<0:raise ValueError('Teacher contains invalid values')
            if abs(np.linalg.norm(mean)-1)>1e-10:raise ValueError('Teacher mean direction is not unit')
            error=float(np.max(abs(moment[0]-mean*(float(kappa)/(float(kappa)+1022.)))))
            if error>1e-10:raise ValueError('Saved Power-Spherical mean moment is inconsistent')
            errors.append(error)
    return dict(teachers=24,maximum_independent_moment_error=max(errors))
