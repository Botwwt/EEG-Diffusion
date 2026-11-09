"""Fail-closed dependency gate for the finite OLD-only engineering profile."""


def dependency_state(dependencies,is_live,read_optional):
    waiting=[]
    for item in dependencies:
        if read_optional(item['failure']) is not None:
            raise RuntimeError('A fixed upstream study failed; never restart it')
        if is_live(item['PID'],item['start_ticks']):
            waiting.append(dict(PID=item['PID'],start_ticks=item['start_ticks']))
            continue
        complete=read_optional(item['completion'])
        if complete is None:
            raise RuntimeError('Missing exact upstream handle without completion; no admission')
        if complete.get('status')!=item['success_status']:
            raise RuntimeError('Wrong upstream success closure')
        if complete.get(item['completion_link_field'])!=item['protocol']:
            raise RuntimeError('Upstream completion belongs to another sealed protocol')
    return dict(ready=not waiting,waiting=waiting)
