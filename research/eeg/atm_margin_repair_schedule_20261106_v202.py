"""Coverage rule for the next full fit; independent of effects and labels at test.

An epoch permutation partitions concepts into the ordinary training batches.
The protection set also uses a fixed rotation by half an epoch, so every
training stimulus is considered twice per pass without picking damaged items.
This is sampling of constraints, not a global feasibility guarantee.
"""
def epoch_batches(order, concepts=1323, concepts_per_batch=16, stimuli_per_concept=10):
    assert len(order)==concepts and set(order)==set(range(concepts))
    offset=concepts//2
    for lo in range(0,concepts,concepts_per_batch):
        indices=list(range(lo,min(lo+concepts_per_batch,concepts)))
        primary=[order[i] for i in indices]
        replay=[order[(i+offset)%concepts] for i in indices]
        assert set(primary).isdisjoint(replay)
        expand=lambda groups:[g*stimuli_per_concept+j for g in groups for j in range(stimuli_per_concept)]
        yield dict(primary_concepts=primary,replay_concepts=replay,
                   response_positions=expand(primary),constraint_positions=expand(primary+replay))

def verify_epoch(order):
    batches=list(epoch_batches(order));loss_count=[0]*13230;constraint_count=[0]*13230
    for b in batches:
        for p in b['response_positions']:loss_count[p]+=1
        for p in b['constraint_positions']:constraint_count[p]+=1
    assert len(batches)==83 and set(loss_count)=={1} and set(constraint_count)=={2}
    assert [len(b['response_positions']) for b in batches]==[160]*82+[110]
    assert [len(b['constraint_positions']) for b in batches]==[320]*82+[220]
    return dict(batches=83,response_exposures=13230,constraint_exposures=26460,
                every_response_once=True,every_constraint_query_twice=True,
                no_primary_replay_overlap=True)
