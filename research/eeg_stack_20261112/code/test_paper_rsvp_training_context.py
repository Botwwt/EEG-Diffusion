from paper_rsvp_training_context import occurrence_slots, select_neighbors


def test_source_preprocessing_reorders_within_session_and_after_merge():
    events = [(0, i) for i in (5, 10)] + [(3, i) for i in (25, 30)]
    result = occurrence_slots(events)
    assert sorted(result) == sorted(events)
    assert result == [(3, 30), (0, 5), (3, 25), (0, 10)]
    assert occurrence_slots(events, merged_shuffle=False) == [(0, 10), (0, 5), (3, 30), (3, 25)]


def test_context_cannot_skip_a_catch_trial_or_jump_between_sequences():
    events = [
        (1, 1, 1, 100, 0), (1, 1, 2, 300, 1),
        (1, 1, 3, 500, -1), (1, 1, 4, 700, 2),
        (1, 2, 1, 900, 3), (1, 2, 2, 1200, 4),
    ]
    assert select_neighbors(events) == {100: 300}
