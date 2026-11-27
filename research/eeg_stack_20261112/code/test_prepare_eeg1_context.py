from prepare_eeg1_context import eligible_epochs, registered_stimulus


def test_source_epoch_uses_exact_photo_and_blocks_nearby_held_out_objects():
    timestamps = [0., .1, .2, .3, .8, .9, 1.0, 1.1]
    photos = [("train-a", "a.jpg"), ("train-a", "other.jpg"), ("novel", "n.jpg"),
              ("train-b", "b.jpg"), ("train-a", "a.jpg"), ("train-b", "b.jpg"),
              ("train-a", "other.jpg"), ("train-b", "p.jpg")]
    allowed = {("train-a", "a.jpg"): 12, ("train-b", "b.jpg"): 18}
    selected = eligible_epochs(timestamps, photos, allowed, {"train-a": 1, "train-b": 3})
    assert [(i, row) for i, row, _ in selected] == [(4, 12), (5, 18)]
    assert all(bits == (1 << 1 | 1 << 3) for _, _, bits in selected)


def test_each_public_participant_supplies_at_most_one_epoch_per_photo():
    a = ("train-a", "a.jpg")
    observed = eligible_epochs([0., 3., 6.], [a, a, a], {a: 8}, {"train-a": 1})
    assert observed == [(0, 8, 2)]


def test_short_public_event_schema_still_checks_the_registered_object():
    path = "stimuli/tape_recorder/tape_recorder_10s.jpg"
    manifest = {path: {}}
    assert registered_stimulus({"stim": path, "object": "tape_recorder"}, manifest)
    assert not registered_stimulus({"stim": path, "object": "pillbox"}, manifest)
    assert not registered_stimulus({"stim": path, "stimname": "wrong.jpg"}, manifest)
