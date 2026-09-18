from meeting_transcriber.progress import ProgressEvent, TimeEstimator


def test_resume_and_loading_are_not_counted_as_new_work():
    estimate = TimeEstimator()
    estimate.update(ProgressEvent("loading", 0, 3600), now=0)
    assert estimate.remaining(now=100) is None
    estimate.update(ProgressEvent("transcribing", 1800, 3600), now=100)
    assert estimate.remaining(now=101) is None
    estimate.update(ProgressEvent("transcribing", 1860, 3600), now=110)
    assert estimate.remaining(now=110) == 290
    assert estimate.remaining(now=120) == 280


def test_phase_switch_and_stalled_estimate():
    estimate = TimeEstimator()
    estimate.update(ProgressEvent("transcribing", 0, 60), now=0)
    estimate.update(ProgressEvent("transcribing", 30, 60), now=10)
    assert estimate.remaining(now=30) is None
    estimate.update(ProgressEvent("diarizing", 0, 10), now=50)
    assert estimate.remaining(now=51) is None
    estimate.update(ProgressEvent("diarizing", 10, 10), now=60)
    assert estimate.remaining(now=61) == 0

