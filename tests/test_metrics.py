"""Fast tests for label and diversity metrics."""

from crl_prompt.metrics import distinct_n, match_emotion


def test_emotion_matching_prefers_first_generated_word() -> None:
    labels = ["anxious", "content", "proud"]
    assert match_emotion("Anxious, because this is uncertain.", labels) == "anxious"


def test_distinct_metrics_are_bounded() -> None:
    distinct1, distinct2 = distinct_n(["a b c", "a b d"])
    assert 0.0 <= distinct1 <= 1.0
    assert 0.0 <= distinct2 <= 1.0

