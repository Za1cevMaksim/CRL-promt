"""Fast tests for deterministic preprocessing helpers."""

from __future__ import annotations

import random

from crl_prompt.data import attach_dialogue_negatives, make_incoherent_response


def test_sentence_permutation_changes_response() -> None:
    source = "First sentence. Second sentence."
    assert make_incoherent_response(source, random.Random(1)) == (
        "Second sentence. First sentence."
    )


def test_offtopic_negative_comes_from_another_example() -> None:
    rows = [
        {"response": "alpha"},
        {"response": "beta"},
        {"response": "gamma"},
        {"response": "delta"},
        {"response": "epsilon"},
    ]
    attach_dialogue_negatives(rows, "test", negative_count=4)
    for row in rows:
        assert row["negative_offtopic"] != row["response"]

