import functools

import numpy as np
import pytest

from echi_structure_generator import (
    exponential_segmenter,
    make_conversation_segment,
    make_parallel_conversations,
    make_speaker_groups,
    make_table,
)


@pytest.mark.parametrize(
    "table_sizes, expected",
    [
        ([2, 3], [[1, 2], [3, 4, 5]]),
        ([1, 4], [[1], [2, 3, 4, 5]]),
        ([3, 2], [[1, 2, 3], [4, 5]]),
        ([5], [[1, 2, 3, 4, 5]]),
        ([0, 5], [[], [1, 2, 3, 4, 5]]),
        ([4, 4, 4], [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
    ],
)
def test_make_speaker_groups(table_sizes, expected):
    speaker_groups = make_speaker_groups(table_sizes)
    assert speaker_groups == expected


def test_exponential_segmenter():
    np.random.seed(0)
    durations = exponential_segmenter(half_life=1, min_duration=1, duration=10)
    assert sum(durations) == 10
    assert all(d >= 1 for d in durations)


def test_make_conversation_segment():
    segment = make_conversation_segment([[1, 2], [3, 4]], 10)
    assert segment["type"] == "splitter"
    assert len(segment["elements"]) == 2
    assert segment["elements"][0]["speakers"] == [1, 2]
    assert segment["elements"][1]["speakers"] == [3, 4]


def test_make_table():
    np.random.seed(0)
    table = make_table(
        [1, 2, 3, 4],
        10,
        segmenter=functools.partial(exponential_segmenter, half_life=1, min_duration=1),
    )
    assert table["type"] == "sequence"
    assert len(table["elements"]) > 1


def test_make_parallel_conversations():
    np.random.seed(0)
    structure = make_parallel_conversations(
        [2, 3],
        10,
        segmenter=functools.partial(exponential_segmenter, half_life=1, min_duration=1),
    )
    assert structure["type"] == "sequence"
    assert len(structure["elements"][0]["elements"]) == 2
