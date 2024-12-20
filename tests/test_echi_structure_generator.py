import pytest

from echi_structure_generator import make_speaker_groups


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
