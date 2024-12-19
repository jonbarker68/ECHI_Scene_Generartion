"""ECHI structure generator.

Generate the structure for an ECHI session suitable for
recording in our recording space and modelling simple cafe
type scenarios. e.g. multiple independent conversations at
a number of tables.
"""

import functools
import json

import hydra
import numpy as np


def exponential_segmenter(half_life, min_duration, duration):
    """Make durations for the conversation segments.

    The seg_controls govern the distribution of segment lengths.
    Standard is an exponential but with an imposed minimum length.
    """
    durations = []
    end_time = 0
    while end_time != duration:
        seg_duration = np.random.exponential(scale=half_life)
        seg_duration = int(max(seg_duration, min_duration))
        seg_duration = min(duration - end_time, seg_duration)
        durations.append(seg_duration)
        end_time += seg_duration
    return durations


def make_speaker_groups(table_sizes):
    """Make a group of speakers for each table."""
    # e.g. [2, 3] -> [[1, 2], [3, 4, 5]]
    cumsum_tables = [0] + np.cumsum(table_sizes).tolist()
    return [list(range(i + 1, j + 1)) for i, j in zip(cumsum_tables, cumsum_tables[1:])]


def make_conversation_segment(speaker_groups, duration):
    if len(speaker_groups) == 1:
        print(speaker_groups)
        return {
            "type": "conversation",
            "speakers": speaker_groups[0],
            "duration": duration,
        }
    else:
        return {
            "type": "splitter",
            "elements": [
                {
                    "type": "conversation",
                    "speakers": speaker_group,
                    "duration": duration,
                }
                for speaker_group in speaker_groups
            ],
        }


def make_table(speakers, duration, stagger=0, segmenter=None):
    """Make a conversation pattern for a table with a given set of speaker."""

    n_speakers = len(speakers)

    # Do not segment conversations with less than 4 speakers
    if n_speakers < 4 or not segmenter:
        node = make_conversation_segment([speakers], duration)
    else:
        # make segments alternating between conversations
        # between all 4 speakers and parallel conversations
        # between 2 groups of 2 speakers
        durations = segmenter(duration=duration)
        n_segments = len(durations)
        speaker_groups = []
        while len(speaker_groups) < n_segments:
            # First all speaker together ...
            speaker_groups.append([speakers])
            # ...then split speakers into two random subgroups
            shuffled = speakers.copy()
            np.random.shuffle(shuffled)
            speaker_groups.append((shuffled[:2], shuffled[2:]))
        speaker_groups = speaker_groups[:n_segments]
        conversations = [
            make_conversation_segment(sgs, d)
            for sgs, d in zip(speaker_groups, durations)
        ]
        node = {"type": "sequence", "talkers": speakers, "elements": conversations}

    if stagger:
        node = {
            "type": "sequence",
            "elements": [{"type": "pause", "duration": stagger}, node],
        }
    return node


def make_parallel_conversations(
    table_sizes, duration, stagger_duration=0, segmenter=None
):
    """Generate a random structure for an ECHI session"""
    n_speakers = sum(table_sizes)
    speakers = list(range(1, n_speakers + 1))
    speaker_groups = make_speaker_groups(table_sizes)
    staggers = np.arange(len(table_sizes)) * stagger_duration
    tables = [
        make_table(speakers, duration, int(stagger), segmenter)
        for speakers, stagger in zip(speaker_groups, staggers)
    ]
    structure = {
        "type": "sequence",
        "talkers": speakers,
        "elements": [{"type": "splitter", "elements": tables}],
    }

    return structure


@hydra.main(
    version_base=None, config_path="conf", config_name="echi_structure_generator_config"
)
def main(cfg):
    """Build a random structure"""

    # The strategy used for segmenting conversations
    segmenter = (
        functools.partial(
            exponential_segmenter,
            half_life=cfg.seg_controls.half_life,
            min_duration=cfg.seg_controls.min_duration,
        )
        if cfg.seg_controls.segment
        else None
    )

    structure = make_parallel_conversations(
        table_sizes=[4, 4, 4],
        duration=cfg.duration,
        stagger_duration=cfg.stagger_duration,
        segmenter=segmenter,
    )

    # write structure to file
    with open(cfg.structure_file, "w", encoding="utf8") as f:
        json.dump(structure, f, indent=4)


if __name__ == "__main__":
    main()