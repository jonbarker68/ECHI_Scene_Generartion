"""Build the echi master

Constructs the master json file that describes all ECHI sessions.
"""

import copy
import functools
import json
import logging
import random

import hydra
import pandas as pd
from tqdm import tqdm

from echi_scene_generator import generate_scene, make_libri_speakers
from echi_structure_generator import exponential_segmenter, make_parallel_conversations

SAMPLE_RATE = 16000


def make_speaker_lists(speakers_df, n_speakers_per_session, min_duration=0):
    """Make a list of speaker ids for each session"""

    total_speakers_needed = sum(n_speakers_per_session)

    speakers_df = speakers_df.groupby("speaker").length.sum().reset_index()
    speakers_df = speakers_df[speakers_df.length >= int(min_duration * SAMPLE_RATE)]
    speaker_ids = speakers_df.speaker.unique()
    print(len(speaker_ids))
    speaker_ids = [int(speaker_id) for speaker_id in speaker_ids]
    speaker_pool = []
    while len(speaker_pool) < total_speakers_needed:
        random.shuffle(speaker_ids)
        speaker_pool.extend(speaker_ids)

    speaker_lists = []
    for n_speakers in n_speakers_per_session:
        speaker_lists.append(list(speaker_pool[:n_speakers]))
        speaker_pool = speaker_pool[n_speakers:]
    return speaker_lists


def add_speakers_to_master(master, speakers_df):
    """Add speaker ids to the master json."""
    master = copy.deepcopy(master)
    n_speakers_per_session = [
        len(session["structure"]["speakers"]) for session in master
    ]
    session_duration = max(master, key=lambda x: x["duration"])["duration"]

    # Note, only use speakers that have at least half the session duration
    speaker_lists = make_speaker_lists(
        speakers_df, n_speakers_per_session, min_duration=session_duration / 2
    )
    for session, speakers_ids in zip(master, speaker_lists):
        session["speakers"] = speakers_ids
    return master


def build_structure(structure_cfg, seg_controls):
    # The strategy used for segmenting conversations

    segmenter = (
        functools.partial(
            exponential_segmenter,
            half_life=seg_controls.half_life,
            min_duration=seg_controls.min_duration,
        )
        if seg_controls.segment
        else None
    )

    structure = make_parallel_conversations(
        table_sizes=structure_cfg.table_sizes,
        duration=structure_cfg.duration,
        segmenter=segmenter,
    )

    return structure


@hydra.main(
    version_base=None, config_path="conf", config_name="echi_build_master_config"
)
def main(cfg):
    """Build the ECHI master file."""

    n_sessions = cfg.n_sessions

    logging.info(f"Build master json for {cfg.n_sessions} sessions")
    master = [
        {
            "session": f"session_{i:03d}",
            "structure": None,
            "speakers": None,
            "scene": None,
        }
        for i in range(1, n_sessions + 1)
    ]

    # Build a structure for each session
    for session_dict in tqdm(master, "Building scene structures"):
        session_dict["duration"] = cfg.structure.duration
        session_dict["structure"] = build_structure(cfg.structure, cfg.seg_controls)

    # Add the speakers for each session
    speakers_df = pd.read_csv(cfg.speaker.libri_index_file)
    master = add_speakers_to_master(master, speakers_df)

    # Generate the scenes
    for session_dict in tqdm(master, "Generating scenes"):
        speaker_ids = session_dict["speakers"]
        speakers = make_libri_speakers(
            cfg.speaker.libri_index_file, speaker_ids, cfg.speaker.offset_scale
        )
        structure = session_dict["structure"]
        session_dict["scene"] = generate_scene(structure, speakers)

    logging.info("Saving master json file.")
    with open(cfg.master_file, "w", encoding="utf8") as f:
        json.dump(master, f, indent=2)


if __name__ == "__main__":
    main()  # type: ignore
