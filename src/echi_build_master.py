"""Build the echi master

Constructs the master json file that describes all ECHI sessions.
"""

import functools
import json
import logging

import hydra
from tqdm import tqdm

from echi_scene_generator import generate_scene, make_libri_speakers
from echi_structure_generator import exponential_segmenter, make_parallel_conversations


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
        stagger_duration=structure_cfg.stagger_duration,
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
        session_dict["structure"] = build_structure(cfg.structure, cfg.seg_controls)

    # Add the speakers for each session
    # TODO: This is currently fixed per session but needs some cross-session design
    for session_dict in tqdm(master, "Selecting speakers"):
        session_dict["speakers"] = list(cfg.speaker.ids)

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
