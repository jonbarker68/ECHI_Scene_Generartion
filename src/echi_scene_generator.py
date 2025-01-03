"""ECHI scene generation script.

Reads a template file and instantiates it to make the low level scene file.

usage:
python src/echi_scene_generator.py paths.structure_file=./test_data/echi_structure.json paths.scene_file=scene.json speaker.ids="[150, 3240, 5463, 6437, 5022, 1553, 32, 6078, 8425, 6367, 8629, 1355]"
"""

import json
import logging
import random

import hydra
import numpy as np
import pandas as pd
from frozendict import frozendict

from conf import Config

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Speaker:
    """Represents a speaker in the scenario."""

    def __init__(self, df, offset_scale=0):
        name = df.iloc[0].speaker
        self.name = name
        self.utterance_index = 0
        self.df = df
        self.max_segments = len(df)
        self.offset_scale = offset_scale

    def next(self):
        """Dummy method for generating the next utterance."""
        item = self.df.iloc[self.utterance_index]
        self.utterance_index += 1
        if self.utterance_index >= self.max_segments:
            logging.warning("Out of segments. Wrapping")
            self.utterance_index = 0
        utterance = {"name": item.file_name, "duration": int(item.length)}
        return utterance

    def get_offset(self):
        """Randomness in speaker start."""
        return int(np.random.normal(loc=0, scale=self.offset_scale))


def get_last_segment(scene):
    """Gets the last segment of the scene."""
    if len(scene) == 0:
        return None
    return max(scene, key=lambda x: x["offset"])


def get_end_time(scene):
    """Gets the end time of the scene."""
    segment = get_last_segment(scene)
    if segment is None:
        return 0
    return segment["offset"]


def get_last_speaker(scene):
    """Gets the id of the last person to have spoken."""
    segment = get_last_segment(scene)
    if segment is None or "speaker" not in segment:
        return 0
    return segment["speaker"]


def process_sequence(scene, sequence, speakers, sample_rate):
    """Processes a sequence element."""
    logger.info("Processing a sequence element.")

    for element in sequence["elements"]:
        scene = generate_scene_node(element, speakers, sample_rate, scene)
    return scene


def process_splitter(scene, splitter, speakers, sample_rate):
    """Processes a splitter element."""
    logger.info("Processing a splitter element.")

    new_scenes = [
        generate_scene_node(element, speakers, sample_rate, scene)
        for element in splitter["elements"]
    ]
    for new_scene in new_scenes:
        scene = scene.union(new_scene)
    return scene


def process_conversation(scene, conversation, speakers, sample_rate):
    """Processes a conversation element."""
    logger.info("Processing a conversation element.")

    end_time = get_end_time(scene)
    last_speaker = get_last_speaker(scene)
    conversation_end_time = end_time + conversation["duration"] * sample_rate

    while True:
        speaker_id = random.choice(conversation["speakers"])
        while speaker_id == last_speaker:
            speaker_id = random.choice(conversation["speakers"])
        speaker = speakers[speaker_id - 1]
        utterance = speaker.next()
        random_offset = speaker.get_offset()
        start_time = max(0, end_time + random_offset)
        end_time = start_time + utterance["duration"]
        if end_time > conversation_end_time:
            break  # Break just before overshooting the conversation duration
        last_speaker = speaker_id
        scene_element = {
            "type": "utterance",
            "onset": start_time,
            "offset": end_time,
            "channel": speaker_id,
            "filename": utterance["name"],
        }
        scene.add(frozendict(scene_element))

    return scene


def process_pause(scene, pause, sample_rate):
    """Processes a pause element."""
    logger.info("Processing a pause element.")
    end_time = get_end_time(scene)
    pause_duration = pause["duration"]
    scene_element = {
        "type": "pause",
        "onset": end_time,
        "offset": end_time + pause_duration * sample_rate,
        "channel": 0,
    }
    scene.add(frozendict(scene_element))
    return scene


def generate_scene_node(structure: dict, speakers: list, sample_rate, scene=None):
    """Generates a node in the scene structure."""
    logger.info("Generating the scene from the structure.")
    scene = scene.copy() if scene is not None else set()
    if structure["type"] == "sequence":
        scene = process_sequence(scene, structure, speakers, sample_rate)
    elif structure["type"] == "splitter":
        scene = process_splitter(scene, structure, speakers, sample_rate)
    elif structure["type"] == "conversation":
        scene = process_conversation(scene, structure, speakers, sample_rate)
    elif structure["type"] == "pause":
        scene = process_pause(scene, structure, sample_rate)
    return scene


def generate_scene(structure: dict, speakers: list, sample_rate, scene=None):
    """Generates the scene from the structure."""
    scene = generate_scene_node(structure, speakers, sample_rate, scene)

    # remove pause elements - these were only used during generation
    scene = {utterance for utterance in scene if utterance["type"] != "pause"}

    scene = [dict(utterance) for utterance in scene]

    return scene


def save_scene(scene, scene_file):
    """Saves the scene to a file."""
    logger.info(f"Saving the scene to {scene_file}.")
    with open(scene_file, "w", encoding="utf8") as f:
        json.dump(scene, f, indent=4)


def make_speakers(utt_index, selected_speakers, offset_scale=0):
    """Makes a list of speakers from the speech dataset."""

    spkr_dfs = [
        utt_index[utt_index.speaker == spkr].sort_values(by="file_name")
        for spkr in selected_speakers
    ]
    speakers = [Speaker(spkr_df, offset_scale) for spkr_df in spkr_dfs]
    return speakers


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """Instantiates the structure."""
    logger.info(
        f"Instantiating {cfg.paths.structure_file} to make {cfg.paths.scene_file}"
    )

    utt_index = pd.read_csv(cfg.paths.utt_index)

    speakers = make_speakers(utt_index, cfg.speaker.ids, cfg.speaker.offset_scale)
    structure = json.load(open(cfg.paths.structure_file, "r", encoding="utf8"))

    scene = generate_scene(structure, speakers, cfg.audio.sample_rate)

    logging.info(f"Saving scene file to {cfg.paths.scene_file}.")
    save_scene(scene, cfg.paths.scene_file)


if __name__ == "__main__":
    main()  # type: ignore
