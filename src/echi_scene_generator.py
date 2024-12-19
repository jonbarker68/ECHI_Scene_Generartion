"""ECHI scene generation script.

Reads a template file and instantiates it to make the low level scene file.
"""

import json
import logging
import random

import hydra
import numpy as np
import pandas as pd
from frozendict import frozendict

logging.basicConfig()

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

SAMPLE_RATE = 16000


class Speaker:
    """Represents a speaker in the scenario."""

    def __init__(self, df, id):
        name = df.iloc[0].speaker
        self.name = name
        self.id = id
        self.utterance_index = 0
        self.df = df
        self.max_segments = len(df)

    def next(self):
        """Dummy method for generating the next utterance."""
        item = self.df.iloc[self.utterance_index]
        self.utterance_index += 1
        if self.utterance_index >= self.max_segments:
            logging.warning("Out of segments. Wrapping")
            self.utterance_index = 0
        utterance = {"name": item.file_name, "duration": int(item.length)}
        return utterance


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


def get_last_talker(scene):
    """Gets the id of the last person to have spoken."""
    segment = get_last_segment(scene)
    if segment is None or "talker" not in segment:
        return 0
    return segment["talker"]


def process_sequence(scene, sequence, speakers):
    """Processes a sequence element."""
    logger.info("Processing a sequence element.")

    for element in sequence["elements"]:
        scene = generate_scene_node(element, speakers, scene)
    return scene


def process_splitter(scene, splitter, speakers):
    """Processes a splitter element."""
    logger.info("Processing a splitter element.")

    new_scenes = [
        generate_scene_node(element, speakers, scene)
        for element in splitter["elements"]
    ]
    for new_scene in new_scenes:
        scene = scene.union(new_scene)
    return scene


def process_conversation(scene, conversation, speakers):
    """Processes a conversation element."""
    logger.info("Processing a conversation element.")

    end_time = get_end_time(scene)
    last_talker = get_last_talker(scene)
    conversation_end_time = end_time + conversation["duration"] * SAMPLE_RATE

    while end_time < conversation_end_time:
        talker_id = random.choice(conversation["talkers"])
        while talker_id == last_talker:
            talker_id = random.choice(conversation["talkers"])
        talker = next(speaker for speaker in speakers if speaker.id == talker_id)
        utterance = talker.next()
        random_offset = int(np.random.normal(loc=0, scale=16000))  # TODO: config
        start_time = max(0, end_time + random_offset)
        end_time = start_time + utterance["duration"]
        last_talker = talker_id
        scene_element = {
            "type": "utterance",
            "onset": start_time,
            "offset": end_time,
            "channel": talker_id,
            "filename": utterance["name"],
        }
        scene.add(frozendict(scene_element))

    return scene


def process_pause(scene, pause):
    """Processes a pause element."""
    logger.info("Processing a pause element.")
    end_time = get_end_time(scene)
    pause_duration = pause["duration"]
    scene_element = {
        "type": "pause",
        "onset": end_time,
        "offset": end_time + pause_duration * SAMPLE_RATE,
        "channel": 0,
    }
    scene.add(frozendict(scene_element))
    return scene


def generate_scene_node(structure: dict, speakers: list, scene=None):
    """Generates a node in the scene structure."""
    logger.info("Generating the scene from the structure.")
    scene = scene.copy() if scene is not None else set()
    if structure["type"] == "sequence":
        scene = process_sequence(scene, structure, speakers)
    elif structure["type"] == "splitter":
        scene = process_splitter(scene, structure, speakers)
    elif structure["type"] == "conversation":
        scene = process_conversation(scene, structure, speakers)
    elif structure["type"] == "pause":
        scene = process_pause(scene, structure)
    return scene


def generate_scene(structure: dict, speakers: list, scene=None):
    """Generates the scene from the structure."""
    scene = generate_scene_node(structure, speakers, scene)

    # remove pause elements - these were only used during generation
    scene = {utterance for utterance in scene if utterance["type"] != "pause"}

    return scene


def save_scene(scene, scene_file):
    """Saves the scene to a file."""
    logger.info(f"Saving the scene to {scene_file}.")
    scene = [dict(utterance) for utterance in scene]
    with open(scene_file, "w", encoding="utf8") as f:
        json.dump(scene, f, indent=4)


def make_libri_speakers(libri_index_filename):
    """Makes a list of speakers from the LibriSpeech dataset."""
    libri_index = pd.read_csv(libri_index_filename)
    libri_speakers = libri_index.speaker.unique()
    selected_speakers = random.sample(list(libri_speakers), 12)

    spkr_dfs = [
        libri_index[libri_index.speaker == spkr].sort_values(by="file_name")
        for spkr in selected_speakers
    ]
    speakers = [
        Speaker(spkr_df, index) for index, spkr_df in enumerate(spkr_dfs, start=1)
    ]
    return speakers


@hydra.main(
    version_base=None, config_path="conf", config_name="echi_scene_generator_config"
)
def main(cfg):
    """Instantiates the structure."""
    logger.info(f"Instantiating {cfg.structure_file} to make {cfg.scene_file}")

    speakers = make_libri_speakers(cfg.libri_index_file)
    structure = json.load(open(cfg.structure_file, "r", encoding="utf8"))

    scene = generate_scene(structure, speakers)

    save_scene(scene, cfg.scene_file)


if __name__ == "__main__":
    main()  # type: ignore
