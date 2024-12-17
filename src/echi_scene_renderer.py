"""ECHI scene rendering script.

Takes the low level scene description and renders it into an audio signal.
The scene description is a list of dictionaries, each dictionary represents
a sound source with a channel, onset, offset and a sound file.
"""

import hydra
import numpy as np
import json
import soundfile as sf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def render_scene(scene, audio_root):
    """Renders the scene into an audio signal.

    Args:
        scene: A list of dictionaries, each dictionary represents a sound source
            with a channel, onset, offset and a sound file.

    Returns:
        The audio signal of the scene.
    """

    # Pre-allocate the audio signal with zeros
    n_channels = max(source["channel"] for source in scene)
    n_samples = max(source["offset"] for source in scene)
    audio = np.zeros((n_channels, n_samples))

    # Add each source to the pre-allocated audio signal
    logger.info(f"Rendering scene with {n_channels} channels and {n_samples} samples.")
    for source in scene:
        channel = source["channel"]
        onset = source["onset"]
        offset = source["offset"]
        with sf.SoundFile(audio_root / source["filename"]) as f:
            audio_segment = f.read()
        audio[channel - 1, onset:offset] += audio_segment

    return audio


@hydra.main(
    version_base=None, config_path="conf", config_name="echi_scene_renderer_config"
)
def main(cfg):

    # Load the scene description
    with open(cfg.scene_file, "r", encoding="utf8") as f:
        scene = json.load(f)

    audio = render_scene(scene, Path(cfg.audio_root))

    # Write the audio signal to a file
    logger.info(f"Writing audio to {cfg.audio_file}")
    sf.write(cfg.audio_file, audio.T, samplerate=cfg.samplerate)


if __name__ == "__main__":
    main()
