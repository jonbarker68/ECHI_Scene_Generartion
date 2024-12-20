"""ECHI scene rendering script.

Takes the low level scene description and renders it into an audio signal.
The scene description is a list of dictionaries, each dictionary represents
a sound source with a channel, onset, offset and a sound file. Applies some
channel normalization to the audio signal so that all channels are at
roughly the same RMS level during the non-silence parts.
"""

import json
import logging
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf  # type: ignore
from tqdm import tqdm

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
    for source in tqdm(scene, desc="Rendering scene"):
        channel = source["channel"]
        onset = source["onset"]
        offset = source["offset"]
        filename = source["filename"]
        with sf.SoundFile(audio_root / filename) as f:
            audio_segment = f.read()
        audio[channel - 1, onset:offset] += audio_segment

    return audio


def channel_normalization(audio, target_rms, clip=True):
    """Normalizes the audio signal to the target RMS level.

    The signals have a lot of silence, so the normalization does a
    crude speech detection by only considering samples with an
    absolute value above the 10th percentile.

    Args:
        audio: The audio signal to normalize.
        target_rms: The target RMS level.
        clip: Whether to clip the audio signal to [-1, 1].

    Returns:
        The normalized audio signal.
    """
    logger.info(f"Normalising audio channels to {target_rms} RMS.")

    audio_copy = audio.copy()
    # zero out all parts in lowest 10% of the signal
    audio_copy = np.where(
        audio_copy < np.percentile(np.abs(audio_copy), 10), 0, audio_copy
    )
    rms = np.sqrt(np.sum(audio_copy**2, axis=1) / np.sum(audio_copy != 0, axis=1))
    gain = target_rms / rms
    audio = audio * gain[:, np.newaxis]
    if clip:
        audio = np.clip(audio, -1, 1)
    return audio


@hydra.main(
    version_base=None, config_path="conf", config_name="echi_scene_renderer_config"
)
def main(cfg):
    # Load the scene description
    with open(cfg.scene_file, "r", encoding="utf8") as f:
        scene = json.load(f)

    audio = render_scene(scene, Path(cfg.audio_root))

    audio = channel_normalization(audio, cfg.target_rms)

    # Write the audio signal to a file
    logger.info(f"Writing audio to {cfg.audio_file}")
    sf.write(cfg.audio_file, audio.T, samplerate=cfg.samplerate)


if __name__ == "__main__":
    main()
