"""ECHI scene rendering script.

Takes the low level scene description and renders it into an audio signal.
The scene description is a list of dictionaries, each dictionary represents
a sound source with a channel, onset, offset and a sound file. Applies some
channel normalization to the audio signal so that all channels are at
roughly the same RMS level during the non-silence parts.
"""

import json
import logging
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf  # type: ignore
from tqdm import tqdm

from babble_generator import generate_babble

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


def process_scene(
    scene,
    audio_root,
    audio_out_filename,
    target_rms,
    samplerate,
    n_diffuse_channels=0,
    babble_generator=None,
):
    """Renders the scene into an audio signal."""

    audio = render_scene(scene, Path(audio_root))

    audio = channel_normalization(audio, target_rms)

    # Add diffuse noise to the audio signal

    if n_diffuse_channels > 0 and babble_generator is not None:
        logger.info(f"Adding {n_diffuse_channels} diffuse channels.")
        babble = np.zeros((n_diffuse_channels, audio.shape[1]))
        for i in range(n_diffuse_channels):
            duration = audio.shape[1]
            # The seed is made from a hash of the scene and the index i
            # to ensure that the babble is different for each channel but reproducible
            seed = hash(json.dumps(scene) + str(i)) % (2**32 - 1)
            babble_chan = babble_generator(
                duration=duration, base_duration=duration * 4, seed=seed
            )
            rms_level = np.sqrt(np.mean(babble_chan**2))
            babble[i, :] = babble_chan / rms_level * target_rms
        audio = np.concatenate((audio, babble), axis=0)

    # Write the audio signal to a file
    logger.info(f"Writing audio to {audio_out_filename}")
    sf.write(audio_out_filename, audio.T, samplerate=samplerate)


@hydra.main(
    version_base=None, config_path="conf", config_name="echi_scene_renderer_config"
)
def main(cfg):
    """Render the scene description into an audio signal."""

    # Load the scene description
    with open(cfg.scene_file, "r", encoding="utf8") as f:
        input_data = json.load(f)

    # Can either process a single scene or a list of sessions
    if "session" in input_data[0]:
        # This is a list of sessions...
        scenes = [session["scene"] for session in input_data]
        outfile_names = [f"{session['session']}.wav" for session in input_data]
    else:
        # ...this is a single scene.
        scenes = [input_data]
        outfile_names = [cfg.audio_out_filename]

    # Add path to the output filenames
    outfile_names = [Path(cfg.audio_out_dir) / name for name in outfile_names]

    # Create a babble generator if needed
    babble_generator = (
        partial(
            generate_babble,
            speech_index=pd.read_csv(cfg.diffuse.utterance_index),
            utterance_root=cfg.audio_in_root,
            n_speakers=cfg.diffuse.n_speaker_babble,
        )
        if cfg.diffuse.n_channels > 0
        else None
    )

    # Run the rendering process for each scene
    for scene, outfile_name in tqdm(zip(scenes, outfile_names), "Processing sessions"):
        process_scene(
            scene,
            cfg.audio_in_root,
            outfile_name,
            cfg.target_rms,
            cfg.samplerate,
            cfg.diffuse.n_channels,
            babble_generator,
        )


if __name__ == "__main__":
    main()
