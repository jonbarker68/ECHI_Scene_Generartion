"""Generate N speaker babble from a single speaker dataset.

Usage:
python src/babble_generator.py +output_file=output.wav
"""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf  # type: ignore
from tqdm import tqdm


def make_base_stream(speech_index, utterance_root, duration):
    """Make a stream of segments from a single speaker.

    duration: The duration of the base stream in samples.
    """
    logging.info("Making base stream.")

    # remove segments with zero rms_level_vad
    speech_index = speech_index[speech_index["rms_level_vad"] > 0]

    # pre-allocate the base stream with zeros
    base_stream = np.zeros(duration)

    # Add add randomly selected segments to the base stream
    start_sample = 0
    while start_sample < duration:
        # Select a random segment
        segment = speech_index.sample()
        filename = segment.iloc[0]["file_name"]
        filename = Path(utterance_root) / filename
        rms_level = segment.iloc[0]["rms_level_vad"]
        with sf.SoundFile(filename) as f:
            segment_samples = f.read()
        if start_sample + len(segment_samples) > duration:
            segment_samples = segment_samples[: duration - start_sample]
        base_stream[start_sample : start_sample + len(segment_samples)] = (
            segment_samples / rms_level
        )
        start_sample += len(segment_samples)

    return base_stream


def mix_speakers(base_stream, n_speakers, duration):
    """Mix segments of the base stream to create a babble with N speakers."""
    base_stream_length = len(base_stream)
    start_range = base_stream_length - duration
    babble = np.zeros(duration)
    for _ in tqdm(range(n_speakers)):
        start = np.random.randint(0, start_range)
        babble += base_stream[start : start + duration]
    return babble


def generate_babble(
    speech_index, utterance_root, duration, n_speakers, base_duration, seed
):
    """Generate N speaker babble from a single speaker dataset."""
    # set the random seed
    np.random.seed(seed)

    base_stream = make_base_stream(speech_index, utterance_root, base_duration)
    babble = mix_speakers(base_stream, n_speakers, duration)
    return babble


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    """Generate N speaker babble"""

    if "output_file" not in cfg or not cfg.output_file:
        raise ValueError("output_file not set in config")

    speech_index = pd.read_csv(cfg.paths.utt_index)

    babble = generate_babble(
        speech_index,
        cfg.paths.libri_root,
        cfg.structure.duration * cfg.audio.sample_rate,
        cfg.diffuse.n_speaker_babble,
        cfg.structure.duration * 4 * cfg.audio.sample_rate,
        cfg.master.seed,
    )

    # Normalize the babble to the target RMS level
    rms_level = np.sqrt(np.mean(babble**2))
    babble = babble / rms_level * cfg.audio.target_rms

    with sf.SoundFile(
        cfg.output_file, "w", samplerate=cfg.audio.sample_rate, channels=1
    ) as f:
        f.write(babble)


if __name__ == "__main__":
    main()
    main()
