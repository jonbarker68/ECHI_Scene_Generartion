"""Generate N speaker babble from a single speaker dataset."""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import soundfile as sf  # type: ignore
from tqdm import tqdm


def make_base_stream(speech_index, utterance_root, duration, samplerate):
    """Make a stream of segments from a single speaker."""
    logging.info("Making base stream.")

    # remove segments with zero rms_level_vad
    speech_index = speech_index[speech_index["rms_level_vad"] > 0]

    # pre-allocate the base stream with zeros
    duration_samples = int(duration * samplerate)
    base_stream = np.zeros(duration_samples)

    # Add add randomly selected segments to the base stream
    start_sample = 0
    while start_sample < duration_samples:
        # Select a random segment
        segment = speech_index.sample()
        filename = segment.iloc[0]["file_name"]
        filename = Path(utterance_root) / filename
        rms_level = segment.iloc[0]["rms_level_vad"]
        with sf.SoundFile(filename) as f:
            segment_samples = f.read()
        if start_sample + len(segment_samples) > duration_samples:
            segment_samples = segment_samples[: duration_samples - start_sample]
        base_stream[start_sample : start_sample + len(segment_samples)] = (
            segment_samples / rms_level
        )
        start_sample += len(segment_samples)

    return base_stream


def mix_speakers(base_stream, n_speakers, duration, samplerate):
    """Mix segments of the base stream to create a babble with N speakers."""
    duration_samples = int(duration * samplerate)
    base_stream_length = len(base_stream)
    start_range = base_stream_length - duration_samples
    babble = np.zeros(duration_samples)
    for _ in tqdm(range(n_speakers)):
        start = np.random.randint(0, start_range)
        babble += base_stream[start : start + duration_samples]
    return babble


def generate_babble(
    speech_index, utterance_root, duration, n_speakers, samplerate, base_duration
):
    """Generate N speaker babble from a single speaker dataset."""
    base_stream = make_base_stream(
        speech_index, utterance_root, base_duration, samplerate
    )
    babble = mix_speakers(base_stream, n_speakers, duration, samplerate)
    return babble


@hydra.main(
    version_base=None, config_path="conf", config_name="babble_generator_config"
)
def main(cfg):
    """Generate N speaker babble"""

    # set the random seed
    np.random.seed(cfg.seed)

    speech_index = pd.read_csv(cfg.speech_index_file)

    babble = generate_babble(
        speech_index,
        cfg.utterance_root,
        cfg.duration,
        cfg.n_speakers,
        cfg.samplerate,
        cfg.base_duration,
    )

    # Normalize the babble to the target RMS level
    rms_level = np.sqrt(np.mean(babble**2))
    babble = babble / rms_level * cfg.target_rms

    with sf.SoundFile(cfg.output_file, "w", samplerate=cfg.samplerate, channels=1) as f:
        f.write(babble)


if __name__ == "__main__":
    main()
