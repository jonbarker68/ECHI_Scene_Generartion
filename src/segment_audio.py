"""segments speech audio files using the WebRTC VAD.

i.e., a longer speech recording with non-speech pauses is turned into a sequence
of end-pointed speech segments.
Adapted from https://github.com/wiseman/py-webrtcvad

usage:
python src/segment_audio.py +input_dir=/path/to/audio +output_dir=/path/to/output
"""

import collections
from pathlib import Path

import hydra
import numpy as np
import soundfile as sf  # type: ignore
import webrtcvad  # type: ignore
from tqdm import tqdm

from conf import Config


def read_flac(path):
    """Reads an audio file (WAV or FLAC).

    Takes the path and returns (PCM audio data, sample rate).
    """
    # Read the audio file
    data, sample_rate = sf.read(path, dtype="int16")

    # Ensure the audio is mono
    if len(data.shape) > 1 and data.shape[1] != 1:
        raise ValueError("Audio file must be mono (single channel).")
    elif len(data.shape) > 1:
        data = data[:, 0]  # Select the first channel if stereo

    # Ensure the sample rate is one of the supported rates
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000 Hz.")

    # Convert the data to bytes
    pcm_data = data.tobytes()

    return pcm_data, sample_rate


def write_flac(path, audio, sample_rate):
    """Writes a .flac file.

    Takes path, PCM audio data, and sample rate.
    """
    # Convert bytes back to int16 array
    audio_array = np.frombuffer(audio, dtype=np.int16)
    sf.write(path, audio_array, sample_rate, format="FLAC")


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, n_bytes, timestamp, duration):
        self.bytes = n_bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOT_TRIGGERED. We start in the
    # NOT_TRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOT_TRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if ring_buffer.maxlen is not None and num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # We want to yield all the audio we see from now until
                # we are NOT_TRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOT_TRIGGERED and yield whatever
            # audio we've collected.
            if (
                ring_buffer.maxlen is not None
                and num_unvoiced > 0.9 * ring_buffer.maxlen
            ):
                triggered = False
                yield b"".join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b"".join([f.bytes for f in voiced_frames])


def process_audio(wavfile, aggressiveness, output_file_root):
    audio, sample_rate = read_flac(wavfile)
    vad = webrtcvad.Vad(aggressiveness)
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    # Write the segments to disk
    for i, segment in enumerate(segments):
        output_file = f"{output_file_root}-{i:03d}.flac"
        write_flac(output_file, segment, sample_rate)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    """Command-line entry point."""

    # Read and segment the audio
    files = Path(cfg.paths.corpus_root).rglob(cfg.segmenter.wavfile)
    for file in tqdm(list(files)):
        # make directories if they don't exist
        output_file = Path(cfg.paths.segmenter_outdir) / file.relative_to(
            cfg.paths.corpus_root
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        process_audio(file, cfg.segmenter.aggressiveness, output_file.with_suffix(""))


if __name__ == "__main__":
    main()
