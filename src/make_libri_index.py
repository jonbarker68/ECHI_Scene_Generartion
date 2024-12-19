"""Build the librispeech index file.

Reads the flac files under the LibriSpeech root and makes an index storing
the file paths and the number of samples in each file. This is handy when
generating the conversations.

Example:
    python make_libri_index.py /path/to/LibriSpeech index.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore
import webrtcvad
from tqdm import tqdm

from librispeech_utils import parse_file_name


def get_file_rms_level(file_name: str | Path) -> float:
    """Compute the RMS level of a file.

    Args:
        file_name: Path to the flac file.

    Returns:
        The number of samples in the file.
    """
    with sf.SoundFile(file_name) as f:
        signal = f.read()
        return np.sqrt(np.mean(signal**2))


def get_file_rms_level_vad(file_name: str | Path) -> float:
    """A better way to compute the RMS level of a file using VAD.

    Args:
        file_name: Path to the flac file.

    Returns:
        The RMS level of the speech signal in the file.
    """
    with sf.SoundFile(file_name) as f:
        signal = f.read(dtype="int16")
        sample_rate = f.samplerate

    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressive mode

    frame_duration = 30  # 30 ms
    frame_size = int(sample_rate * frame_duration / 1000)
    num_frames = len(signal) // frame_size
    speech_frames = []
    for i in range(num_frames):
        frame = signal[i * frame_size : (i + 1) * frame_size]
        if len(frame) == frame_size and vad.is_speech(frame.tobytes(), sample_rate):
            speech_frames.extend(frame)

    if len(speech_frames) == 0:
        return 0  # No speech detected

    speech_signal = np.array(speech_frames, dtype="int16")
    speech_signal = speech_signal / 2**15  # Normalize to [-1, 1]
    return np.sqrt(np.mean(speech_signal**2))


def get_file_length(file_name: str | Path) -> int:
    """Get the number of samples in a flac file.

    Args:
        file_name: Path to the flac file.

    Returns:
        The number of samples in the file.
    """
    with sf.SoundFile(file_name) as f:
        return len(f)


def build_chapter_index(utterance_index: list[dict]) -> list[dict]:
    """Build the index of LibriSpeech chapters.

    Sums over all the utterances in the chapter to compute a length and rms level.
    """
    chapter_index = []
    speaker_chapters = {
        (entry["speaker"], entry["chapter"]) for entry in utterance_index
    }
    for speaker, chapter in tqdm(speaker_chapters, desc="Building chapter index"):
        chapter_entries = [
            entry
            for entry in utterance_index
            if entry["speaker"] == speaker and entry["chapter"] == chapter
        ]
        chapter_length = sum(entry["length"] for entry in chapter_entries)
        chapter_sum_squared_level = sum(
            entry["rms_level_vad"] ** 2 * entry["length"] for entry in chapter_entries
        )
        chapter_rms_level = np.sqrt(chapter_sum_squared_level / chapter_length)
        chapter_index.append(
            {
                "speaker": speaker,
                "chapter": chapter,
                "length": chapter_length,
                "rms_level": chapter_rms_level,
            }
        )
    return chapter_index


def build_utterance_index(root: str) -> list[dict]:
    """Build the index of LibriSpeech files.

    Args:
        root: Root directory of LibriSpeech.

    Returns:
        A dictionary mapping the file paths to the number of samples in each
        file.
    """
    file_names: list[str] = [
        str(file.relative_to(root)) for file in Path(root).rglob("*.flac")
    ]

    file_data = [
        {
            "file_name": str(file_name),
            "length": 0,
            "rms_level_raw": 0,
            "rms_level_vad": 0,
            "speaker": 0,
            "chapter": 0,
            "utterance": 0,
        }
        for file_name in file_names
    ]

    for file_entry in tqdm(file_data, desc="Building index"):
        file_name_str = str(file_entry["file_name"])
        file_entry["speaker"], file_entry["chapter"], file_entry["utterance"] = (
            parse_file_name(file_name_str)
        )
        try:
            file_entry["length"] = get_file_length(Path(root) / file_name_str)
            file_entry["rms_level_raw"] = get_file_rms_level(Path(root) / file_name_str)
            file_entry["rms_level_vad"] = get_file_rms_level_vad(
                Path(root) / file_name_str
            )
        except Exception as e:
            logging.error(f"Error processing {file_name_str}: {e}")

    return file_data


def main():
    """Build an index for the LibriSpeech files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Root directory of LibriSpeech")
    parser.add_argument("utt_index", help="Output index file")
    parser.add_argument("chapter_index", help="Output index file")
    args = parser.parse_args()

    utterance_index = build_utterance_index(args.root)
    chapter_index = build_chapter_index(utterance_index)

    # print list of dict as a csv file
    with open(args.utt_index, "w", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=utterance_index[0].keys())
        writer.writeheader()
        writer.writerows(utterance_index)

    # print list of dict as a csv file
    with open(args.chapter_index, "w", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=chapter_index[0].keys())
        writer.writeheader()
        writer.writerows(chapter_index)


if __name__ == "__main__":
    main()
