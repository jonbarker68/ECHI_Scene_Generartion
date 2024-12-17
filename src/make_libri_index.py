"""Build the librispeech index file.

Reads the flac files under the LibriSpeech root and makes an index storing
the file paths and the number of samples in each file. This is handy when
generating the conversations.

Example:
    python make_libri_index.py /path/to/LibriSpeech index.csv
"""

import argparse
import csv
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def get_file_length(file_name: str) -> int:
    """Get the number of samples in a flac file.

    Args:
        file_name: Path to the flac file.

    Returns:
        The number of samples in the file.
    """
    try:
        with sf.SoundFile(file_name) as f:
            return len(f)
    except RuntimeError as e:
        print(f"Error reading {file_name}: {e}")
        return 0


def parse_file_name(file_name: str) -> tuple[int, int, int]:
    """Parse the file name to get the speaker and chapter information.

    Args:
        file_name: Path to the flac file.

    Returns:
        A tuple containing the speaker ID, the chapter ID, and the utterance ID.
    """
    parts = Path(file_name).stem.split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])


def build_index(root: str) -> list[dict]:
    """Build the index of LibriSpeech files.

    Args:
        root: Root directory of LibriSpeech.

    Returns:
        A dictionary mapping the file paths to the number of samples in each
        file.
    """
    file_names = [str(file.relative_to(root)) for file in Path(root).rglob("*.flac")]

    file_data = [
        {
            "file_name": file_name,
            "length": 0,
            "speaker": 0,
            "chapter": 0,
            "utterance": 0,
        }
        for file_name in file_names
    ]
    for file_entry in tqdm(file_data, desc="Building index"):
        file_entry["length"] = get_file_length(Path(root) / file_entry["file_name"])
        file_entry["speaker"], file_entry["chapter"], file_entry["utterance"] = (
            parse_file_name(file_entry["file_name"])
        )

    return file_data


def main():
    """Build an index for the LibriSpeech files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Root directory of LibriSpeech")
    parser.add_argument("index", help="Output index file")
    args = parser.parse_args()

    index = build_index(args.root)

    # print list of dict as a csv file
    with open(args.index, "w", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=index[0].keys())
        writer.writeheader()
        writer.writerows(index)


if __name__ == "__main__":
    main()
