from pathlib import Path


def parse_file_name(file_name: str) -> tuple[int, int, int]:
    """Parse a librispeech file name to get the speaker and chapter information.

    Args:
        file_name: Path to the flac file.

    Returns:
        A tuple containing the speaker ID, the chapter ID, and the utterance ID.
    """
    parts = Path(file_name).stem.split("-")
    return int(parts[0]), int(parts[1]), int(parts[2])
