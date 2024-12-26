from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    target_rms: float = 0.05


@dataclass
class PathConfig:
    data_root: str = "./data"
    libri_speech: str = "${paths.data_root}/LibriSpeech/train-clean-100"
    output: str = "${paths.data_root}/echi_audio"
    libri_root: str = "${paths.data_root}/LibriSpeech/train-clean-100"
    utt_index: str = "${paths.data_root}/libri_index.csv"
    chapter_index: str = "${paths.data_root}/libri_chapters.csv"


@dataclass
class SpeakerConfig:
    ids: List[int]
    offset_scale: int = 4000
    libri_index_file: str = "${paths.data_root}/libri_index.csv"


@dataclass
class StructureConfig:
    duration: int = 1800
    table_sizes: List[int] = (4, 4, 4)
    segment: bool = True
    half_life: int = 600
    min_duration: int = 30


@dataclass
class SegmenterConfig:
    aggressiveness: int = 1


@dataclass
class DiffuseConfig:
    n_channels: int = 4
    n_speaker_babble: int = 20


@dataclass
class Config:
    paths: PathConfig = PathConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
