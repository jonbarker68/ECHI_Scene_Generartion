from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class MasterConfig:
    n_sessions: int = MISSING
    seed: int = MISSING


@dataclass
class AudioConfig:
    sample_rate: int = MISSING
    target_rms: float = MISSING


@dataclass
class PathConfig:
    data_root: str = MISSING
    corpus_root: str = MISSING
    segmenter_outdir: str = MISSING
    audio_dir: str = MISSING
    audio_file: str = MISSING

    utt_index: str = MISSING
    chapter_index: str = MISSING
    master_file: str = MISSING
    scene_file: str = MISSING
    structure_file: str = MISSING


@dataclass
class SpeakerConfig:
    ids: List[int] = MISSING
    offset_scale: int = MISSING
    libri_index_file: str = MISSING
    min_speaker_duration: int = MISSING


@dataclass
class StructureConfig:
    duration: int = MISSING
    table_sizes: tuple[int, ...] = MISSING
    segment: bool = MISSING
    half_life: int = MISSING
    min_duration: int = MISSING


@dataclass
class SegmenterConfig:
    aggressiveness: int = MISSING
    wavfile: str = MISSING


@dataclass
class DiffuseConfig:
    n_channels: int = MISSING
    n_speaker_babble: int = MISSING


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    diffuse: DiffuseConfig = field(default_factory=DiffuseConfig)
    master: MasterConfig = field(default_factory=MasterConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    session: str = MISSING
    seed: int = MISSING


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
