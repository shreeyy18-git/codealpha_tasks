"""
config.py
Central configuration for the Music Generation project.
All paths, model hyperparameters, and training settings live here.
"""

import os
from dataclasses import dataclass, field

# Project root (one level above src/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
MIDI_DIR = os.path.join(DATA_DIR, "MIDI")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def _pick_existing_dir(*candidates: str) -> str:
    """Return the first existing path from candidates, else the first candidate."""
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


HIPHOP_MIDI_DIR = _pick_existing_dir(
    os.path.join(MIDI_DIR, "hiphop"),
    os.path.join(MIDI_DIR, "HipHop"),
)
RETRO_MIDI_DIR = _pick_existing_dir(
    os.path.join(MIDI_DIR, "retro"),
    os.path.join(MIDI_DIR, "RetroGame"),
)
MIXED_MIDI_DIR = _pick_existing_dir(
    os.path.join(MIDI_DIR, "mixed"),
    os.path.join(MIDI_DIR, "Mixed"),
)

# Output directories
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# Auto-create dirs at import time
for _d in [PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# Supported modes
MODES = ("hiphop", "retro", "mixed")


# Preprocessing
@dataclass
class PreprocessConfig:
    sequence_length: int = 64
    min_notes: int = 20
    transpose_semitones: list = field(default_factory=lambda: [-2, -1, 0, 1, 2])
    include_chords: bool = True
    duration_buckets: tuple = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


# Model
@dataclass
class ModelConfig:
    hidden_size: int = 512
    num_layers: int = 3
    embedding_dim: int = 256
    dropout: float = 0.3
    bidirectional: bool = False  # keep False for autoregressive generation


# Training
@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    lr_step_size: int = 20
    lr_gamma: float = 0.5
    save_every: int = 10
    early_stop_patience: int = 15
    grad_clip: float = 5.0
    use_amp: bool = True
    seed: int = 42
    num_workers: int = 0  # Set to 0 for Windows compatibility, 4 for Linux/Mac
    val_split: float = 0.1


# Generation
@dataclass
class GenerateConfig:
    num_tokens: int = 256
    seed_length: int = 32
    temperature: float = 1.0
    top_k: int = 10
    top_p: float = 0.9
    bpm: int = 90
    instrument_program: int = 0
    output_prefix: str = "generated"


# Convenience: bundle everything
@dataclass
class Config:
    mode: str = "mixed"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    processed_path: str = ""
    vocab_path: str = ""
    checkpoint_dir: str = ""
    best_checkpoint: str = ""

    def __post_init__(self):
        self._refresh_paths()

    def _refresh_paths(self):
        self.processed_path = os.path.join(PROCESSED_DIR, f"{self.mode}_sequences.pkl")
        self.vocab_path = os.path.join(PROCESSED_DIR, f"{self.mode}_vocab.pkl")
        self.checkpoint_dir = os.path.join(MODELS_DIR, self.mode)
        self.best_checkpoint = os.path.join(self.checkpoint_dir, "best_model.pt")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


def get_config(mode: str = "mixed") -> Config:
    assert mode in MODES, f"mode must be one of {MODES}, got '{mode}'"
    cfg = Config(mode=mode)
    cfg._refresh_paths()
    return cfg


def midi_dirs_for_mode(mode: str) -> list:
    if mode == "mixed":
        if os.path.isdir(MIXED_MIDI_DIR):
            return [MIXED_MIDI_DIR]
        return [HIPHOP_MIDI_DIR, RETRO_MIDI_DIR]

    mapping = {
        "hiphop": [HIPHOP_MIDI_DIR],
        "retro": [RETRO_MIDI_DIR],
    }
    return mapping[mode]
