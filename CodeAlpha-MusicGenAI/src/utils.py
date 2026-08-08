"""
utils.py
Shared utility functions: vocab I/O, seeding, logging, checkpointing.
"""

import os
import pickle
import random
import logging
import time
import shutil
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch


def get_logger(name: str = "MusicGen", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = get_logger()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {name} ({mem:.1f} GB)")
    else:
        dev = torch.device("cpu")
        logger.warning("CUDA not available - training on CPU (will be slow!)")
    return dev


def save_vocab(vocab: Dict[str, int], idx2token: Dict[int, str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"vocab": vocab, "idx2token": idx2token}, f)
    logger.info(f"Vocabulary saved -> {path} (size={len(vocab)})")


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    vocab = obj["vocab"]
    idx2token = obj["idx2token"]
    logger.info(f"Vocabulary loaded <- {path} (size={len(vocab)})")
    return vocab, idx2token


def save_sequences(sequences: List[List[int]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(sequences, f)
    logger.info(f"Sequences saved -> {path} (count={len(sequences)})")


def load_sequences(path: str) -> List[List[int]]:
    with open(path, "rb") as f:
        sequences = pickle.load(f)
    logger.info(f"Sequences loaded <- {path} (count={len(sequences)})")
    return sequences


def save_checkpoint(
    state: dict, path: str, is_best: bool = False, best_path: Optional[str] = None
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved -> {path}")
    if is_best and best_path:
        shutil.copyfile(path, best_path)
        logger.info(f"Best checkpoint updated -> {best_path}")


def load_checkpoint(path: str, device: torch.device) -> dict:
    logger.info(f"Loading checkpoint <- {path}")
    return torch.load(path, map_location=device)


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    threshold = values[:, -1].unsqueeze(-1)
    return torch.where(
        logits < threshold, torch.full_like(logits, float("-inf")), logits
    )


def top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > p
    sorted_logits[sorted_remove] = float("-inf")
    out = torch.zeros_like(logits)
    out.scatter_(1, sorted_idx, sorted_logits)
    return out


def sample_token(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0
) -> int:
    logits = logits / max(temperature, 1e-8)
    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.elapsed:.2f}s"


def progress_bar(current: int, total: int, width: int = 30, suffix: str = "") -> str:
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / max(total, 1)
    return f"[{bar}] {pct:5.1f}%  {suffix}"
