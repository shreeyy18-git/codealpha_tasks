"""
preprocess.py
Parse MIDI files with music21, build a token vocabulary, create fixed-length
integer sequences, and persist everything for the trainer.
"""

import os
import sys
import argparse
import glob
from collections import Counter
from typing import List, Dict, Tuple

os.environ.setdefault("MUSIC21_BRAILLE_NO_AUTO_UPDATE", "1")

from music21 import converter, instrument, note, chord

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SRC_DIR)
from config import get_config, midi_dirs_for_mode, Config
from utils import get_logger, save_vocab, save_sequences, set_seed, Timer

logger = get_logger("Preprocess")
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "REST"]


def quantise_duration(duration_ql: float, buckets: tuple) -> float:
    return min(buckets, key=lambda b: abs(b - duration_ql))


def parse_midi_file(filepath: str, cfg: Config) -> List[str]:
    try:
        score = converter.parse(filepath)
    except Exception as exc:
        logger.warning(f"  Skipping {os.path.basename(filepath)}: {exc}")
        return []

    parts = instrument.partitionByInstrument(score)
    target = parts.parts[0] if parts else score.flat

    tokens: List[str] = []
    preproc = cfg.preprocess

    for element in target.flat.notesAndRests:
        dur = quantise_duration(
            element.duration.quarterLength, preproc.duration_buckets
        )
        dur_token = f"DUR_{dur}"

        if isinstance(element, note.Rest):
            tokens.append("REST")
            tokens.append(dur_token)
        elif isinstance(element, note.Note):
            tokens.append(str(element.pitch.midi))
            tokens.append(dur_token)
        elif isinstance(element, chord.Chord) and preproc.include_chords:
            pitches = sorted(n.pitch.midi for n in element.notes)
            tokens.append(".".join(map(str, pitches)))
            tokens.append(dur_token)

    if len([t for t in tokens if not t.startswith("DUR_")]) < cfg.preprocess.min_notes:
        return []
    return tokens


def augment_with_transpositions(
    tokens: List[str], semitones_list: List[int]
) -> List[List[str]]:
    results = []
    for semitones in semitones_list:
        transposed = []
        for tok in tokens:
            if tok.startswith("DUR_") or tok == "REST":
                transposed.append(tok)
            elif "." in tok:
                pitches = [int(p) + semitones for p in tok.split(".")]
                pitches = [max(0, min(127, p)) for p in pitches]
                transposed.append(".".join(map(str, sorted(pitches))))
            else:
                try:
                    new_pitch = max(0, min(127, int(tok) + semitones))
                    transposed.append(str(new_pitch))
                except ValueError:
                    transposed.append(tok)
        results.append(transposed)
    return results


def build_vocabulary(all_tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter(all_tokens)
    vocab: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for tok, _ in counter.most_common():
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab, {idx: tok for tok, idx in vocab.items()}


def tokens_to_sequences(
    token_lists: List[List[str]], vocab: Dict[str, int], seq_len: int
) -> List[List[int]]:
    unk_id = vocab.get("<PAD>", 0)
    sequences: List[List[int]] = []
    for toks in token_lists:
        ids = [vocab.get(t, unk_id) for t in toks]
        for start in range(0, len(ids) - seq_len):
            sequences.append(ids[start : start + seq_len + 1])
    return sequences


def run_preprocessing(mode: str) -> None:
    cfg = get_config(mode)
    set_seed(cfg.train.seed)
    midi_dirs = midi_dirs_for_mode(mode)

    midi_files = []
    for d in midi_dirs:
        midi_files.extend(
            glob.glob(os.path.join(d, "*.mid")) + glob.glob(os.path.join(d, "*.midi"))
        )

    if not midi_files:
        logger.error("No MIDI files found!")
        sys.exit(1)

    all_token_lists = []
    with Timer():
        for fp in midi_files:
            tokens = parse_midi_file(fp, cfg)
            if tokens:
                all_token_lists.extend(
                    augment_with_transpositions(
                        tokens, cfg.preprocess.transpose_semitones
                    )
                )

    vocab, idx2token = build_vocabulary([tok for seq in all_token_lists for tok in seq])
    sequences = tokens_to_sequences(
        all_token_lists, vocab, cfg.preprocess.sequence_length
    )

    save_vocab(vocab, idx2token, cfg.vocab_path)
    save_sequences(sequences, cfg.processed_path)
    logger.info(f"Preprocessed {len(sequences)} sequences for mode '{mode}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hiphop", "retro", "mixed"], default="mixed")
    args = parser.parse_args()
    run_preprocessing(args.mode)
