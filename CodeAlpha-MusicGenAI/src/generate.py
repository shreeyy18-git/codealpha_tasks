"""
generate.py
Load a trained MusicLSTM checkpoint and generate new MIDI music.
"""

import os
import sys
import argparse
import datetime
from typing import List

import torch

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SRC_DIR)
from config import get_config, Config, OUTPUTS_DIR
from utils import (
    get_logger,
    get_device,
    load_vocab,
    load_checkpoint,
    sample_token,
    Timer,
)

logger = get_logger("Generate")

try:
    from music21 import stream, note, chord, instrument as m21inst, tempo, midi

    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False
    logger.warning("music21 not installed - MIDI export disabled.")


def _load_model(cfg: Config, device: torch.device):
    """Import model lazily to avoid tight coupling and circular imports."""
    from train import MusicLSTM

    vocab, idx2token = load_vocab(cfg.vocab_path)
    vocab_size = len(vocab)

    model = MusicLSTM(vocab_size, cfg).to(device)

    if not os.path.exists(cfg.best_checkpoint):
        logger.error(
            "No checkpoint found at %s. Train the model first.", cfg.best_checkpoint
        )
        sys.exit(1)

    ckpt = load_checkpoint(cfg.best_checkpoint, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    logger.info(
        "Model loaded - epoch=%s val_loss=%.4f vocab_size=%s",
        epoch,
        val_loss,
        vocab_size,
    )
    return model, vocab, idx2token


def _random_seed(sequences_path: str, seed_length: int) -> List[int]:
    import random
    from utils import load_sequences

    sequences = load_sequences(sequences_path)
    seq = random.choice(sequences)
    seed = seq[:seed_length]
    logger.info("Random seed drawn from training data (len=%s)", len(seed))
    return seed


def _file_seed(midi_path: str, seed_length: int, vocab: dict, cfg: Config) -> List[int]:
    from preprocess import parse_midi_file

    tokens = parse_midi_file(midi_path, cfg)
    if not tokens:
        logger.warning("Could not extract tokens from seed file - using random seed.")
        return _random_seed(cfg.processed_path, seed_length)

    pad_id = vocab.get("<PAD>", 0)
    ids = [vocab.get(t, pad_id) for t in tokens[:seed_length]]
    logger.info("Seed from file: %s (len=%s)", os.path.basename(midi_path), len(ids))
    return ids


@torch.no_grad()
def generate_sequence(
    model,
    seed_ids: List[int],
    num_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> List[int]:
    model.eval()
    generated = list(seed_ids)

    x = torch.tensor([seed_ids], dtype=torch.long, device=device)
    _, hidden = model(x)

    for _ in range(num_tokens):
        last_token = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        logits, hidden = model(last_token, hidden)
        logits = logits[:, -1, :]

        next_id = sample_token(
            logits, temperature=temperature, top_k=top_k, top_p=top_p
        )
        generated.append(next_id)

    return generated


def _parse_duration_token(tok: str) -> float:
    try:
        return float(tok.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0.5


def tokens_to_midi_stream(
    token_ids: List[int],
    idx2token: dict,
    bpm: int = 90,
    instrument_program: int = 0,
) -> "stream.Score":
    tokens = [idx2token.get(i, "<PAD>") for i in token_ids]

    events = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("<PAD>", "<SOS>", "<EOS>"):
            i += 1
            continue
        if tok.startswith("DUR_"):
            i += 1
            continue

        dur_ql = 0.5
        if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_"):
            dur_ql = _parse_duration_token(tokens[i + 1])
            i += 2
        else:
            i += 1

        events.append((tok, dur_ql))

    score = stream.Score()
    part = stream.Part()
    instr = m21inst.Piano()
    instr.midiProgram = instrument_program
    part.append(instr)
    part.append(tempo.MetronomeMark(number=bpm))

    for ev_tok, dur_ql in events:
        ql = max(dur_ql, 0.125)
        if ev_tok == "REST":
            # Keep only very short rests to avoid silence accumulation.
            if ql <= 0.5:
                part.append(note.Rest(quarterLength=ql))
            continue
        elif "." in ev_tok:
            try:
                pitches = [int(p) for p in ev_tok.split(".")]
                part.append(chord.Chord(pitches, quarterLength=ql))
            except ValueError:
                continue
        else:
            try:
                part.append(note.Note(int(ev_tok), quarterLength=ql))
            except ValueError:
                continue

    score.append(part)
    return score


def save_midi(score: "stream.Score", output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mf = midi.translate.music21ObjectToMidiFile(score)
    mf.open(output_path, "wb")
    mf.write()
    mf.close()
    logger.info("MIDI saved -> %s", output_path)


def generate(mode: str, args: argparse.Namespace) -> None:
    cfg = get_config(mode)
    device = get_device()

    gc = cfg.generate
    if args.num_tokens:
        gc.num_tokens = args.num_tokens
    if args.temperature:
        gc.temperature = args.temperature
    if args.top_k is not None:
        gc.top_k = args.top_k
    if args.top_p is not None:
        gc.top_p = args.top_p
    if args.bpm:
        gc.bpm = args.bpm
    if args.seed_length:
        gc.seed_length = args.seed_length

    model, vocab, idx2token = _load_model(cfg, device)

    if args.seed_file and os.path.exists(args.seed_file):
        seed_ids = _file_seed(args.seed_file, gc.seed_length, vocab, cfg)
    else:
        if not os.path.exists(cfg.processed_path):
            logger.error(
                "No processed sequences found and no seed file given. Run preprocess.py first."
            )
            sys.exit(1)
        seed_ids = _random_seed(cfg.processed_path, gc.seed_length)

    logger.info(
        "Generating %s tokens temp=%s top_k=%s top_p=%s",
        gc.num_tokens,
        gc.temperature,
        gc.top_k,
        gc.top_p,
    )

    with Timer() as t:
        token_ids = generate_sequence(
            model,
            seed_ids,
            gc.num_tokens,
            device,
            temperature=gc.temperature,
            top_k=gc.top_k,
            top_p=gc.top_p,
        )

    logger.info("Generated %s tokens in %s", len(token_ids), t)

    if not HAS_MUSIC21:
        logger.warning("music21 unavailable - skipping MIDI export.")
        return

    score = tokens_to_midi_stream(
        token_ids,
        idx2token,
        bpm=gc.bpm,
        instrument_program=gc.instrument_program,
    )

    output_name = None
    if args.output_filename:
        output_name = os.path.basename(args.output_filename.strip())
        if output_name and not output_name.lower().endswith((".mid", ".midi")):
            output_name += ".mid"

    if not output_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{gc.output_prefix}_{mode}_{timestamp}.mid"

    output_path = os.path.join(args.output_dir or OUTPUTS_DIR, output_name)
    save_midi(score, output_path)

    note_events = [
        t
        for t in [idx2token.get(i) for i in token_ids]
        if t
        and not t.startswith("DUR_")
        and t not in ("<PAD>", "<SOS>", "<EOS>", "REST")
    ]
    logger.info("Note/chord events generated: %s", len(note_events))
    logger.info("Done! Output: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate music with a trained MusicLSTM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        "--dataset",
        "-m",
        dest="mode",
        choices=["hiphop", "retro", "mixed"],
        default="mixed",
        help="Which trained model to use",
    )
    parser.add_argument(
        "--num_tokens",
        "--length",
        "--len",
        dest="num_tokens",
        type=int,
        default=None,
        help="Tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        "--temp",
        dest="temperature",
        type=float,
        default=None,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling (0=off)"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Nucleus sampling (1.0=off)"
    )
    parser.add_argument("--bpm", type=int, default=None, help="Output MIDI BPM")
    parser.add_argument(
        "--seed_length", type=int, default=None, help="Seed sequence length"
    )
    parser.add_argument(
        "--seed_file", type=str, default=None, help="Path to a seed MIDI file"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Optional output MIDI filename",
    )
    parser.add_argument(
        "--output_dir",
        "--out",
        dest="output_dir",
        type=str,
        default=None,
        help="Where to save the MIDI",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.mode, args)
