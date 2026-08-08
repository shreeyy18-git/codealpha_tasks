"""
train.py
LSTM-based music generation model + training loop.
"""

import os
import sys
import math
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, SRC_DIR)
from config import get_config, Config
from utils import (
    get_logger,
    set_seed,
    get_device,
    load_vocab,
    load_sequences,
    save_checkpoint,
    load_checkpoint,
    Timer,
    progress_bar,
)

logger = get_logger("Train")


class MusicDataset(Dataset):
    """Each item is (input_ids, target_ids), with target shifted by 1."""

    def __init__(self, sequences: List[List[int]]):
        self.data = [torch.tensor(seq, dtype=torch.long) for seq in sequences]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        return seq[:-1], seq[1:]


class MusicLSTM(nn.Module):
    """Embedding -> stacked LSTM -> classifier over vocabulary."""

    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        mc = cfg.model

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=mc.embedding_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=mc.embedding_dim,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            dropout=mc.dropout if mc.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=mc.bidirectional,
        )

        lstm_out_size = mc.hidden_size * (2 if mc.bidirectional else 1)
        self.dropout = nn.Dropout(mc.dropout)
        self.layer_norm = nn.LayerNorm(lstm_out_size)
        self.fc = nn.Linear(lstm_out_size, vocab_size)

        if lstm_out_size == mc.embedding_dim:
            self.fc.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        out = self.layer_norm(out)
        logits = self.fc(out)
        return logits, hidden

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_dataloaders(
    sequences: List[List[int]], cfg: Config
) -> Tuple[DataLoader, DataLoader]:
    dataset = MusicDataset(sequences)
    val_size = max(1, int(len(dataset) * cfg.train.val_split))
    trn_size = len(dataset) - val_size

    trn_ds, val_ds = random_split(
        dataset,
        [trn_size, val_size],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    pin = torch.cuda.is_available()
    trn_dl = DataLoader(
        trn_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
    )
    logger.info(
        "DataLoaders ready - train: %s samples  val: %s samples  batch_size: %s",
        f"{len(trn_ds):,}",
        f"{len(val_ds):,}",
        cfg.train.batch_size,
    )
    return trn_dl, val_dl


def build_optimizer_and_scheduler(model: nn.Module, cfg: Config):
    tc = cfg.train
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )

    if tc.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tc.epochs,
            eta_min=tc.learning_rate * 0.01,
        )
    elif tc.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=tc.lr_step_size,
            gamma=tc.lr_gamma,
        )
    else:
        scheduler = None

    return optimizer, scheduler


def run_epoch(
    model: MusicLSTM,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    scaler: Optional[GradScaler],
    cfg: Config,
    training: bool = True,
) -> float:
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        use_amp = training and scaler is not None and device.type == "cuda"

        with autocast(enabled=use_amp):
            logits, _ = model(x)
            bsz, tsz, voc = logits.shape
            loss = criterion(logits.reshape(bsz * tsz, voc), y.reshape(bsz * tsz))

        if training:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

        n_tokens = bsz * tsz
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if training and (step + 1) % max(1, len(loader) // 5) == 0:
            bar = progress_bar(step + 1, len(loader))
            logger.info("  %s  loss=%.4f", bar, loss.item())

    return total_loss / max(total_tokens, 1)


def train(mode: str, cli_overrides: argparse.Namespace) -> None:
    cfg = get_config(mode)
    device = get_device()
    set_seed(cfg.train.seed)

    if cli_overrides.epochs:
        cfg.train.epochs = cli_overrides.epochs
    if cli_overrides.batch_size:
        cfg.train.batch_size = cli_overrides.batch_size
    if cli_overrides.lr:
        cfg.train.learning_rate = cli_overrides.lr

    if not os.path.exists(cfg.processed_path) or not os.path.exists(cfg.vocab_path):
        logger.info("Preprocessed data not found - running preprocessing first ...")
        from preprocess import run_preprocessing

        run_preprocessing(mode)

    vocab, _ = load_vocab(cfg.vocab_path)
    sequences = load_sequences(cfg.processed_path)
    vocab_size = len(vocab)

    logger.info("Vocab size: %s  |  Sequences: %s", vocab_size, f"{len(sequences):,}")

    trn_dl, val_dl = build_dataloaders(sequences, cfg)
    model = MusicLSTM(vocab_size, cfg).to(device)
    logger.info(
        "Model: MusicLSTM  |  Params: %s  |  hidden=%s  layers=%s",
        f"{model.num_parameters:,}",
        cfg.model.hidden_size,
        cfg.model.num_layers,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.get("<PAD>", 0))
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    scaler = GradScaler() if cfg.train.use_amp and device.type == "cuda" else None
    if scaler:
        logger.info("AMP (FP16) enabled")

    start_epoch = 1
    best_val_loss = float("inf")
    patience_ctr = 0

    if cli_overrides.resume and os.path.exists(cfg.best_checkpoint):
        ckpt = load_checkpoint(cfg.best_checkpoint, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info(
            "Resumed from epoch %s (val_loss=%.4f)", start_epoch - 1, best_val_loss
        )

    logger.info("=" * 60)
    logger.info("Training mode=%s epochs=%s device=%s", mode, cfg.train.epochs, device)
    logger.info("=" * 60)

    history = {"trn_loss": [], "val_loss": [], "lr": []}

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "\\n-- Epoch %s/%s  lr=%.2e --", epoch, cfg.train.epochs, current_lr
        )

        with Timer() as t:
            trn_loss = run_epoch(
                model,
                trn_dl,
                optimizer,
                criterion,
                device,
                scaler,
                cfg,
                training=True,
            )
        trn_ppl = math.exp(min(trn_loss, 20))
        logger.info("  Train  loss=%.4f  ppl=%.1f  [%s]", trn_loss, trn_ppl, t)

        with Timer() as t:
            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_dl,
                    None,
                    criterion,
                    device,
                    None,
                    cfg,
                    training=False,
                )
        val_ppl = math.exp(min(val_loss, 20))
        logger.info("  Val    loss=%.4f  ppl=%.1f  [%s]", val_loss, val_ppl, t)

        if scheduler is not None:
            scheduler.step()

        history["trn_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1

        ckpt_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "trn_loss": trn_loss,
            "vocab_size": vocab_size,
            "mode": mode,
        }

        if cfg.train.save_every > 0 and epoch % cfg.train.save_every == 0:
            epoch_path = os.path.join(cfg.checkpoint_dir, f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_state, epoch_path, is_best=False)

        if is_best:
            save_checkpoint(ckpt_state, cfg.best_checkpoint, is_best=False)
            logger.info("  New best val_loss=%.4f", best_val_loss)

        if (
            cfg.train.early_stop_patience > 0
            and patience_ctr >= cfg.train.early_stop_patience
        ):
            logger.info(
                "Early stopping triggered - no improvement for %s epochs.",
                cfg.train.early_stop_patience,
            )
            break

    import json

    hist_path = os.path.join(cfg.checkpoint_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training complete. Best val_loss = %.4f", best_val_loss)
    logger.info("Best model saved to: %s", cfg.best_checkpoint)
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Music LSTM on MIDI data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        "--dataset",
        "-m",
        dest="mode",
        choices=["hiphop", "retro", "mixed"],
        default="mixed",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size", "--batch", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from best checkpoint if available"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.mode, args)
