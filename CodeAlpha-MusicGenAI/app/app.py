"""
app/app.py
Interactive command-line interface for the Music Generation project.
"""

import os
import sys
import glob
import argparse
import subprocess

APP_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

from config import get_config, MODES, midi_dirs_for_mode
from utils import get_logger

logger = get_logger("App")


def apply_compatibility_wrapper() -> None:
    """Translate legacy app invocations into the new --run format when possible."""
    args = sys.argv[1:]

    if not args:
        return

    if any(a in ("-h", "--help") for a in args):
        return

    has_run = any(a == "--run" or a.startswith("--run=") for a in args)
    if has_run:
        return

    has_mode = any(a == "--mode" or a.startswith("--mode=") for a in args)
    if has_mode:
        print("[compat] Legacy command detected. Using '--run full_pipeline'.")
        sys.argv.insert(1, "--run")
        sys.argv.insert(2, "full_pipeline")


apply_compatibility_wrapper()

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"


def c(text: str, color: str = CYAN) -> str:
    return f"{color}{text}{RESET}"


def check_status(mode: str) -> dict:
    cfg = get_config(mode)
    midi_dirs = midi_dirs_for_mode(mode)

    midi_files = []
    for d in midi_dirs:
        midi_files.extend(glob.glob(os.path.join(d, "*.mid")))
        midi_files.extend(glob.glob(os.path.join(d, "*.midi")))

    return {
        "midi_count": len(midi_files),
        "preprocessed": os.path.exists(cfg.processed_path)
        and os.path.exists(cfg.vocab_path),
        "trained": os.path.exists(cfg.best_checkpoint),
        "processed_path": cfg.processed_path,
        "vocab_path": cfg.vocab_path,
        "checkpoint": cfg.best_checkpoint,
    }


def print_status(mode: str) -> None:
    st = check_status(mode)
    ok = c("OK", GREEN)
    no = c("NO", RED)

    print(f"\n{c('Project Status', BOLD)} [mode={c(mode, YELLOW)}]")
    print(f"  MIDI files    : {c(str(st['midi_count']))} found")
    print(f"  Preprocessed  : {ok if st['preprocessed'] else no}")
    print(f"  Model trained : {ok if st['trained'] else no}")
    if st["trained"]:
        size_mb = os.path.getsize(st["checkpoint"]) / 1e6
        print(f"  Checkpoint    : {st['checkpoint']} ({size_mb:.1f} MB)")
    print()


def run_step(script: str, extra_args=None) -> int:
    cmd = [sys.executable, script] + (extra_args or [])
    print(c(f"\n> {' '.join(cmd)}\n", CYAN))
    result = subprocess.run(cmd)
    return result.returncode


def step_preprocess(mode: str) -> None:
    script = os.path.join(SRC_DIR, "preprocess.py")
    rc = run_step(script, ["--mode", mode])
    print(
        c("  Preprocessing complete.", GREEN)
        if rc == 0
        else c("  Preprocessing failed!", RED)
    )


def step_train(mode: str, extra=None) -> None:
    script = os.path.join(SRC_DIR, "train.py")
    args = ["--mode", mode] + (extra or [])
    rc = run_step(script, args)
    print(c("  Training complete.", GREEN) if rc == 0 else c("  Training failed!", RED))


def step_generate(mode: str, extra=None) -> None:
    script = os.path.join(SRC_DIR, "generate.py")
    args = ["--mode", mode] + (extra or [])
    rc = run_step(script, args)
    print(
        c("  Generation complete.", GREEN)
        if rc == 0
        else c("  Generation failed!", RED)
    )


def step_web(host: str, port: int, reload_server: bool = False) -> None:
    script = os.path.join(APP_DIR, "web_server.py")
    args = ["--host", host, "--port", str(port)]
    if reload_server:
        args.append("--reload")
    run_step(script, args)


BANNER = f"""
{c("=======================================", CYAN)}
{c("      AI Music Generator CLI", BOLD)}
{c("      hiphop | retro | mixed", CYAN)}
{c("=======================================", CYAN)}
"""

MENU = f"""
{c("Select an option:", BOLD)}
  {c("1", YELLOW)} - Preprocess MIDI data
  {c("2", YELLOW)} - Train model
  {c("3", YELLOW)} - Generate music
  {c("4", YELLOW)} - Full pipeline (preprocess -> train -> generate)
  {c("5", YELLOW)} - Change mode (current: {{mode}})
  {c("6", YELLOW)} - Show status
    {c("7", YELLOW)} - Launch web dashboard
  {c("q", YELLOW)} - Quit
"""


def prompt_mode() -> str:
    print(f"\n{c('Available modes:', BOLD)} hiphop | retro | mixed")
    while True:
        choice = input("  Enter mode: ").strip().lower()
        if choice in MODES:
            return choice
        print(c("  Invalid choice.", RED))


def prompt_generate_options() -> list:
    opts = []
    try:
        tokens = input(f"  Tokens to generate [{c('256', YELLOW)}]: ").strip()
        if tokens:
            opts += ["--num_tokens", tokens]

        temp = input(f"  Temperature [{c('1.0', YELLOW)}]: ").strip()
        if temp:
            opts += ["--temperature", temp]

        top_k = input(f"  Top-k [{c('10', YELLOW)}]: ").strip()
        if top_k:
            opts += ["--top_k", top_k]

        top_p = input(f"  Top-p [{c('0.9', YELLOW)}]: ").strip()
        if top_p:
            opts += ["--top_p", top_p]

        bpm = input(f"  BPM [{c('90', YELLOW)}]: ").strip()
        if bpm:
            opts += ["--bpm", bpm]

        seed = input(f"  Seed MIDI file [{c('none', YELLOW)}]: ").strip()
        if seed and os.path.exists(seed):
            opts += ["--seed_file", seed]
        elif seed:
            print(c("  File not found - using random seed.", YELLOW))
    except (EOFError, KeyboardInterrupt):
        pass
    return opts


def prompt_train_options() -> list:
    opts = []
    try:
        resume = (
            input(f"  Resume from checkpoint? [{c('n', YELLOW)}]: ").strip().lower()
        )
        if resume in ("y", "yes"):
            opts += ["--resume"]

        epochs = input(
            f"  Override epochs? [{c('blank = config default', YELLOW)}]: "
        ).strip()
        if epochs:
            opts += ["--epochs", epochs]

        batch_size = input(
            f"  Override batch size? [{c('blank = config default', YELLOW)}]: "
        ).strip()
        if batch_size:
            opts += ["--batch_size", batch_size]

        lr = input(
            f"  Override learning rate? [{c('blank = config default', YELLOW)}]: "
        ).strip()
        if lr:
            opts += ["--lr", lr]
    except (EOFError, KeyboardInterrupt):
        pass
    return opts


def run_menu(mode: str) -> None:
    print(BANNER)

    while True:
        print_status(mode)
        print(MENU.format(mode=c(mode, YELLOW)))

        try:
            choice = input(c("  -> ", CYAN)).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if choice == "1":
            step_preprocess(mode)
        elif choice == "2":
            extra = prompt_train_options()
            step_train(mode, extra)
        elif choice == "3":
            if not check_status(mode)["trained"]:
                print(c("  No trained model found. Train first.", RED))
            else:
                extra = prompt_generate_options()
                step_generate(mode, extra)
        elif choice == "4":
            print(c("\nFull pipeline", BOLD))
            step_preprocess(mode)
            step_train(mode)
            step_generate(mode)
        elif choice == "5":
            mode = prompt_mode()
            print(c(f"  Mode changed to: {mode}", GREEN))
        elif choice == "6":
            print_status(mode)
        elif choice == "7":
            print(c("  Starting web dashboard on http://127.0.0.1:8000", GREEN))
            step_web("127.0.0.1", 8000)
        elif choice in ("q", "quit", "exit"):
            print("Bye!")
            break
        else:
            print(c("  Unknown option.", RED))


def run_noninteractive(
    mode: str, run: str, host: str, port: int, web_reload: bool
) -> None:
    steps = {
        "preprocess": step_preprocess,
        "train": step_train,
        "generate": step_generate,
        "full_pipeline": None,
        "web": None,
    }
    if run not in steps:
        print(
            c(
                f"Unknown --run value '{run}'. Use: preprocess | train | generate | full_pipeline | web",
                RED,
            )
        )
        sys.exit(1)
    if run == "full_pipeline":
        step_preprocess(mode)
        step_train(mode)
        step_generate(mode)
        return
    if run == "web":
        step_web(host, port, reload_server=web_reload)
        return
    steps[run](mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Music Generation App - interactive CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES),
        default="mixed",
        help="Dataset mode to start with",
    )
    parser.add_argument(
        "--run",
        choices=["preprocess", "train", "generate", "full_pipeline", "web"],
        default=None,
        help="Run a specific step without the interactive menu",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for --run web")
    parser.add_argument("--port", type=int, default=8000, help="Port for --run web")
    parser.add_argument(
        "--web_reload", action="store_true", help="Enable auto-reload for --run web"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run:
        run_noninteractive(args.mode, args.run, args.host, args.port, args.web_reload)
    else:
        run_menu(args.mode)
