# Contributing to MusicGenAI

## Setup

```bash
git clone https://github.com/shreeyy18-git/codealpha_tasks.git
cd codealpha_tasks/CodeAlpha-MusicGenAI
python3.10 -m pip install -r requirements.txt
pip install ruff pytest
```

Add your MIDI files to `data/MIDI/hiphop/`, `data/MIDI/retro/`, or `data/MIDI/mixed/`.

## Running the App

```bash
cd app
python3.10 app.py --run web --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

## Before Opening a PR

```bash
python3.10 -m ruff check src/ app/
python3.10 -m ruff format --check src/ app/
```

Both must pass with zero errors. To auto-fix:

```bash
python3.10 -m ruff format src/ app/
python3.10 -m ruff check --fix src/ app/
```

## Rules

- `src/` is ML only. `app/` is web only. Never mix them.
- Do not modify `preprocess.py` or training data to fix generation quality — fixes go in `generate.py` only.
- Do not add `ruff` or `pytest` to `requirements.txt`.
- If you add or change a field in `web_server.py` (`GenerateOptions` or `TrainOptions`), update the matching HTML input in `app/web/index.html` in the same PR.
- Python 3.10 only. No 3.11+ syntax.

## Git Workflow

- Branch from `main`: `git checkout -b your-feature-name`
- Keep commits focused and descriptive.
- Open a PR against `main` with a clear description of what changed and why.

## AI-Assisted Development

This project includes a Copilot agent with project-specific rules baked in.

If you use GitHub Copilot, the agent at `.github/agents/copilot.agent.md` will
automatically apply the correct coding standards, linting rules, and architecture
constraints for this project. Use it — it will save you a lot of back-and-forth.
