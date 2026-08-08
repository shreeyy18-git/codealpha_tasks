# MusicGenAI

<img width="1921" height="1159" alt="MusicGenAI Dashboard" src="https://github.com/user-attachments/assets/feaed821-ae08-47b4-a659-373441e5a6a0" />

[![CI](https://github.com/shreeyy18-git/codealpha_tasks/actions/workflows/ci.yml/badge.svg)](https://github.com/shreeyy18-git/codealpha_tasks/actions/workflows/ci.yml)

I built MusicGenAI to train an LSTM on MIDI files and generate original music from a web dashboard. It supports hip-hop, retro game, and mixed datasets out of the box.

## Requirements

- Python 3.10
- A folder of `.mid` files

## Quickstart

```bash
git clone https://github.com/shreeyy18-git/codealpha_tasks.git
cd codealpha_tasks/CodeAlpha-MusicGenAI
pip install -r requirements.txt
```

Drop your MIDI files into `data/MIDI/hiphop/`, `data/MIDI/retro/`, or `data/MIDI/mixed/`, then:

```bash
cd app
python3.10 app.py --run web
```

Open `http://127.0.0.1:8000` and hit **Full Pipeline** to preprocess, train, and generate in one shot.

## What it does

- **Preprocess** — parses MIDI files into token sequences with optional transposition augmentation
- **Train** — fits a multi-layer LSTM language model on your sequences
- **Generate** — samples new MIDI from the trained model with controls for temperature, BPM, top-k/p, and seed file
- **Dashboard** — live job status, streaming logs, and a built-in MIDI player for generated outputs

## CLI

If you prefer the terminal over the web UI:

```bash
python3.10 app.py --run preprocess --mode hiphop
python3.10 app.py --run train --mode hiphop
python3.10 app.py --run generate --mode hiphop
```

## Author

**Shreeyansh**
GitHub: [@shreeyy18-git](https://github.com/shreeyy18-git)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
