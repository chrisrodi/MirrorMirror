# Mirror Mirror

A macOS overlay app: drag a glass pane over part of your screen and Mirror Mirror reads the text underneath (OCR) and shows an answer from an OpenAI model inside the pane.

## Run from source

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python mirror_mirror.py
```

On first launch you'll be asked for an OpenAI API key (or copy `.env.example` to `.env` and fill it in). macOS will ask for Screen Recording permission.

## Build a DMG

```bash
./build_release.sh
```

Outputs `release/MirrorMirror-<version>.dmg`. Run `./build_release.sh --help` for signing and notarization options.
