# Mirror Mirror

A macOS overlay app: drag a glass pane over part of your screen and Mirror Mirror reads the text underneath (OCR) and shows an answer from an OpenAI model inside the pane.

## Why I made this
As I saw the use of AI grow combined with my childhood of watching Shrek; I created a program that allows you to ask questions to anything you hover over and get a response back quickly all in one spot.

## Run from source

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python mirror_mirror.py
```

On first launch you'll be asked for an OpenAI API key, macOS will then ask for Screen Recording permission. Give permission and restart the app.

## Build a DMG

```bash
./build_release.sh
```

Outputs `release/MirrorMirror-<version>.dmg`.

## License

MIT — see [LICENSE](LICENSE). The built app bundles [PyQt6](https://www.riverbankcomputing.com/software/pyqt/), which is licensed under GPL v3.
