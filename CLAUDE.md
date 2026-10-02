# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small PyTorch + Tkinter app that recognizes handwritten digits (0-9). A CNN (`model.py`) is trained on MNIST (`train.py`) and a Tkinter canvas GUI (`predict_gui.py`) lets the user draw a digit with the mouse and get a live prediction. There's also a browser version (`index.html` / `app.js`) that runs the same model client-side via ONNX Runtime Web, served through GitHub Pages at https://littlestarkr.github.io/handwritten-digit-recognizer/.

## Commands

```bash
pip install -r requirements.txt   # torch, torchvision, pillow, numpy

python predict_gui.py             # run the drawing GUI (uses the committed digit_cnn.pth)
python train.py                   # retrain from scratch; downloads MNIST into ./data and overwrites digit_cnn.pth
```

On Windows, `손글씨숫자인식_실행.bat` double-click-launches `predict_gui.py` (auto-detects `py` vs `python` on PATH, so it's portable — this is the one tracked in git). `손글씨숫자인식_로컬실행.bat` is a machine-specific copy with a hardcoded interpreter path and is gitignored; regenerate it locally rather than committing it.

There is no test suite or linter configured in this repo.

## Architecture

- `model.py` — `DigitCNN`: 2x (Conv2d -> ReLU -> MaxPool) then 2 FC layers, expects a `(N, 1, 28, 28)` float tensor normalized with MNIST's mean/std (`0.1307`, `0.3081`). Both `train.py` and `predict_gui.py` import this class and must stay in sync with its input shape/normalization.
- `train.py` — downloads MNIST into `./data` (gitignored), trains for `EPOCHS=5`, prints per-epoch loss/test-accuracy, and saves weights to `digit_cnn.pth`.
- `predict_gui.py` — `DigitRecognizerApp`: a 280x280 Tkinter `Canvas` (10x the model's 28x28 input) that the user draws on. Every mouse-drag stroke is drawn twice in parallel: once visually on the `Canvas`, once on an offscreen `PIL.Image` (`self.image`) that mirrors it pixel-for-pixel for inference. `_preprocess()` crops that offscreen image to the drawn content's bounding box, pads it into a square with a margin, then resizes to 28x28 and applies the same MNIST normalization used in `train.py` — this centering/cropping step is what makes recognition robust to where/how large the user draws; a naive full-canvas resize produces poor accuracy on digits drawn small or off-center.
- `digit_cnn.pth` — pretrained weights are committed directly so the GUI runs immediately after cloning, without requiring a `train.py` run.
- `index.html` / `app.js` — browser port of `predict_gui.py`. Draws into an HTML `<canvas>`, replicates the exact same bounding-box crop/pad/center/normalize steps as `_preprocess()` in JS, and runs inference with `digit_cnn.onnx` via onnxruntime-web (loaded from CDN in `index.html`). GitHub Pages serves this repo's root directly (`master` branch, `/` path), so `index.html` is what visitors see at the Pages URL. `digit_cnn.onnx` is exported from `digit_cnn.pth` via `torch.onnx.export(..., dynamo=False)` (the newer dynamo-based exporter needs `onnxscript`, which isn't a project dependency) — re-export it whenever `digit_cnn.pth` is retrained.

## Conventions

- All code comments are written in English, including inline comments explaining non-obvious logic (e.g. normalization constants, the crop/pad/center preprocessing steps).
