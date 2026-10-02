# Handwritten Digit Recognizer

A CNN trained on MNIST that recognizes handwritten digits (0-9) drawn with the mouse.

**[Try it in your browser](https://littlestarkr.github.io/handwritten-digit-recognizer/)** - no install needed, runs the model client-side with onnxruntime-web.

![Screenshot](screenshot.png)

## Files

- `model.py` - CNN architecture
- `train.py` - trains the model on MNIST and saves `digit_cnn.pth`
- `predict_gui.py` - Tkinter desktop app: draw a digit, click Predict
- `digit_cnn.pth` - pretrained weights (test accuracy ~99.1%)
- `index.html` / `app.js` / `digit_cnn.onnx` - browser version of the same app, served via GitHub Pages

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python predict_gui.py
```

On Windows you can also just double-click `손글씨숫자인식_실행.bat`.

## Retrain (optional)

The pretrained `digit_cnn.pth` is included, so this step isn't required. To retrain from scratch:

```bash
python train.py
```
