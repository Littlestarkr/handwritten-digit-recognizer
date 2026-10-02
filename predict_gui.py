import tkinter as tk
from tkinter import messagebox

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import DigitCNN

MODEL_PATH = "digit_cnn.pth"
CANVAS_SIZE = 280          # on-screen drawing area (28 * 10 for easier drawing)
IMAGE_SIZE = 28            # size expected by the model (MNIST)
BRUSH_RADIUS = 8


class DigitRecognizerApp:
    """Tkinter app where the user draws a digit and a CNN predicts it."""

    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

        # Canvas for the user to draw on (visual, black background like MNIST)
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._reset_stroke)

        # Offscreen PIL image mirrors the canvas pixels so we can feed it to the model
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.result_label = tk.Label(root, text="Draw a digit (0-9)", font=("Arial", 20))
        self.result_label.grid(row=1, column=0, columnspan=3, pady=5)

        predict_btn = tk.Button(root, text="Predict", command=self._predict, width=10)
        predict_btn.grid(row=2, column=0, pady=10)

        clear_btn = tk.Button(root, text="Clear", command=self._clear, width=10)
        clear_btn.grid(row=2, column=1, pady=10)

    def _load_model(self):
        model = DigitCNN().to(self.device)
        try:
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
        except FileNotFoundError:
            messagebox.showerror(
                "Model not found",
                f"Could not find '{MODEL_PATH}'. Run train.py first to train and save the model.",
            )
            raise SystemExit(1)
        model.eval()
        return model

    def _paint(self, event):
        x, y = event.x, event.y
        if self.last_x is not None:
            # Draw on the visible canvas
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=BRUSH_RADIUS * 2, fill="white", capstyle=tk.ROUND, smooth=True,
            )
            # Mirror the same stroke on the offscreen image used for prediction
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=BRUSH_RADIUS * 2,
            )
        self.last_x, self.last_y = x, y

    def _reset_stroke(self, event):
        self.last_x, self.last_y = None, None

    def _clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="Draw a digit (0-9)")

    def _preprocess(self):
        # Crop to the drawn content and center it like MNIST digits, instead of
        # resizing the whole (mostly empty) canvas, which distorts the shape.
        bbox = self.image.getbbox()
        if bbox is None:
            img = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        else:
            cropped = self.image.crop(bbox)
            w, h = cropped.size

            # Pad into a square with margin so the digit isn't stretched and
            # keeps breathing room, matching MNIST's centered digit layout.
            side = max(w, h)
            margin = int(side * 0.4)
            square_size = side + margin * 2
            square = Image.new("L", (square_size, square_size), color=0)
            square.paste(cropped, ((square_size - w) // 2, (square_size - h) // 2))

            img = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

        tensor = torch.tensor(list(img.getdata()), dtype=torch.float32)
        tensor = tensor.view(1, 1, IMAGE_SIZE, IMAGE_SIZE) / 255.0

        # Apply the same normalization used during training
        tensor = (tensor - 0.1307) / 0.3081
        return tensor.to(self.device)

    def _predict(self):
        tensor = self._preprocess()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)

        digit = predicted.item()
        confidence_pct = confidence.item() * 100
        self.result_label.config(text=f"Prediction: {digit}  ({confidence_pct:.1f}%)")


def main():
    root = tk.Tk()
    DigitRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
