// Mirrors the preprocessing used in predict_gui.py / train.py so the
// in-browser model sees the same kind of input it was trained on.
const CANVAS_SIZE = 280;
const IMAGE_SIZE = 28;
const BRUSH_WIDTH = 16;
const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const resultEl = document.getElementById("result");
const barsEl = document.getElementById("bars");
const statusEl = document.getElementById("status");
const clearBtn = document.getElementById("clearBtn");

let session = null;
let drawing = false;
let lastX = null;
let lastY = null;
let hasDrawn = false;

function resetCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.strokeStyle = "white";
  ctx.lineWidth = BRUSH_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}
resetCanvas();

function canvasPoint(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = CANVAS_SIZE / rect.width;
  const scaleY = CANVAS_SIZE / rect.height;
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  const clientY = e.touches ? e.touches[0].clientY : e.clientY;
  return [(clientX - rect.left) * scaleX, (clientY - rect.top) * scaleY];
}

function startStroke(e) {
  e.preventDefault();
  drawing = true;
  hasDrawn = true;
  [lastX, lastY] = canvasPoint(e);
}

function moveStroke(e) {
  if (!drawing) return;
  e.preventDefault();
  const [x, y] = canvasPoint(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
}

function endStroke(e) {
  if (!drawing) return;
  drawing = false;
  if (hasDrawn) predict();
}

canvas.addEventListener("mousedown", startStroke);
canvas.addEventListener("mousemove", moveStroke);
window.addEventListener("mouseup", endStroke);

canvas.addEventListener("touchstart", startStroke, { passive: false });
canvas.addEventListener("touchmove", moveStroke, { passive: false });
canvas.addEventListener("touchend", endStroke);

clearBtn.addEventListener("click", () => {
  resetCanvas();
  hasDrawn = false;
  resultEl.innerHTML = "Draw a digit";
  barsEl.innerHTML = "";
});

function findBoundingBox(imageData, size) {
  let minX = size, minY = size, maxX = -1, maxY = -1;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      // Canvas is grayscale-on-black; use the red channel as brightness.
      if (imageData.data[idx] > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;
  return { minX, minY, maxX, maxY };
}

function preprocess() {
  const imageData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const bbox = findBoundingBox(imageData, CANVAS_SIZE);

  // Offscreen canvas holding the cropped-and-padded square before final resize
  const squareCanvas = document.createElement("canvas");
  const squareCtx = squareCanvas.getContext("2d");

  if (!bbox) {
    squareCanvas.width = IMAGE_SIZE;
    squareCanvas.height = IMAGE_SIZE;
  } else {
    const w = bbox.maxX - bbox.minX + 1;
    const h = bbox.maxY - bbox.minY + 1;
    const side = Math.max(w, h);
    const margin = Math.round(side * 0.4);
    const squareSize = side + margin * 2;

    squareCanvas.width = squareSize;
    squareCanvas.height = squareSize;
    squareCtx.fillStyle = "black";
    squareCtx.fillRect(0, 0, squareSize, squareSize);
    squareCtx.drawImage(
      canvas,
      bbox.minX, bbox.minY, w, h,
      (squareSize - w) / 2, (squareSize - h) / 2, w, h
    );
  }

  // Final downscale to the 28x28 size the model expects
  const finalCanvas = document.createElement("canvas");
  finalCanvas.width = IMAGE_SIZE;
  finalCanvas.height = IMAGE_SIZE;
  const finalCtx = finalCanvas.getContext("2d");
  finalCtx.imageSmoothingEnabled = true;
  finalCtx.imageSmoothingQuality = "high";
  finalCtx.drawImage(squareCanvas, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const data = finalCtx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
  const float32 = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    const gray = data[i * 4] / 255; // red channel, already grayscale
    float32[i] = (gray - MNIST_MEAN) / MNIST_STD;
  }
  return float32;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function renderBars(probs, topIdx) {
  barsEl.innerHTML = "";
  probs.forEach((p, digit) => {
    const bar = document.createElement("div");
    bar.className = "bar" + (digit === topIdx ? " top" : "");
    bar.style.height = Math.max(2, p * 48) + "px";
    const label = document.createElement("span");
    label.textContent = digit;
    bar.appendChild(label);
    barsEl.appendChild(bar);
  });
}

async function predict() {
  if (!session) return;
  const input = preprocess();
  const tensor = new ort.Tensor("float32", input, [1, 1, IMAGE_SIZE, IMAGE_SIZE]);
  const output = await session.run({ input: tensor });
  const logits = Array.from(output.output.data);
  const probs = softmax(logits);
  const topIdx = probs.indexOf(Math.max(...probs));

  resultEl.innerHTML = `${topIdx} <span class="conf">(${(probs[topIdx] * 100).toFixed(1)}%)</span>`;
  renderBars(probs, topIdx);
}

async function init() {
  try {
    session = await ort.InferenceSession.create("digit_cnn.onnx");
    statusEl.textContent = "Model ready";
  } catch (err) {
    statusEl.textContent = "Failed to load model: " + err.message;
    console.error(err);
  }
}

init();
