// zone_calibrator — a minimal static Streamlit component.
// Draws the video frame on a canvas, lets you drag out a rectangle with a
// LIVE preview while dragging (unlike streamlit-image-coordinates, which
// only reports the result on mouse-up), and sends the finished rectangle
// back to Python via Streamlit.setComponentValue.

function sendValue(value) {
  Streamlit.setComponentValue(value);
}

let img = new Image();
let canvas, ctx;
let dragging = false;
let startX = 0, startY = 0;
let curX = 0, curY = 0;
let existingZones = [];   // list of zones, each a list of [x_norm, y_norm]
let imgLoaded = false;

function drawAll() {
  if (!imgLoaded) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  // Already-saved zones (solid indigo)
  ctx.strokeStyle = "#6366f1";
  ctx.lineWidth = 3;
  existingZones.forEach(function (zone, zi) {
    ctx.beginPath();
    zone.forEach(function (pt, i) {
      const x = pt[0] * canvas.width;
      const y = pt[1] * canvas.height;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();

    // label
    const cx = zone.reduce((s, p) => s + p[0], 0) / zone.length * canvas.width;
    const cy = zone.reduce((s, p) => s + p[1], 0) / zone.length * canvas.height;
    ctx.fillStyle = "#6366f1";
    ctx.font = "16px monospace";
    ctx.fillText("Zone " + zi, cx - 25, cy);
  });

  // Live in-progress rectangle (dashed amber) — this is the part that
  // actually updates while the mouse is still moving, mid-drag.
  if (dragging) {
    ctx.strokeStyle = "#f59e0b";
    ctx.fillStyle = "rgba(245, 158, 11, 0.12)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    const rx = Math.min(startX, curX);
    const ry = Math.min(startY, curY);
    const rw = Math.abs(curX - startX);
    const rh = Math.abs(curY - startY);
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
  }
}

function getCanvasPos(e) {
  const rect = canvas.getBoundingClientRect();
  // canvas.width/height are the true pixel dimensions; rect is the
  // possibly-CSS-scaled displayed size, so rescale click position into
  // true canvas pixel space right here rather than downstream in Python.
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function onMouseDown(e) {
  const pos = getCanvasPos(e);
  startX = pos.x;
  startY = pos.y;
  curX = startX;
  curY = startY;
  dragging = true;
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
}

function onMouseMove(e) {
  if (!dragging) return;
  const pos = getCanvasPos(e);
  curX = pos.x;
  curY = pos.y;
  drawAll();   // redraw on every mouse move — this is the live preview
}

function onMouseUp(e) {
  if (!dragging) return;
  const pos = getCanvasPos(e);
  curX = pos.x;
  curY = pos.y;
  dragging = false;
  window.removeEventListener("mousemove", onMouseMove);
  window.removeEventListener("mouseup", onMouseUp);
  drawAll();

  const rw = Math.abs(curX - startX);
  const rh = Math.abs(curY - startY);
  if (rw > 5 && rh > 5) {
    sendValue({
      x1: startX, y1: startY, x2: curX, y2: curY,
      canvas_width: canvas.width, canvas_height: canvas.height,
      unix_time: Date.now(),
    });
  }
}

function onRender(event) {
  const { src, zones, canvas_width, canvas_height } = event.detail.args;
  existingZones = zones || [];

  if (!canvas) {
    canvas = document.getElementById("calib-canvas");
    ctx = canvas.getContext("2d");
    canvas.addEventListener("mousedown", onMouseDown);
  }

  canvas.width = canvas_width;
  canvas.height = canvas_height;
  canvas.style.width = "100%";
  canvas.style.height = "auto";

  if (img.src !== src) {
    imgLoaded = false;
    img.onload = function () {
      imgLoaded = true;
      drawAll();
      Streamlit.setFrameHeight(canvas.getBoundingClientRect().height);
    };
    img.src = src;
  } else {
    drawAll();
    Streamlit.setFrameHeight(canvas.getBoundingClientRect().height);
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();