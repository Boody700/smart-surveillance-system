# zone_calibrator/__init__.py
# Minimal custom Streamlit component: drag a rectangle over a video frame,
# see it drawn LIVE while dragging, get the finished rectangle back in
# Python. No build step / npm required — this is a static HTML+JS component,
# using Streamlit's own official component-glue file (streamlit-component-lib.js).

import os
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "zone_calibrator",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
)


def zone_calibrator(image_rgb, zones_norm=None, key=None):
    """
    image_rgb   : HxWx3 numpy array, RGB order (e.g. from cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    zones_norm  : list of already-saved zones to draw, each a list of [x_norm, y_norm]
                  points in 0-1 range (same format as zones.json)
    key         : Streamlit widget key

    Returns a dict once a drag completes:
        {x1, y1, x2, y2, canvas_width, canvas_height, unix_time}
    All coordinates are already in the ORIGINAL image's pixel space — no
    rescaling needed on the Python side, unlike streamlit-image-coordinates.
    Returns None if no drag has completed yet (or after a fresh rerun with
    no new event).
    """
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image_rgb must be an HxWx3 numpy array (RGB)")

    h, w = image_rgb.shape[:2]
    pil_img = Image.fromarray(image_rgb)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    src = "data:image/jpeg;base64," + b64

    return _component_func(
        src=src,
        zones=zones_norm or [],
        canvas_width=int(w),
        canvas_height=int(h),
        key=key,
        default=None,
    )