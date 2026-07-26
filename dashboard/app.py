# app.py — Smart Surveillance System
# Run with: streamlit run app.py

import streamlit as st
import sqlite3
import subprocess
import sys
import os
import time
import json
import numpy as np
import cv2
from pathlib import Path

try:
    from zone_calibrator import zone_calibrator
    HAS_IMG_COORD = True
except ImportError:
    HAS_IMG_COORD = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import DATABASE_PATH, VIDEO_PATH

LIVE_FRAME_PATH = os.path.join(ROOT_DIR, "live_frame.jpg")

st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

* { font-family: 'Inter', sans-serif; }
.stApp {
    background:
        repeating-linear-gradient(180deg, rgba(255,255,255,0.012) 0px, rgba(255,255,255,0.012) 1px, transparent 1px, transparent 3px),
        radial-gradient(ellipse 900px 500px at 50% -10%, rgba(79,70,229,0.10), transparent 60%),
        #05070c;
    color: #e6ecf5;
}
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* Hide streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a0f1c 0%, #151030 55%, #0a0f1c 100%);
    border: 1px solid #1c2636;
    border-radius: 14px;
    padding: 2.25rem 2rem 2rem 2rem;
    margin-bottom: 1.75rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent);
    height: 1px; top: 0;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #13182b;
    border: 1px solid #4338ca;
    color: #a5b4fc;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 1.1rem;
}
.hero-badge .dot-live {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 6px 1px rgba(74,222,128,0.8);
}
.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.6px;
    line-height: 1.15;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    color: #5b6b82;
    font-size: 0.88rem;
    margin-top: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0a0f1c;
    border: 1px solid #1c2636;
    border-radius: 10px;
    padding: 5px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #45536b;
    border-radius: 7px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding: 9px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #171633 !important;
    color: #c7d2fe !important;
    box-shadow: inset 0 0 0 1px #4338ca;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Cards */
.card {
    background: #0a0f1c;
    border: 1px solid #1c2636;
    border-left: 3px solid #4338ca;
    border-radius: 10px;
    padding: 1.4rem 1.5rem;
    height: 100%;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #45536b;
    margin-bottom: 0.4rem;
}
.card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.card-sub {
    font-size: 0.75rem;
    color: #5b6b82;
    margin-top: 0.3rem;
}

/* Terminal */
.terminal {
    background: #020408;
    border: 1px solid #1c2636;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #4ade80;
    line-height: 1.7;
    max-height: 380px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
.terminal-header {
    background: #0a0f1c;
    border: 1px solid #1c2636;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    padding: 0.55rem 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.dot-r { background: #ef4444; }
.dot-y { background: #f59e0b; }
.dot-g { background: #22c55e; }

/* Step indicator */
.step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid #1c2636;
    color: #45536b;
    font-size: 0.84rem;
}
.step:last-child { border-bottom: none; }
.step.active { color: #e6ecf5; }
.step.done { color: #4ade80; }
.step-num {
    width: 23px; height: 23px;
    border-radius: 50%;
    background: #141b2b;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step.done .step-num { background: #052e16; color: #4ade80; }
.step.active .step-num { background: #1e1b4b; color: #818cf8; box-shadow: 0 0 0 1px #4338ca; }

/* Violation cards */
.vcard {
    background: #0a0f1c;
    border: 1px solid #1c2636;
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.vcard-afk  { border-left: 3px solid #f59e0b; }
.vcard-left { border-left: 3px solid #ef4444; }
.vcard-unauth { border-left: 3px solid #8b5cf6; }

.pill {
    display: inline-block;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 12px;
}
.pill-afk   { background: #451a03; color: #fbbf24; }
.pill-left  { background: #450a0a; color: #f87171; }
.pill-unauth { background: #2e1065; color: #a78bfa; }

/* Buttons */
.stButton > button {
    background: #13182b;
    color: #a5b4fc;
    border: 1px solid #4338ca;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.875rem;
    letter-spacing: 0.01em;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #4338ca;
    color: #fff;
    border-color: #6366f1;
}
.stButton > button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

/* Upload area */
[data-testid="stFileUploader"] section {
    background: #0a0f1c;
    border: 2px dashed #1c2636;
    border-radius: 10px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #4338ca;
}

/* Divider */
hr { border-color: #1c2636; margin: 1.5rem 0; }

/* Metrics */
[data-testid="metric-container"] {
    background: #0a0f1c;
    border: 1px solid #1c2636;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #fff; font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricLabel"] { color: #45536b; }

/* Status pill */
.status-ok  { color: #4ade80; font-weight: 600; }
.status-err { color: #f87171; font-weight: 600; }
.status-run { color: #818cf8; font-weight: 600; }

/* Live preview frame — camera-viewfinder treatment */
.live-frame-wrap {
    border: 1px solid #1c2636;
    border-radius: 10px;
    overflow: hidden;
    background: #020408;
    position: relative;
    padding: 3px;
}
.live-frame-wrap::before, .live-frame-wrap::after,
.live-frame-wrap .corner-tl, .live-frame-wrap .corner-br {
    content: "";
    position: absolute;
    width: 22px; height: 22px;
    z-index: 11;
    pointer-events: none;
}
.live-frame-wrap::before {
    top: 8px; left: 8px;
    border-top: 2px solid #4f46e5;
    border-left: 2px solid #4f46e5;
    border-radius: 4px 0 0 0;
}
.live-frame-wrap::after {
    bottom: 8px; right: 8px;
    border-bottom: 2px solid #4f46e5;
    border-right: 2px solid #4f46e5;
    border-radius: 0 0 4px 0;
}
.live-frame-badge {
    position: absolute;
    top: 10px;
    left: 10px;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid #ef4444;
    color: #fca5a5;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 999px;
    z-index: 10;
}
.scanline {
    position: absolute;
    left: 0; right: 0;
    height: 40%;
    background: linear-gradient(180deg, transparent, rgba(99,102,241,0.10), transparent);
    animation: scan 3.2s linear infinite;
    pointer-events: none;
    z-index: 9;
}
@keyframes scan {
    0%   { top: -40%; }
    100% { top: 100%; }
}
@media (prefers-reduced-motion: reduce) {
    .scanline { animation: none; display: none; }
}
.progress-readout {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.85rem;
    color: #7d8ba3;
    margin: 0.6rem 0 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
}
.progress-readout b { color: #e6ecf5; }

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #4338ca, #818cf8);
}
.stProgress > div > div > div {
    background: #141b2b;
}
</style>
""", unsafe_allow_html=True)

# ── ACCESS CONTROL ────────────────────────────────────────────────────────────
AUTH_USERS = {
    "admin": "123",
    "gamal" : "Jimmy"
}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def render_login():
    st.markdown("""
    <div style="max-width:420px;margin:9vh auto 0 auto;">
      <div class="hero" style="padding:2rem 1.75rem;">
        <div class="hero-badge">
          <span class="dot-live" style="background:#f59e0b;box-shadow:0 0 6px 1px rgba(245,158,11,.8);"></span>
          ACCESS TERMINAL
        </div>
        <div class="hero-title" style="font-size:1.5rem;">Restricted <span>Access</span></div>
        <div class="hero-sub">Authorized operators only</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.3, 1])
    with mid:
        with st.form("login_form"):
            username = st.text_input("Operator ID")
            password = st.text_input("Passphrase", type="password")
            submitted = st.form_submit_button("Authenticate", use_container_width=True)
        if submitted:
            if AUTH_USERS.get(username) == password:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Access denied — check operator ID and passphrase.")

if not st.session_state["authenticated"]:
    render_login()
    st.stop()

op_l, op_r = st.columns([6, 1])
with op_l:
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
        f'color:#45536b;margin-bottom:0.5rem;">'
        f'OPERATOR &rarr; <span style="color:#a5b4fc;">{st.session_state.get("username", "?")}</span>'
        f'</div>',
        unsafe_allow_html=True
    )
with op_r:
    if st.button("Sign out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state.pop("username", None)
        st.rerun()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge"><span class="dot-live"></span> AGENTIC CV PIPELINE // v2</div>
    <div class="hero-title">Smart <span>Surveillance</span> System</div>
    <div class="hero-sub">FEED.IN &rarr; TRACK.PERSON &rarr; FLAG.VIOLATION &rarr; REPORT.OUT</div>
</div>
""", unsafe_allow_html=True)

# ── VIDEO SOURCE (persistent, shown above all tabs) ───────────────────────────
src_col1, src_col2 = st.columns([2.2, 1], gap="large")
with src_col1:
    uploaded = st.file_uploader(
        "Video source — used by both Calibrate Zones and Run Detection",
        type=["mp4", "avi", "mov", "mkv"],
    )
    if uploaded:
        vdir = os.path.join(ROOT_DIR, "videos")
        os.makedirs(vdir, exist_ok=True)
        vpath = os.path.join(vdir, uploaded.name)
        with open(vpath, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state["video_path"] = vpath

with src_col2:
    active_video = st.session_state.get("video_path", VIDEO_PATH)
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.78rem;'
        f'color:#5b6b82;margin-top:1.9rem;">'
        f'SOURCE &rarr;<br><span style="color:#a5b4fc;">{os.path.basename(active_video)}</span>'
        f'{"<br>(default — nothing uploaded yet)" if "video_path" not in st.session_state else ""}'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_stats():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        people     = conn.execute("SELECT COUNT(DISTINCT person_id) FROM events WHERE event_type='detected'").fetchone()[0]
        detections = conn.execute("SELECT COUNT(*) FROM events WHERE event_type='detected'").fetchone()[0]
        violations = conn.execute("SELECT COUNT(*) FROM events WHERE event_type LIKE 'violation_%'").fetchone()[0]
        conn.close()
        return people, detections, violations
    except:
        return 0, 0, 0

def get_violations():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        rows = conn.execute("""
            SELECT person_id, event_type, duration_seconds, timestamp, crop_path, vlm_summary
            FROM events WHERE event_type LIKE 'violation_%'
            ORDER BY timestamp
        """).fetchall()
        conn.close()
        return rows
    except:
        return []

def format_timestamp(seconds):
    """Format a point in time as M:SS (e.g. 192.4 -> '3:12'). Durations
    (how long something lasted) stay as seconds/minutes elsewhere - this
    is specifically for WHEN something happened in the video."""
    seconds = max(0, int(seconds or 0))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"

def person_label(pid):
    """person_id can be NULL (legacy rows) or 0 (the reserved sentinel) for
    a general/whole-room violation that was never tied to one tracked
    person - e.g. "everyone AFK". Neither of those is actually "Person X",
    so render them as a clear "General" label instead of the literal
    "Person None" / "Person 0" that f-string interpolation would otherwise
    produce."""
    if pid is None or pid == 0:
        return "General"
    return f"Person {pid}"

def run_agent(script_path, log_placeholder, cwd=ROOT_DIR, extra_args=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, script_path] + (extra_args or [])
    lines = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8', 
        errors='replace',
        env=env,
        cwd=cwd
    )
    for line in process.stdout:
        line = line.rstrip()
        if line:
            lines.append(line)
            display = "\n".join(lines[-60:])
            log_placeholder.markdown(
                f'<div class="terminal">{display}</div>',
                unsafe_allow_html=True
            )
    process.wait()
    return process.returncode, lines


def run_agent_with_progress(script_path, progress_bar, readout_placeholder,
                             image_placeholder, log_placeholder, cwd=ROOT_DIR,
                             extra_args=None):
    UPDATE_EVERY_FRAMES = 15

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, script_path] + (extra_args or [])
    lines = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=env,
        cwd=cwd
    )
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue

        if line.startswith("PROGRESS:"):
            try:
                _, frame_no_s, total_s = line.split(":")
                frame_no, total = int(frame_no_s), int(total_s)
                pct = min(frame_no / total, 1.0) if total else 0.0
            except (ValueError, ZeroDivisionError):
                continue

            if frame_no % UPDATE_EVERY_FRAMES == 0 or frame_no >= total:
                pct_int = int(pct * 100)
                progress_bar.progress(pct)
                readout_placeholder.markdown(
                    f'<div class="progress-readout">'
                    f'<span>Frame <b>{frame_no:,}</b> / {total:,}</span>'
                    f'<span><b>{pct_int}%</b></span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                if os.path.exists(LIVE_FRAME_PATH):
                    try:
                        with open(LIVE_FRAME_PATH, "rb") as img_f:
                            image_placeholder.image(
                                img_f.read(),
                                use_container_width=True
                            )
                    except Exception:
                        pass
        else:
            lines.append(line)
            display = "\n".join(lines[-60:])
            log_placeholder.markdown(
                f'<div class="terminal">{display}</div>',
                unsafe_allow_html=True
            )

    process.wait()
    progress_bar.progress(1.0)
    return process.returncode, lines

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab0, tab2, tab3 = st.tabs([
    "  📹  Step 1 — Run Detection  ",
    "  🗺️  Calibrate Zones  ",
    "  🚨  Step 2 — View Violations  ",
    "  📄  Step 3 — Report  "
])

# ═══════════════════════════════════════════════════════
# TAB 0 — CALIBRATE ZONES
# ═══════════════════════════════════════════════════════
with tab0:
    st.markdown("#### Calibrate desk zones")
    st.markdown(
        '<p style="color:#5b6b82;font-size:0.85rem;line-height:1.6;">'
        'Click and drag a box around each desk — you\'ll see the rectangle '
        'live as you drag, release to create the zone. Zones are saved as '
        'normalized coordinates, so they still work if the video resolution changes.</p>',
        unsafe_allow_html=True
    )

    if not HAS_IMG_COORD:
        st.error(
            "The zone_calibrator component folder wasn't found next to config.py. "
            "Make sure the zone_calibrator/ folder (with its frontend/ subfolder) "
            "sits at your project root, same level as config.py."
        )
    else:
        calib_video = st.session_state.get("video_path", VIDEO_PATH)
        if not os.path.exists(calib_video):
            st.warning("Upload a video in the box above first, or make sure the default video in config.py exists.")
        else:
            cap = cv2.VideoCapture(calib_video)
            ok, frame_bgr = cap.read()
            cap.release()

            if not ok:
                st.error("Couldn't read a frame from the selected video.")
            else:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                fh, fw = frame_rgb.shape[:2]
                zones_json_path = os.path.join(os.path.dirname(DATABASE_PATH), "zones.json")

                if "calib_zones_norm" not in st.session_state:
                    loaded = []
                    if os.path.exists(zones_json_path):
                        try:
                            with open(zones_json_path) as f:
                                data = json.load(f)
                            loaded = data.get("zones", [])
                        except Exception:
                            loaded = []
                    st.session_state["calib_zones_norm"] = loaded
                if "calib_last_drag_time" not in st.session_state:
                    st.session_state["calib_last_drag_time"] = None

                col_img, col_ctl = st.columns([2.2, 1], gap="large")

                with col_img:
                    drag_value = zone_calibrator(
                        frame_rgb,
                        zones_norm=st.session_state["calib_zones_norm"],
                        key="zone_calib_widget",
                    )

                with col_ctl:
                    n_zones = len(st.session_state["calib_zones_norm"])
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-label">Zones saved</div>
                        <div class="card-value">{n_zones}</div>
                        <div class="card-sub">drag a box on the image to add one</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if st.button("Remove last saved zone", use_container_width=True):
                        if st.session_state["calib_zones_norm"]:
                            st.session_state["calib_zones_norm"].pop()
                            st.rerun()
                    if st.button("Clear all zones", use_container_width=True):
                        st.session_state["calib_zones_norm"] = []
                        st.rerun()

                    st.markdown("<hr>", unsafe_allow_html=True)

                    if st.button("💾 Save zones.json", use_container_width=True):
                        os.makedirs(os.path.dirname(zones_json_path), exist_ok=True)
                        with open(zones_json_path, "w") as f:
                            json.dump({"zones": st.session_state["calib_zones_norm"]}, f, indent=2)
                        st.success(f"✓ Saved {n_zones} zone(s) to zones.json")

                if drag_value is not None and drag_value.get("x1") is not None:
                    drag_time = drag_value.get("unix_time")
                    if drag_time != st.session_state["calib_last_drag_time"]:
                        st.session_state["calib_last_drag_time"] = drag_time

                        cw = drag_value.get("canvas_width") or fw
                        ch = drag_value.get("canvas_height") or fh
                        x1, x2 = sorted([drag_value["x1"], drag_value["x2"]])
                        y1, y2 = sorted([drag_value["y1"], drag_value["y2"]])

                        zone_norm = [
                            [x1 / cw, y1 / ch], [x2 / cw, y1 / ch],
                            [x2 / cw, y2 / ch], [x1 / cw, y2 / ch],
                        ]
                        st.session_state["calib_zones_norm"].append(zone_norm)
                        st.rerun()

# ═══════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ═══════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        agent1_done = st.session_state.get("agent1_done", False)
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Pipeline</div>
            <div class="step {'done' if agent1_done else 'active'}">
                <span class="step-num">{'✓' if agent1_done else '1'}</span>
                <span>Agent 1 — Detect &amp; Track</span>
            </div>
            <div class="step">
                <span class="step-num">2</span>
                <span>Agent 2 — Rule Engine</span>
            </div>
            <div class="step">
                <span class="step-num">3</span>
                <span>Agent 3 — VLM Confirmation</span>
            </div>
            <div class="step">
                <span class="step-num">4</span>
                <span>Report Generation</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        people, detections, violations = get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("People", people)
        c2.metric("Detections", f"{detections:,}")
        c3.metric("Violations", violations)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Start Detection", use_container_width=True)

    with right:
        st.markdown("#### Live preview")

        preview_wrap = st.container()
        with preview_wrap:
            scan_html = '<div class="scanline"></div>' if run_btn else ''
            badge_html = '<div class="live-frame-badge">● REC</div>' if run_btn else ''
            st.markdown(f'<div class="live-frame-wrap">{badge_html}{scan_html}', unsafe_allow_html=True)
            image_box = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        progress_bar = st.progress(0.0)
        readout_box = st.empty()

        if not run_btn:
            image_box.markdown(
                '<div style="height:320px;display:flex;align-items:center;justify-content:center;'
                'color:#2a3549;font-size:0.85rem;font-family:\'JetBrains Mono\',monospace;background:#020408;">'
                'AWAITING INPUT — upload a video and start detection'
                '</div>',
                unsafe_allow_html=True
            )

        with st.expander("Show detailed logs", expanded=False):
            log_box = st.empty()
            log_box.markdown(
                '<div class="terminal" style="color:#1c2636;">Waiting for video...</div>',
                unsafe_allow_html=True
            )

        if run_btn:
            selected_video = st.session_state.get("video_path", VIDEO_PATH)
            if not os.path.exists(selected_video):
                st.error("Please upload a video first.")
            else:
                agent1_path = os.path.join(ROOT_DIR, "agent1_tracking", "agent1.py")
                if not os.path.exists(agent1_path):
                    st.error(f"agent1.py not found at {agent1_path}")
                else:
                    readout_box.markdown(
                        '<div class="progress-readout"><span>Starting detection…</span></div>',
                        unsafe_allow_html=True
                    )
                    code, lines = run_agent_with_progress(
                        agent1_path, progress_bar, readout_box, image_box, log_box,
                        extra_args=[selected_video]
                    )

                    if code == 0:
                        st.session_state["agent1_done"] = True
                        st.success("✓ Detection complete — database populated. Move to Step 2.")
                        st.rerun()
                    else:
                        st.error("Agent 1 failed. Check the detailed logs above.")

# ═══════════════════════════════════════════════════════
# TAB 2 — VIOLATIONS
# ═══════════════════════════════════════════════════════
with tab2:
    top_left, top_right = st.columns([1, 2], gap="large")

    with top_left:
        st.markdown("#### Run violation analysis")
        st.markdown(
            '<p style="color:#5b6b82;font-size:0.85rem;line-height:1.6;">Agent 2 scans the database for rule violations — AFK, unauthorized zones, and people who left. Agent 3 confirms each finding using LLaVA.</p>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        check_btn = st.button("🔍  Check Violations", use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        violations = get_violations()
        total_v = len(violations)
        afk_v   = sum(1 for v in violations if "afk" in v[1])
        unauth_v = sum(1 for v in violations if "unauth" in v[1])
        left_v  = sum(1 for v in violations if "left" in v[1])

        st.markdown(f"""
        <div class="card">
            <div class="card-label">Summary</div>
            <div class="card-value">{total_v}</div>
            <div class="card-sub">total violation(s)</div>
            <hr style="margin:1rem 0;">
            <div style="display:flex;flex-direction:column;gap:0.5rem;">
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;">
                    <span style="color:#fbbf24;">⚠ AFK</span><span style="color:#fff;font-weight:600;">{afk_v}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;">
                    <span style="color:#a78bfa;">⊘ Unauthorized Zone</span><span style="color:#fff;font-weight:600;">{unauth_v}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:0.82rem;">
                    <span style="color:#f87171;">↗ Left Frame</span><span style="color:#fff;font-weight:600;">{left_v}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top_right:
        log2_box = st.empty()
        log2_box.markdown(
            '<div class="terminal" style="color:#1c2636;">Click Check Violations to start analysis...</div>',
            unsafe_allow_html=True
        )

        if check_btn:
            agent2_path = os.path.join(ROOT_DIR, "agent2_rules", "agent2.py")
            if not os.path.exists(agent2_path):
                st.error(f"agent2.py not found at {agent2_path}")
            else:
                code2, lines2 = run_agent(agent2_path, log2_box)
                if code2 == 0:
                    st.success("✓ Violation analysis complete.")
                    st.rerun()
                else:
                    st.error("Agent 2 failed. Check the log above.")

    st.markdown("<hr>", unsafe_allow_html=True)

    violations = get_violations()
    if violations:
        st.markdown(f"### Detected Violations")
        for v in violations:
            pid, etype, duration, ts, crop_path, vlm = v

            if "afk" in etype:
                pill = '<span class="pill pill-afk">AFK</span>'
                card_cls = "vcard-afk"
                label = "Away from desk"
            elif "unauth" in etype:
                pill = '<span class="pill pill-unauth">Unauthorized Zone</span>'
                card_cls = "vcard-unauth"
                label = "In wrong zone"
            elif "left" in etype:
                pill = '<span class="pill pill-left">Left Frame</span>'
                card_cls = "vcard-left"
                label = "Left and didn't return"
            else:
                pill = f'<span class="pill">{etype}</span>'
                card_cls = ""
                label = etype

            vlm_display = vlm if vlm else "—"
            vlm_color = "#4ade80" if vlm in ("AFK", "LEFT", "OTHER_ZONE", "LOITERING") else "#7d8ba3"

            img_col, info_col = st.columns([1.2, 1], gap="medium")

            with img_col:
                if crop_path and os.path.exists(crop_path):
                    st.image(crop_path, use_container_width=True, caption=f"Captured at {format_timestamp(ts)}")
                else:
                    st.markdown(
                        '<div style="background:#0a0f1c;border:1px solid #1c2636;border-radius:10px;height:200px;display:flex;align-items:center;justify-content:center;color:#45536b;font-size:0.8rem;">No image available</div>',
                        unsafe_allow_html=True
                    )

            with info_col:
                st.markdown(f"""
                <div class="vcard {card_cls}">
                    {pill}
                    <div style="margin-top:1rem;font-size:0.8rem;color:#5b6b82;">{label}</div>
                    <div style="font-size:1.6rem;font-weight:700;color:#fff;margin-top:0.25rem;">{person_label(pid)}</div>
                    <hr style="margin:1rem 0;border-color:#1c2636;">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                        <div>
                            <div style="color:#45536b;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;">Duration</div>
                            <div style="color:#fff;font-weight:600;font-size:1.1rem;margin-top:2px;">{int(duration)}s</div>
                            <div style="color:#5b6b82;font-size:0.75rem;">{duration/60:.1f} minutes</div>
                        </div>
                        <div>
                            <div style="color:#45536b;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;">Video Timestamp</div>
                            <div style="color:#fff;font-weight:600;font-size:1.1rem;margin-top:2px;">{format_timestamp(ts)}</div>
                            <div style="color:#5b6b82;font-size:0.75rem;">{ts/60:.1f} min mark</div>
                        </div>
                    </div>
                    <hr style="margin:1rem 0;border-color:#1c2636;">
                    <div>
                        <div style="color:#45536b;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;">LLaVA Verdict</div>
                        <div style="color:{vlm_color};font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:600;margin-top:4px;">→ {vlm_display}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#45536b;">
            <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
            <div style="font-size:1rem;font-weight:500;color:#5b6b82;">No violations found yet</div>
            <div style="font-size:0.8rem;margin-top:0.5rem;">Run Detection first, then click Check Violations above.</div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TAB 3 — REPORT
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Session Summary")

    people, detections, violations_count = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("People Tracked", people)
    c2.metric("Total Detections", f"{detections:,}")
    c3.metric("Violations Found", violations_count)
    c4.metric("System Status", "Ready" if detections > 0 else "No Data")

    st.markdown("<hr>", unsafe_allow_html=True)

    violations = get_violations()
    if violations:
        st.markdown("#### Violation Log")
        import pandas as pd
        rows = []
        for v in violations:
            pid, etype, duration, ts, crop, vlm = v
            rows.append({
                "Person": person_label(pid),
                "Violation": etype.replace("violation_", "").replace("_", " ").upper(),
                "Duration": f"{int(duration)}s ({duration/60:.1f} min)",
                "Timestamp": format_timestamp(ts),
                "VLM Verdict": vlm or "—"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Generate Report")
    st.markdown(
        '<p style="color:#5b6b82;font-size:0.85rem;">Generates a PDF report summarising all tracked people, violations, and LLaVA verdicts, with an LLM-written narrative synthesizing patterns per person.</p>',
        unsafe_allow_html=True
    )

    report_btn = st.button("📄  Generate PDF Report", use_container_width=False)

    log4_box = st.empty()
    log4_box.markdown(
        '<div class="terminal" style="color:#1c2636;">Click Generate PDF Report to start...</div>',
        unsafe_allow_html=True
    )

    if report_btn:
        agent4_path = os.path.join(ROOT_DIR, "agent4_reporting", "agent4.py")
        if os.path.exists(agent4_path):
            code4, lines4 = run_agent(agent4_path, log4_box)

            if code4 == 0:
                st.success("✓ Report generated successfully.")
                report_path = os.path.join(ROOT_DIR, "report.pdf")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button("⬇  Download PDF", f, file_name="surveillance_report.pdf", mime="application/pdf")
            else:
                st.error("Agent 4 failed. Check the log above.")
        else:
            st.info("Agent 4 not set up yet. Wire your PDF generator to agent4_reporting/agent4.py")