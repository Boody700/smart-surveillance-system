# app.py — Smart Surveillance System
# Run with: streamlit run app.py

import streamlit as st
import sqlite3
import subprocess
import sys
import os
import time
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from config import DATABASE_PATH, VIDEO_PATH

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
.stApp { background: #080c14; color: #e2e8f0; }

/* Hide streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero-badge {
    display: inline-block;
    background: #1e1b4b;
    border: 1px solid #4f46e5;
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #475569;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 8px 20px;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #e2e8f0 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Cards */
.card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    height: 100%;
}
.card-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 0.35rem;
}
.card-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}
.card-sub {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Terminal */
.terminal {
    background: #020408;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.25rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #4ade80;
    line-height: 1.7;
    max-height: 450px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}
.terminal-header {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-bottom: none;
    border-radius: 10px 10px 0 0;
    padding: 0.6rem 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot-r { background: #ef4444; }
.dot-y { background: #f59e0b; }
.dot-g { background: #22c55e; }

/* Step indicator */
.step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #1e293b;
    color: #475569;
    font-size: 0.85rem;
}
.step:last-child { border-bottom: none; }
.step.active { color: #e2e8f0; }
.step.done { color: #4ade80; }
.step-num {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: #1e293b;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step.done .step-num { background: #052e16; color: #4ade80; }
.step.active .step-num { background: #1e1b4b; color: #818cf8; }

/* Violation cards */
.vcard {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
.vcard-afk  { border-left: 4px solid #f59e0b; }
.vcard-left { border-left: 4px solid #ef4444; }
.vcard-unauth { border-left: 4px solid #8b5cf6; }

.pill {
    display: inline-block;
    border-radius: 999px;
    font-size: 0.68rem;
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
    background: #1e1b4b;
    color: #a5b4fc;
    border: 1px solid #4f46e5;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.875rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #4f46e5;
    color: #fff;
    border-color: #6366f1;
}
.stButton > button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

/* Upload area */
[data-testid="stFileUploader"] section {
    background: #0f172a;
    border: 2px dashed #1e293b;
    border-radius: 10px;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #4f46e5;
}

/* Divider */
hr { border-color: #1e293b; margin: 1.5rem 0; }

/* Metrics */
[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: #fff; }
[data-testid="stMetricLabel"] { color: #475569; }

/* Status pill */
.status-ok  { color: #4ade80; font-weight: 600; }
.status-err { color: #f87171; font-weight: 600; }
.status-run { color: #818cf8; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ &nbsp; Agentic AI System</div>
    <div class="hero-title">Smart <span>Surveillance</span> System</div>
    <div class="hero-sub">Upload a video · Detect violations · Generate a report</div>
</div>
""", unsafe_allow_html=True)

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

def run_agent(script_path, log_placeholder, cwd=ROOT_DIR):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Force the agent to use UTF-8 encoding for its internal print statements
    env["PYTHONIOENCODING"] = "utf-8"
    
    lines = []
    # Add encoding='utf-8' and errors='replace' here
    process = subprocess.Popen(
        [sys.executable, script_path],
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

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  📹  Step 1 — Run Detection  ", "  🚨  Step 2 — View Violations  ", "  📄  Step 3 — Report  "])

# ═══════════════════════════════════════════════════════
# TAB 1 — DETECTION
# ═══════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown("#### Upload your video")
        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed"
        )

        if uploaded:
            vdir = os.path.join(ROOT_DIR, "videos")
            os.makedirs(vdir, exist_ok=True)
            vpath = os.path.join(vdir, uploaded.name)
            with open(vpath, "wb") as f:
                f.write(uploaded.getbuffer())
            st.session_state["video_path"] = vpath
            st.success(f"✓ {uploaded.name}")
            st.video(uploaded)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Pipeline steps
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

        # Stats
        people, detections, violations = get_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("People", people)
        c2.metric("Detections", f"{detections:,}")
        c3.metric("Violations", violations)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Start Detection", use_container_width=True)

    with right:
        st.markdown("#### Live output")
        log_box = st.empty()
        log_box.markdown(
            '<div class="terminal" style="color:#1e293b;">Waiting for video...\n\nUpload a video and click Start Detection to begin.</div>',
            unsafe_allow_html=True
        )

        if run_btn:
            if "video_path" not in st.session_state and not os.path.exists(VIDEO_PATH):
                st.error("Please upload a video first.")
            else:
                agent1_path = os.path.join(ROOT_DIR, "agent1_tracking", "agent1cl.py")
                if not os.path.exists(agent1_path):
                    st.error(f"agent1.py not found at {agent1_path}")
                else:
                    with st.spinner(""):
                        code, lines = run_agent(agent1_path, log_box)

                    if code == 0:
                        st.session_state["agent1_done"] = True
                        st.success("✓ Detection complete — database populated. Move to Step 2.")
                        st.rerun()
                    else:
                        st.error("Agent 1 failed. Check the log above.")

# ═══════════════════════════════════════════════════════
# TAB 2 — VIOLATIONS
# ═══════════════════════════════════════════════════════
with tab2:
    top_left, top_right = st.columns([1, 2], gap="large")

    with top_left:
        st.markdown("#### Run violation analysis")
        st.markdown(
            '<p style="color:#64748b;font-size:0.85rem;line-height:1.6;">Agent 2 scans the database for rule violations — AFK, unauthorized zones, and people who left. Agent 3 confirms each finding using LLaVA.</p>',
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
            '<div class="terminal" style="color:#1e293b;">Click Check Violations to start analysis...</div>',
            unsafe_allow_html=True
        )

        if check_btn:
            agent2_path = os.path.join(ROOT_DIR, "agent2_rules", "agent2cl.py")
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

    # Show violations
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
            vlm_color = "#4ade80" if vlm in ("AFK", "LEFT", "OTHER_ZONE", "LOITERING") else "#94a3b8"

            img_col, info_col = st.columns([1.2, 1], gap="medium")

            with img_col:
                if crop_path and os.path.exists(crop_path):
                    st.image(crop_path, use_container_width=True, caption=f"Captured at {ts:.0f}s")
                else:
                    st.markdown(
                        '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;height:200px;display:flex;align-items:center;justify-content:center;color:#475569;font-size:0.8rem;">No image available</div>',
                        unsafe_allow_html=True
                    )

            with info_col:
                st.markdown(f"""
                <div class="vcard {card_cls}">
                    {pill}
                    <div style="margin-top:1rem;font-size:0.8rem;color:#64748b;">{label}</div>
                    <div style="font-size:1.6rem;font-weight:700;color:#fff;margin-top:0.25rem;">Person {pid}</div>
                    <hr style="margin:1rem 0;border-color:#1e293b;">
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                        <div>
                            <div style="color:#475569;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;">Duration</div>
                            <div style="color:#fff;font-weight:600;font-size:1.1rem;margin-top:2px;">{int(duration)}s</div>
                            <div style="color:#64748b;font-size:0.75rem;">{duration/60:.1f} minutes</div>
                        </div>
                        <div>
                            <div style="color:#475569;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;">Video Timestamp</div>
                            <div style="color:#fff;font-weight:600;font-size:1.1rem;margin-top:2px;">{ts:.0f}s</div>
                            <div style="color:#64748b;font-size:0.75rem;">{ts/60:.1f} min mark</div>
                        </div>
                    </div>
                    <hr style="margin:1rem 0;border-color:#1e293b;">
                    <div>
                        <div style="color:#475569;font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;">LLaVA Verdict</div>
                        <div style="color:{vlm_color};font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:600;margin-top:4px;">→ {vlm_display}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#475569;">
            <div style="font-size:3rem;margin-bottom:1rem;">🔍</div>
            <div style="font-size:1rem;font-weight:500;color:#64748b;">No violations found yet</div>
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
                "Person": f"Person {pid}",
                "Violation": etype.replace("violation_", "").replace("_", " ").upper(),
                "Duration": f"{int(duration)}s ({duration/60:.1f} min)",
                "Timestamp": f"{ts:.0f}s into video",
                "VLM Verdict": vlm or "—"
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Generate Report")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">Generates a PDF report summarising all tracked people, violations, and LLaVA verdicts.</p>',
        unsafe_allow_html=True
    )

    report_btn = st.button("📄  Generate PDF Report", use_container_width=False)
    if report_btn:
        agent4_path = os.path.join(ROOT_DIR, "agent4_dashboard", "agent4.py")
        if os.path.exists(agent4_path):
            with st.spinner("Generating report..."):
                result = subprocess.run(
                    [sys.executable, agent4_path],
                    cwd=ROOT_DIR, capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("✓ Report generated successfully.")
                # Try to offer download if PDF exists
                report_path = os.path.join(ROOT_DIR, "report.pdf")
                if os.path.exists(report_path):
                    with open(report_path, "rb") as f:
                        st.download_button("⬇  Download PDF", f, file_name="surveillance_report.pdf", mime="application/pdf")
            else:
                st.error(f"Report generation failed:\n{result.stderr}")
        else:
            st.info("Agent 4 not set up yet. Wire your PDF generator to agent4_dashboard/agent4.py")