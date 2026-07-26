# dashboard/chat_assistant.py
#
# "Chat with your footage" - lets an operator ask natural-language questions
# about the CURRENT session's tracked people and violations, answered by an
# LLM grounded only in this run's own database contents.
#
# Deliberately reuses the exact same local Ollama text model Agent 4 already
# uses for its PDF narrative (llama3.1:8b, see agent4_reporting/agent4.py) -
# no new dependency, no new service, no external API key. If Ollama /
# llama3.1:8b isn't reachable, ask() returns a plain error string instead of
# raising, so a broken LLM connection can't crash the dashboard tab.

import sqlite3
import ollama

CHAT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a surveillance-analysis assistant embedded in a workplace monitoring dashboard.
You answer questions ONLY using the SESSION DATA provided below - a summary of one specific video run (people tracked, their zones, and every violation logged for it).

Rules:
- Never invent people, timestamps, zones, or violations that aren't in the data below.
- If the answer isn't in the data, say so plainly - do not guess or speculate.
- Be concise and factual (a few sentences unless the question asks for a list).
- Use M:SS format for timestamps when relevant (e.g. "3:12").
- "Person N" refers to that person's persistent tracking ID - there is no name attached to it.
- Do not give opinions on the person's behavior or intent - stick to what was observed.
"""


def format_timestamp(seconds):
    """Same M:SS convention used everywhere else in the dashboard/agents."""
    seconds = max(0, int(seconds or 0))
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def build_session_context(database_path, max_violations=150):
    """Pulls a compact, human-readable snapshot of the current session's DB
    contents and formats it as plain text for the LLM's context window.

    Deliberately pre-aggregated rather than a raw SQL dump: a text-instruct
    model answers far more reliably from readable summaries ("Person 2:
    first seen 0:05, last seen 4:12") than from hundreds of undifferentiated
    rows, and it keeps the prompt small regardless of session length.
    """
    conn = sqlite3.connect(database_path)
    try:
        people_ids = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT person_id FROM events "
                "WHERE event_type = 'detected' ORDER BY person_id"
            ).fetchall()
        ]

        total_detections = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'detected'"
        ).fetchone()[0]

        violations = conn.execute("""
            SELECT person_id, event_type, timestamp, duration_seconds,
                   zone_name, vlm_summary
            FROM events WHERE event_type LIKE 'violation_%'
            ORDER BY timestamp
            LIMIT ?
        """, (max_violations,)).fetchall()

        # Per-person presence window (first/last seen + how many frames) -
        # cheap proxy for "how long were they actually around", doesn't
        # require re-deriving zone logic here (that's Agent 2's job, not
        # this feature's).
        presence = {}
        for pid in people_ids:
            row = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM events WHERE event_type = 'detected' AND person_id = ?
            """, (pid,)).fetchone()
            presence[pid] = row
    finally:
        conn.close()

    lines = []
    lines.append(
        f"SESSION OVERVIEW: {len(people_ids)} person(s) tracked, "
        f"{total_detections} total detection frames recorded."
    )
    lines.append("")
    lines.append("PEOPLE:")
    if not people_ids:
        lines.append("  (no one was tracked in this session)")
    for pid in people_ids:
        first_ts, last_ts, count = presence[pid]
        lines.append(
            f"  Person {pid}: first seen {format_timestamp(first_ts)}, "
            f"last seen {format_timestamp(last_ts)}, {count} detection frame(s)."
        )

    lines.append("")
    lines.append("VIOLATIONS (chronological):")
    if not violations:
        lines.append("  (no violations recorded)")
    for pid, etype, ts, duration, zone_name, vlm in violations:
        # pid can be NULL (legacy rows) or 0 (reserved sentinel) for a
        # whole-room violation not tied to one tracked identity - same
        # convention as person_label() in app.py / agent4.py.
        who = "General (whole room)" if pid is None or pid == 0 else f"Person {pid}"
        vtype = etype.replace("violation_", "").replace("_", " ").upper()
        dur = f"{duration:.0f}s" if duration else "n/a"
        zone_part = f", zone={zone_name}" if zone_name else ""
        vlm_part = f", VLM verdict={vlm}" if vlm else ""
        lines.append(
            f"  - {who} | {vtype} | at {format_timestamp(ts)} | "
            f"duration {dur}{zone_part}{vlm_part}"
        )

    return "\n".join(lines)


def ask(question, database_path, history=None):
    """Answers one question, grounded in the current session's DB contents.

    `history` is an optional list of prior {"role", "content"} turns (NOT
    including the system/context message) so follow-up questions keep
    conversational continuity without re-summarizing the DB from scratch
    into the visible chat log.

    Rebuilds the session context on every call (cheap - a handful of
    lightweight SQL queries) rather than caching it, so an answer always
    reflects whatever's in the DB right now - e.g. right after re-running
    Agent 2, without needing to restart the chat.
    """
    context = build_session_context(database_path)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nSESSION DATA:\n" + context},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        response = ollama.chat(model=CHAT_MODEL, messages=messages)
        return response["message"]["content"].strip()
    except Exception as e:
        return (
            f"⚠️ Couldn't reach the local Ollama model ({CHAT_MODEL}). "
            f"Make sure Ollama is running and the model is pulled "
            f"(`ollama pull {CHAT_MODEL}`).\n\nDetails: {e}"
        )
