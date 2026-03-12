"""
Smart City Traffic Command Center Dashboard
============================================
Unified dashboard with 8 panels:
  Left   → Live Traffic Monitor (video + YOLO detection)
  Center → Traffic Density Analytics + AI Signal Optimization
  Right  → Emergency Alerts + Green Corridor + AI Decision Log
  Bottom → City Traffic Map & Congestion Heatmap
"""

import os
import time
import random
import html
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.process_video import process_frame
from backend.traffic_brain import get_traffic_brain, make_decision, calculate_density, decide_signal_timing, check_emergency, generate_ai_decision
from backend.traffic_controller import get_traffic_controller


# ─────────────────────── helpers ───────────────────────

def _to_int(v, d=0):
    try:
        return int(float(v)) if v is not None else d
    except Exception:
        return d


def _init_state():
    defaults = {
        "processed": False,
        "df": None,
        "metrics": {
            "total_vehicles": 0, "queue_count": 0, "total_violations": 0,
            "red_light_violations": 0, "rash_driving": 0,
            "no_helmet_violations": 0, "mobile_usage_violations": 0,
            "triple_riding_violations": 0, "heavy_load_violations": 0,
            "autos": 0,
        },
        "dash_search": "",
        "cmd_live_running": False,
        "cmd_live_paused": False,
        "cmd_live_video_path": None,
        "cmd_live_frame_idx": 0,
        "cmd_live_fps": 24.0,
        "cmd_vehicle_counts": {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "autos": 0},
        "cmd_density_history": [],
        "cmd_event_log": [],
        "cmd_signal_countdown": 30,
        "cmd_emergency_active": False,
        "cmd_emergency_info": None,
        "cmd_corridor_active": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _density_level(total_vehicles):
    """Legacy wrapper — used only for heatmap cells."""
    if total_vehicles >= 20:
        return "High", "#ef4444"
    if total_vehicles >= 8:
        return "Medium", "#f59e0b"
    return "Low", "#22c55e"


def _density_color_from_level(level):
    """Color from AI decision engine density level."""
    return {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}.get(level, "#22c55e")


def _density_icon(level):
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "🟢")


def _save_uploaded(uploaded):
    os.makedirs("uploads", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("uploads", f"{ts}_{os.path.basename(uploaded.name)}")
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# ─────────────────────── CSS ───────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-dark: #0a0e1a;
    --bg-panel: #111827;
    --bg-card: #1a2236;
    --border: #1e293b;
    --accent-cyan: #06b6d4;
    --accent-green: #22c55e;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

.cmd-root {
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #f1f5f9 !important;
}
/* Force ALL text inside our custom panels to be light */
.cmd-root, .cmd-root div, .cmd-root span, .cmd-root p,
.cmd-root strong, .cmd-root h1, .cmd-root h2, .cmd-root h3, .cmd-root h4 {
    color: #f1f5f9 !important;
}
.cmd-root .sub { color: #94a3b8 !important; }
.cmd-hero h1 {
    background: linear-gradient(135deg, #06b6d4, #8b5cf6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* ── HERO HEADER ── */
.cmd-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(6,182,212,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
}
.cmd-hero::before {
    content: '';
    position: absolute;
    top: -80%;
    right: -10%;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(6,182,212,0.12) 0%, transparent 70%);
}
.cmd-hero::after {
    content: '';
    position: absolute;
    bottom: -80%;
    left: 5%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.1) 0%, transparent 70%);
}
.cmd-hero-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 1;
}
.cmd-hero h1 {
    margin: 0;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #06b6d4, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.cmd-hero .sub {
    margin: 4px 0 0 0;
    color: #94a3b8;
    font-size: 13px;
}
.cmd-status {
    display: flex;
    gap: 18px;
    align-items: center;
}
.cmd-status-item {
    text-align: center;
}
.cmd-status-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #94a3b8 !important;
}
.cmd-status-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 700;
}

/* ── PANELS ── */
.panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
    min-height: 100px;
}
.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.panel-title {
    font-size: 14px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e2e8f0 !important;
}
.panel-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-live { background: rgba(239,68,68,0.2); color: #ef4444 !important; animation: pulse-badge 1.5s ease-in-out infinite; }
.badge-active { background: rgba(34,197,94,0.2); color: #22c55e !important; }
.badge-idle { background: rgba(100,116,139,0.2); color: #cbd5e1 !important; }
@keyframes pulse-badge { 0%,100%{opacity:1;} 50%{opacity:0.5;} }

/* ── VEHICLE COUNTER GRID ── */
.v-counter-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.v-counter {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px;
    text-align: center;
}
.v-counter .icon { font-size: 20px; margin-bottom: 4px; }
.v-counter .count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 800;
    line-height: 1;
}
.v-counter .label {
    font-size: 11px;
    color: #94a3b8 !important;
    margin-top: 2px;
}

/* ── DENSITY INDICATOR ── */
.density-indicator {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 8px;
}
.density-dot {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.density-text .level {
    font-size: 18px;
    font-weight: 800;
}
.density-text .detail {
    font-size: 11px;
    color: #94a3b8 !important;
}

/* ── SIGNAL DISPLAY ── */
.signal-intersection {
    display: grid;
    grid-template-areas:
        ". north ."
        "west center east"
        ". south .";
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: 1fr 1fr 1fr;
    gap: 6px;
    width: 180px;
    height: 180px;
    margin: 0 auto 10px auto;
}
.sig-light {
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    font-size: 9px;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    transition: all 0.3s ease;
    border: 2px solid rgba(255,255,255,0.1);
}
.sig-light.north { grid-area: north; }
.sig-light.south { grid-area: south; }
.sig-light.east  { grid-area: east; }
.sig-light.west  { grid-area: west; }
.sig-center {
    grid-area: center;
    background: #0f172a;
    border: 2px solid rgba(6,182,212,0.3);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}
.sig-light .countdown {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    font-weight: 800;
    margin-top: 2px;
}
.sig-green { background: radial-gradient(circle, #22c55e, #16a34a); box-shadow: 0 0 20px rgba(34,197,94,0.5); }
.sig-red { background: radial-gradient(circle, #ef4444, #dc2626); box-shadow: 0 0 20px rgba(239,68,68,0.5); }
.sig-amber { background: radial-gradient(circle, #f59e0b, #d97706); box-shadow: 0 0 20px rgba(245,158,11,0.5); }

/* Signal timing bars */
.sig-timing {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
}
.sig-timing-label {
    width: 45px;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: capitalize;
}
.sig-timing-bar {
    flex: 1;
    height: 18px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
}
.sig-timing-fill {
    height: 100%;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    color: white;
    transition: width 0.6s ease;
}
.sig-timing-state {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    width: 42px;
    text-align: center;
}

/* ── EMERGENCY BANNER ── */
.emergency-alert {
    background: linear-gradient(135deg, #7f1d1d, #991b1b, #7f1d1d);
    background-size: 200% 200%;
    animation: gradient-pulse 2s ease-in-out infinite;
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 10px;
}
@keyframes gradient-pulse {
    0%,100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.emergency-flex {
    display: flex;
    align-items: center;
    gap: 12px;
}
.emergency-icon-wrap {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    background: rgba(255,255,255,0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    animation: icon-pulse 1.2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes icon-pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.15);} }
.emergency-details h4 { margin: 0; font-size: 15px; font-weight: 800; color: white; }
.emergency-details p { margin: 2px 0 0 0; font-size: 12px; color: rgba(255,255,255,0.8); }

.no-emergency {
    background: var(--bg-card);
    border: 1px dashed rgba(100,116,139,0.3);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
}

/* ── GREEN CORRIDOR ── */
.corridor-active {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 12px;
    padding: 14px;
    position: relative;
    overflow: hidden;
}
.corridor-active::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    animation: corridor-shine 2.5s infinite;
}
@keyframes corridor-shine { 0%{left:-100%;} 100%{left:100%;} }
.corridor-header {
    font-size: 14px;
    font-weight: 800;
    color: #4ade80 !important;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}
.corridor-status-list {
    position: relative;
    z-index: 1;
}
.corridor-status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 0;
    font-size: 12px;
    color: rgba(255,255,255,0.85) !important;
}
.corridor-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ade80;
    animation: blink-dot 1s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes blink-dot { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.corridor-path-row {
    display: flex;
    gap: 6px;
    margin-top: 8px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}
.corridor-node {
    background: rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 700;
    color: #bbf7d0;
}
.corridor-arrow { color: rgba(255,255,255,0.4); font-size: 14px; }

.no-corridor {
    background: var(--bg-card);
    border: 1px dashed rgba(100,116,139,0.3);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: var(--text-muted);
    font-size: 13px;
}

/* ── DECISION LOG ── */
.decision-entry {
    display: flex;
    gap: 8px;
    padding: 6px 8px;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 11px;
    align-items: flex-start;
    border: 1px solid transparent;
}
.decision-entry.d-emergency { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.15); }
.decision-entry.d-signal { background: rgba(6,182,212,0.08); border-color: rgba(6,182,212,0.15); }
.decision-entry.d-corridor { background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.15); }
.decision-entry.d-general { background: rgba(100,116,139,0.06); border-color: rgba(100,116,139,0.1); }
.decision-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}
.decision-msg { color: #cbd5e1 !important; line-height: 1.4; }
.decision-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #94a3b8 !important;
    margin-top: 2px;
}

/* ── MAP PLACEHOLDER ── */
.map-panel {
    background: var(--bg-card);
    border-radius: 10px;
    min-height: 260px;
    overflow: hidden;
}

/* ── MISC ── */
.cmd-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(6,182,212,0.2), transparent);
    margin: 6px 0;
}

</style>
"""


# ─────────────────────── SHOW ───────────────────────

def show():
    _init_state()
    brain = get_traffic_brain()

    # Tick brain each render
    counts = st.session_state.get("cmd_vehicle_counts", {})
    total_v = int(sum(counts.values()))
    brain.tick(
        detection_results={
            "vehicle_counts": counts,
            "total_vehicles": total_v,
            "emergency_vehicles": [],
        },
        elapsed_seconds=1.0,
    )

    # ── AI Decision Engine ──
    controller = get_traffic_controller()
    decision = controller.process_pipeline(counts)
    traffic_score = decision["traffic_score"]
    density_level = decision["density"]
    signal_duration = decision["signal_duration"]
    emergency_mode = decision["emergency_mode"]
    emergency_type = decision["emergency_type"]
    ai_decision_message = decision["ai_decision"]
    density_color = _density_color_from_level(density_level)
    lane_data = decision.get("lane_data", {})

    grid_state = brain.get_grid_state()
    decision_log = brain.get_decision_log(30)
    active_corridors = grid_state.get("active_corridors", [])

    # ── inject CSS ──
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="cmd-root">', unsafe_allow_html=True)

    # ════════════════════ HERO HEADER ════════════════════
    now = datetime.now()
    username = html.escape(str(st.session_state.get("username", "Operator")))

    st.markdown(f"""
    <div class="cmd-hero">
        <div class="cmd-hero-flex">
            <div>
                <h1>🏙️ Smart City Traffic Command Center</h1>
                <p class="sub">Dynamic AI Traffic Flow Optimizer &amp; Emergency Grid &bull; Operator: {username}</p>
            </div>
            <div class="cmd-status">
                <div class="cmd-status-item">
                    <div class="cmd-status-label">Time</div>
                    <div class="cmd-status-value" style="color:#06b6d4;">{now.strftime("%H:%M:%S")}</div>
                </div>
                <div class="cmd-status-item">
                    <div class="cmd-status-label">Score</div>
                    <div class="cmd-status-value" style="color:{density_color};">{traffic_score}</div>
                </div>
                <div class="cmd-status-item">
                    <div class="cmd-status-label">Density</div>
                    <div class="cmd-status-value" style="color:{density_color};">{density_level}</div>
                </div>
                <div class="cmd-status-item">
                    <div class="cmd-status-label">Signal</div>
                    <div class="cmd-status-value" style="color:#06b6d4;">{signal_duration}s</div>
                </div>
                <div class="cmd-status-item">
                    <div class="cmd-status-label">Emergency</div>
                    <div class="cmd-status-value" style="color:{'#ef4444' if emergency_mode else '#22c55e'};">{'🚨 YES' if emergency_mode else '✅ NO'}</div>
                </div>
                <div class="cmd-status-item">
                    <div class="cmd-status-label">Corridors</div>
                    <div class="cmd-status-value" style="color:{'#ef4444' if active_corridors else '#22c55e'};">{len(active_corridors)}</div>
                </div>
            </div>
        </div>
        <div style="margin-top:10px;padding:8px 14px;background:rgba(6,182,212,0.08);border:1px solid rgba(6,182,212,0.15);border-radius:8px;font-size:12px;color:#94a3b8;">
            🧠 <strong style="color:#06b6d4;">AI Decision:</strong> {html.escape(ai_decision_message)}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ════════════════════ ARCHITECTURE VISUALIZATION ════════════════════
    st.markdown("""
    <div style="background:#1e293b; padding:12px; border-radius:8px; margin-top:16px; margin-bottom:16px; border:1px solid #334155;">
        <div style="font-size:12px; color:#94a3b8; margin-bottom:8px; font-weight:bold; letter-spacing:1px; text-transform:uppercase;">System Architecture Pipeline</div>
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
            <div style="background:#0f172a; padding:8px 16px; border-radius:6px; border-left:3px solid #3b82f6; font-size:14px; font-weight:600; color:#cbd5e1; flex:1; text-align:center;">📷 Camera Feed</div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#0f172a; padding:8px 16px; border-radius:6px; border-left:3px solid #8b5cf6; font-size:14px; font-weight:600; color:#cbd5e1; flex:1; text-align:center;">🧠 AI Detection</div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#0f172a; padding:8px 16px; border-radius:6px; border-left:3px solid #06b6d4; font-size:14px; font-weight:600; color:#cbd5e1; flex:1; text-align:center;">🤖 Traffic Brain</div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#0f172a; padding:8px 16px; border-radius:6px; border-left:3px solid #f59e0b; font-size:14px; font-weight:600; color:#cbd5e1; flex:1; text-align:center;">🚦 Signal Control</div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#0f172a; padding:8px 16px; border-radius:6px; border-left:3px solid #22c55e; font-size:14px; font-weight:600; color:#cbd5e1; flex:1; text-align:center;">🚑 Emergency Grid</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════ MAIN 3-COLUMN LAYOUT ════════════════════
    col_left, col_center, col_right = st.columns([4, 4, 3])

    # ──────── LEFT PANEL: Live Traffic Monitor ────────
    with col_left:
        is_running = st.session_state.get("cmd_live_running", False)
        badge = '<span class="panel-badge badge-live">● LIVE</span>' if is_running else '<span class="panel-badge badge-idle">IDLE</span>'

        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">📹 Live Traffic Monitor</div>
                {badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Video upload + controls
        uploaded = st.file_uploader("Upload Traffic Video", type=["mp4", "avi", "mov", "mkv"], key="cmd_upload", label_visibility="collapsed")

        ctrl1, ctrl2, ctrl3 = st.columns(3)
        start = ctrl1.button("▶ Start", key="cmd_start", use_container_width=True)
        pause = ctrl2.button("⏸ Pause", key="cmd_pause", use_container_width=True)
        stop = ctrl3.button("⏹ Stop", key="cmd_stop", use_container_width=True)

        if start:
            if uploaded:
                path = _save_uploaded(uploaded)
                st.session_state.cmd_live_video_path = path
            elif not st.session_state.cmd_live_video_path:
                last = st.session_state.get("last_uploaded_video_path")
                if last and os.path.exists(last):
                    st.session_state.cmd_live_video_path = last
            if st.session_state.cmd_live_video_path:
                st.session_state.cmd_live_running = True
                st.session_state.cmd_live_paused = False
                st.session_state.cmd_live_frame_idx = 0
                st.session_state.cmd_vehicle_counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "autos": 0}
                st.session_state.cmd_density_history = []
                st.session_state.cmd_event_log = []
        if pause:
            st.session_state.cmd_live_paused = True
        if stop:
            st.session_state.cmd_live_running = False
            st.session_state.cmd_live_paused = False
            st.session_state.cmd_live_frame_idx = 0

        frame_ph = st.empty()

        # Render video feed
        vid_path = st.session_state.get("cmd_live_video_path")
        if st.session_state.cmd_live_running and not st.session_state.cmd_live_paused and vid_path:
            cap = cv2.VideoCapture(vid_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                st.session_state.cmd_live_fps = fps
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.cmd_live_frame_idx)

                for _ in range(300):  # process up to 300 frames per render
                    if not st.session_state.cmd_live_running or st.session_state.cmd_live_paused:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        st.session_state.cmd_live_running = False
                        break

                    fidx = st.session_state.cmd_live_frame_idx
                    processed, vc, events = process_frame(
                        frame,
                        settings={"conf_threshold": 0.35, "detect_imgsz": 416,
                                  "bounding_boxes": True, "vehicle_ids": True,
                                  "queue_zones": True, "violation_alerts": True,
                                  "highlight_violation_red": True},
                        frame_number=fidx,
                        camera_id="CAM-01",
                        draw_overlays=True,
                    )

                    st.session_state.cmd_vehicle_counts = vc
                    total_now = int(sum(vc.values()))

                    # Track density history
                    st.session_state.cmd_density_history.append({
                        "frame": fidx,
                        "total": total_now,
                        "cars": vc.get("cars", 0),
                        "bikes": vc.get("bikes", 0),
                        "buses": vc.get("buses", 0),
                        "trucks": vc.get("trucks", 0),
                    })
                    st.session_state.cmd_density_history = st.session_state.cmd_density_history[-200:]

                    # Feed to brain
                    brain.tick(
                        detection_results={
                            "vehicle_counts": vc,
                            "total_vehicles": total_now,
                            "emergency_vehicles": [],
                        },
                        elapsed_seconds=1.0 / max(1, fps),
                    )

                    for ev in events:
                        st.session_state.cmd_event_log.append(ev)
                    st.session_state.cmd_event_log = st.session_state.cmd_event_log[-100:]

                    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                    frame_ph.image(rgb, channels="RGB", use_container_width=True)
                    st.session_state.cmd_live_frame_idx += 1
                    time.sleep(1.0 / max(1, fps))

                cap.release()
            else:
                frame_ph.error("Unable to open video file.")
        elif vid_path and st.session_state.cmd_live_paused:
            cap = cv2.VideoCapture(vid_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, st.session_state.cmd_live_frame_idx))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    p, _, _ = process_frame(frame, settings={"bounding_boxes": True, "vehicle_ids": True,
                                                              "queue_zones": True, "conf_threshold": 0.35, "detect_imgsz": 416},
                                             frame_number=st.session_state.cmd_live_frame_idx, camera_id="CAM-01")
                    rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
                    frame_ph.image(rgb, channels="RGB", use_container_width=True)
        else:
            frame_ph.markdown("""
            <div style="background:#1a2236;border:2px dashed #1e293b;border-radius:12px;padding:60px 20px;text-align:center;color:#94a3b8;">
                <div style="font-size:48px;margin-bottom:10px;">📹</div>
                <p style="font-size:14px;margin:0;">Upload a traffic video and click <strong>▶ Start</strong></p>
                <p style="font-size:12px;margin:4px 0 0 0;">Supports MP4, AVI, MOV, MKV</p>
            </div>
            """, unsafe_allow_html=True)

        # Vehicle count overlay
        vc = st.session_state.cmd_vehicle_counts
        lane_html = f"""
        <div class="panel" style="padding:10px;">
            <div class="v-counter-grid">
                <div class="v-counter"><div class="icon">🚗</div><div class="count" style="color:#06b6d4;">{vc.get('cars',0)}</div><div class="label">Cars (×1)</div></div>
                <div class="v-counter"><div class="icon">🏍️</div><div class="count" style="color:#f59e0b;">{vc.get('bikes',0)}</div><div class="label">Bikes (×0.5)</div></div>
                <div class="v-counter"><div class="icon">🚌</div><div class="count" style="color:#8b5cf6;">{vc.get('buses',0)}</div><div class="label">Buses (×3)</div></div>
                <div class="v-counter"><div class="icon">🚛</div><div class="count" style="color:#ef4444;">{vc.get('trucks',0)}</div><div class="label">Trucks (×3)</div></div>
                <div class="v-counter"><div class="icon">🛺</div><div class="count" style="color:#ec4899;">{vc.get('autos',0)}</div><div class="label">Autos (×1)</div></div>
                <div class="v-counter"><div class="icon">📊</div><div class="count" style="color:{density_color};">{total_v}</div><div class="label">Total</div></div>
            </div>
            <div class="density-indicator">
                <div class="density-dot" style="background:{density_color};box-shadow:0 0 16px {density_color}40;">{_density_icon(density_level)}</div>
                <div class="density-text">
                    <div class="level" style="color:{density_color};">Density: {density_level} (Score: {traffic_score})</div>
                    <div class="detail">Green signal: {signal_duration}s &bull; {total_v} vehicles in frame</div>
                </div>
            </div>
            
            <div style="margin-top:16px;">
                <div style="font-size:12px; color:#94a3b8; font-weight:bold; margin-bottom:8px; text-transform:uppercase;">4-Direction Simulation</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
"""
        for lane, l_counts in lane_data.items():
            l_total = sum(l_counts.values())
            l_score, l_den = calculate_density(l_counts)
            l_c = _density_color_from_level(l_den)
            lane_html += f"""
                    <div style="background:#0f172a; padding:8px; border-radius:6px; border-left:3px solid {l_c};">
                        <div style="font-size:13px; font-weight:bold; color:#cbd5e1;">{lane} Lane</div>
                        <div style="font-size:11px; color:#94a3b8;">Total: {l_total} ({l_den})</div>
                    </div>
            """
        lane_html += """
                </div>
            </div>
"""
        st.markdown(lane_html + """
        </div>
        """, unsafe_allow_html=True)

    # ──────── CENTER PANEL: Analytics + Signal Control ────────
    with col_center:
        # Traffic Density Analytics
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">📊 Traffic Density Analytics</div>
                <span class="panel-badge badge-active">REAL-TIME</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        hist = st.session_state.get("cmd_density_history", [])
        if hist and len(hist) > 2:
            df_hist = pd.DataFrame(hist)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_hist["frame"], y=df_hist["total"],
                mode="lines", name="Total",
                line=dict(color="#06b6d4", width=2),
                fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
            ))
            fig.add_trace(go.Scatter(x=df_hist["frame"], y=df_hist["cars"], mode="lines", name="Cars", line=dict(color="#3b82f6", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=df_hist["frame"], y=df_hist["bikes"], mode="lines", name="Bikes", line=dict(color="#f59e0b", width=1.5, dash="dot")))
            fig.update_layout(
                height=180, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=10),
                legend=dict(orientation="h", y=1.1, font=dict(size=10)),
                xaxis=dict(showgrid=False, title="", color="#475569"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", title="", color="#475569"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("""
            <div style="background:#1a2236;border-radius:10px;padding:30px;text-align:center;color:#94a3b8;font-size:13px;">
                📈 Analytics chart will appear when video is playing
            </div>
            """, unsafe_allow_html=True)

        # AI Signal Optimization
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🚦 AI Signal Optimization</div>
                <span class="panel-badge badge-active">DYNAMIC</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Get signal states from brain's first intersection
        first_int = list(brain.intersections.values())[0] if brain.intersections else None
        if first_int:
            plan = first_int.optimizer.get_timing_plan()
            signals = first_int.optimizer.get_signal_states()
            approaches = plan.get("approaches", {})
            override_approach = plan.get("emergency_override")

            # Render intersection signal display
            signal_html = '<div class="signal-intersection">'
            for direction in ["north", "south", "east", "west"]:
                state = signals.get(direction, "RED")
                approach_data = approaches.get(direction, {})
                green_dur = int(approach_data.get("green_duration", 15))
                elapsed = int(plan.get("phase_elapsed", 0))
                countdown = max(0, green_dur - elapsed) if state == "GREEN" else green_dur

                css_cls = "sig-green" if state == "GREEN" else "sig-red"
                if override_approach == direction:
                    css_cls = "sig-green"
                    countdown = "∞"
                    state = "GREEN"

                signal_html += f"""
                <div class="sig-light {direction} {css_cls}">
                    {direction[0].upper()}
                    <span class="countdown">{countdown}</span>
                </div>"""
            signal_html += '<div class="sig-center">🚗</div></div>'
            st.markdown(signal_html, unsafe_allow_html=True)

            # Timing bars
            timing_html = ""
            for direction in ["north", "south", "east", "west"]:
                ad = approaches.get(direction, {})
                green_dur = ad.get("green_duration", 15)
                state = signals.get(direction, "RED")
                density = ad.get("density", 0)
                pct = min(100, int((green_dur / 90.0) * 100))
                bar_color = "#22c55e" if state == "GREEN" else "#ef4444"
                state_bg = f"background:{bar_color};"

                timing_html += f"""
                <div class="sig-timing">
                    <span class="sig-timing-label">{direction[0].upper()}</span>
                    <div class="sig-timing-bar"><div class="sig-timing-fill" style="width:{pct}%;background:{bar_color};">{green_dur:.0f}s</div></div>
                    <span class="sig-timing-state" style="{state_bg};color:white;">{state}</span>
                </div>"""
            st.markdown(timing_html, unsafe_allow_html=True)

        # Simulate emergency button
        sim_cols = st.columns([2, 2, 3])
        with sim_cols[0]:
            em_type = st.selectbox("Vehicle", ["ambulance", "fire_truck", "police"], key="cmd_em_type", label_visibility="collapsed")
        with sim_cols[1]:
            em_dir = st.selectbox("Approach", ["north", "south", "east", "west"], key="cmd_em_dir", label_visibility="collapsed")
        with sim_cols[2]:
            if st.button("🚨 Simulate Emergency", type="primary", use_container_width=True, key="cmd_sim_em"):
                brain.simulate_emergency(vehicle_type=em_type, approach=em_dir)
                st.session_state.cmd_emergency_active = True
                st.session_state.cmd_corridor_active = True
                st.rerun()

    # ──────── RIGHT PANEL: Emergency + Corridor + Decision Log ────────
    with col_right:
        # Emergency Vehicle Detection
        active_corridor = grid_state.get("active_corridor")
        em_active = True if active_corridor else False
        if em_active:
            st.session_state.cmd_emergency_active = True
        else:
            st.session_state.cmd_emergency_active = False

        em_badge = '<span class="panel-badge badge-live">⚠️ ALERT</span>' if em_active else '<span class="panel-badge badge-idle">CLEAR</span>'
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🚨 Emergency Detection</div>
                {em_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if active_corridor:
            vtype = active_corridor.get("vehicle_type", "Ambulance")
            start_node = active_corridor.get("route", ["Unknown"])[0]
            icon = "🚑" if "Ambulance" in vtype else "🚒"
            st.markdown(f"""
            <div class="emergency-alert">
                <div class="emergency-flex">
                    <div class="emergency-icon-wrap" style="font-size:32px;">{icon}</div>
                    <div class="emergency-details">
                        <h4 style="margin:0 0 8px 0; color:#ef4444;">⚠️ Emergency Vehicle Detected</h4>
                        <p style="margin:2px 0; font-size:13px; color:#cbd5e1;"><strong>Vehicle Type:</strong> {vtype}</p>
                        <p style="margin:2px 0; font-size:13px; color:#cbd5e1;"><strong>Current Location:</strong> {start_node}</p>
                        <p style="margin:2px 0; font-size:13px; color:#cbd5e1;"><strong>Destination:</strong> Hospital</p>
                        <p style="margin:6px 0 0 0; font-size:13px; color:#cbd5e1;"><strong>Status:</strong> <span style="color:#4ade80; font-weight:bold;">Green Corridor Activated</span></p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-emergency">
                <div style="font-size:24px;margin-bottom:6px;">✅</div>
                <strong>No Emergency Vehicles Detected</strong><br>
                <span style="font-size:11px;color:#94a3b8 !important;">System monitoring all camera feeds</span>
            </div>
            """, unsafe_allow_html=True)

        # Green Corridor
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🟢 Green Corridor</div>
                {'<span class="panel-badge badge-live">ACTIVE</span>' if active_corridor else '<span class="panel-badge badge-idle">STANDBY</span>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if active_corridor:
            route = active_corridor.get("route", [])
            plan = active_corridor.get("plan", {})
            vtype = active_corridor.get("vehicle_type", "Ambulance")
            
            path_html = ""
            for i, node in enumerate(route):
                if i > 0:
                    path_html += '<span class="corridor-arrow">→</span>'
                path_html += f'<span class="corridor-node">{node}</span>'

            signal_html = "<div style='margin-top:12px; margin-bottom:8px;'><strong style='color:#94a3b8;'>Signal Preemption:</strong></div>"
            for node in route:
                if node == "Hospital": continue
                state = plan.get(node, "UNKNOWN")
                c = "#22c55e" if state == "GREEN" else "#f59e0b" if state == "PREPARE GREEN" else "#ef4444"
                signal_html += f"<div style='font-size:13px;color:#cbd5e1;margin-bottom:4px;'>&bull; {node} &rarr; <span style='color:{c};font-weight:800;'>{state}</span></div>"

            st.markdown(f"""
            <div class="corridor-active">
                <div class="corridor-header">🟢 Active Route: {vtype}</div>
                
                {signal_html}
                
                <div style="margin-top:14px; margin-bottom:8px;"><strong style='color:#94a3b8;'>Route:</strong></div>
                <div class="corridor-path-row">{path_html}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-corridor">
                <div style="font-size:24px;margin-bottom:6px;">🛤️</div>
                <strong>No Active Corridors</strong><br>
                <span style="font-size:11px;color:#94a3b8 !important;">Corridors activate when emergency vehicles are detected</span>
            </div>
            """, unsafe_allow_html=True)

        # AI Decision Log
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🧠 AI Decision Log</div>
                <span class="panel-badge badge-active">AI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if decision_log:
            for entry in reversed(decision_log[-12:]):
                d_type = entry.get("type", "unknown")
                details = entry.get("details", {})
                ts = entry.get("timestamp", 0)
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
                ai_msg = entry.get("ai_message", "")

                if ai_msg:
                    # Use the AI decision engine's human-readable message
                    if "emergency" in d_type.lower() or "simulated" in d_type.lower():
                        css = "d-emergency"
                        dot_c = "#ef4444"
                    elif details.get("traffic_density") == "HIGH":
                        css = "d-signal"
                        dot_c = "#ef4444"
                    elif details.get("traffic_density") == "MEDIUM":
                        css = "d-signal"
                        dot_c = "#f59e0b"
                    else:
                        css = "d-general"
                        dot_c = "#06b6d4"
                    msg = html.escape(ai_msg)
                else:
                    # Fallback for older entries
                    if "emergency" in d_type or "simulated" in d_type:
                        css = "d-emergency"
                        dot_c = "#ef4444"
                        vt = details.get("vehicle_type", "").replace("_", " ").title()
                        ap = details.get("approach", "").title()
                        msg = f"🚨 {d_type.replace('_', ' ').title()}: {vt} on {ap}" if vt else f"🚨 {d_type.replace('_', ' ').title()}"
                    elif "timing" in d_type:
                        css = "d-signal"
                        dot_c = "#06b6d4"
                        msg = "⏱️ Signal timing recalculated based on density"
                    else:
                        css = "d-general"
                        dot_c = "#64748b"
                        msg = f"📊 System update"

                st.markdown(f"""
                <div class="decision-entry {css}">
                    <div class="decision-dot" style="background:{dot_c};"></div>
                    <div>
                        <div class="decision-msg">{msg}</div>
                        <div class="decision-time">{time_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8;font-size:12px;text-align:center;padding:12px;">AI decisions will appear here as the system operates</div>', unsafe_allow_html=True)

    # ════════════════════ BOTTOM: Map + Heatmap ════════════════════
    st.markdown('<div class="cmd-divider"></div>', unsafe_allow_html=True)

    map_col, heat_col = st.columns([3, 2])

    with map_col:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🗺️ City Traffic Map</div>
                <span class="panel-badge badge-active">LIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        import pydeck as pdk
        from backend.green_corridor import visualize_corridor_on_map

        # Simulated intersections on a city grid
        active_corridor = grid_state.get("active_corridor")
        corridor_plan = active_corridor.get("plan", {}) if active_corridor else {}
        
        intersection_data = []
        int_positions = {
            "INT-01": {"lat": 12.9716, "lon": 77.5946, "name": "Junction A"},
            "INT-02": {"lat": 12.9716, "lon": 77.6046, "name": "Junction B"},
            "INT-03": {"lat": 12.9616, "lon": 77.5946, "name": "Junction C"},
            "INT-04": {"lat": 12.9616, "lon": 77.6046, "name": "Junction D"},
        }

        # Base intersections layer
        for int_id, pos in int_positions.items():
            state = corridor_plan.get(int_id)
            is_emergency = brain.intersections.get(int_id, None)
            em = is_emergency.emergency_detected if is_emergency else False
            
            if state == "GREEN":
                color = [34, 197, 94, 200]    # Green (Cleared)
            elif state == "PREPARE GREEN":
                color = [245, 158, 11, 200]   # Yellow (Preparing)
            elif em:
                color = [239, 68, 68, 200]    # Red fallback
            else:
                color = [6, 182, 212, 100]    # Cyan (Normal)

            intersection_data.append({
                "lat": pos["lat"],
                "lon": pos["lon"],
                "name": f"{int_id}: {pos['name']} (State: {state or 'NORMAL'})",
                "color": color,
                "radius": 150 if state else 80,
            })

        # Congestion heatmap points
        heatmap_points = []
        density_map = brain.get_density_heatmap()
        for int_id, densities in density_map.items():
            if int_id in int_positions:
                avg_d = sum(densities.values()) / max(1, len(densities))
                pos = int_positions[int_id]
                for _ in range(int(avg_d * 30) + 1):
                    heatmap_points.append({
                        "lat": pos["lat"] + random.uniform(-0.001, 0.001),
                        "lon": pos["lon"] + random.uniform(-0.001, 0.001),
                        "weight": avg_d * 100,
                    })

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=intersection_data,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "HeatmapLayer",
                data=heatmap_points,
                get_position=["lon", "lat"],
                get_weight="weight",
                radiusPixels=60,
                intensity=1,
                threshold=0.1,
                opacity=0.5,
            ),
        ]

        # Add Green Corridor Viz Layers if active
        if active_corridor:
            gc_layers = visualize_corridor_on_map(
                route=active_corridor.get("route", []),
                vehicle_type=active_corridor.get("vehicle_type", "Ambulance")
            )
            layers.extend(gc_layers)

        view = pdk.ViewState(
            latitude=12.9666,
            longitude=77.6046,
            zoom=14,
            pitch=45,
            bearing=0,
        )

        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view,
            map_style="mapbox://styles/mapbox/dark-v11",
            tooltip={"text": "{name}"},
        )
        st.pydeck_chart(deck, use_container_width=True)

    with heat_col:
        st.markdown("""
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">🌡️ Congestion Heatmap</div>
                <span class="panel-badge badge-active">ANALYTICS</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Heatmap grid per intersection
        for int_id, densities in density_map.items():
            avg_d = sum(densities.values()) / max(1, len(densities))
            level, color = _density_level(int(avg_d * 50))

            st.markdown(f"""
            <div style="background:#1a2236;border:1px solid #1e293b;border-radius:10px;padding:10px;margin-bottom:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-size:13px;font-weight:700;color:#f1f5f9;">{int_id}</span>
                    <span style="font-size:11px;font-weight:700;color:{color};padding:2px 8px;background:{color}20;border-radius:999px;">{level}</span>
                </div>
            """, unsafe_allow_html=True)

            dir_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;">'
            for direction in ["north", "south", "east", "west"]:
                d = densities.get(direction, 0)
                pct = int(d * 100)
                dc = "#ef4444" if d >= 0.7 else "#f59e0b" if d >= 0.4 else "#22c55e"
                dir_html += f"""
                <div style="text-align:center;background:#111827;border-radius:6px;padding:6px;">
                    <div style="font-size:10px;color:#94a3b8;">{direction[0].upper()}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:800;color:{dc};">{pct}%</div>
                </div>"""
            dir_html += "</div></div>"
            st.markdown(dir_html, unsafe_allow_html=True)

        # Intersection grid summary
        st.markdown("""
        <div class="panel" style="margin-top:8px;">
            <div class="panel-header">
                <div class="panel-title">📊 Grid Status</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        stats = [
            ("AI Ticks", grid_state.get("tick_count", 0), "#06b6d4"),
            ("Active Corridors", 1 if active_corridor else 0, "#ef4444" if active_corridor else "#22c55e"),
            ("Intersections", len(brain.intersections), "#8b5cf6"),
            ("Decisions", len(decision_log), "#f59e0b"),
        ]
        stats_html = '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px;">'
        for label, val, col in stats:
            stats_html += f"""
            <div style="background:#1a2236;border:1px solid #1e293b;border-radius:8px;padding:8px;text-align:center;">
                <div style="font-size:10px;color:#94a3b8;">{label}</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:800;color:{col};">{val}</div>
            </div>"""
        stats_html += "</div>"
        st.markdown(stats_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
