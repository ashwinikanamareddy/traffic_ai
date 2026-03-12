"""
Emergency Grid Dashboard
========================
Premium, visually stunning dashboard showing:
1. Intersection Grid with animated signal states
2. Dynamic Signal Timing panels
3. Emergency Vehicle Alerts with pulsing animation
4. Green Corridor status visualization
5. AI Decision Log
6. Traffic Density Heatmap
"""

import time
from datetime import datetime

import streamlit as st

from backend.traffic_brain import get_traffic_brain


def _init_state():
    if "brain_initialized" not in st.session_state:
        st.session_state.brain_initialized = True
        brain = get_traffic_brain()
        # Initial tick so grid is populated
        brain.tick(detection_results={"vehicle_counts": {}, "total_vehicles": 0, "emergency_vehicles": []})


def _signal_color_css(state: str) -> str:
    colors = {
        "GREEN": "#22c55e",
        "RED": "#ef4444",
        "YELLOW": "#eab308",
    }
    return colors.get(state, "#6b7280")


def _signal_glow(state: str) -> str:
    glows = {
        "GREEN": "0 0 18px rgba(34,197,94,0.6)",
        "RED": "0 0 18px rgba(239,68,68,0.6)",
        "YELLOW": "0 0 18px rgba(234,179,8,0.6)",
    }
    return glows.get(state, "none")


def _density_color(density: float) -> str:
    if density >= 0.7:
        return "#ef4444"
    if density >= 0.4:
        return "#f59e0b"
    return "#22c55e"


def _density_label(density: float) -> str:
    if density >= 0.7:
        return "High"
    if density >= 0.4:
        return "Moderate"
    return "Low"


def show():
    _init_state()
    brain = get_traffic_brain()

    # Auto-tick the brain to keep it alive
    brain.tick(
        detection_results={
            "vehicle_counts": st.session_state.get("vehicle_counts", {}),
            "total_vehicles": sum(st.session_state.get("vehicle_counts", {}).values()),
            "emergency_vehicles": [],
        },
        elapsed_seconds=1.0,
    )

    grid_state = brain.get_grid_state()
    decision_log = brain.get_decision_log(last_n=25)
    density_heatmap = brain.get_density_heatmap()
    active_corridors = grid_state.get("active_corridors", [])

    # ──────────────── CSS ────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    .eg-root { font-family: 'Inter', sans-serif; padding: 4px 0; }

    .eg-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 18px;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .eg-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(34,197,94,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .eg-hero h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #ffffff, #22c55e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .eg-hero p {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 14px;
    }

    .eg-section-title {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
        margin: 18px 0 10px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .eg-section-title .icon {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }

    /* Intersection card */
    .int-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(15,23,42,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .int-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(15,23,42,0.1);
    }
    .int-card.emergency {
        border: 2px solid #ef4444;
        animation: pulse-border 1.5s ease-in-out infinite;
    }
    @keyframes pulse-border {
        0%, 100% { border-color: #ef4444; box-shadow: 0 0 0 0 rgba(239,68,68,0.3); }
        50% { border-color: #f87171; box-shadow: 0 0 20px 4px rgba(239,68,68,0.2); }
    }
    .int-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }
    .int-id {
        font-size: 16px;
        font-weight: 800;
        color: #0f172a;
    }
    .int-badge {
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 999px;
    }
    .int-badge.normal { background: #dcfce7; color: #166534; }
    .int-badge.emergency-badge { background: #fee2e2; color: #991b1b; animation: pulse-text 1s ease-in-out infinite; }
    @keyframes pulse-text {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    /* Signal light display */
    .signal-grid {
        display: grid;
        grid-template-areas:
            ". north ."
            "west center east"
            ". south .";
        grid-template-columns: 1fr 1fr 1fr;
        grid-template-rows: 1fr 1fr 1fr;
        gap: 4px;
        width: 130px;
        height: 130px;
        margin: 0 auto 12px auto;
    }
    .signal-light {
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 9px;
        font-weight: 700;
        color: white;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .signal-light.north { grid-area: north; }
    .signal-light.south { grid-area: south; }
    .signal-light.east { grid-area: east; }
    .signal-light.west { grid-area: west; }
    .signal-center {
        grid-area: center;
        background: #1e293b;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }

    /* Density bar */
    .density-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
        font-size: 12px;
        color: #475569;
    }
    .density-bar-wrap {
        flex: 1;
        height: 6px;
        background: #e5e7eb;
        border-radius: 999px;
        margin: 0 8px;
        overflow: hidden;
    }
    .density-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
    }

    /* Emergency alert banner */
    .emergency-banner {
        background: linear-gradient(135deg, #7f1d1d, #dc2626, #7f1d1d);
        background-size: 200% 200%;
        animation: gradient-shift 2s ease-in-out infinite;
        border-radius: 16px;
        padding: 18px 24px;
        color: white;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 14px;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .emergency-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        animation: pulse-icon 1.2s ease-in-out infinite;
        flex-shrink: 0;
    }
    @keyframes pulse-icon {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.15); }
    }
    .emergency-text h3 { margin: 0; font-size: 18px; font-weight: 800; }
    .emergency-text p { margin: 4px 0 0 0; font-size: 13px; opacity: 0.9; }

    /* Corridor card */
    .corridor-card {
        background: linear-gradient(135deg, #065f46, #059669);
        border-radius: 14px;
        padding: 16px 20px;
        color: white;
        margin-bottom: 10px;
        position: relative;
        overflow: hidden;
    }
    .corridor-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: corridor-shine 2s infinite;
    }
    @keyframes corridor-shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    .corridor-header { font-size: 16px; font-weight: 800; margin-bottom: 8px; }
    .corridor-meta { font-size: 12px; opacity: 0.85; }
    .corridor-path {
        display: flex;
        gap: 8px;
        margin-top: 10px;
        align-items: center;
        flex-wrap: wrap;
    }
    .corridor-node {
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 700;
    }
    .corridor-arrow {
        font-size: 16px;
        opacity: 0.7;
    }

    /* No corridor */
    .no-corridor {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        color: #64748b;
    }
    .no-corridor .icon { font-size: 28px; margin-bottom: 8px; }

    /* Decision log */
    .decision-item {
        display: flex;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 6px;
        font-size: 12px;
        align-items: flex-start;
    }
    .decision-item.emergency-decision {
        background: #fef2f2;
        border: 1px solid #fecaca;
    }
    .decision-item.timing-decision {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .decision-item.update-decision {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }
    .decision-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
        margin-top: 4px;
    }
    .decision-text {
        color: #334155;
        line-height: 1.4;
    }
    .decision-time {
        color: #94a3b8;
        font-size: 11px;
    }

    /* Heatmap cell */
    .heatmap-cell {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .heatmap-cell .label {
        font-size: 11px;
        color: #64748b;
        margin-bottom: 4px;
    }
    .heatmap-cell .value {
        font-size: 22px;
        font-weight: 800;
    }
    .heatmap-cell .level {
        font-size: 11px;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Timing bar */
    .timing-approach {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }
    .timing-label {
        width: 50px;
        font-size: 12px;
        font-weight: 700;
        color: #334155;
        text-transform: capitalize;
    }
    .timing-bar-outer {
        flex: 1;
        height: 22px;
        background: #f1f5f9;
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .timing-bar-inner {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 8px;
        font-size: 10px;
        font-weight: 700;
        color: white;
    }
    .timing-state {
        font-size: 12px;
        font-weight: 700;
        width: 50px;
        text-align: center;
        padding: 2px 6px;
        border-radius: 6px;
    }

    .panel-card {
        background: #ffffff;
        border: 1px solid #e7edf4;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(15,23,42,0.05);
        padding: 16px;
        margin-bottom: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ──────────────── HERO ────────────────
    st.markdown("""
    <div class="eg-root">
    <div class="eg-hero">
        <h1>🚨 Emergency Grid & AI Traffic Brain</h1>
        <p>Dynamic signal optimization • Emergency vehicle priority • Green corridor management</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ──────────────── EMERGENCY BANNER ────────────────
    if active_corridors:
        for corridor in active_corridors:
            vehicle_type = corridor.get("vehicle_type", "Emergency Vehicle").replace("_", " ").title()
            approach = corridor.get("approach", "unknown").title()
            remaining = corridor.get("remaining_seconds", 0)
            affected = corridor.get("affected_intersections", [])
            st.markdown(f"""
            <div class="emergency-banner">
                <div class="emergency-icon">🚑</div>
                <div class="emergency-text">
                    <h3>⚠️ EMERGENCY — {vehicle_type} Detected</h3>
                    <p>Green corridor active on {approach} approach • {len(affected)} intersection(s) cleared • {remaining:.0f}s remaining</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ──────────────── SIMULATE EMERGENCY ────────────────
    st.markdown('<div class="eg-section-title"><div class="icon" style="background:#fef2f2;color:#dc2626;">⚡</div>Emergency Simulation</div>', unsafe_allow_html=True)

    sim_cols = st.columns([2, 2, 2, 3])
    with sim_cols[0]:
        em_type = st.selectbox("Vehicle Type", ["ambulance", "fire_truck", "police"], key="sim_em_type")
    with sim_cols[1]:
        em_approach = st.selectbox("Approach", ["north", "south", "east", "west"], key="sim_em_approach")
    with sim_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚨 Simulate Emergency", type="primary", use_container_width=True):
            result = brain.simulate_emergency(vehicle_type=em_type, approach=em_approach)
            st.session_state["last_simulation"] = result
            st.rerun()
    with sim_cols[3]:
        if st.session_state.get("last_simulation"):
            sim = st.session_state["last_simulation"]
            st.success(f"✅ Corridor {sim['corridor']['corridor_id']} activated for {sim['vehicle_type']} on {sim['approach']}")

    # ──────────────── INTERSECTION GRID ────────────────
    st.markdown('<div class="eg-section-title"><div class="icon" style="background:#eff6ff;color:#2563eb;">🏙️</div>Intersection Grid — Live Signal States</div>', unsafe_allow_html=True)

    grid = grid_state.get("grid", [])
    for row in grid:
        cols = st.columns(len(row))
        for col_st, intersection in zip(cols, row):
            with col_st:
                int_id = intersection["id"]
                is_emergency = intersection.get("emergency_detected", False)
                signals = intersection.get("signal_states", {})
                timing = intersection.get("timing_plan", {})
                approaches = timing.get("approaches", {})

                card_class = "int-card emergency" if is_emergency else "int-card"
                badge_class = "int-badge emergency-badge" if is_emergency else "int-badge normal"
                badge_text = "🚨 EMERGENCY" if is_emergency else "✅ Normal"

                # Build signal lights HTML
                signal_html = '<div class="signal-grid">'
                for direction in ["north", "south", "east", "west"]:
                    state = signals.get(direction, "RED")
                    color = _signal_color_css(state)
                    glow = _signal_glow(state)
                    signal_html += f'<div class="signal-light {direction}" style="background:{color};box-shadow:{glow};">{direction[0].upper()}</div>'
                signal_html += '<div class="signal-center">🚗</div></div>'

                # Density bars
                density_html = ""
                for direction in ["north", "south", "east", "west"]:
                    approach_data = approaches.get(direction, {})
                    density = approach_data.get("density", 0)
                    pct = int(density * 100)
                    d_color = _density_color(density)
                    density_html += f"""
                    <div class="density-row">
                        <span>{direction[0].upper()}</span>
                        <div class="density-bar-wrap">
                            <div class="density-bar-fill" style="width:{pct}%;background:{d_color};"></div>
                        </div>
                        <span>{pct}%</span>
                    </div>"""

                st.markdown(f"""
                <div class="{card_class}">
                    <div class="int-card-header">
                        <span class="int-id">{int_id}</span>
                        <span class="{badge_class}">{badge_text}</span>
                    </div>
                    {signal_html}
                    {density_html}
                </div>
                """, unsafe_allow_html=True)

    # ──────────────── SIGNAL TIMING + CORRIDOR (side by side) ────────────────
    left_panel, right_panel = st.columns([3, 2])

    with left_panel:
        st.markdown('<div class="eg-section-title"><div class="icon" style="background:#f0fdf4;color:#16a34a;">⏱️</div>Dynamic Signal Timing</div>', unsafe_allow_html=True)

        # Show timing for first intersection as primary example
        first_int = list(brain.intersections.values())[0] if brain.intersections else None
        if first_int:
            plan = first_int.optimizer.get_timing_plan()
            current_phase = plan.get("current_phase", "north")
            override = plan.get("emergency_override")

            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown(f"**{first_int.id}** — Current Phase: **{current_phase.title()}**" + (f" ⚡ Override: **{override.title()}**" if override else ""), unsafe_allow_html=True)

            for approach in ["north", "south", "east", "west"]:
                approach_data = plan.get("approaches", {}).get(approach, {})
                green_dur = approach_data.get("green_duration", 15)
                state = approach_data.get("signal_state", "RED")
                density = approach_data.get("density", 0)
                pct = min(100, int((green_dur / 90.0) * 100))
                bar_color = _signal_color_css(state)
                state_bg = f"background:{bar_color};color:white;"

                st.markdown(f"""
                <div class="timing-approach">
                    <span class="timing-label">{approach.title()}</span>
                    <div class="timing-bar-outer">
                        <div class="timing-bar-inner" style="width:{pct}%;background:{bar_color};">{green_dur:.0f}s</div>
                    </div>
                    <span class="timing-state" style="{state_bg}">{state}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with right_panel:
        st.markdown('<div class="eg-section-title"><div class="icon" style="background:#ecfdf5;color:#059669;">🟢</div>Green Corridor Status</div>', unsafe_allow_html=True)

        if active_corridors:
            for c in active_corridors:
                path_html = ""
                for i, int_id in enumerate(c.get("affected_intersections", [])):
                    if i > 0:
                        path_html += '<span class="corridor-arrow">→</span>'
                    path_html += f'<span class="corridor-node">{int_id}</span>'

                remaining = c.get("remaining_seconds", 0)
                st.markdown(f"""
                <div class="corridor-card">
                    <div class="corridor-header">🟢 {c.get("corridor_id", "GC-???")}</div>
                    <div class="corridor-meta">
                        Vehicle: {c.get("vehicle_type", "unknown").replace("_", " ").title()} •
                        Approach: {c.get("approach", "?").title()} •
                        Remaining: {remaining:.0f}s
                    </div>
                    <div class="corridor-path">{path_html}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-corridor">
                <div class="icon">🛤️</div>
                <p><strong>No Active Corridors</strong></p>
                <p style="font-size:12px;">Corridors activate when emergency vehicles are detected</p>
            </div>
            """, unsafe_allow_html=True)

    # ──────────────── DECISION LOG + HEATMAP ────────────────
    log_col, heat_col = st.columns([3, 2])

    with log_col:
        st.markdown('<div class="eg-section-title"><div class="icon" style="background:#faf5ff;color:#7c3aed;">🧠</div>AI Decision Log</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)

        if decision_log:
            for entry in reversed(decision_log[-15:]):
                d_type = entry.get("type", "unknown")
                details = entry.get("details", {})
                ts = entry.get("timestamp", 0)
                time_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""

                if "emergency" in d_type:
                    css_class = "decision-item emergency-decision"
                    dot_color = "#ef4444"
                    text = f"🚨 {d_type.replace('_', ' ').title()}"
                    if "vehicle_type" in details:
                        text += f" — {details['vehicle_type'].replace('_', ' ').title()}"
                    if "approach" in details:
                        text += f" ({details['approach'].title()})"
                elif "timing" in d_type:
                    css_class = "decision-item timing-decision"
                    dot_color = "#22c55e"
                    text = f"⏱️ Signal timing recalculated"
                else:
                    css_class = "decision-item update-decision"
                    dot_color = "#94a3b8"
                    vehicles = details.get("total_vehicles", 0)
                    corridors = details.get("active_corridors", 0)
                    text = f"📊 Update: {vehicles} vehicles • {corridors} corridors"

                st.markdown(f"""
                <div class="{css_class}">
                    <div class="decision-dot" style="background:{dot_color};"></div>
                    <div>
                        <div class="decision-text">{text}</div>
                        <div class="decision-time">{time_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No decisions recorded yet. Upload a video or simulate an emergency.")

        st.markdown('</div>', unsafe_allow_html=True)

    with heat_col:
        st.markdown('<div class="eg-section-title"><div class="icon" style="background:#fff7ed;color:#ea580c;">🗺️</div>Traffic Density Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)

        for int_id, densities in density_heatmap.items():
            st.markdown(f"**{int_id}**", unsafe_allow_html=True)
            h_cols = st.columns(4)
            for i, direction in enumerate(["north", "south", "east", "west"]):
                density = densities.get(direction, 0)
                d_color = _density_color(density)
                d_label = _density_label(density)
                pct = int(density * 100)
                with h_cols[i]:
                    st.markdown(f"""
                    <div class="heatmap-cell">
                        <div class="label">{direction[0].upper()}</div>
                        <div class="value" style="color:{d_color};">{pct}%</div>
                        <div class="level" style="color:{d_color};">{d_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ──────────────── SYSTEM STATS ────────────────
    st.markdown('<div class="eg-section-title"><div class="icon" style="background:#f1f5f9;color:#475569;">📈</div>System Statistics</div>', unsafe_allow_html=True)

    stat_cols = st.columns(4)
    stats = [
        ("Total Ticks", grid_state.get("tick_count", 0), "#3b82f6"),
        ("Active Corridors", len(active_corridors), "#22c55e" if not active_corridors else "#ef4444"),
        ("Intersections", len(brain.intersections), "#8b5cf6"),
        ("Decisions Logged", len(decision_log), "#f59e0b"),
    ]
    for col_st, (label, value, color) in zip(stat_cols, stats):
        with col_st:
            st.markdown(f"""
            <div class="heatmap-cell">
                <div class="label">{label}</div>
                <div class="value" style="color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)
