"""
UrbanFlow AI - Traffic Scenario Simulator
=========================================
Page 10: Master control panel to force edge-case events and visualize AI Brain telemetry.
"""
import streamlit as st
import time
from backend.city_control_engine import get_control_engine

def show():
    engine = get_control_engine()
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Scenario Simulator</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Trigger real-world traffic crisis events to demonstrate the UrbanFlow AI Brain's dynamic adaptation capabilities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Grid 1: System Intelligence Panel
    st.markdown("<h3 style='color:#0F172A; font-size:16px; font-weight:700; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;'>System Intelligence Status</h3>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    # Get active scenario from session state
    scenario = st.session_state.get("active_scenario", "Normal")
    
    # Dynamic Coloring Rules
    proc_col = "#22C55E" if scenario == "Normal" else "#EAB308"
    proc_bg = "#F0FDF4" if scenario == "Normal" else "#FEFCE8"
    proc_tx = "Optimizing" if scenario == "Normal" else "Adapting Route"
    
    curr_em = "#64748B" if scenario != "Ambulance" else "#EF4444"
    curr_em_bg = "#F8FAFC" if scenario != "Ambulance" else "#FEF2F2"
    curr_em_tx = "Standby" if scenario != "Ambulance" else "Priority Active"
    
    curr_net = "#22C55E" if scenario not in ["Heavy Traffic", "Blockage", "Accident"] else "#EF4444"
    curr_net_bg = "#F0FDF4" if scenario not in ["Heavy Traffic", "Blockage", "Accident"] else "#FEF2F2"
    curr_net_tx = "Clear" if scenario not in ["Heavy Traffic", "Blockage", "Accident"] else "Congested Focus"

    with c1:
        st.markdown(f"""
        <div class="saas-card" style="padding:16px; background:{proc_bg}; border-color:#E2E8F0;">
            <div style="color:{proc_col}; font-size:24px; margin-bottom:8px;">🧠</div>
            <div style="color:#0F172A; font-weight:700; font-size:14px;">AI Processing</div>
            <div style="color:#64748B; font-size:12px; font-weight:600;">{proc_tx}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="saas-card" style="padding:16px; background:{curr_net_bg}; border-color:#E2E8F0;">
            <div style="color:{curr_net}; font-size:24px; margin-bottom:8px;">🚦</div>
            <div style="color:#0F172A; font-weight:700; font-size:14px;">Signal Network</div>
            <div style="color:#64748B; font-size:12px; font-weight:600;">{curr_net_tx}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="saas-card" style="padding:16px; background:{curr_em_bg}; border-color:#E2E8F0;">
            <div style="color:{curr_em}; font-size:24px; margin-bottom:8px;">🚨</div>
            <div style="color:#0F172A; font-weight:700; font-size:14px;">Emergency System</div>
            <div style="color:#64748B; font-size:12px; font-weight:600;">{curr_em_tx}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="saas-card" style="padding:16px; background:#EFF6FF; border-color:#E2E8F0;">
            <div style="color:#2563EB; font-size:24px; margin-bottom:8px;">📈</div>
            <div style="color:#0F172A; font-weight:700; font-size:14px;">Prediction Engine</div>
            <div style="color:#64748B; font-size:12px; font-weight:600;">Forecasting Active</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3 style='color:#0F172A; font-size:16px; font-weight:700; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;'>Simulation Triggers</h3>", unsafe_allow_html=True)
    
    # Interactive Sandbox Sandbox buttons
    b1, b2, b3, b4 = st.columns(4)
    
    if b1.button("🚑 Simulate Ambulance Emergency", use_container_width=True):
        st.session_state.active_scenario = "Ambulance"
        engine.process_event("ambulance_detected", {"location": "INT-03"})
        st.rerun()
    if b2.button("🚗 Simulate Heavy Traffic", use_container_width=True):
        st.session_state.active_scenario = "Heavy Traffic"
        engine.process_event("vehicle_detected", {"counts": {"cars": 150, "buses": 10}, "log_congestion": True})
        st.rerun()
    if b3.button("💥 Simulate Traffic Accident", use_container_width=True):
        st.session_state.active_scenario = "Accident"
        engine.process_event("incident_triggered", {"type": "Major Accident", "location": "INT-02"})
        st.rerun()
    if b4.button("🚧 Simulate Road Blockage", use_container_width=True):
        st.session_state.active_scenario = "Blockage"
        engine.process_event("incident_triggered", {"type": "Road Blockage", "location": "INT-05"})
        st.rerun()

    # Reset Control
    if st.button("🔄 Reset Global State & Clear Simulators"):
        st.session_state.active_scenario = "Normal"
        engine.process_event("emergency_cleared")
        engine.process_event("incident_cleared")
        st.rerun()

    st.markdown("<br><h3 style='color:#0F172A; font-size:16px; font-weight:700; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;'>AI Response Visualization Telemetry</h3>", unsafe_allow_html=True)
    
    # Output Console View (Matches the Scenario State Request)
    log_stream = []
    if scenario == "Normal":
        log_stream = [
            "AI System: Monitoring 6 active intersections.",
            "AI System: Signal intervals synchronized. Density nominal."
        ]
    elif scenario == "Ambulance":
        log_stream = [
            "<span style='color:#EF4444;'>AI Alert: 🚑 HIGH-PRIORITY AMBULANCE VEHICLE DETECTED AT INT-03</span>",
            "AI Calculation: Parsing hospital coordinate geometry...",
            "AI Decision: Route isolated. Triggering GREEN CORRIDOR protocol.",
            "<span style='color:#22C55E;'>AI Action: Forcing signals North-West to solid GREEN. Delaying lateral traffic flows by +35s.</span>",
            "AI Status: Hospital reception bay notified of ETA: 58 seconds."
        ]
    elif scenario == "Heavy Traffic":
        log_stream = [
            "<span style='color:#EAB308;'>AI Alert: ⚠️ HEAVY CONGESTION ACCUMULATION DETECTED (INT-02 NORTH LANE)</span>",
            "AI Processing: Calculating localized density load...",
            "AI Decision: Current queue density exceeds 89% capacity.",
            "AI Action: Signal duration increased dynamically +45s to flush excessive queue buildup.",
            "AI System: Monitoring adjacent corridors for spillback friction."
        ]
    elif scenario == "Accident":
        log_stream = [
            "<span style='color:#EF4444;'>AI Alert: 💥 MAJOR COLLISION DETECTED (INT-02 QUADRANT)</span>",
            "AI Engine: Immediate throughput loss detected. Lane friction 100%.",
            "AI Action: Alerting emergency services directly. ETA 3 minutes.",
            "<span style='color:#06B6D4;'>AI Decision: Commencing Master Reroute algorithm. Calculating alternative flow graphs.</span>",
            "AI Action: Upstream digital signage updated to divert inbound vehicles through INT-04 and INT-06 grid paths to avoid bottleneck."
        ]
    elif scenario == "Blockage":
        log_stream = [
            "<span style='color:#EA580C;'>AI Alert: 🚧 ROADWAY BLOCKAGE / CONSTRUCTION DETECTED (INT-05 SOUTH)</span>",
            "AI Processing: Analyzing flow diversion potential...",
            "AI Action: Adjusting local signal loops to favor East/West corridor escapes.",
            "<span style='color:#06B6D4;'>AI Decision: Vehicles actively rerouted to alternate parallel intersections limiting downstream delays to < 2 minutes.</span>"
        ]

    # Matrix render box
    box_html = '<div class="saas-card" style="background:#0F172A; border:none; padding:24px; font-family:monospace; min-height:200px;">'
    for line in log_stream:
        box_html += f"<div style='color:#94A3B8; font-size:14px; margin-bottom:8px;'>{line}</div>"
    box_html += '</div>'

    st.markdown(box_html, unsafe_allow_html=True)
