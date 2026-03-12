"""
UrbanFlow AI - Dashboard Overview
=================================
Page 1: Full system status of the 10-stage architecture.
"""
import streamlit as st
import datetime
from backend.traffic_brain import get_traffic_brain
from backend.city_control_engine import get_control_engine

def show():
    brain = get_traffic_brain()
    grid_state = brain.get_grid_state()
    active_corridor = grid_state.get("active_corridor")
    
    engine = get_control_engine()
    st.markdown("""
    <div style="margin-bottom:24px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Dashboard Overview</h2>
            <p style="color:#64748B; margin-top:0; font-size:14px;">Full system status mapping across the UrbanFlow AI network.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Global Demo Mode Toggle
    demo_status = engine.state.get("demo_mode", False)
    demo_toggle = st.toggle("🧪 Enable Pitch/Demo Mode (Simulated AI Telemetry)", value=demo_status)
    if demo_toggle != demo_status:
        engine.process_event("demo_mode_toggled", {"status": demo_toggle})
        st.rerun()

    # System Health Monitor
    st.markdown("<h3 style='color:#0F172A; font-size:14px; font-weight:700; margin-bottom:12px; margin-top:24px; text-transform:uppercase; letter-spacing:0.5px;'>Live System Health</h3>", unsafe_allow_html=True)
    health = engine.state["system_health"]
    
    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    def render_health_pill(label, status):
        color = "#22C55E" if status in ["ONLINE", "ACTIVE"] else ("#EAB308" if status == "SIMULATING" else "#EF4444")
        bg = "#F0FDF4" if status in ["ONLINE", "ACTIVE"] else ("#FEFCE8" if status == "SIMULATING" else "#FEF2F2")
        return f"""
        <div style="background:{bg}; border:1px solid {color}33; border-radius:8px; padding:12px; text-align:center;">
            <div style="color:#0F172A; font-size:12px; font-weight:700; margin-bottom:4px;">{label}</div>
            <div style="color:{color}; font-size:11px; font-weight:800; letter-spacing:0.5px;">{status}</div>
        </div>
        """
        
    hc1.markdown(render_health_pill("AI Model", health["ai_model"]), unsafe_allow_html=True)
    hc2.markdown(render_health_pill("YOLOv8 Engine", health["detection_engine"]), unsafe_allow_html=True)
    hc3.markdown(render_health_pill("Signal Network", health["signal_network"]), unsafe_allow_html=True)
    hc4.markdown(render_health_pill("Prediction", health["prediction_engine"]), unsafe_allow_html=True)
    hc5.markdown(render_health_pill("Incident Sys", health["incident_response"]), unsafe_allow_html=True)
    
    st.markdown("<hr style='margin:24px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    # Row 1 Grid
    with c1:
        st.markdown("""
        <div class="saas-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="color:#64748B; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Traffic Cameras</div>
                <div style="background:#EFF6FF; color:#2563EB; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">📹</div>
            </div>
            <div style="color:#0F172A; font-size:32px; font-weight:800; line-height:1;">4 / 4</div>
            <div style="color:#22C55E; font-size:13px; margin-top:12px; font-weight:600; display:flex; align-items:center; gap:4px;"><div style="width:6px; height:6px; background:#22C55E; border-radius:50%;"></div> Live Feed Connected</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        ints = len(brain.intersections)
        st.markdown(f"""
        <div class="saas-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="color:#64748B; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Intersections</div>
                <div style="background:#FFF7ED; color:#EA580C; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">🚦</div>
            </div>
            <div style="color:#0F172A; font-size:32px; font-weight:800; line-height:1;">{ints}</div>
            <div style="color:#64748B; font-size:13px; margin-top:12px; font-weight:600; display:flex; align-items:center; gap:4px;"><div style="width:6px; height:6px; background:#64748B; border-radius:50%;"></div> AI Optimization Active</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        em_count = 1 if active_corridor else 0
        bg_col = "#FEF2F2" if em_count > 0 else "#FFFFFF" # Clean white unless active
        text_col = "#EF4444" if em_count > 0 else "#0F172A"
        bd_col = "#FECACA" if em_count > 0 else "#E2E8F0"
        msg = "Tracking in realtime" if em_count > 0 else "No active emergencies"
        msg_col = "#EF4444" if em_count > 0 else "#64748B"
        st.markdown(f"""
        <div class="saas-card" style="background:{bg_col}; border-color:{bd_col};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="color:#64748B; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Emergency Vehicles</div>
                <div style="background:#FEF2F2; color:#EF4444; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">🚑</div>
            </div>
            <div style="color:{text_col}; font-size:32px; font-weight:800; line-height:1;">{em_count}</div>
            <div style="color:{msg_col}; font-size:13px; margin-top:12px; font-weight:600; display:flex; align-items:center; gap:4px;"><div style="width:6px; height:6px; background:{msg_col}; border-radius:50%;"></div> {msg}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Row 2 Grid
    with c4:
        st.markdown(f"""
        <div class="saas-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="color:#64748B; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Active Corridors</div>
                <div style="background:#F0FDF4; color:#22C55E; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">🛡️</div>
            </div>
            <div style="color:#0F172A; font-size:32px; font-weight:800; line-height:1;">{em_count}</div>
            <div style="color:#64748B; font-size:13px; margin-top:12px; font-weight:600; display:flex; align-items:center; gap:4px;"><div style="width:6px; height:6px; background:#64748B; border-radius:50%;"></div> Synchronized routes</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        st.markdown("""
        <div class="saas-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="color:#64748B; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Signal Timing</div>
                <div style="background:#FFFBEB; color:#D97706; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">⏱️</div>
            </div>
            <div style="color:#0F172A; font-size:32px; font-weight:800; line-height:1;">32s</div>
            <div style="color:#22C55E; font-size:13px; margin-top:12px; font-weight:600; display:flex; align-items:center; gap:4px;"><div style="width:6px; height:6px; background:#22C55E; border-radius:50%;"></div> ↓ 15% Wait Reduction</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c6:
        log_count = len(brain.get_decision_log(100))
        st.markdown(f"""
        <div class="saas-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <div style="color:#64748B; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">AI Decisions</div>
                <div style="background:#ECFEFF; color:#06B6D4; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px;">🧠</div>
            </div>
            <div style="color:#0F172A; font-size:32px; font-weight:800; line-height:1;">{log_count}</div>
            <div style="color:#64748B; font-size:13px; margin-top:12px; font-weight:600; display:flex; align-items:center; gap:4px;"><div style="width:6px; height:6px; background:#64748B; border-radius:50%;"></div> Autonomous adjustments</div>
        </div>
        """, unsafe_allow_html=True)
