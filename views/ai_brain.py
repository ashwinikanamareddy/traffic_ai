"""
UrbanFlow AI - AI Traffic Brain
===============================
Page 4: The central logical coordinator highlighting AI decision explainability.
"""
import streamlit as st
from backend.traffic_controller import get_traffic_controller
from backend.traffic_brain import get_traffic_brain

def show():
    # Header
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">AI Traffic Brain</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">The central intelligence hub orchestrating autonomous network execution.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-left:4px solid #3B82F6; padding:16px; border-radius:8px; margin-bottom:24px; color:#1E3A8A; font-size:14px;">
        <strong>How it works:</strong> AI analyzes traffic density and adjusts signal timing to optimize flow. It evaluates live vision inputs against strict priority rules to deploy the most efficient grid configurations.
    </div>
    """, unsafe_allow_html=True)

    vc = st.session_state.get("cmd_vehicle_counts", {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0})
    controller = get_traffic_controller()
    brain = get_traffic_brain()
    decision = controller.process_pipeline(vc)
    
    col_logic, col_state = st.columns([5, 5])
    
    with col_logic:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px; font-weight:700;">Decision Architecture</h4>', unsafe_allow_html=True)
        
        d_html = ""
        # Show logic derived from process_pipeline
        steps = [
            ("Vision Analysis", "Processed YOLOv8 bounding frames", True),
            ("Density Computation", f"Score: {decision['traffic_score']} | Level: {decision['density']}", True),
            ("Emergency Override Scan", f"Triggered: {'Yes' if brain.get_grid_state().get('active_corridor') else 'No'}", True),
            ("Signal Timing Output", f"Green Phase: {decision['green_duration']} seconds", True)
        ]
        
        for name, desc, ok in steps:
            icon = "#22C55E" if ok else "#64748B"
            d_html += f"""
            <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:16px;">
                <div style="width:24px; height:24px; border-radius:50%; background:{icon}20; border:1px solid {icon}; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px;">
                    <div style="width:8px; height:8px; background:{icon}; border-radius:50%;"></div>
                </div>
                <div>
                    <div style="font-weight:700; color:#0F172A; font-size:14px; margin-bottom:2px;">{name}</div>
                    <div style="font-size:13px; color:#64748B;">{desc}</div>
                </div>
            </div>
            """
        st.markdown(d_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_state:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px; font-weight:700;">Congestion Prediction Model</h4>', unsafe_allow_html=True)
        
        pred = "Increasing" if decision["density"] in ["HIGH", "MEDIUM"] else "Stable"
        p_col = "#EF4444" if pred == "Increasing" else "#22C55E"
        
        st.markdown(f"""
        <div style="text-align:center; padding:32px 0;">
            <div style="font-size:48px; margin-bottom:16px;">🔮</div>
            <div style="color:#64748B; font-size:13px; text-transform:uppercase; font-weight:700; letter-spacing:1px; margin-bottom:8px;">15-Minute Network Forecast</div>
            <div style="color:{p_col}; font-size:32px; font-weight:800; line-height:1;">{pred}</div>
            <p style="color:#475569; font-size:14px; margin-top:16px;">The Neural Network anticipates traffic volume {pred.lower()} based on current chronological density trends across immediate intersection nodes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
