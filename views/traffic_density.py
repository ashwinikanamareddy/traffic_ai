"""
UrbanFlow AI - Traffic Density Analyzer
=======================================
Page 3: Deep dive into congestion scores, lane density, and analytical charting.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.traffic_controller import get_traffic_controller

def show():
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Traffic Density Analyzer</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Translating raw visual counts into actionable mathematical density scores.</p>
    </div>
    """, unsafe_allow_html=True)
    
    vc = st.session_state.get("cmd_vehicle_counts", {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0})
    controller = get_traffic_controller()
    decision = controller.process_pipeline(vc)
    
    score = decision["traffic_score"]
    density = decision["density"]
    
    c = "#EF4444" if density == "HIGH" else "#F59E0B" if density == "MEDIUM" else "#22C55E"
    
    col_kpi, col_chart = st.columns([3, 7])
    
    with col_kpi:
        st.markdown(f"""
        <div class="saas-card" style="margin-bottom:16px;">
            <div style="color:#64748B; font-size:12px; font-weight:700; text-transform:uppercase; margin-bottom:8px;">Calculated Congestion Score</div>
            <div style="display:flex; align-items:baseline; gap:8px;">
                <div style="color:#0F172A; font-size:48px; font-weight:800; line-height:1;">{score}</div>
                <div style="color:#64748B; font-size:14px; font-weight:600;">pts</div>
            </div>
            <div style="margin-top:16px; padding-top:16px; border-top:1px solid #F1F5F9;">
                <div style="color:#64748B; font-size:12px; font-weight:700; text-transform:uppercase; margin-bottom:8px;">Total Saturation Level</div>
                <div style="display:inline-block; background:{c}15; color:{c}; padding:6px 16px; border-radius:99px; font-size:14px; font-weight:800;">{density}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="saas-card">
            <h4 style="color:#0F172A; margin:0 0 16px 0; font-size:14px; font-weight:700;">Weight Map</h4>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:13px; color:#475569;"><span>Car</span> <span>1x</span></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:13px; color:#475569;"><span>Bike</span> <span>0.5x</span></div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px; font-size:13px; color:#475569;"><span>Bus/Truck</span> <strong style="color:#EF4444;">3x</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_chart:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 8px 0; font-size:16px; font-weight:700;">Density Trends (Vehicles per minute)</h4>', unsafe_allow_html=True)
        
        hist = st.session_state.get("cmd_density_history", [])
        if len(hist) > 2:
            df = pd.DataFrame(hist)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["frame"], y=df["total"],
                marker_color="#2563EB", marker_line_width=0
            ))
            fig.update_layout(
                height=260, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#64748B", size=12),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div style="height:250px; display:flex; align-items:center; justify-content:center; color:#94A3B8; font-size:14px;">Awaiting sufficient data frames...</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
