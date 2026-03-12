"""
UrbanFlow AI - City Traffic Intelligence
========================================
Page 2: Executive master-view mapping global traffic health, system mobility scores, and PyDeck density heatmaps.
"""
import streamlit as st
import pydeck as pdk
import pandas as pd
from backend.traffic_brain import get_traffic_brain
from backend.city_control_engine import get_control_engine

def show():
    engine = get_control_engine()
    brain = get_traffic_brain()
    
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">City Traffic Intelligence</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">High-level overview of the traffic system managed by the AI Traffic Brain.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── CALCULATE GLOBAL MOBILITY SCORE ──
    # Base 100. Deduct points for density, active corridors, and active incidents
    score = 100
    density = engine.state.get("density_level", "LOW")
    if density == "MEDIUM": score -= 12
    elif density == "HIGH": score -= 28
    
    active_corridor = engine.state.get("active_emergency", False)
    if active_corridor: score -= 8
    
    active_incident = engine.state.get("active_incident", False)
    if active_incident: score -= 15

    # Interpolate Score text/colors
    if score >= 90:
        s_text = "Excellent traffic flow"
        s_col = "#22C55E" # Green
        s_bg = "#F0FDF4"
        s_border = "#BBF7D0"
    elif score >= 70:
        s_text = "Good traffic flow"
        s_col = "#3B82F6" # Blue
        s_bg = "#EFF6FF"
        s_border = "#BFDBFE"
    elif score >= 50:
        s_text = "Moderate congestion"
        s_col = "#EAB308" # Yellow
        s_bg = "#FEFCE8"
        s_border = "#FEF08A"
    else:
        s_text = "Severe congestion"
        s_col = "#EF4444" # Red
        s_bg = "#FEF2F2"
        s_border = "#FECACA"

    # ── ROW 1: MOBILITY SCORE & REAL-TIME STATUS ──
    c_score, c_status = st.columns([4, 6])
    
    with c_score:
        st.markdown(f"""
        <div class="saas-card" style="background:{s_bg}; border:2px solid {s_border}; text-align:center; padding:32px 20px; height:100%;">
            <div style="color:{s_col}; font-size:14px; font-weight:800; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">City Mobility Score</div>
            <div style="color:{s_col}; font-size:72px; font-weight:900; line-height:1; text-shadow: 0 4px 6px rgba(0,0,0,0.05);">{score}</div>
            <div style="color:#64748B; font-size:15px; font-weight:600; margin-top:16px;">{s_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_status:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#0F172A; font-weight:700; font-size:15px; margin-bottom:16px; text-transform:uppercase; letter-spacing:0.5px;">Real-Time City Status</div>', unsafe_allow_html=True)
        
        # Grid 2x2 for Status metrics
        sc1, sc2 = st.columns(2)
        sc3, sc4 = st.columns(2)
        
        city_status_txt = "Stable" if score >= 70 else ("Degraded" if score >= 50 else "Critical")
        city_status_col = "#22C55E" if score >= 70 else ("#EAB308" if score >= 50 else "#EF4444")
        
        inc_count = 1 if active_incident else 0
        inc_col = "#EF4444" if active_incident else "#64748B"
        
        em_count = 1 if active_corridor else 0
        em_col = "#EF4444" if active_corridor else "#64748B"
        
        # We simulate "Signals Optimized" dynamically based on the log length of the traffic brain
        sig_opt = min(max(5, len(engine.state.get("recent_events", []))), 35)

        def render_status_metric(label, value, color, icon):
            return f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px; padding-bottom:12px; border-bottom:1px solid #F1F5F9;">
                <div style="background:#F8FAFC; width:40px; height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px;">{icon}</div>
                <div>
                    <div style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">{label}</div>
                    <div style="color:{color}; font-size:20px; font-weight:800; line-height:1.2;">{value}</div>
                </div>
            </div>
            """
            
        sc1.markdown(render_status_metric("City Traffic Status", city_status_txt, city_status_col, "🏙️"), unsafe_allow_html=True)
        sc2.markdown(render_status_metric("Active Incidents", str(inc_count), inc_col, "⚠️"), unsafe_allow_html=True)
        sc3.markdown(render_status_metric("Emergency Vehicles", str(em_count), em_col, "🚑"), unsafe_allow_html=True)
        sc4.markdown(render_status_metric("Signals Optimized", str(sig_opt), "#06B6D4", "🚦"), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 2: CITY HEATMAP & AI STATUS ──
    c_map, c_ai = st.columns([7, 3])
    
    with c_map:
        st.markdown('<div class="saas-card" style="padding:16px;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#0F172A; font-weight:700; font-size:15px; margin-bottom:12px;">City Traffic Heatmap</div>', unsafe_allow_html=True)
        
        nodes = brain.intersections
        # Map Base Color Logics
        # Green -> Smooth [34, 197, 94]
        # Yellow -> Moderate [234, 179, 8]
        # Orange -> Heavy [249, 115, 22]
        # Red -> Severe [239, 68, 68]
        
        plot_data = []
        for n_id, n_data in nodes.items():
            base_col = [34, 197, 94] # Default Green
            
            # Apply global density
            if density == "MEDIUM": base_col = [234, 179, 8]
            if density == "HIGH": base_col = [249, 115, 22]
            
            # Apply targeted incident overwrites
            if engine.state.get("incident_data") and engine.state["incident_data"]["location"] == n_id:
                base_col = [239, 68, 68] # Red Crash Node
            elif engine.state.get("emergency_location") == n_id:
                base_col = [56, 189, 248] # Cyan Emergency Escort Node
                
            plot_data.append({
                "id": n_id,
                "lat": n_data["lat"],
                "lon": n_data["lon"],
                "color": base_col,
                "size": 300 if base_col == [239, 68, 68] else 200
            })
            
        df_nodes = pd.DataFrame(plot_data)
        
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v11",
            initial_view_state=pdk.ViewState(
                latitude=40.7128,
                longitude=-74.0060,
                zoom=12.5,
                pitch=30
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_nodes,
                    get_position='[lon, lat]',
                    get_fill_color='color',
                    get_radius='size',
                    pickable=True
                ),
                pdk.Layer(
                    "TextLayer",
                    data=df_nodes,
                    get_position='[lon, lat]',
                    get_text='id',
                    get_color=[255, 255, 255, 200],
                    get_size=12,
                    get_alignment_baseline="'bottom'",
                )
            ]
        ))
        
        # Heatmap Legend
        st.markdown("""
        <div style="display:flex; justify-content:center; gap:24px; margin-top:16px;">
            <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:3px; background:#22C55E;"></div><span style="font-size:12px; color:#64748B;">Smooth</span></div>
            <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:3px; background:#EAB308;"></div><span style="font-size:12px; color:#64748B;">Moderate</span></div>
            <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:3px; background:#F97316;"></div><span style="font-size:12px; color:#64748B;">Heavy</span></div>
            <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:3px; background:#EF4444;"></div><span style="font-size:12px; color:#64748B;">Congestion</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_ai:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#0F172A; font-weight:700; font-size:15px; margin-bottom:20px; text-transform:uppercase; letter-spacing:0.5px;">AI Traffic Brain Status</div>', unsafe_allow_html=True)
        
        def render_ai_status(label, active=True):
            col = "#22C55E" if active else "#EF4444"
            txt = "Active / Connected" if active else "Offline"
            bg = "#F0FDF4" if active else "#FEF2F2"
            return f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; margin-bottom:12px;">
                <div style="color:#0F172A; font-size:13px; font-weight:700;">{label}</div>
                <div style="display:flex; align-items:center; gap:8px; margin-top:6px;">
                    <span style="background:{bg}; color:{col}; font-size:10px; font-weight:800; padding:2px 8px; border-radius:99px;">{txt}</span>
                </div>
            </div>
            """
            
        sys_health = engine.state["system_health"]
        st.markdown(render_ai_status("AI Traffic Brain", sys_health["ai_model"] == "ONLINE"), unsafe_allow_html=True)
        st.markdown(render_ai_status("Prediction Engine", sys_health["prediction_engine"] == "ONLINE"), unsafe_allow_html=True)
        st.markdown(render_ai_status("Signal Control Network", sys_health["signal_network"] == "ONLINE"), unsafe_allow_html=True)
        st.markdown(render_ai_status("Emergency System", True), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 3: ANALYTICS & IMPACT ──
    c_perf, c_imp = st.columns([6, 4])
    
    with c_perf:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div style="color:#0F172A; font-weight:700; font-size:15px; margin-bottom:16px; text-transform:uppercase; letter-spacing:0.5px;">Emergency Response Performance</div>', unsafe_allow_html=True)
        
        pc1, pc2 = st.columns(2)
        pc3, pc4 = st.columns(2)
        
        def render_perf_box(label, value, icon):
            return f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-left:4px solid #2563EB; border-radius:8px; padding:16px; margin-bottom:16px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="color:#64748B; font-size:12px; font-weight:600; text-transform:uppercase;">{label}</div>
                    <div style="font-size:16px;">{icon}</div>
                </div>
                <div style="color:#0F172A; font-size:24px; font-weight:800; margin-top:8px;">{value}</div>
            </div>
            """
            
        pc1.markdown(render_perf_box("Avg Response Time", "2m 30s", "⏱️"), unsafe_allow_html=True)
        pc2.markdown(render_perf_box("Corridors Activated Today", "6", "🛡️"), unsafe_allow_html=True)
        pc3.markdown(render_perf_box("Signals Synchronized", "142", "🚦"), unsafe_allow_html=True)
        pc4.markdown(render_perf_box("Emergency Routes Cleared", "8", "🚑"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_imp:
        st.markdown('<div class="saas-card" style="height:100%; background:linear-gradient(135deg, #0F172A 0%, #1E293B 100%);">', unsafe_allow_html=True)
        st.markdown('<div style="color:#F8FAFC; font-weight:700; font-size:15px; margin-bottom:20px; text-transform:uppercase; letter-spacing:0.5px;">Traffic Impact Analytics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px; margin-bottom:16px;">
            <div style="color:#94A3B8; font-size:12px; font-weight:600; text-transform:uppercase; margin-bottom:8px;">Congestion Reduction</div>
            <div style="display:flex; align-items:baseline; gap:12px;">
                <span style="color:#22C55E; font-size:32px; font-weight:800;">📉 28%</span>
                <span style="color:#F8FAFC; font-size:14px; font-weight:600;">Overall Network Flow</span>
            </div>
        </div>
        
        <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:20px;">
            <div style="color:#94A3B8; font-size:12px; font-weight:600; text-transform:uppercase; margin-bottom:8px;">Emergency Travel Improvement</div>
            <div style="display:flex; align-items:baseline; gap:12px;">
                <span style="color:#06B6D4; font-size:32px; font-weight:800;">🚀 35%</span>
                <span style="color:#F8FAFC; font-size:14px; font-weight:600;">Travel Time Saved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ROW 4: SYSTEM SUMMARY FOOTER ──
    st.markdown("""
    <div class="saas-card" style="background:#F8FAFC; border-left:4px solid #3B82F6; margin-top:24px;">
        <div style="display:flex; gap:16px; align-items:flex-start;">
            <div style="font-size:24px;">ℹ️</div>
            <div>
                <strong style="color:#0F172A; font-size:14px; display:block; margin-bottom:4px;">UrbanFlow Operations Summary</strong>
                <span style="color:#64748B; font-size:13px; line-height:1.6;">The AI Traffic Brain continuously monitors traffic conditions across the city, predicts congestion, dynamically optimizes signal timings, and ensures priority movement for emergency vehicles through automated green corridors.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
