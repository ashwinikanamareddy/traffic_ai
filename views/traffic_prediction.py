"""
UrbanFlow AI - Traffic Prediction AI
====================================
Page 4: Predictive modeling and congestion forecasting across multiple time horizons.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from backend.traffic_brain import get_traffic_brain

def show():
    brain = get_traffic_brain()
    
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Traffic Prediction AI</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Machine learning forecasts for proactive congestion prevention and signal optimization.</p>
    </div>
    """, unsafe_allow_html=True)

    # Simulated Prediction Data Model
    predictions = {
        "INT-01": {"traffic": "LOW", "color": "#22C55E", "bg": "#F0FDF4", "border": "#BBF7D0", "load": 24, "status": "Smooth"},
        "INT-02": {"traffic": "MEDIUM", "color": "#F59E0B", "bg": "#FEF3C7", "border": "#FDE68A", "load": 58, "status": "Moderate"},
        "INT-03": {"traffic": "HIGH", "color": "#EF4444", "bg": "#FEF2F2", "border": "#FECACA", "load": 89, "status": "Heavy"},
        "INT-04": {"traffic": "MEDIUM", "color": "#F59E0B", "bg": "#FEF3C7", "border": "#FDE68A", "load": 42, "status": "Moderate"}
    }

    # --- 1. Traffic Forecast Panel ---
    st.markdown('<div class="saas-card" style="margin-bottom:24px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px;">AI Traffic Forecast (Next 5 Minutes)</h4>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for i, (int_id, data) in enumerate(predictions.items()):
        with cols[i]:
            st.markdown(f"""
            <div style="background:{data['bg']}; border:1px solid {data['border']}; border-radius:12px; padding:16px; text-align:center;">
                <div style="font-size:12px; font-weight:700; color:#475569; margin-bottom:8px;">{int_id}</div>
                <div style="font-size:16px; font-weight:800; color:{data['color']};">{data['traffic']}</div>
                <div style="font-size:11px; color:#64748B; margin-top:4px;">Predicted traffic</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col_chart, col_actions = st.columns([6, 4])
    
    # --- 2. Congestion Prediction Chart (Plotly) ---
    with col_chart:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px;">Congestion Evolution Timeline</h4>', unsafe_allow_html=True)
        
        # Plotly Time-horizon Line Chart
        time_horizons = ['1 min', '3 min', '5 min', '10 min']
        
        fig = go.Figure()
        
        # Define the Y-axis conceptually: 0=Low, 50=Medium, 100=High
        fig.add_trace(go.Scatter(x=time_horizons, y=[20, 25, 24, 30], mode='lines+markers', name='INT-01', line=dict(color='#22C55E', width=3)))
        fig.add_trace(go.Scatter(x=time_horizons, y=[45, 50, 58, 48], mode='lines+markers', name='INT-02', line=dict(color='#F59E0B', width=3)))
        fig.add_trace(go.Scatter(x=time_horizons, y=[60, 75, 89, 95], mode='lines+markers', name='INT-03', line=dict(color='#EF4444', width=4, dash='solid')))
        fig.add_trace(go.Scatter(x=time_horizons, y=[30, 35, 42, 38], mode='lines+markers', name='INT-04', line=dict(color='#3B82F6', width=3)))

        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=250,
            xaxis=dict(title="Time Horizon", gridcolor="#F1F5F9", showgrid=True),
            yaxis=dict(
                title="Predicted Density", 
                gridcolor="#F1F5F9", 
                showgrid=True,
                tickmode='array',
                tickvals=[20, 50, 85],
                ticktext=['Low', 'Medium', 'High Congestion']
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- 3. AI Preventive Actions ---
    with col_actions:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px;">AI Preventive Actions</h4>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#F8FAFC; border-left:4px solid #3B82F6; padding:16px; border-radius:4px; margin-bottom:12px;">
            <div style="color:#1E3A8A; font-weight:700; font-size:13px; margin-bottom:4px;">Automated Signal Adjustment</div>
            <div style="color:#475569; font-size:13px;">AI predicted congestion at <strong>INT-03</strong> in the next 5 minutes. Signal timing adjusted to reduce upcoming traffic buildup.</div>
        </div>
        
        <div style="background:#F8FAFC; border-left:4px solid #10B981; padding:16px; border-radius:4px;">
            <div style="color:#065F46; font-weight:700; font-size:13px; margin-bottom:4px;">Dynamic Traffic Rerouting</div>
            <div style="color:#475569; font-size:13px;">AI actively rerouted approach vehicles from INT-03 to alternative intersection <strong>INT-02</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 4. Intersection Load Visualization & 5. Proactive Signal Optimization ---
    st.markdown('<div class="saas-card" style="margin-top:24px; margin-bottom:24px;">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:15px;">Predicted Volume Load</h4>', unsafe_allow_html=True)
        load_html = '<div style="display:flex; flex-direction:column; gap:8px;">'
        for int_id, data in predictions.items():
            load_html += f"""
            <div style="display:flex; justify-content:space-between; align-items:center; background:#F8FAFC; border:1px solid #E2E8F0; padding:12px 16px; border-radius:8px;">
                <div style="font-weight:700; color:#0F172A; font-size:14px;">{int_id} <span style="font-weight:600; font-size:12px; color:#64748B; margin-left:8px;">→ {data['status']}</span></div>
                <div style="font-weight:800; color:{data['color']}; font-size:15px;">{data['load']}% Load</div>
            </div>
            """
        load_html += '</div>'
        st.markdown(load_html, unsafe_allow_html=True)

    with c2:
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:15px;">Proactive Signal Optimization</h4>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#EEF2FF; border:1px solid #C7D2FE; border-radius:12px; padding:24px; text-align:center; height:calc(100% - 32px); display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:32px; margin-bottom:12px;">⚡</div>
            <div style="color:#3730A3; font-weight:800; font-size:16px; margin-bottom:8px;">AI adjusted signal timing in advance to prevent congestion.</div>
            <div style="color:#4F46E5; font-size:14px; font-weight:600; padding:8px; background:white; border-radius:8px; display:inline-block; margin:0 auto;">
                Signal duration increased for north lane by <span style="font-weight:800; color:#1E3A8A;">10 seconds</span>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- 6. PyDeck Map Visualization ---
    st.markdown('<div class="saas-card" style="margin-bottom:24px; position:relative; padding:0; overflow:hidden;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="position:absolute; top:20px; left:20px; z-index:10; background:rgba(255,255,255,0.95); padding:16px; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.1); border:1px solid #E2E8F0;">
        <div style="font-size:12px; font-weight:800; color:#0F172A; text-transform:uppercase; margin-bottom:8px;">Prediction Map Legend</div>
        <div style="display:flex; align-items:center; gap:8px; font-size:12px; color:#475569; font-weight:600;"><div style="width:10px;height:10px;border-radius:50%;background:#22C55E;"></div> Smooth Traffic</div>
        <div style="display:flex; align-items:center; gap:8px; font-size:12px; color:#475569; font-weight:600;"><div style="width:10px;height:10px;border-radius:50%;background:#EAB308;"></div> Moderate Traffic</div>
        <div style="display:flex; align-items:center; gap:8px; font-size:12px; color:#475569; font-weight:600;"><div style="width:10px;height:10px;border-radius:50%;background:#EF4444;box-shadow:0 0 10px #EF4444;"></div> Predicted Congestion Zone</div>
    </div>
    """, unsafe_allow_html=True)

    map_nodes = [
        {"name": "INT-01", "position": [80.2707, 13.0827], "color": [34, 197, 94, 200], "radius": 60},
        {"name": "INT-02", "position": [80.2718, 13.0850], "color": [234, 179, 8, 200], "radius": 70},
        {"name": "INT-03", "position": [80.2750, 13.0870], "color": [239, 68, 68, 200], "radius": 120},  # Glowing predicted congestion
        {"name": "INT-04", "position": [80.2780, 13.0840], "color": [234, 179, 8, 200], "radius": 70}
    ]
    df_map = pd.DataFrame(map_nodes)

    layer_glow = pdk.Layer(
        "ScatterplotLayer",
        df_map[df_map["name"] == "INT-03"],
        get_position="position",
        get_color=[239, 68, 68, 50],
        get_radius=250,
        pickable=False
    )
    
    layer_nodes = pdk.Layer("ScatterplotLayer", df_map, get_position="position", get_color="color", get_radius="radius", pickable=True)
    layer_text = pdk.Layer("TextLayer", df_map, get_position="position", get_text="name", get_color=[15, 23, 42, 255], get_size=14, get_alignment_baseline="'bottom'")

    view_state = pdk.ViewState(latitude=13.085, longitude=80.274, zoom=14.8, pitch=35)
    r = pdk.Deck(layers=[layer_glow, layer_nodes, layer_text], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10")
    
    st.markdown('<div class="saas-map-container" style="height:400px;">', unsafe_allow_html=True)
    st.pydeck_chart(r, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- 8. Explanation Panel ---
    st.markdown('<div class="saas-card" style="background:#F8FAFC; border:1px solid #E2E8F0;">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#0F172A; margin:0 0 8px 0; font-size:15px;">🧠 Traffic Prediction Engine Architecture</h4>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569; font-size:14px; margin:0; line-height:1.6;">The AI Traffic Brain analyzes traffic density patterns and predicts congestion trends using machine learning models. Based on predictions, signal timing and routing strategies are adjusted proactively to prevent traffic jams before they materialize across the urban grid.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
