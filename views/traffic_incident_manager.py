"""
UrbanFlow AI - Traffic Incident Manager
=======================================
Page 11: Monitors and visualizes the AI Brain's self-healing capabilities during gridlock events.
"""
import streamlit as st
import pydeck as pdk
import pandas as pd
import time
from backend.traffic_brain import get_traffic_brain
from backend.city_control_engine import get_control_engine

def show():
    engine = get_control_engine()
    brain = get_traffic_brain()
    # Initialize the self-healing state
    if "incident_active" not in st.session_state:
        st.session_state.incident_active = False
        st.session_state.incident_stage = 0  # 0: None, 1: Detected, 2: Analyzed, 3: Rerouted, 4: Adjusted, 5: Restored
        
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Traffic Incident Manager</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Monitor the AI Traffic Brain's autonomous self-healing response to sudden grid disruptions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Top Control & Alert Banner ──
    col_alert, col_btn = st.columns([7, 3])
    
    with col_btn:
        st.markdown('<div class="saas-card" style="text-align:center; padding:16px;">', unsafe_allow_html=True)
        if not st.session_state.incident_active:
            if st.button("🚧 Simulate Traffic Accident", use_container_width=True, type="primary"):
                st.session_state.incident_active = True
                st.session_state.incident_stage = 1
                engine.process_event("incident_triggered", {"type": "Major Accident", "location": "INT-03"})
                st.rerun()
        else:
            if st.button("✅ Resolve Incident", use_container_width=True):
                st.session_state.incident_active = False
                st.session_state.incident_stage = 0
                engine.process_event("incident_cleared")
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_alert:
        if st.session_state.incident_active:
            st.markdown("""
            <div style="background:#FEF2F2; border:2px solid #FECACA; border-radius:12px; padding:16px 24px; display:flex; gap:20px; align-items:center; box-shadow:0 10px 15px -3px rgba(220,38,38,0.1);">
                <div style="font-size:40px; line-height:1;">⚠️</div>
                <div style="flex:1;">
                    <h3 style="color:#DC2626; margin:0 0 8px 0; font-size:18px; font-weight:800; text-transform:uppercase; letter-spacing:1px;">Traffic Incident Detected</h3>
                    <div style="display:flex; gap:24px;">
                        <div style="font-size:14px; color:#991B1B;"><strong style="color:#7F1D1D;">Type:</strong> Major Accident (100% Lane Blockage)</div>
                        <div style="font-size:14px; color:#991B1B;"><strong style="color:#7F1D1D;">Location:</strong> Intersection INT-03</div>
                        <div style="font-size:14px; color:#991B1B;"><strong style="color:#7F1D1D;">Severity:</strong> HIGH CRITICAL</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px; padding:16px 24px; display:flex; gap:20px; align-items:center;">
                <div style="font-size:40px; line-height:1;">✅</div>
                <div style="flex:1;">
                    <h3 style="color:#166534; margin:0 0 4px 0; font-size:18px; font-weight:800; text-transform:uppercase; letter-spacing:1px;">Network Stable</h3>
                    <div style="font-size:14px; color:#15803D;">All interconnected grids operating at optimal flow states. Zero critical incidents detected.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Progression Logic
    if st.session_state.incident_active and st.session_state.incident_stage < 5:
        time.sleep(1.5) # Simulate AI thinking time visually
        st.session_state.incident_stage += 1
        st.rerun()

    stage = st.session_state.incident_stage

    if stage > 0:
        st.markdown("<h3 style='color:#0F172A; font-size:16px; font-weight:700; margin-top:24px; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;'>AI Incident Response Pipeline</h3>", unsafe_allow_html=True)
        
        # Pipeline Nodes Visual
        colors = ["#E2E8F0"] * 5
        text_colors = ["#94A3B8"] * 5
        for i in range(stage):
            colors[i] = "#2563EB"
            text_colors[i] = "#1E3A8A"
            if i == 4: colors[i] = "#22C55E" # Final node green
            
        st.markdown(f"""
        <div class="saas-card" style="padding:24px; margin-bottom:24px;">
            <div style="display:flex; justify-content:space-between; position:relative; z-index:1;">
                <!-- Connecting Line -->
                <div style="position:absolute; top:20px; left:10%; right:10%; height:4px; background:#E2E8F0; z-index:-1;"></div>
                <div style="position:absolute; top:20px; left:10%; width:{min(100, (stage-1)*25)}%; height:4px; background:#2563EB; z-index:-1; transition:width 0.5s ease;"></div>
                
                <div style="display:flex; flex-direction:column; align-items:center; width:20%;">
                    <div style="width:44px; height:44px; border-radius:50%; background:{colors[0]}; color:white; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:20px; transition:all 0.3s; border:4px solid white;">1</div>
                    <div style="margin-top:12px; font-size:13px; font-weight:700; color:{text_colors[0]};">Incident Detected</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; width:20%;">
                    <div style="width:44px; height:44px; border-radius:50%; background:{colors[1]}; color:white; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:20px; transition:all 0.3s; border:4px solid white;">2</div>
                    <div style="margin-top:12px; font-size:13px; font-weight:700; color:{text_colors[1]};">Analyzed Impact</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; width:20%;">
                    <div style="width:44px; height:44px; border-radius:50%; background:{colors[2]}; color:white; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:20px; transition:all 0.3s; border:4px solid white;">3</div>
                    <div style="margin-top:12px; font-size:13px; font-weight:700; color:{text_colors[2]};">Rerouted Vehicles</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; width:20%;">
                    <div style="width:44px; height:44px; border-radius:50%; background:{colors[3]}; color:white; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:20px; transition:all 0.3s; border:4px solid white;">4</div>
                    <div style="margin-top:12px; font-size:13px; font-weight:700; color:{text_colors[3]};">Signals Adjusted</div>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; width:20%;">
                    <div style="width:44px; height:44px; border-radius:50%; background:{colors[4]}; color:white; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:20px; transition:all 0.3s; border:4px solid white;">✓</div>
                    <div style="margin-top:12px; font-size:13px; font-weight:700; color:{text_colors[4]};">Flow Restored</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── Middle Row: Maps and Telemetry ──
        c_map, c_data = st.columns([6, 4])
        
        with c_map:
            st.markdown('<div class="saas-card" style="padding:16px;">', unsafe_allow_html=True)
            st.markdown('<div style="color:#0F172A; font-weight:700; font-size:15px; margin-bottom:12px;">City Reroute Topology Map</div>', unsafe_allow_html=True)
            
            nodes = brain.intersections
            df_nodes = pd.DataFrame([{
                "id": n_id,
                "lat": n_data["lat"],
                "lon": n_data["lon"],
                "color": [239, 68, 68] if n_id == "INT-03" else ([234, 179, 8] if n_id in ["INT-02", "INT-04"] and stage >= 2 else [34, 197, 94]),
                "size": 300 if n_id == "INT-03" else 150
            } for n_id, n_data in nodes.items()])
            
            layers = [
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_nodes,
                    get_position='[lon, lat]',
                    get_fill_color='color',
                    get_radius='size',
                    pickable=True
                )
            ]
            
            # Add reroute line if stage >= 3
            if stage >= 3:
                df_lines = pd.DataFrame([{
                    "start": [nodes["INT-02"]["lon"], nodes["INT-02"]["lat"]],
                    "end": [nodes["INT-04"]["lon"], nodes["INT-04"]["lat"]],
                    "color": [6, 182, 212]
                }])
                layers.append(
                    pdk.Layer(
                        "LineLayer",
                        data=df_lines,
                        get_source_position="start",
                        get_target_position="end",
                        get_color="color",
                        get_width=10
                    )
                )

            st.pydeck_chart(pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v10",
                initial_view_state=pdk.ViewState(
                    latitude=40.7128,
                    longitude=-74.0060,
                    zoom=13,
                    pitch=45
                ),
                layers=layers
            ))
            
            st.markdown("""
            <div style="background:#FFFFFF; border-top:1px solid #E2E8F0; padding:12px; display:flex; gap:16px; justify-content:center; flex-wrap:wrap; margin-top:-16px; border-bottom-left-radius:12px; border-bottom-right-radius:12px;">
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; border-radius:50%; background:#22C55E;"></div><span style="font-size:11px; color:#475569; font-weight:600;">Clear</span></div>
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:10px; height:10px; border-radius:50%; background:#EF4444;"></div><span style="font-size:11px; color:#475569; font-weight:600;">Incident</span></div>
                <div style="display:flex; align-items:center; gap:6px;"><div style="width:20px; height:3px; border-radius:2px; background:#06B6D4;"></div><span style="font-size:11px; color:#475569; font-weight:600;">Reroute Path</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        with c_data:
            st.markdown('<div class="saas-card" style="padding:20px; height:100%;">', unsafe_allow_html=True)
            st.markdown('<div style="color:#0F172A; font-weight:700; font-size:15px; margin-bottom:16px;">AI Rerouting Logic</div>', unsafe_allow_html=True)
            
            if stage >= 2:
                st.markdown("""
                <div style="background:#FEFCE8; border-left:4px solid #EAB308; padding:12px; margin-bottom:16px;">
                    <strong style="color:#A16207; font-size:13px;">IMPACT ANALYSIS</strong>
                    <div style="color:#854D0E; font-size:13px; margin-top:4px;">100% capacity loss on Node INT-03. Upstream queue predicted to exceed 120 cars within 3 minutes.</div>
                </div>
                """, unsafe_allow_html=True)
                
            if stage >= 3:
                st.markdown("""
                <div style="background:#ECFEFF; border-left:4px solid #06B6D4; padding:12px; margin-bottom:16px;">
                    <strong style="color:#155E75; font-size:13px;">TOPOLOGICAL REROUTE</strong>
                    <div style="color:#164E63; font-size:13px; margin-top:4px;">Bypassing INT-03. Redirecting inbound vehicles across path <strong>INT-02 → INT-04</strong> to maintain sector flow.</div>
                </div>
                """, unsafe_allow_html=True)
                
            if stage >= 4:
                st.markdown("""
                <div style="background:#F3E8FF; border-left:4px solid #A855F7; padding:12px; margin-bottom:16px;">
                    <strong style="color:#6B21A8; font-size:13px;">SIGNAL OPTIMIZATION</strong>
                    <div style="color:#581C87; font-size:13px; margin-top:4px;">INT-02 Eastbound signal duration increased by <strong>+45s</strong>.<br>INT-04 Northbound cycle modified to accept overflow volume.</div>
                </div>
                """, unsafe_allow_html=True)

            if stage == 5:
                st.markdown("""
                <div style="background:#F0FDF4; border-left:4px solid #22C55E; padding:12px;">
                    <strong style="color:#166534; font-size:13px;">NETWORK RECOVERY METRICS</strong>
                    <ul style="color:#15803D; font-size:13px; margin-top:4px; margin-bottom:0; padding-left:20px;">
                        <li>Congestion stabilized (Reduced by 40%)</li>
                        <li>87 vehicles successfully rerouted</li>
                        <li>Surrounding grid flow returned to nominal 74% capacity</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)

    # Passive History Log (always visible)
    st.markdown("<h3 style='color:#0F172A; font-size:16px; font-weight:700; margin-top:24px; margin-bottom:12px; text-transform:uppercase; letter-spacing:0.5px;'>Recent Incident History</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="saas-card" style="padding:0;">
        <table style="width:100%; text-align:left; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #E2E8F0; background:#F8FAFC; color:#64748B; font-size:12px; text-transform:uppercase;">
                <th style="padding:16px 24px; font-weight:700;">Timestamp</th>
                <th style="padding:16px 24px; font-weight:700;">Incident Type</th>
                <th style="padding:16px 24px; font-weight:700;">Location</th>
                <th style="padding:16px 24px; font-weight:700;">AI Action</th>
                <th style="padding:16px 24px; font-weight:700;">Result</th>
            </tr>
            <tr style="border-bottom:1px solid #E2E8F0;">
                <td style="padding:16px 24px; color:#64748B; font-size:14px; font-weight:600;">10:42 AM</td>
                <td style="padding:16px 24px; color:#0F172A; font-size:14px; font-weight:600;">Vehicle Breakdown</td>
                <td style="padding:16px 24px; color:#64748B; font-size:14px;">INT-06 (South Lane)</td>
                <td style="padding:16px 24px; color:#64748B; font-size:14px;">Shifted lane allocation; dynamic signal timing</td>
                <td style="padding:16px 24px;"><span style="background:#F0FDF4; color:#166534; padding:4px 10px; border-radius:99px; font-size:11px; font-weight:700;">RESOLVED</span></td>
            </tr>
            <tr>
                <td style="padding:16px 24px; color:#64748B; font-size:14px; font-weight:600;">08:15 AM</td>
                <td style="padding:16px 24px; color:#0F172A; font-size:14px; font-weight:600;">Extreme Congestion</td>
                <td style="padding:16px 24px; color:#64748B; font-size:14px;">INT-01 (Highway Exit)</td>
                <td style="padding:16px 24px; color:#64748B; font-size:14px;">Activated extended green flush; delayed cross-traffic</td>
                <td style="padding:16px 24px;"><span style="background:#F0FDF4; color:#166534; padding:4px 10px; border-radius:99px; font-size:11px; font-weight:700;">RESOLVED</span></td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
