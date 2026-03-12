"""
UrbanFlow AI - City Traffic Map
===============================
Page 9: Map architecture defining intersections, congestion mapping, and isolated Emergency vehicles.
"""
import streamlit as st
import pydeck as pdk
import pandas as pd
from backend.traffic_brain import get_traffic_brain
from backend.green_corridor import visualize_corridor_on_map

def show():
    brain = get_traffic_brain()
    grid_state = brain.get_grid_state()
    active_corridor = grid_state.get("active_corridor")
    
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">City Traffic Map</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Live geographic rendering of the UrbanFlow AI network and precise target tracking.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="saas-card" style="position:relative; padding:0; overflow:hidden;">', unsafe_allow_html=True)
    
    # Legend Overlay
    st.markdown("""
    <div style="position:absolute; top:20px; left:20px; z-index:10; background:rgba(255,255,255,0.95); padding:16px; border-radius:12px; box-shadow:0 10px 15px -3px rgba(0,0,0,0.1); border:1px solid #E2E8F0; backdrop-filter:blur(4px);">
        <div style="font-size:12px; font-weight:800; color:#0F172A; text-transform:uppercase; margin-bottom:12px; letter-spacing:0.5px;">Network Legend</div>
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#475569; font-weight:600;">
                <div style="width:12px;height:12px;border-radius:50%;background:#22C55E;box-shadow:0 0 4px #22C55E60;"></div> Smooth Traffic
            </div>
            <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#475569; font-weight:600;">
                <div style="width:12px;height:12px;border-radius:50%;background:#EAB308;box-shadow:0 0 4px #EAB30860;"></div> Moderate Volume
            </div>
            <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#475569; font-weight:600;">
                <div style="width:12px;height:12px;border-radius:50%;background:#EF4444;box-shadow:0 0 4px #EF444460;"></div> Heavy Congestion
            </div>
            <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#475569; font-weight:600; margin-top:4px; padding-top:8px; border-top:1px solid #E2E8F0;">
                <div style="width:12px;height:12px;border-radius:50%;background:#DC2626; border:2px solid #FECACA; box-shadow:0 0 10px #DC2626;"></div> Active Ambulance
            </div>
            <div style="display:flex; align-items:center; gap:8px; font-size:13px; color:#475569; font-weight:600; margin-top:2px;">
                <div style="width:24px;height:4px;border-radius:2px;background:#22C55E;box-shadow:0 0 6px #22C55E;"></div> Green Corridor Path
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nodes_data = []
    ambulance_data = []
    
    coords_map = {
        "INT-01": [80.2707, 13.0827],
        "INT-02": [80.2718, 13.0850],
        "INT-03": [80.2750, 13.0870],
        "Hospital": [80.2800, 13.0900]
    }

    for nid, coords in coords_map.items():
        if nid == "Hospital":
            color = [37, 99, 235, 200]
            radius = 60
        else:
            intersection = brain.intersections.get(nid)
            den = intersection.get_state().get("state", {}).get("density", "LOW") if intersection else "LOW"
            if den == "HIGH":
                color = [239, 68, 68, 180] 
                radius = 80
            elif den == "MEDIUM":
                color = [234, 179, 8, 180]  
                radius = 60
            else:
                color = [34, 197, 94, 180]   
                radius = 50
                
        nodes_data.append({"name": nid, "position": coords, "color": color, "radius": radius})

    if active_corridor:
        route = active_corridor.get("route", [])
        story_state = active_corridor.get("story_state", "DETECTED")
        eta = active_corridor.get("eta", 58)
        
        # Calculate interpolated position
        if route and story_state != "IDLE":
            if story_state == "ARRIVED" or eta <= 0:
                current_pos = coords_map["Hospital"]
            elif story_state == "DETECTED":
                current_pos = coords_map.get(route[0], coords_map["INT-02"])
            else:
                # Interpolate across the 58 seconds
                progress = max(0.0, min(1.0, 1.0 - (eta / 58.0)))
                # Route has len(route) total segments
                total_segs = len(route)
                seg_float = progress * total_segs
                seg_idx = min(int(seg_float), total_segs - 1)
                
                start_n = route[seg_idx]
                end_n = route[seg_idx + 1] if seg_idx + 1 < len(route) else "Hospital"
                
                p1 = coords_map.get(start_n, coords_map["INT-02"])
                p2 = coords_map.get(end_n, coords_map["Hospital"])
                
                seg_prog = seg_float - seg_idx
                # Linear interpolate
                lat = p1[0] + (p2[0] - p1[0]) * seg_prog
                lon = p1[1] + (p2[1] - p1[1]) * seg_prog
                current_pos = [lat, lon]
                
            ambulance_data.append({
                "name": "🚑 Ambulance Detected",
                "position": current_pos,
                "color": [220, 38, 38, 255] # Explicit Red
            })

    df_nodes = pd.DataFrame(nodes_data)
    nodes_layer = pdk.Layer("ScatterplotLayer", df_nodes, get_position="position", get_color="color", get_radius="radius", pickable=True)
    text_layer = pdk.Layer("TextLayer", df_nodes, get_position="position", get_text="name", get_color=[15, 23, 42, 255], get_size=13, get_alignment_baseline="'bottom'")

    layers = [nodes_layer, text_layer]

    if active_corridor:
        layers.extend(visualize_corridor_on_map(active_corridor))
        if ambulance_data:
            df_amb = pd.DataFrame(ambulance_data)
            amb_layer = pdk.Layer(
                "ScatterplotLayer", df_amb,
                get_position="position", get_color="color", get_radius=100,
                pickable=True, stroked=True, get_line_color=[254, 202, 202, 255], line_width_min_pixels=3
            )
            layers.append(amb_layer)

    view_state = pdk.ViewState(latitude=13.085, longitude=80.275, zoom=14.5, pitch=45)

    r = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10", tooltip={"text": "{name}"})

    st.markdown('<div class="saas-map-container" style="height:600px;">', unsafe_allow_html=True)
    st.pydeck_chart(r, use_container_width=True)
    
    st.markdown("""
    <div style="background:#FFFFFF; border-top:1px solid #E2E8F0; padding:16px; display:flex; gap:24px; justify-content:center; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#22C55E;"></div><span style="font-size:12px; color:#475569; font-weight:600;">Smooth Traffic</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#EAB308;"></div><span style="font-size:12px; color:#475569; font-weight:600;">Moderate Traffic</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#F97316;"></div><span style="font-size:12px; color:#475569; font-weight:600;">Heavy Traffic</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:12px; height:12px; border-radius:50%; background:#EF4444;"></div><span style="font-size:12px; color:#475569; font-weight:600;">Severe Congestion</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><div style="width:24px; height:4px; border-radius:2px; background:#06B6D4;"></div><span style="font-size:12px; color:#475569; font-weight:600;">Emergency Corridor</span></div>
        <div style="display:flex; align-items:center; gap:8px;"><span style="font-size:14px;">🚑</span><span style="font-size:12px; color:#475569; font-weight:600;">Emergency Vehicle</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
