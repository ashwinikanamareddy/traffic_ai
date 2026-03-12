"""
UrbanFlow AI - City Traffic Digital Twin
========================================
Page 7: A highly interactive CSS-animated 2x3 node grid demonstrating live traffic flow and emergency corridor routing.
"""
import streamlit as st
import time

def show():
    # --- State Management ---
    if "sim_mode" not in st.session_state:
        st.session_state.sim_mode = "NORMAL"
    if "sim_tick" not in st.session_state:
        st.session_state.sim_tick = 0
        
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">City Traffic Digital Twin</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Live CSS-animated simulation of the AI Traffic Brain controlling a 6-node urban grid.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_sim, col_panel = st.columns([7, 3])
    
    # --- Network State Variables ---
    mode = st.session_state.sim_mode
    tick = st.session_state.sim_tick
    
    # Dynamic active states
    ambulance_active = False
    amb_step = 0
    route = ["INT-02", "INT-03", "INT-05", "Hospital"]
    
    # Default node signals (Green/Red)
    nodes = {
        "INT-01": {"ns": "red", "ew": "green"},
        "INT-02": {"ns": "green", "ew": "red"},
        "INT-03": {"ns": "red", "ew": "green"},
        "INT-04": {"ns": "green", "ew": "red"},
        "INT-05": {"ns": "red", "ew": "green"},
        "INT-06": {"ns": "green", "ew": "red"}
    }
    
    message_log = ["System initialized. Normal AI synchronization running."]
    
    if mode == "NORMAL":
        # Toggle signals automatically every few ticks
        if tick % 4 < 2:
            for n in nodes:
                nodes[n]["ns"], nodes[n]["ew"] = "green", "red"
        else:
            for n in nodes:
                nodes[n]["ns"], nodes[n]["ew"] = "red", "green"
            
    elif mode == "CONGESTION":
        # Simulate heavy load at INT-04 forcing long greens
        for n in nodes:
             nodes[n]["ns"], nodes[n]["ew"] = "red", "green"
        nodes["INT-04"]["ns"] = "red"
        nodes["INT-04"]["ew"] = "green" # Force flush
        message_log.append("Heavy congestion detected at INT-04.")
        message_log.append("AI extended East-West signal timing at INT-04 to 45s to clear volume.")
        
    elif mode == "EMERGENCY":
        ambulance_active = True
        amb_step = min(tick, 3) # 0 to 3
        
        message_log.append("🚑 Emergency vehicle detected at INT-02.")
        
        # Override grid logic
        for n in nodes:
            nodes[n]["ns"], nodes[n]["ew"] = "red", "red" # Lock everything
            
        # Target node gets green, others hold red
        if amb_step == 0:
            nodes["INT-02"]["ns"] = "green"
            message_log.append("Clearing signal at INT-02.")
            message_log.append("Synchronizing signal at INT-03.")
        elif amb_step == 1:
            nodes["INT-03"]["ns"] = "green" 
            message_log.append("Clearing signal at INT-03.")
            message_log.append("Synchronizing signal at INT-05.")
        elif amb_step == 2:
            nodes["INT-05"]["ns"] = "green"
            message_log.append("Clearing signal at INT-05.")
            message_log.append("Hospital approaching.")
        else:
            message_log.append("Emergency vehicle successfully reached destination.")
            
    # --- CSS ANIMATION INJECTION ---
    st.markdown("""
    <style>
    .twin-grid {
        position: relative;
        background: #F1F5F9;
        border-radius: 16px;
        border: 2px solid #E2E8F0;
        height: 500px;
        width: 100%;
        overflow: hidden;
    }
    .road-h {
        position: absolute;
        height: 40px;
        width: 100%;
        background: #94A3B8;
        z-index: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .road-h::after {
        content: ""; height: 2px; width: 100%;
        background: repeating-linear-gradient(90deg, transparent, transparent 10px, white 10px, white 20px);
    }
    .road-v {
        position: absolute;
        width: 40px;
        height: 100%;
        background: #94A3B8;
        z-index: 1;
        display: flex;
        justify-content: center;
    }
    .road-v::after {
        content: ""; width: 2px; height: 100%;
        background: repeating-linear-gradient(0deg, transparent, transparent 10px, white 10px, white 20px);
    }
    .node {
        position: absolute;
        width: 60px; height: 60px;
        background: #334155;
        z-index: 2;
        border-radius: 8px;
        display: flex;
        align-items: center; justify-content: center;
        transform: translate(-50%, -50%);
        color: white; font-size: 10px; font-weight: 800;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Signals */
    .signal { width: 12px; height: 12px; border-radius: 50%; position: absolute; background: red; transition: background 0.3s; }
    .sig-n { top: 4px; left: 24px; }
    .sig-s { bottom: 4px; left: 24px; }
    .sig-e { right: 4px; top: 24px; }
    .sig-w { left: 4px; top: 24px; }
    
    /* Cars */
    .car {
        position: absolute; width:12px; height:8px; border-radius:2px; z-index: 3;
    }
    
    @keyframes drive-east { from { left: -20px; } to { left: 100%; } }
    @keyframes drive-south { from { top: -20px; } to { top: 100%; } }
    
    /* Variable Traffic Flow Arrays */
    .flow-normal { animation: drive-east 6s linear infinite; background: #3B82F6; }
    .flow-normal-2 { animation: drive-east 7s linear infinite 1s; background: #0EA5E9; }
    .flow-south { animation: drive-south 5s linear infinite; background: #10B981; }
    
    .flow-heavy { animation: drive-east 10s linear infinite; background: #EF4444; width:14px; }
    .flow-heavy-2 { animation: drive-east 12s linear infinite 1s; background: #F87171; width:14px;}
    .flow-heavy-3 { animation: drive-east 11s linear infinite 2s; background: #DC2626; width:14px;}
    
    /* The Ambulance */
    .ambulance-icon {
        position: absolute; z-index: 10; font-size: 24px;
        transition: all 1s ease-in-out;
        transform: translate(-50%, -50%);
        text-shadow: 0 0 10px rgba(220,38,38,0.8);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Grid coordinates mapping (percentages)
    grid_coords = {
        "INT-01": {"x": 30, "y": 25},
        "INT-02": {"x": 70, "y": 25},
        "INT-03": {"x": 30, "y": 50},
        "INT-04": {"x": 70, "y": 50},
        "INT-05": {"x": 30, "y": 75},
        "INT-06": {"x": 70, "y": 75},
        "Hospital": {"x": 30, "y": 95} # Off bottom edge
    }

    with col_sim:
        st.markdown('<div class="saas-card" style="padding:0; overflow:hidden;">', unsafe_allow_html=True)
        
        # Build Grid HTML
        html = '<div class="twin-grid">'
        
        # Roads
        html += '<div class="road-h" style="top: 25%; transform:translateY(-50%);"></div>'
        html += '<div class="road-h" style="top: 50%; transform:translateY(-50%);"></div>'
        html += '<div class="road-h" style="top: 75%; transform:translateY(-50%);"></div>'
        
        html += '<div class="road-v" style="left: 30%; transform:translateX(-50%);"></div>'
        html += '<div class="road-v" style="left: 70%; transform:translateX(-50%);"></div>'
        
        # Traffic Flows (CSS Animated cars based on state)
        if mode == "NORMAL":
            html += '<div class="car flow-normal" style="top: calc(25% + 5px);"></div>'
            html += '<div class="car flow-normal-2" style="top: calc(50% + 5px);"></div>'
            html += '<div class="car flow-south" style="left: calc(70% - 10px);"></div>'
            html += '<div class="car flow-south" style="left: calc(30% - 10px); animation-delay:2s;"></div>'
        elif mode == "CONGESTION":
            html += '<div class="car flow-heavy" style="top: calc(50% + 5px);"></div>'
            html += '<div class="car flow-heavy-2" style="top: calc(50% + 5px);"></div>'
            html += '<div class="car flow-heavy-3" style="top: calc(50% + 5px);"></div>'
            html += '<div class="car flow-south" style="left: calc(30% - 10px);"></div>'
            
        # Green Corridor Override Styling
        if mode == "EMERGENCY":
            # Highlight the path vertically along the 30% line
            html += '<div style="position:absolute; width:16px; height:70%; left:30%; top:25%; transform:translateX(-50%); background:rgba(34,197,94,0.3); z-index:1; box-shadow:0 0 15px #22C55E;"></div>'
            
        # Draw Intersections & Signals
        for nid, pos in grid_coords.items():
            if nid == "Hospital":
                html += f'<div class="node" style="left:{pos["x"]}%; top:{pos["y"]}%; background:#DC2626; z-index:5;"><div style="font-size:24px;">🏥</div></div>'
            else:
                ns_color = "#22C55E" if nodes[nid]["ns"] == "green" else "#EF4444"
                ew_color = "#22C55E" if nodes[nid]["ew"] == "green" else "#EF4444"
                
                # Dynamic node background if emergency clears it
                bg = "#16A34A" if mode == "EMERGENCY" and nid in route and nodes[nid]["ns"] == "green" else "#334155"
                
                html += f"""
                <div class="node" style="left:{pos["x"]}%; top:{pos["y"]}%; background:{bg};">
                    {nid}
                    <div class="signal sig-n" style="background:{ns_color};"></div>
                    <div class="signal sig-s" style="background:{ns_color};"></div>
                    <div class="signal sig-e" style="background:{ew_color};"></div>
                    <div class="signal sig-w" style="background:{ew_color};"></div>
                </div>
                """
                
        # Draw Ambulance Pin directly interpolated by amb_step index
        if ambulance_active:
            target_node = route[min(amb_step, 3)]
            amb_pos = grid_coords[target_node]
            html += f'<div class="ambulance-icon" style="left:{amb_pos["x"]}%; top:{amb_pos["y"]}%;">🚑</div>'

        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)
        
    with col_panel:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px;">Simulation Control</h4>', unsafe_allow_html=True)
        
        if st.button("▶ Reset City Traffic", use_container_width=True):
            st.session_state.sim_mode = "NORMAL"
            st.session_state.sim_tick = 0
            st.rerun()
            
        st.markdown('<hr style="margin:16px 0; border:none; border-top:1px solid #E2E8F0;">', unsafe_allow_html=True)
        
        if st.button("🚨 Trigger Emergency Route", use_container_width=True, type="primary"):
            st.session_state.sim_mode = "EMERGENCY"
            st.session_state.sim_tick = 0
            st.rerun()
            
        if st.button("⚠️ Simulate INT-04 Congestion", use_container_width=True):
            st.session_state.sim_mode = "CONGESTION"
            st.session_state.sim_tick = 0
            st.rerun()
            
        st.markdown('<h4 style="color:#0F172A; margin:24px 0 12px 0; font-size:14px; text-transform:uppercase; letter-spacing:1px;">System Event Log</h4>', unsafe_allow_html=True)
        
        log_html = '<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px; height:180px; overflow-y:auto; font-family:monospace; font-size:12px; color:#475569; display:flex; flex-direction:column; gap:8px;">'
        for msg in message_log:
            if "🚑" in msg or "Emergency" in msg or "Green" in msg:
                log_html += f'<div style="color:#DC2626; font-weight:700;">&gt; {msg}</div>'
            elif "Congestion" in msg or "extended" in msg:
                log_html += f'<div style="color:#D97706; font-weight:700;">&gt; {msg}</div>'
            else:
                log_html += f'<div>&gt; {msg}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
    # --- Explanation Panel ---
    st.markdown('<div class="saas-card" style="margin-top:24px; background:#F8FAFC; border:1px solid #E2E8F0;">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#0F172A; margin:0 0 8px 0; font-size:15px;">🌐 Dynamic AI Adaptation</h4>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569; font-size:14px; margin:0; line-height:1.6;">This simulation demonstrates how the AI Traffic Brain coordinates multiple intersections across the city in real-time. By dynamically shifting signal phases, it preemptively offloads traffic spikes (like at INT-04), and instantly coordinates Green Corridors for maximum flow priority during critical emergency events.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Auto-tick the simulation timeline
    if mode == "EMERGENCY" and tick < 4:
        time.sleep(1.5)
        st.session_state.sim_tick += 1
        st.rerun()
    elif mode == "NORMAL":
        time.sleep(2)
        st.session_state.sim_tick += 1
        st.rerun()
