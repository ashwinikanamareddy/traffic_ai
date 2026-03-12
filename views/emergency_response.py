"""
UrbanFlow AI - Emergency Response
=================================
Page 5: Emergency routing alerts and ETA tracking.
"""
import streamlit as st
import html
import random
from backend.traffic_brain import get_traffic_brain

def show():
    brain = get_traffic_brain()
    grid_state = brain.get_grid_state()
        
    # We use the new backend fast-forwarding ETA now.
    active_corridor = grid_state.get("active_corridor")
    if active_corridor:
        eta = active_corridor.get("eta", 58) 
        story_state = active_corridor.get("story_state", "DETECTED")
    else:
        eta = 58
        story_state = "IDLE"
        
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Emergency Response</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Instantaneous threat triage and routing preparation triggers.</p>
    </div>
    """, unsafe_allow_html=True)

    col_sim, col_viz = st.columns([4, 6])

    with col_sim:
        st.markdown('<div class="saas-card" style="text-align:center; height:100%;">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:16px;">System Dispatch Trigger</h4>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:13px; color:#64748B; margin:0 0 24px 0;">Manually inject an emergency vehicle into the detection pipeline to test routing and ETA synchronization.</p>', unsafe_allow_html=True)
        
        if st.button("🚨 Simulate Ambulance Detection", use_container_width=True, type="primary"):
            brain.simulate_emergency()
            st.rerun()
            
        if active_corridor:
            st.markdown('<div style="margin:24px 0;"><hr style="border-top:1px solid #F1F5F9;"></div>', unsafe_allow_html=True)
            if st.button("✅ Clear Grid", use_container_width=True):
                brain.clear_emergency()
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        st.markdown('<div class="saas-card" style="height:100%;">', unsafe_allow_html=True)
        
        if not active_corridor:
            st.markdown("""
            <div style="background:#F8FAFC; border:1px dashed #E2E8F0; border-radius:12px; height:100%; min-height:200px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center;">
                <div style="font-size:40px; filter:grayscale(1); opacity:0.3; margin-bottom:12px;">🚑</div>
                <div style="color:#64748B; font-size:14px; font-weight:600;">Standby Mode</div>
                <div style="color:#94A3B8; font-size:13px;">Awaiting CV detection hook...</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            v_type = active_corridor.get("vehicle_type", "ambulance").replace("_", " ").title()
            start = active_corridor.get("start_node", "Unknown")
            end = active_corridor.get("end_node", "City Hospital")
            
            # --- ETA Tracking ---
            st.markdown(f"""
            <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:12px; padding:24px; text-align:center; margin-bottom:24px;">
                <h3 style="color:#991B1B; margin:0 0 4px 0; font-size:24px; font-weight:800;">{html.escape(v_type)} Detected</h3>
                <div style="color:#DC2626; font-size:48px; font-weight:800; line-height:1; display:flex; align-items:baseline; justify-content:center; gap:8px;">
                    {eta} <span style="font-size:18px; color:#991B1B; text-transform:uppercase;">Seconds ETA</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- 6-Stage Progress Pipeline ---
            st.markdown('<h4 style="color:#0F172A; font-size:16px; margin:0 0 16px 0;">Emergency Response Progress</h4>', unsafe_allow_html=True)
            
            steps = ["DETECTED", "CALCULATING_ROUTE", "PREPARING_CORRIDOR", "SYNCHRONIZING", "CLEARING", "ARRIVED"]
            labels = ["Ambulance Detected", "Route Calculation", "Green Corridor Preparing", "Signal Synchronization", "Traffic Clearing", "Hospital Arrival"]
            
            curr_idx = steps.index(story_state) if story_state in steps else 0
            
            pipe_html = '<div style="display:flex; flex-direction:column; gap:12px; margin-bottom:24px;">'
            for i, (code, lbl) in enumerate(zip(steps, labels)):
                if i < curr_idx:
                    # Completed
                    bg, border, col, icon = "#F0FDF4", "#BBF7D0", "#16A34A", "✓"
                elif i == curr_idx:
                    # Active
                    bg, border, col, icon = "#EFF6FF", "#BFDBFE", "#2563EB", "🔄"
                else:
                    # Pending
                    bg, border, col, icon = "#F8FAFC", "#E2E8F0", "#94A3B8", "○"
                    
                anim = "animation:pulse_blue 2s infinite;" if i == curr_idx else ""
                
                pipe_html += f"""
                <div style="display:flex; align-items:center; gap:12px; background:{bg}; border:1px solid {border}; padding:12px 16px; border-radius:8px; {anim}">
                    <div style="font-size:16px; font-weight:800; color:{col}; width:24px; text-align:center;">{icon}</div>
                    <div style="font-size:14px; font-weight:700; color:{col};">{lbl}</div>
                </div>
                """
            pipe_html += '</div>'
            st.markdown(pipe_html, unsafe_allow_html=True)
            
            # --- Hospital Readiness Card ---
            if story_state == "ARRIVED":
                st.markdown("""
                <div style="background:#F0FDF4; border:2px solid #22C55E; padding:24px; border-radius:12px; text-align:center; margin-bottom:24px;">
                    <div style="font-size:48px; margin-bottom:8px;">✅</div>
                    <h3 style="color:#16A34A; margin:0 0 8px 0; font-size:20px; font-weight:800;">Emergency vehicle reached hospital successfully.</h3>
                    <p style="color:#15803D; margin:0; font-size:14px; font-weight:600;">AI Traffic Brain cleared the route and ensured uninterrupted travel for the ambulance.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:20px; border-radius:12px; margin-bottom:24px;">
                    <h4 style="color:#0F172A; font-size:15px; margin:0 0 12px 0;">🏥 Hospital Readiness System</h4>
                    <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #F1F5F9; padding-bottom:8px; margin-bottom:8px;">
                        <span style="font-size:14px; color:#475569;">Hospital Alerted</span>
                        <span style="color:#22C55E; font-weight:800;">✓ Yes</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:14px; color:#475569;">Emergency Team Ready</span>
                        <span style="color:#22C55E; font-weight:800;">✓ Standing By</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("""
            <style>
            @keyframes pulse_blue {
                0% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(37, 99, 235, 0); }
                100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Auto-tick the simulation by rerunning until ARRIVAL state is met
            if eta > 0:
                import time
                time.sleep(0.5)
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
