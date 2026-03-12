"""
UrbanFlow AI - Smart Signal Control
===================================
Page 8: Highlighting dynamic signal timing and overrides across interfaces.
"""
import streamlit as st
from backend.traffic_brain import get_traffic_brain

def show():
    brain = get_traffic_brain()
    grid_state = brain.get_grid_state()
    
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Smart Signal Control</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Live traffic signals actively modifying timing based on AI density assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    first_int = list(brain.intersections.values())[0] if brain.intersections else None
    
    if not first_int:
        st.warning("No active intersections detected.")
        return
        
    plan = first_int.optimizer.get_timing_plan()
    signals = first_int.optimizer.get_signal_states()
    approaches = plan.get("approaches", {})
    override_approach = plan.get("emergency_override")

    active_corridor = grid_state.get("active_corridor")
    story_state = active_corridor.get("story_state", "IDLE") if active_corridor else "IDLE"

    st.markdown('<div class="saas-card" style="margin-bottom:24px;">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#0F172A; margin:0 0 8px 0; font-size:16px;">Dynamic Timing Adjustments</h4>', unsafe_allow_html=True)
    
    if story_state in ["PREPARING_CORRIDOR", "SYNCHRONIZING", "CLEARING"]:
        st.markdown("""
        <div style="background:#FEF2F2; border-left:4px solid #DC2626; padding:12px 16px; border-radius:4px; margin-bottom:24px;">
            <div style="color:#991B1B; font-weight:800; font-size:13px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">⚠️ Normal Signal Timing Disabled</div>
            <div style="color:#DC2626; font-size:14px; font-weight:600;">Emergency priority activated. AI clearing traffic for emergency vehicle.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:#64748B; font-size:13px; margin:0 0 24px 0;">Green parameters extended dynamically if lane density registers high priority flags.</p>', unsafe_allow_html=True)
    
    timing_cols = st.columns(4)
    dirs = ["north", "south", "east", "west"]
    
    for i, direction in enumerate(dirs):
        ad = approaches.get(direction, {})
        green_dur = ad.get("green_duration", 15)
        state = signals.get(direction, "RED")
        elapsed = int(plan.get("phase_elapsed", 0))
        countdown = max(0, green_dur - elapsed) if state == "GREEN" else green_dur
        
        timing_reason = "Base allocation"
        if green_dur >= 45:
            timing_reason = f"<span style='color:#EF4444; font-weight:700;'>Heavy traffic detected → green extended</span>"
        elif green_dur >= 30:
            timing_reason = f"<span style='color:#F59E0B; font-weight:700;'>Moderate density → slightly extended</span>"
            
        if override_approach == direction:
            countdown = "∞"
            state = "OVERRIDE"
            color = "#06B6D4"
            bg = "#ECFEFF"
            border = "#CFFAFE"
            timing_reason = f"<span style='color:#06B6D4; font-weight:800;'>Ambulance detected → override forcing green</span>"
        else:
            color = "#22C55E" if state == "GREEN" else "#EF4444"
            bg = "#F0FDF4" if state == "GREEN" else "#FEF2F2"
            border = "#DCFCE7" if state == "GREEN" else "#FEE2E2"
            
        with timing_cols[i]:
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:12px; padding:20px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-weight:700; color:#0F172A; text-transform:capitalize; font-size:15px; margin-bottom:8px;">{direction} Lane</div>
                <div style="color:{color}; font-weight:800; font-size:28px; line-height:1; margin-bottom:4px;">{state}</div>
                <div style="font-size:15px; color:{color}; opacity:0.8; font-weight:600; margin-bottom:16px;">{countdown}s remaining</div>
                <div style="font-size:12px; color:#64748B; background:#FFFFFF; padding:8px; border-radius:8px; border:1px solid {border};">{timing_reason}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)
