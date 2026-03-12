"""
UrbanFlow AI - Green Corridor Manager
=====================================
Page 7: Monitoring active emergency corridors and forced signal synchronization.
"""
import streamlit as st
import html
from backend.traffic_brain import get_traffic_brain

def show():
    brain = get_traffic_brain()
    grid_state = brain.get_grid_state()
    active = grid_state.get("active_corridor")
    
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Green Corridor Manager</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Live tracking of synchronized traffic signals clearing the emergency route.</p>
    </div>
    """, unsafe_allow_html=True)

    if not active:
        st.markdown('<div class="saas-card" style="text-align:center; padding:80px 20px;">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:48px; filter:grayscale(1); opacity:0.3; margin-bottom:16px;">🛡️</div>', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#0F172A; font-size:18px; margin:0 0 8px 0;">No Active Corridors</h3>', unsafe_allow_html=True)
        st.markdown('<p style="color:#64748B; font-size:14px; margin:0;">The traffic network is operating under normal AI synchronization rules.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        route = active.get("route", [])
        plan = active.get("signal_plan", {})
        
        st.markdown("""
        <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:12px; padding:24px; display:flex; justify-content:space-between; align-items:center; margin-bottom:24px;">
            <div>
                <div style="display:flex; align-items:center; gap:8px; color:#16A34A; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                    <span style="display:block; width:8px; height:8px; border-radius:50%; background:#16A34A; box-shadow:0 0 8px #16A34A;"></span> NETWORK OVERRIDE
                </div>
                <h3 style="color:#14532D; margin:0; font-size:20px; font-weight:800;">Corridor Sequence Initiated</h3>
            </div>
            <div style="font-size:32px; color:#22C55E;">🟩🟩🟩</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; font-size:16px; margin:0 0 8px 0;">Intersections Turning Green</h4>', unsafe_allow_html=True)
        st.markdown('<p style="color:#64748B; font-size:13px; margin:0 0 24px 0;">AI clearing traffic for emergency vehicle.</p>', unsafe_allow_html=True)
        
        story_state = active.get("story_state", "DETECTED")
        active_idx = 0
        if story_state == "SYNCHRONIZING" or story_state == "CLEARING":
            active_idx = 1
        elif story_state == "ARRIVED":
            active_idx = len(route) - 1
        
        for i, node in enumerate(route):
            if i < active_idx or story_state == "ARRIVED":
                # Fully Cleared
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px; background:#F8FAFC; border:1px solid #E2E8F0; padding:16px; border-radius:12px;">
                    <div style="width:48px; height:48px; border-radius:12px; background:#F0FDF4; color:#16A34A; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:800; border:1px solid #BBF7D0;">
                        ✓
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:4px;">{html.escape(node)}</div>
                        <div style="font-size:13px; color:#64748B;">Route execution completed</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:#E2E8F0; color:#475569; padding:6px 16px; border-radius:99px; font-size:12px; font-weight:800; letter-spacing:1px;">PASSED</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif i == active_idx:
                # Active Green
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px; background:#F8FAFC; border:1px solid #E2E8F0; padding:16px; border-radius:12px;">
                    <div style="width:48px; height:48px; border-radius:12px; background:#DCFCE7; color:#16A34A; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:800; border:1px solid #BBF7D0;">
                        🟩
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:4px;">{html.escape(node)}</div>
                        <div style="font-size:13px; color:#64748B;">Forcing Green Signal Synchronization</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:#22C55E; color:white; padding:6px 16px; border-radius:99px; font-size:12px; font-weight:800; letter-spacing:1px; box-shadow:0 0 10px rgba(34,197,94,0.4); animation: pulse_green 2s infinite;">GREEN</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Preparing Green
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:16px; margin-bottom:16px; background:#F8FAFC; border:1px solid #E2E8F0; padding:16px; border-radius:12px; opacity:0.7;">
                    <div style="width:48px; height:48px; border-radius:12px; background:#FEFCE8; color:#CA8A04; display:flex; align-items:center; justify-content:center; font-size:20px; font-weight:800; border:1px solid #FEF08A;">
                        ⏳
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:15px; font-weight:700; color:#0F172A; margin-bottom:4px;">{html.escape(node)}</div>
                        <div style="font-size:13px; color:#64748B;">Next node calculating timing parameters</div>
                    </div>
                    <div style="text-align:right;">
                        <span style="background:#EAB308; color:white; padding:6px 16px; border-radius:99px; font-size:12px; font-weight:800; letter-spacing:1px; box-shadow:0 0 10px rgba(234,179,8,0.4);">PREPARING GREEN</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("""
        <style>
        @keyframes pulse_green {
            0% { box-shadow: 0 0 0 0 rgba(34,197,94, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(34,197,94, 0); }
            100% { box-shadow: 0 0 0 0 rgba(34,197,94, 0); }
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
