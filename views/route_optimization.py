"""
UrbanFlow AI - Route Optimization
=================================
Page 6: Graph algorithms for shortest path calculations.
"""
import streamlit as st
import html
from backend.traffic_brain import get_traffic_brain

def show():
    brain = get_traffic_brain()
    
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Route Optimization Engine</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Simulating A* / Dijkstra shortest-path calculations across the city grid.</p>
    </div>
    """, unsafe_allow_html=True)

    grid_state = brain.get_grid_state()
    active = grid_state.get("active_corridor")

    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown('<h4 style="color:#0F172A; margin:0 0 20px 0; font-size:16px;">A* Algorithm Path Execution</h4>', unsafe_allow_html=True)
    
    if not active:
        st.markdown("""
        <div style="background:#F8FAFC; padding:40px; text-align:center; border-radius:12px; color:#64748B; font-size:14px;">
            Awaiting routing trigger to execute A* Search over network nodes.
        </div>
        """, unsafe_allow_html=True)
    else:
        route = active.get("route", [])
        start = active.get("start_node", "Start")
        end = active.get("end_node", "End")
        
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#F8FAFC; border:1px solid #E2E8F0; padding:16px 24px; border-radius:12px; margin-bottom:32px;">
            <div>
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px;">Origin Node</div>
                <div style="font-size:16px; font-weight:800; color:#0F172A;">{html.escape(start)}</div>
            </div>
            <div style="font-size:24px; color:#CBD5E1;">→</div>
            <div style="text-align:right;">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:4px;">Target Node</div>
                <div style="font-size:16px; font-weight:800; color:#0F172A;">{html.escape(end)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Draw node graph horizontally
        st.markdown('<div style="font-size:12px; font-weight:700; color:#64748B; text-transform:uppercase; margin-bottom:16px;">Calculated Optimal Path</div>', unsafe_allow_html=True)
        
        graph_html = '<div style="display:flex; align-items:center; gap:8px; overflow-x:auto; padding-bottom:16px;">'
        
        story_state = active.get("story_state", "DETECTED")
        
        # Determine the currently active intersection based on story timeline relative mapping
        active_idx = 0
        if story_state == "SYNCHRONIZING" or story_state == "CLEARING":
            active_idx = 1
        elif story_state == "ARRIVED":
            active_idx = len(route) - 1

        for i, node in enumerate(route):
            if i < active_idx:
                # Passed
                bg, col, shadow = "#F0FDF4", "#16A34A", "none"
                border = "1px solid #BBF7D0"
            elif i == active_idx:
                # Active
                bg, col, shadow = "#2563EB", "white", "0 4px 6px rgba(37,99,235,0.3)"
                border = "none"
            else:
                # Pending
                bg, col, shadow = "#F8FAFC", "#64748B", "none"
                border = "1px dashed #CBD5E1"
                
            graph_html += f"""
            <div style="background:{bg}; color:{col}; border:{border}; padding:12px 20px; border-radius:8px; font-weight:700; font-size:14px; box-shadow:{shadow}; white-space:nowrap; transition:all 0.3s ease;">
                {html.escape(node)}
            </div>
            """
            if i < len(route) - 1:
                graph_html += '<div style="color:#94A3B8; font-weight:800; font-size:16px;">→</div>'
                
        graph_html += '</div>'
        st.markdown(graph_html, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
