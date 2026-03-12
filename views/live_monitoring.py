"""
UrbanFlow AI - Live Traffic Detection
=====================================
Page 2: Video detection feed and immediate emergency alerts.
"""
import streamlit as st
import cv2
import tempfile
import numpy as np
import time
from backend.traffic_controller import get_traffic_controller
from backend.traffic_brain import get_traffic_brain
from backend.city_control_engine import get_control_engine

# Lazy load model to avoid locking startup
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO('yolov8n.pt')

def show():
    engine = get_control_engine()
    st.markdown("""
    <div style="margin-bottom:24px;">
        <h2 style="color:#0F172A; font-weight:700; margin-bottom:4px; font-size:24px;">Live Traffic Detection</h2>
        <p style="color:#64748B; margin-top:0; font-size:14px;">Real-time computer vision processing capturing live vehicle metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "cmd_live_running" not in st.session_state:
        st.session_state.cmd_live_running = False
    if "cmd_vehicle_counts" not in st.session_state:
        st.session_state.cmd_vehicle_counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0}

    # Simulated emergency hook for the top alert banner
    brain = get_traffic_brain()
    grid_state = brain.get_grid_state()
    active_corridor = grid_state.get("active_corridor")
    
    if active_corridor:
        v_type = active_corridor.get("vehicle_type", "Ambulance").title()
        start = active_corridor.get("start_node", "INT-02")
        eta = active_corridor.get("eta", 58)
        
        st.markdown(f"""
        <div style="background:#FEF2F2; border:2px solid #FECACA; border-radius:12px; padding:24px; margin-bottom:24px; display:flex; gap:24px; align-items:center; box-shadow:0 10px 15px -3px rgba(220,38,38,0.1); animation:pulse_red 2s infinite;">
            <div style="font-size:56px; line-height:1;">🚑</div>
            <div style="flex:1;">
                <h3 style="color:#DC2626; margin:0 0 12px 0; font-size:22px; font-weight:800; text-transform:uppercase; letter-spacing:1px;">Emergency Vehicle Detected</h3>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                    <div style="font-size:15px; color:#991B1B;"><strong style="color:#7F1D1D;">Vehicle type:</strong> {v_type}</div>
                    <div style="font-size:15px; color:#991B1B;"><strong style="color:#7F1D1D;">Estimated Arrival Time:</strong> {eta} seconds</div>
                    <div style="font-size:15px; color:#991B1B;"><strong style="color:#7F1D1D;">Location:</strong> Intersection {start}</div>
                    <div style="font-size:15px; color:#991B1B;"><strong style="color:#7F1D1D;">Destination:</strong> City Hospital</div>
                </div>
            </div>
            <div>
                <button onclick="window.parent.postMessage({{type: 'streamlit:setComponentValue', value: true}}, '*')" style="background:#DC2626; color:white; border:none; padding:12px 24px; border-radius:8px; font-weight:700; cursor:pointer;">Tracking Timeline ➔</button>
            </div>
        </div>
        <style>
        @keyframes pulse_red {{
            0% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }}
            70% {{ box-shadow: 0 0 0 15px rgba(220, 38, 38, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }}
        }}
        </style>
        """, unsafe_allow_html=True)

    col_vid, col_alert = st.columns([7, 3])
    
    with col_vid:
        st.markdown('<div class="saas-card" style="padding:16px;">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns([5, 2, 2])
        with c1:
            uploaded_file = st.file_uploader("Upload Traffic Feed", type=["mp4", "avi"], label_visibility="collapsed")
        with c2:
            if st.button("▶ Connect Feed", type="primary", use_container_width=True):
                st.session_state.cmd_live_running = True if uploaded_file else False
        with c3:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.cmd_live_running = False
                
        # Main Video Rendering Block
        video_placeholder = st.empty()
        log_placeholder = st.empty()
        
        if not st.session_state.cmd_live_running:
            video_placeholder.markdown("""
            <div style="background:#F8FAFC; border:1px dashed #E2E8F0; border-radius:12px; padding:80px 20px; text-align:center; color:#64748B; margin-top:16px;">
                <div style="background:#FFFFFF; width:64px; height:64px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:28px; margin:0 auto 16px auto; box-shadow:0 1px 3px rgba(0,0,0,0.05); border:1px solid #F1F5F9;">📹</div>
                <h3 style="color:#0F172A; font-size:16px; margin:0 0 4px 0;">Camera Feed Offline</h3>
                <p style="font-size:13px; margin:0;">Upload a video source and connect to begin YOLOv8 detection.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if uploaded_file is not None:
                # 1. Save uploaded bytes to temp file for CV2
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                
                vid_cap = cv2.VideoCapture(tfile.name)
                yolo = load_yolo_model()
                
                # Class mapping based on COCO
                class_map = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                
                frame_skip = 2 # Process every Nth frame to keep Streamlit responsive
                frame_idx = 0
                
                # 10. Dashboard Event Log initialization
                event_logs = ["AI Decision: Listening to active feed..."]
                
                while vid_cap.isOpened() and st.session_state.cmd_live_running:
                    ret, frame = vid_cap.read()
                    if not ret:
                        break
                        
                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue # Skip to maintain FPS in browser
                        
                    # 2. YOLOv8 Detection
                    results = yolo(frame, classes=[2, 3, 5, 7], conf=0.3, verbose=False)
                    res = results[0]
                    
                    # 3. Vehicle Counting
                    c_car, c_bike, c_bus, c_truck = 0, 0, 0, 0
                    
                    # Draw boxes & annotate
                    annotated_frame = res.plot()
                    
                    # Parse detections
                    boxes = res.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id == 2: c_car += 1
                        elif cls_id == 3: c_bike += 1
                        elif cls_id == 5: c_bus += 1
                        elif cls_id == 7: c_truck += 1
                    
                    # Store to session state for the right-hand panel
                    st.session_state.cmd_vehicle_counts = {"cars": c_car, "bikes": c_bike, "buses": c_bus, "trucks": c_truck}
                    
                    # 4. Traffic Density Score & 5. Emergency Override (Now handed to the Engine)
                    # We broadcast the frame payload to the centralized event broker
                    engine.process_event("vehicle_detected", {"counts": st.session_state.cmd_vehicle_counts, "log_congestion": True})
                    
                    # If Demo mode bypasses cv2 inference speed, we fake extreme counts
                    if engine.state.get("demo_mode"):
                        st.session_state.cmd_vehicle_counts["cars"] = 45 
                        
                    prob = 0.05
                    if (c_bus > 0 or c_truck > 0) and not active_corridor:
                        if np.random.rand() < prob:
                            # 6. Activate AI Traffic Brain
                            engine.process_event("ambulance_detected", {"location": "INT-02"})
                            # We force a bare rerun so the top banner catches the active_corridor state
                            st.rerun()
                    
                    # Pull Unified Logs from Engine instead of local list
                    event_logs = []
                    for ev in engine.state.get("recent_events", [])[:10]:
                        col = "#94A3B8"
                        if ev['level'] == "warning": col = "#EAB308"
                        if ev['level'] == "error": col = "#EF4444"
                        event_logs.append(f"<span style='color:{col};'>[{ev['time']}] {ev['msg']}</span>")
                    
                    # Convert BGR to RGB for Streamlit render
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Render Feed with Density overlay
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                    
                    # Render Console logs below video
                    log_html = "<div style='background:#0F172A; padding:12px; border-radius:8px; margin-top:12px; height:120px; overflow-y:auto; font-family:monospace; font-size:12px; line-height:1.5; color:#94A3B8;'>"
                    for lg in event_logs[:10]:
                        log_html += f"<div>> {lg}</div>"
                    log_html += "</div>"
                    log_placeholder.markdown(log_html, unsafe_allow_html=True)
                    
                    time.sleep(0.01) # Small breath for streamlit loop
                    
                vid_cap.release()
            else:
                st.warning("Please upload a video file first.")
                st.session_state.cmd_live_running = False
                
        st.markdown('</div>', unsafe_allow_html=True)

    with col_alert:
        vc = st.session_state.cmd_vehicle_counts
        
        if active_corridor:
            # Replaced with the Top Banner, so just show a clear button here
            st.markdown('<div class="saas-card" style="text-align:center; margin-bottom:16px;">', unsafe_allow_html=True)
            if st.button("Cancel Emergency Route", use_container_width=True):
                engine.process_event("emergency_cleared")
                st.session_state.cmd_live_running = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="saas-card" style="text-align:center; margin-bottom:16px; min-height:160px; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:32px; filter:grayscale(1); opacity:0.3; margin-bottom:12px;">🚑</div>
                <div style="color:#64748B; font-size:14px; font-weight:600;">No Emergency Vehicles</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#0F172A; margin:0 0 16px 0; font-size:15px; font-weight:700;">Live Bounding Frame Counts</h4>', unsafe_allow_html=True)
        counts_html = f"""
        <div style="display:flex; flex-direction:column; gap:8px;">
            <div style="display:flex; justify-content:space-between; padding:8px 12px; background:#F8FAFC; border-radius:8px;">
                <span>🚗 Cars</span> <strong>{vc.get('cars',0)}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 12px; background:#F8FAFC; border-radius:8px;">
                <span>🏍️ Bikes</span> <strong>{vc.get('bikes',0)}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 12px; background:#F8FAFC; border-radius:8px;">
                <span>🚌 Buses</span> <strong>{vc.get('buses',0)}</strong>
            </div>
            <div style="display:flex; justify-content:space-between; padding:8px 12px; background:#F8FAFC; border-radius:8px;">
                <span>🚛 Trucks</span> <strong>{vc.get('trucks',0)}</strong>
            </div>
        </div>
        """
        st.markdown(counts_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
