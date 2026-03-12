"""
Central City Control Engine
===========================
Master event-driven singleton coordinating all frontend and backend modules.
Maintains the global truth state for the UrbanFlow AI dashboard.
"""

import streamlit as st
import time
from backend.traffic_brain import get_traffic_brain

class CityControlEngine:
    def __init__(self):
        # Master Global State
        self.state = {
            "demo_mode": False,
            "density_level": "LOW",
            "active_emergency": False,
            "emergency_type": None,
            "emergency_location": None,
            "signal_status": "NORMAL",
            "active_incident": False,
            "incident_data": None,
            "system_health": {
                "ai_model": "ONLINE",
                "detection_engine": "ONLINE",
                "signal_network": "ONLINE",
                "prediction_engine": "ONLINE",
                "incident_response": "ONLINE"
            },
            "recent_events": []
        }
        self.brain = get_traffic_brain()
        self.log_event("System initialized. AI Broker active.")

    def log_event(self, message: str, level: str = "info"):
        """Append an event to the global circular log history."""
        timestamp = time.strftime("%H:%M:%S")
        self.state["recent_events"].insert(0, {"time": timestamp, "msg": message, "level": level})
        # Keep only last 50 events
        if len(self.state["recent_events"]) > 50:
            self.state["recent_events"].pop()

    def process_event(self, event_type: str, payload: dict = None):
        """Unified Event Bus Router"""
        if payload is None:
            payload = {}
            
        if event_type == "demo_mode_toggled":
            self.state["demo_mode"] = payload.get("status", False)
            if self.state["demo_mode"]:
                self.state["system_health"]["detection_engine"] = "SIMULATING"
                self.log_event("Demo Mode engaged. System running purely on synthetic heuristics.", "warning")
            else:
                self.state["system_health"]["detection_engine"] = "ONLINE"
                self.log_event("Demo Mode disabled. Awaiting live telemetry.", "info")

        elif event_type == "vehicle_detected":
            # Pass vehicles to math model
            counts = payload.get("counts", {"cars":0, "bikes":0, "buses":0, "trucks":0})
            from backend.traffic_brain import calculate_density, decide_signal_timing
            score, level = calculate_density(counts)
            self.state["density_level"] = level
            
            if level == "HIGH" and payload.get("log_congestion", True):
                self.log_event(f"Heavy traffic detected globally (Score: {score}). Adjusting macro timers.", "warning")
                
        elif event_type == "ambulance_detected":
            loc = payload.get("location", "INT-02")
            self.state["active_emergency"] = True
            self.state["emergency_type"] = "Ambulance"
            self.state["emergency_location"] = loc
            self.state["signal_status"] = "OVERRIDE"
            self.brain.trigger_emergency("Ambulance", loc)
            self.log_event(f"Ambulance detected at {loc}. Route to Hospital forced to GREEN priority.", "error")
            
        elif event_type == "emergency_cleared":
            self.state["active_emergency"] = False
            self.state["emergency_type"] = None
            self.state["emergency_location"] = None
            self.state["signal_status"] = "NORMAL"
            self.brain.clear_emergency()
            self.log_event("Emergency state cleared. Network returning to Density-Timed optimization.", "info")

        elif event_type == "incident_triggered":
            self.state["active_incident"] = True
            loc = payload.get("location", "INT-03")
            itype = payload.get("type", "Accident")
            self.state["incident_data"] = {"location": loc, "type": itype}
            self.state["signal_status"] = "REROUTING"
            self.state["system_health"]["incident_response"] = "ACTIVE"
            self.log_event(f"Critical {itype} registered at {loc}. Activating spatial reroute.", "error")

        elif event_type == "incident_cleared":
            self.state["active_incident"] = False
            self.state["incident_data"] = None
            self.state["signal_status"] = "NORMAL"
            self.state["system_health"]["incident_response"] = "ONLINE"
            self.log_event("Incident resolved. Flow restored and macro rerouting deactivated.", "info")
            
        elif event_type == "video_upload_failed":
            self.state["system_health"]["detection_engine"] = "ERROR"
            self.log_event("CV Pipeline crash: Detection error. Switching to simulation mode.", "error")
            self.process_event("demo_mode_toggled", {"status": True})

@st.cache_resource
def get_control_engine():
    """Singleton accessor for the central engine."""
    return CityControlEngine()
