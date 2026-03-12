"""
Central AI Traffic Brain Controller
===================================
Master coordinator module that processes vehicle detections,
evaluates priority rules, and dictates traffic network behavior.
"""

import time
import random
from typing import Dict, Any, List

from backend.traffic_brain import calculate_density, decide_signal_timing

class CentralTrafficController:
    """
    Step 1: Central controller module.
    Orchestrates vehicle_detection, traffic_density_analysis,
    signal_optimization, emergency_detection, and green_corridor.
    """
    def __init__(self):
        self.lanes = ["North", "South", "East", "West"]
        self.system_log = []
        
    def simulate_four_directions(self, base_counts: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """
        Step 7: Simulate multi-intersection behavior natively for 4 directions.
        Distributes or simulates counts for North, South, East, West based on base_counts.
        """
        simulated = {}
        # Assume the base_counts are for the main 'North' camera
        simulated["North"] = dict(base_counts)
        
        # Simulate other lanes with some randomization
        for lane in ["South", "East", "West"]:
            variance = random.uniform(0.2, 1.2)
            sim_counts = {}
            for k, v in base_counts.items():
                if k in ["ambulances", "fire_trucks"]:
                    sim_counts[k] = 0 # Assume emergency is only where detected
                else:
                    sim_counts[k] = int(v * variance)
            simulated[lane] = sim_counts
            
        return simulated

    def process_pipeline(self, vehicle_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Step 3: Decision Pipeline
        1. Analyze traffic density
        2. Detect emergency vehicles
        3. Decide signal timings
        4. Activate green corridor if needed
        5. Generate AI decision explanation
        """
        from backend.green_corridor import detect_emergency, generate_emergency_route, activate_green_corridor
        
        # We start by simulating the 4 directions (Step 7)
        lane_data = self.simulate_four_directions(vehicle_counts)
        
        # In a real scenario we'd aggregate, but here we base decision on the most congested lane or emergency
        
        primary_lane = "North"
        max_score = 0
        best_density = "LOW"
        is_em = False
        
        for lane, counts in lane_data.items():
            score, den = calculate_density(counts)
            if score > max_score:
                max_score = score
                best_density = den
                primary_lane = lane
            if detect_emergency(counts):
                is_em = True
                primary_lane = lane
                
        # 1. Analyze density
        density = best_density
        
        # 2. Detect emergency
        emergency_mode = is_em
        
        # 3. Decide signal timings
        signal_duration = decide_signal_timing(density)
        
        # 4 & 5. Priority Rules & Green Corridor
        green_corridor = False
        ai_decision = ""
        em_type = "ambulance" if vehicle_counts.get("ambulances", 0) > 0 else "fire truck"

        if emergency_mode:
            green_corridor = True
            signal_duration = 60
            ai_decision = f"Emergency {em_type} detected in {primary_lane} lane. Activating green corridor."
        elif density == "HIGH":
            ai_decision = f"AI detected heavy traffic in {primary_lane} lane. Signal extended to {signal_duration} seconds."
        elif density == "MEDIUM":
            ai_decision = f"Moderate traffic detected in {primary_lane} lane. Signal duration set to {signal_duration} seconds."
        else:
            ai_decision = f"Traffic flow normalized. Returning to adaptive signal mode."

        # Step 5: AI Decision Object
        traffic_state = {
            "traffic_score": max_score,
            "density": density,
            "signal_duration": signal_duration,
            "emergency_mode": emergency_mode,
            "emergency_type": em_type if emergency_mode else "",
            "green_corridor": green_corridor,
            "ai_decision": ai_decision,
            "lane_data": lane_data, # Include the simulated 4 directions for dashboard
            "primary_lane": primary_lane
        }
        
        self.system_log.append({
            "timestamp": time.time(),
            "state": traffic_state
        })
        self.system_log = self.system_log[-100:]
        
        return traffic_state

# Expose a singleton
_CONTROLLER = CentralTrafficController()

def get_traffic_controller() -> CentralTrafficController:
    return _CONTROLLER
