"""
AI Traffic Decision Engine
===========================
Core decision engine for the Dynamic AI Traffic Flow Optimizer & Emergency Grid.

Functions:
    calculate_density(vehicle_counts) → traffic_score, density_level
    decide_signal_timing(density)     → signal_duration
    check_emergency(vehicle_counts)   → emergency_mode, emergency_type
    generate_ai_decision(...)         → ai_decision_message

All outputs feed directly into the Streamlit dashboard.
"""

import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.signal_optimizer import DynamicSignalOptimizer
from backend.green_corridor import (
    detect_emergency,
    generate_emergency_route,
    activate_green_corridor,
    visualize_corridor_on_map
)


# ═══════════════════════════════════════════════════════
#  STEP 1: Calculate Traffic Density Score
# ═══════════════════════════════════════════════════════

# Weighted scoring
WEIGHTS = {
    "cars": 1.0,
    "bikes": 0.5,
    "buses": 3.0,
    "trucks": 3.0,
    "autos": 1.0,
}


def calculate_density(vehicle_counts: Dict[str, int]) -> Tuple[float, str]:
    """
    Calculate a weighted traffic density score and determine the traffic level.

    Weights:
        car   = 1.0
        bike  = 0.5
        bus   = 3.0
        truck = 3.0

    Returns:
        (traffic_score, density_level)
        where density_level is "LOW", "MEDIUM", or "HIGH"
    """
    traffic_score = 0.0
    for vehicle_type, count in vehicle_counts.items():
        weight = WEIGHTS.get(vehicle_type, 1.0)
        traffic_score += int(count) * weight

    # ── Step 2: Determine traffic level ──
    if traffic_score < 20:
        density = "LOW"
    elif traffic_score <= 50:
        density = "MEDIUM"
    else:
        density = "HIGH"

    return round(traffic_score, 1), density


# ═══════════════════════════════════════════════════════
#  STEP 3: Dynamic Signal Timing
# ═══════════════════════════════════════════════════════

SIGNAL_DURATIONS = {
    "LOW": 15,
    "MEDIUM": 30,
    "HIGH": 45,
}

EMERGENCY_SIGNAL_DURATION = 60


def decide_signal_timing(density: str) -> int:
    """
    Return the green signal duration (seconds) for a given density level.

    LOW    → 15 seconds
    MEDIUM → 30 seconds
    HIGH   → 45 seconds
    """
    return SIGNAL_DURATIONS.get(density, 15)


# ═══════════════════════════════════════════════════════
#  STEP 4: Emergency Vehicle Priority
# ═══════════════════════════════════════════════════════

def check_emergency(vehicle_counts: Dict[str, int]) -> Tuple[bool, str]:
    """
    Check if any emergency vehicles are present.

    If ambulance or fire truck detected:
        - Activate emergency priority mode
        - Override normal signal cycle
        - Extend green signal to 60 seconds

    Returns:
        (emergency_mode, emergency_type)
        emergency_type is "ambulance", "fire_truck", or "" if none
    """
    ambulances = int(vehicle_counts.get("ambulances", 0))
    fire_trucks = int(vehicle_counts.get("fire_trucks", 0))

    if ambulances > 0:
        return True, "ambulance"
    if fire_trucks > 0:
        return True, "fire_truck"

    return False, ""


# ═══════════════════════════════════════════════════════
#  STEP 5: AI Decision Log Generation
# ═══════════════════════════════════════════════════════

def generate_ai_decision(
    traffic_score: float,
    density: str,
    signal_duration: int,
    emergency_mode: bool,
    emergency_type: str,
) -> str:
    """
    Generate a human-readable AI decision explanation.

    Examples:
        "AI Decision: Heavy traffic detected. Green signal extended to 45 seconds."
        "AI Decision: Ambulance detected. Emergency priority mode activated."
        "AI Decision: Low traffic. Reduced signal duration."
    """
    if emergency_mode:
        vehicle_name = emergency_type.replace("_", " ").title()
        return (
            f"AI Decision: {vehicle_name} detected. "
            f"Emergency priority mode activated. "
            f"Green signal extended to {EMERGENCY_SIGNAL_DURATION} seconds."
        )

    if density == "HIGH":
        return (
            f"AI Decision: Heavy traffic detected (score: {traffic_score}). "
            f"Green signal extended to {signal_duration} seconds."
        )
    elif density == "MEDIUM":
        return (
            f"AI Decision: Moderate traffic detected (score: {traffic_score}). "
            f"Signal duration set to {signal_duration} seconds."
        )
    else:
        return (
            f"AI Decision: Low traffic (score: {traffic_score}). "
            f"Reduced signal duration to {signal_duration} seconds."
        )


# ═══════════════════════════════════════════════════════
#  STEP 6: Full Decision Pipeline (Integrate All Steps)
# ═══════════════════════════════════════════════════════

def make_decision(vehicle_counts: Dict[str, int]) -> Dict:
    """
    Run the full AI decision pipeline.

    Input:
        vehicle_counts = {"cars": int, "bikes": int, "buses": int,
                          "trucks": int, "ambulances": int, "fire_trucks": int}

    Returns dict with:
        traffic_score       → float
        traffic_density     → "LOW" / "MEDIUM" / "HIGH"
        signal_duration     → int (seconds)
        emergency_mode      → bool
        emergency_type      → str
        ai_decision_message → str
    """
    # Step 1 & 2: Calculate density
    traffic_score, density = calculate_density(vehicle_counts)

    # Step 3: Decide signal timing
    signal_duration = decide_signal_timing(density)

    # Step 4: Check emergency
    emergency_mode, emergency_type = check_emergency(vehicle_counts)

    # Override signal duration if emergency
    if emergency_mode:
        signal_duration = EMERGENCY_SIGNAL_DURATION

    # Step 5: Generate AI decision message
    ai_decision_message = generate_ai_decision(
        traffic_score, density, signal_duration, emergency_mode, emergency_type
    )

    return {
        "traffic_score": traffic_score,
        "traffic_density": density,
        "signal_duration": signal_duration,
        "emergency_mode": emergency_mode,
        "emergency_type": emergency_type,
        "ai_decision_message": ai_decision_message,
    }


# ═══════════════════════════════════════════════════════
#  STEP 7: Intersection & TrafficBrain (Grid Coordinator)
# ═══════════════════════════════════════════════════════

class Intersection:
    """Represents a single intersection in the grid."""

    def __init__(self, intersection_id: str, row: int, col: int):
        self.id = intersection_id
        self.row = row
        self.col = col
        self.optimizer = DynamicSignalOptimizer(intersection_id=intersection_id)
        self.vehicle_count = 0
        self.emergency_detected = False
        self.last_update = time.time()
        self.last_decision: Dict = {}

    def to_dict(self) -> Dict:
        timing = self.optimizer.get_timing_plan()
        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "vehicle_count": self.vehicle_count,
            "emergency_detected": self.emergency_detected,
            "signal_states": self.optimizer.get_signal_states(),
            "timing_plan": timing,
            "last_update": self.last_update,
            "last_decision": self.last_decision,
        }


class TrafficBrain:
    """AI Traffic Brain — central coordinator for the intersection grid."""

    def __init__(self, grid_rows: int = 2, grid_cols: int = 2):
        self.grid_rows = max(1, int(grid_rows))
        self.grid_cols = max(1, int(grid_cols))

        # Build intersection grid
        self.intersections: Dict[str, Intersection] = {}
        self._grid: List[List[str]] = []
        idx = 1
        for r in range(self.grid_rows):
            row_ids = []
            for c in range(self.grid_cols):
                int_id = f"INT-{idx:02d}"
                self.intersections[int_id] = Intersection(int_id, r, c)
                row_ids.append(int_id)
                idx += 1
            self._grid.append(row_ids)

        # Corridor state
        self.active_corridor: Optional[Dict] = None

        # Decision log
        self._decision_log: List[Dict] = []
        self._tick_count = 0

    def tick(self, detection_results: Optional[Dict] = None, elapsed_seconds: float = 1.0):
        """
        Main brain tick.  Runs the AI decision pipeline each frame.
        """
        self._tick_count += 1
        detection_results = detection_results or {}

        vehicle_counts = detection_results.get("vehicle_counts", {})
        emergency_vehicles = detection_results.get("emergency_vehicles", [])
        total_vehicles = int(detection_results.get("total_vehicles", 0))

        # ── Run decision pipeline ──
        # Merge emergency vehicles from vision into counts
        merged_counts = dict(vehicle_counts)
        merged_counts["ambulances"] = len([
            e for e in emergency_vehicles
            if e.get("type") in ("ambulance",)
        ])
        merged_counts["fire_trucks"] = len([
            e for e in emergency_vehicles
            if e.get("type") in ("fire_truck",)
        ])

        decision = make_decision(merged_counts)

        # Distribute density across intersections
        self._distribute_density(total_vehicles, vehicle_counts, decision)

        # Update each intersection's signal optimizer
        for intersection in self.intersections.values():
            intersection.optimizer.tick(elapsed_seconds)
            intersection.last_update = time.time()

        # Handle emergency vehicles via new green_corridor functions
        is_emergency = detect_emergency(merged_counts)
        
        # If emergency detected and no active corridor, or we want to update it
        if is_emergency and not self.active_corridor:
            # Em type for logging
            em_type = "Fire Truck" if merged_counts.get("fire_trucks") else "Ambulance"
            
            # For prototype, just start at a default intersection based on which camera triggered
            # In a real app we'd map bbox to intersection ID
            start_node = "INT-01"
            
            # Step 2: Identify Emergency Route
            route = generate_emergency_route(start_node)
            
            # Step 3: Signal Preemption
            plan = activate_green_corridor(route)
            
            # Save state
            self.active_corridor = {
                "route": route,
                "plan": plan,
                "vehicle_type": em_type,
                "activated_at": time.time(),
                "ttl": EMERGENCY_SIGNAL_DURATION
            }
            
            # Step 7: AI Decision Explanation
            self._decision_log.append({
                "timestamp": time.time(),
                "tick": self._tick_count,
                "type": "corridor_activated",
                "details": {"route": route},
                "ai_message": f"AI Decision: {em_type} detected at {start_node}."
            })
            self._decision_log.append({
                "timestamp": time.time(),
                "tick": self._tick_count,
                "type": "corridor_activated",
                "details": {"route": route},
                "ai_message": "AI Decision: Activating green corridor."
            })
            self._decision_log.append({
                "timestamp": time.time(),
                "tick": self._tick_count,
                "type": "corridor_activated",
                "details": {"route": route},
                "ai_message": "AI Decision: Clearing signals along emergency route."
            })

        # Cleanup expired corridors
        if self.active_corridor:
            if time.time() - self.active_corridor["activated_at"] > self.active_corridor["ttl"]:
                self.active_corridor = None

        # Apply corridor overrides
        for int_id, intersection in self.intersections.items():
            if self.active_corridor and int_id in self.active_corridor["route"]:
                state = self.active_corridor["plan"].get(int_id)
                if state == "GREEN":
                    intersection.optimizer.set_emergency_override("north") # force green
                # PREPARE GREEN is conceptual right now in the optimizer, but we pass it to dashboard
                intersection.emergency_detected = True
            else:
                intersection.optimizer.clear_emergency_override()
                intersection.emergency_detected = False

        # Log the normal traffic decision
        self._log_decision(decision)

    def _distribute_density(self, total_vehicles: int, vehicle_counts: Dict, decision: Dict):
        """Distribute detected vehicles across intersection approaches using the decision engine."""
        approaches = DynamicSignalOptimizer.APPROACHES
        score = decision["traffic_score"]
        # Normalize to 0-1 range using the score
        base_density = min(1.0, score / 100.0)

        for intersection in self.intersections.values():
            intersection.vehicle_count = total_vehicles
            intersection.last_decision = decision

            # Dynamic min/max green based on decision engine
            sig_dur = decision["signal_duration"]
            intersection.optimizer.min_green = max(10.0, float(sig_dur) * 0.5)
            intersection.optimizer.max_green = max(float(sig_dur), 45.0)

            density_map = {}
            for approach in approaches:
                noise = random.uniform(-0.1, 0.1)
                d = max(0.0, min(1.0, base_density + noise))
                density_map[approach] = d
            intersection.optimizer.update_densities(density_map)

    def simulate_emergency(self, vehicle_type: str = "ambulance", approach: str = "north") -> Dict:
        """Simulate an emergency vehicle detection for demo purposes."""
        vehicle_id = f"SIM-EV-{int(time.time()) % 10000}"
        
        start_node = "INT-01"
        route = generate_emergency_route(start_node)
        plan = activate_green_corridor(route)

        vt_str = vehicle_type.replace('_', ' ').title()

        self.active_corridor = {
            "route": route,
            "plan": plan,
            "vehicle_type": vt_str,
            "activated_at": time.time(),
            "ttl": EMERGENCY_SIGNAL_DURATION,
            "story_state": "DETECTED",  # New state tracking parameter
            "eta": 58                   # Initial ETA value
        }

        # Step 7: AI Decision Explanation
        self._decision_log.append({
            "timestamp": time.time(),
            "tick": self._tick_count,
            "type": "corridor_activated",
            "details": {"route": route},
            "ai_message": f"AI Decision: {vt_str} detected at {start_node}."
        })
        self._decision_log.append({
            "timestamp": time.time(),
            "tick": self._tick_count,
            "type": "corridor_activated",
            "details": {"route": route},
            "ai_message": "AI Decision: Activating green corridor."
        })
        self._decision_log.append({
            "timestamp": time.time(),
            "tick": self._tick_count,
            "type": "corridor_activated",
            "details": {"route": route},
            "ai_message": "AI Decision: Clearing signals along emergency route."
        })

        return {
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "start_node": start_node,
            "route": route,
            "plan": plan
        }

    def get_grid_state(self) -> Dict:
        # Dynamically evaluate the story state based on elapsed time if corridor active
        if self.active_corridor:
            elapsed = time.time() - self.active_corridor["activated_at"]
            eta = max(0, 58 - int(elapsed * 2)) # Fast forward ETA for simulation
            
            # Map elapsed time to story states
            if eta == 0:
                state = "ARRIVED"
            elif elapsed < 2:
                state = "DETECTED"
            elif elapsed < 4:
                state = "CALCULATING_ROUTE"
            elif elapsed < 6:
                state = "PREPARING_CORRIDOR"
            elif elapsed < 10:
                state = "SYNCHRONIZING"
            else:
                state = "CLEARING"
                
            self.active_corridor["story_state"] = state
            self.active_corridor["eta"] = eta

        grid_visual = []
        for row in self._grid:
            grid_row = []
            for int_id in row:
                grid_row.append(self.intersections[int_id].to_dict())
            grid_visual.append(grid_row)

        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "grid": grid_visual,
            "intersections": {k: v.to_dict() for k, v in self.intersections.items()},
            "active_corridor": self.active_corridor,
            "tick_count": self._tick_count,
        }

    def get_decision_log(self, last_n: int = 30) -> List[Dict]:
        return self._decision_log[-last_n:]

    def get_density_heatmap(self) -> Dict[str, Dict[str, float]]:
        heatmap = {}
        for int_id, intersection in self.intersections.items():
            heatmap[int_id] = intersection.optimizer.get_densities()
        return heatmap

    def get_latest_decision(self) -> Dict:
        """Get the most recent decision from any intersection."""
        for intersection in self.intersections.values():
            if intersection.last_decision:
                return intersection.last_decision
        return make_decision({})


# Singleton
_BRAIN: Optional[TrafficBrain] = None


def get_traffic_brain() -> TrafficBrain:
    """Get or create the singleton TrafficBrain instance."""
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = TrafficBrain(grid_rows=2, grid_cols=2)
    return _BRAIN


def reset_traffic_brain():
    """Reset the singleton TrafficBrain."""
    global _BRAIN
    _BRAIN = TrafficBrain(grid_rows=2, grid_cols=2)
