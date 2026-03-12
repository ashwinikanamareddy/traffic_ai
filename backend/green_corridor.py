"""
Emergency Green Corridor System
===============================
Manages the automatic clearing of traffic signals along a route when an 
emergency vehicle is detected. Matches the exact 8-step spec.
"""
import pydeck as pdk
from typing import Dict, List, Tuple


# Predefined coordinates mapped to our system's INT-XX IDs for pydeck visualization
# Bangalore coords roughly
INTERSECTION_COORDS = {
    "INT-01": [77.5946, 12.9716],
    "INT-02": [77.6046, 12.9716],
    "INT-03": [77.5946, 12.9616],
    "INT-04": [77.6046, 12.9616],
    "Hospital": [77.6150, 12.9750]  # Destination Hospital
}

# ═══════════════════════════════════════════════════════
# STEP 1: Emergency Detection Trigger
# ═══════════════════════════════════════════════════════
def detect_emergency(vehicle_counts: Dict[str, int]) -> bool:
    """
    Input from object detection: vehicle_counts
    If ambulances > 0 OR fire_trucks > 0 -> emergency_mode = True
    """
    ambulances = int(vehicle_counts.get("ambulances", 0))
    fire_trucks = int(vehicle_counts.get("fire_trucks", 0))
    
    return (ambulances > 0) or (fire_trucks > 0)


# ═══════════════════════════════════════════════════════
# STEP 2: Identify Emergency Route
# ═══════════════════════════════════════════════════════
def generate_emergency_route(start_intersection: str = "INT-01") -> List[str]:
    """
    Simulate a route from: Current Intersection → Destination Hospital.
    Uses a predefined list of intersections for the prototype.
    """
    # Simple predefined routes based on start point
    routes = {
        "INT-01": ["INT-01", "INT-02", "INT-04", "Hospital"],
        "INT-02": ["INT-02", "INT-04", "Hospital"],
        "INT-03": ["INT-03", "INT-04", "Hospital"],
        "INT-04": ["INT-04", "Hospital"],
    }
    return routes.get(start_intersection, ["INT-01", "INT-02", "Hospital"])


# ═══════════════════════════════════════════════════════
# STEP 3: Signal Preemption
# ═══════════════════════════════════════════════════════
def activate_green_corridor(route: List[str]) -> Dict[str, str]:
    """
    For each intersection in the route:
    Override normal signal cycle.
    Set signal to GREEN for the emergency lane.
    Example: 
      Intersection_A -> GREEN
      Intersection_B -> PREPARE GREEN
      Intersection_C -> GREEN
    Returns dict mapping: intersection_id -> signal_status
    """
    preemption_plan = {}
    
    for i, node in enumerate(route):
        if node == "Hospital":
            continue
            
        if i == 0:
            preemption_plan[node] = "GREEN"
        elif i == 1:
            preemption_plan[node] = "PREPARE GREEN"
        else:
            preemption_plan[node] = "GREEN"
            
    return preemption_plan


# ═══════════════════════════════════════════════════════
# STEP 4 & 5: Green Corridor Visualization & Map Integration
# ═══════════════════════════════════════════════════════
def visualize_corridor_on_map(route: List[str], vehicle_type: str = "Ambulance"):
    """
    Use pydeck to display a city map.
    Plot:
      - Emergency vehicle current location
      - Hospital destination
      - Green corridor route (highlighted bright green)
    Returns: list of pydeck layers
    """
    path_coords = []
    points_data = []
    
    for i, node in enumerate(route):
        if node in INTERSECTION_COORDS:
            coord = INTERSECTION_COORDS[node]
            path_coords.append(coord)
            
            # Identify marker type
            if i == 0:
                name = f"🚨 {vehicle_type} (Current)"
                color = [239, 68, 68, 255] # Red
            elif node == "Hospital":
                name = "🏥 Destination Hospital"
                color = [59, 130, 246, 255] # Blue
            else:
                name = f"🚦 {node} (Cleared)"
                color = [34, 197, 94, 255] # Green
                
            points_data.append({
                "name": name,
                "coordinates": coord,
                "color": color
            })

    # Route Line Layer
    route_layer = pdk.Layer(
        "PathLayer",
        data=[{"path": path_coords, "name": "Green Corridor"}],
        get_path="path",
        get_color=[34, 197, 94, 255], # Bright green
        width_scale=20,
        width_min_pixels=5,
        get_width=5,
    )
    
    # Points Layer
    points_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_data,
        get_position="coordinates",
        get_color="color",
        get_radius=150,
        radius_min_pixels=8,
        radius_max_pixels=15,
        pickable=True,
    )
    
    return [route_layer, points_layer]
