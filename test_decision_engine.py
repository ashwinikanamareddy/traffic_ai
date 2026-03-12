"""Test the AI Traffic Decision Engine."""
import sys
sys.path.insert(0, r'c:\Users\Ashwini\OneDrive\Desktop\traffic_ai_round2')

from backend.traffic_brain import (
    calculate_density, decide_signal_timing, check_emergency,
    generate_ai_decision, make_decision, get_traffic_brain,
)

# Test 1: LOW traffic (score < 20)
score, level = calculate_density({"cars": 5, "bikes": 4, "buses": 0, "trucks": 0})
print(f"Test LOW: score={score}, level={level}")
assert level == "LOW" and score == 7.0, f"Failed: {score}, {level}"

# Test 2: MEDIUM traffic (20 <= score <= 50)
score, level = calculate_density({"cars": 10, "bikes": 6, "buses": 2, "trucks": 1})
print(f"Test MEDIUM: score={score}, level={level}")
assert level == "MEDIUM" and score == 22.0, f"Failed: {score}, {level}"

# Test 3: HIGH traffic (score > 50)
score, level = calculate_density({"cars": 20, "bikes": 10, "buses": 5, "trucks": 5})
print(f"Test HIGH: score={score}, level={level}")
assert level == "HIGH" and score == 55.0, f"Failed: {score}, {level}"

# Test 4: Signal timing
assert decide_signal_timing("LOW") == 15
assert decide_signal_timing("MEDIUM") == 30
assert decide_signal_timing("HIGH") == 45
print("Signal timing OK: LOW=15s, MEDIUM=30s, HIGH=45s")

# Test 5: Emergency check
em, et = check_emergency({"cars": 5, "ambulances": 1})
assert em is True and et == "ambulance"
em, et = check_emergency({"cars": 5, "fire_trucks": 1})
assert em is True and et == "fire_truck"
em, et = check_emergency({"cars": 5})
assert em is False
print("Emergency check OK")

# Test 6: AI decision messages
msg = generate_ai_decision(55.0, "HIGH", 45, False, "")
print(f"HIGH msg: {msg}")
assert "Heavy traffic" in msg and "45 seconds" in msg

msg = generate_ai_decision(25.0, "MEDIUM", 30, False, "")
print(f"MED msg: {msg}")
assert "Moderate traffic" in msg and "30 seconds" in msg

msg = generate_ai_decision(10.0, "LOW", 15, False, "")
print(f"LOW msg: {msg}")
assert "Low traffic" in msg and "15 seconds" in msg

msg = generate_ai_decision(5.0, "LOW", 60, True, "ambulance")
print(f"EM msg: {msg}")
assert "Ambulance" in msg and "60 seconds" in msg

# Test 7: Full pipeline - emergency overrides to 60s
result = make_decision({"cars": 5, "ambulances": 1})
print(f"Emergency pipeline: score={result['traffic_score']}, dur={result['signal_duration']}s")
print(f"  Message: {result['ai_decision_message']}")
assert result["emergency_mode"] is True
assert result["signal_duration"] == 60

# Test 8: Brain integration
brain = get_traffic_brain()
brain.tick({"vehicle_counts": {"cars": 20, "bikes": 10, "buses": 5, "trucks": 5},
            "total_vehicles": 40, "emergency_vehicles": []})
dec = brain.get_latest_decision()
print(f"Brain decision: density={dec['traffic_density']}, dur={dec['signal_duration']}s")
assert dec["traffic_density"] == "HIGH"
assert dec["signal_duration"] == 45

# Test 9: Brain simulation
result = brain.simulate_emergency("ambulance", "north")
print(f"Simulation: corridor={result['corridor']['corridor_id']}")
print(f"  AI msg: {result['decision']['ai_decision_message']}")
assert result["decision"]["emergency_mode"] is True
assert result["decision"]["signal_duration"] == 60

print("")
print("=== ALL 9 TESTS PASSED ===")
