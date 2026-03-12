"""
Dynamic Signal Optimizer
========================
Replaces fixed-cycle signal timing with density-responsive optimization.
Each approach (N/S/E/W) at an intersection gets green time proportional
to its traffic density, bounded by configurable min/max durations.
"""

import time
from typing import Dict, List, Optional


class DynamicSignalOptimizer:
    """Manages adaptive signal timing for a single intersection."""

    APPROACHES = ("north", "south", "east", "west")

    def __init__(
        self,
        intersection_id: str = "INT-01",
        min_green_seconds: float = 15.0,
        max_green_seconds: float = 90.0,
        yellow_seconds: float = 3.0,
        base_cycle_seconds: float = 120.0,
    ):
        self.intersection_id = intersection_id
        self.min_green = max(5.0, float(min_green_seconds))
        self.max_green = max(self.min_green + 5, float(max_green_seconds))
        self.yellow_seconds = max(1.0, float(yellow_seconds))
        self.base_cycle = max(30.0, float(base_cycle_seconds))

        # Current densities per approach (0.0 - 1.0)
        self._densities: Dict[str, float] = {a: 0.0 for a in self.APPROACHES}
        # Computed green durations
        self._green_durations: Dict[str, float] = {a: self.min_green for a in self.APPROACHES}
        # Current active phase
        self._phase_index: int = 0
        self._phase_start: float = time.time()
        self._phase_elapsed: float = 0.0
        # Override for emergency
        self._emergency_override: Optional[str] = None
        # Decision history
        self._decisions: List[Dict] = []

    def update_densities(self, density_map: Dict[str, float]):
        """Update traffic density for each approach. Values should be 0.0-1.0."""
        for approach in self.APPROACHES:
            if approach in density_map:
                self._densities[approach] = max(0.0, min(1.0, float(density_map[approach])))
        self._recalculate_timings()

    def _recalculate_timings(self):
        """Recalculate green durations based on current densities."""
        total_density = sum(self._densities.values())
        if total_density <= 0:
            for a in self.APPROACHES:
                self._green_durations[a] = self.min_green
            return

        available_green = self.base_cycle - (self.yellow_seconds * len(self.APPROACHES))
        available_green = max(len(self.APPROACHES) * self.min_green, available_green)

        for approach in self.APPROACHES:
            proportion = self._densities[approach] / max(0.001, total_density)
            raw_green = proportion * available_green
            clamped = max(self.min_green, min(self.max_green, raw_green))
            self._green_durations[approach] = round(clamped, 1)

        self._decisions.append({
            "timestamp": time.time(),
            "type": "timing_update",
            "densities": dict(self._densities),
            "green_durations": dict(self._green_durations),
        })
        # Keep last 100 decisions
        self._decisions = self._decisions[-100:]

    def set_emergency_override(self, approach: Optional[str]):
        """Force a specific approach to green (emergency corridor)."""
        if approach and approach in self.APPROACHES:
            self._emergency_override = approach
            self._decisions.append({
                "timestamp": time.time(),
                "type": "emergency_override",
                "approach": approach,
            })
        else:
            self._emergency_override = None

    def clear_emergency_override(self):
        """Remove emergency override."""
        if self._emergency_override:
            self._decisions.append({
                "timestamp": time.time(),
                "type": "emergency_cleared",
                "approach": self._emergency_override,
            })
        self._emergency_override = None

    def get_signal_states(self) -> Dict[str, str]:
        """Get current signal state for all approaches."""
        if self._emergency_override:
            return {
                a: "GREEN" if a == self._emergency_override else "RED"
                for a in self.APPROACHES
            }

        states = {}
        for i, approach in enumerate(self.APPROACHES):
            if i == self._phase_index:
                states[approach] = "GREEN"
            else:
                states[approach] = "RED"
        return states

    def get_signal_state(self, approach: str) -> str:
        """Get signal state for a single approach."""
        states = self.get_signal_states()
        return states.get(approach, "RED")

    def advance_phase(self):
        """Move to the next phase in the cycle."""
        self._phase_index = (self._phase_index + 1) % len(self.APPROACHES)
        self._phase_start = time.time()
        self._phase_elapsed = 0.0

    def tick(self, elapsed_seconds: float = 1.0):
        """Advance the optimizer by elapsed_seconds. Auto-cycles phases."""
        if self._emergency_override:
            return

        self._phase_elapsed += elapsed_seconds
        current_approach = self.APPROACHES[self._phase_index]
        green_duration = self._green_durations[current_approach]

        if self._phase_elapsed >= green_duration:
            self.advance_phase()

    def get_timing_plan(self) -> Dict:
        """Return the full timing plan for display."""
        return {
            "intersection_id": self.intersection_id,
            "approaches": {
                a: {
                    "density": round(self._densities[a], 3),
                    "green_duration": self._green_durations[a],
                    "signal_state": self.get_signal_states()[a],
                }
                for a in self.APPROACHES
            },
            "current_phase": self.APPROACHES[self._phase_index],
            "phase_elapsed": round(self._phase_elapsed, 1),
            "emergency_override": self._emergency_override,
        }

    def get_decisions(self, last_n: int = 20) -> List[Dict]:
        """Return recent decisions."""
        return self._decisions[-last_n:]

    def get_densities(self) -> Dict[str, float]:
        return dict(self._densities)
