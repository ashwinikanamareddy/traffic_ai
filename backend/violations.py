from collections import defaultdict, deque
from typing import Dict, List, Tuple


Line = Tuple[Tuple[float, float], Tuple[float, float]]


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return float((dx * dx + dy * dy) ** 0.5)


def _line_side(point: Tuple[float, float], line: Line) -> float:
    (x1, y1), (x2, y2) = line
    px, py = point
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def _heading_deg(prev: Tuple[float, float], curr: Tuple[float, float]) -> float:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if dx == 0 and dy == 0:
        return 0.0
    import math

    return float(math.degrees(math.atan2(dy, dx)))


class SignalController:
    def __init__(self, red_frames: int = 120, green_frames: int = 120):
        self.red_frames = max(1, int(red_frames))
        self.green_frames = max(1, int(green_frames))
        self.cycle = self.red_frames + self.green_frames

    def state_for_frame(self, frame_id: int) -> str:
        idx = int(frame_id) % self.cycle
        return "RED" if idx < self.red_frames else "GREEN"


class ViolationDetector:
    def __init__(
        self,
        stop_line: Line,
        signal_controller: SignalController,
        speed_threshold: float = 32.0,
        acceleration_threshold: float = 18.0,
        direction_change_threshold: float = 48.0,
        zigzag_heading_threshold: float = 20.0,
        zigzag_window: int = 6,
    ):
        self.stop_line = stop_line
        self.signal_controller = signal_controller

        self.speed_threshold = float(speed_threshold)
        self.acceleration_threshold = float(acceleration_threshold)
        self.direction_change_threshold = float(direction_change_threshold)
        self.zigzag_heading_threshold = float(zigzag_heading_threshold)
        self.zigzag_window = max(4, int(zigzag_window))

        self._prev_center: Dict[int, Tuple[float, float]] = {}
        self._prev_speed: Dict[int, float] = defaultdict(float)
        self._prev_side: Dict[int, float] = {}
        self._heading_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.zigzag_window))
        self._red_crossing_cooldown: Dict[int, int] = defaultdict(int)
        self._rash_cooldown: Dict[int, int] = defaultdict(int)

    def _detect_zigzag(self, track_id: int) -> bool:
        headings = list(self._heading_history[track_id])
        if len(headings) < 4:
            return False

        diffs = []
        for i in range(1, len(headings)):
            diffs.append(_normalize_angle_deg(headings[i] - headings[i - 1]))

        sign_changes = 0
        for i in range(1, len(diffs)):
            a = diffs[i - 1]
            b = diffs[i]
            if abs(a) < self.zigzag_heading_threshold or abs(b) < self.zigzag_heading_threshold:
                continue
            if (a > 0 > b) or (a < 0 < b):
                sign_changes += 1

        return sign_changes >= 2

    def update(self, tracks: List[Dict], frame_id: int, fps: float, timestamp: str):
        fps = max(1.0, float(fps))
        signal_state = self.signal_controller.state_for_frame(frame_id)

        frame_events = []

        for track in tracks:
            track_id = int(track["track_id"])
            center = (float(track["center_x"]), float(track["center_y"]))

            prev_center = self._prev_center.get(track_id)
            speed = 0.0
            acceleration = 0.0
            heading_change = 0.0

            if prev_center is not None:
                dist = _distance(prev_center, center)
                speed = dist * fps
                prev_speed = float(self._prev_speed.get(track_id, 0.0))
                acceleration = (speed - prev_speed) * fps

                prev_heading = self._heading_history[track_id][-1] if self._heading_history[track_id] else None
                heading = _heading_deg(prev_center, center)
                self._heading_history[track_id].append(heading)
                if prev_heading is not None:
                    heading_change = abs(_normalize_angle_deg(heading - prev_heading))
            else:
                self._heading_history[track_id].append(0.0)

            # Red-light jump detection (crossing stop line while red)
            previous_side = self._prev_side.get(track_id)
            current_side = _line_side(center, self.stop_line)
            crossed = previous_side is not None and ((previous_side > 0 >= current_side) or (previous_side < 0 <= current_side))

            if self._red_crossing_cooldown[track_id] > 0:
                self._red_crossing_cooldown[track_id] -= 1

            if crossed and signal_state == "RED" and self._red_crossing_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Red Light Jump",
                        "signal_state": signal_state,
                        "vehicle_type": track.get("vehicle_type", "unknown"),
                    }
                )
                self._red_crossing_cooldown[track_id] = int(fps)

            # Rash driving detection
            is_zigzag = self._detect_zigzag(track_id)
            rash = (
                (speed >= self.speed_threshold and abs(acceleration) >= self.acceleration_threshold)
                or (heading_change >= self.direction_change_threshold and speed >= (self.speed_threshold * 0.6))
                or is_zigzag
            )

            if self._rash_cooldown[track_id] > 0:
                self._rash_cooldown[track_id] -= 1

            if rash and self._rash_cooldown[track_id] == 0:
                frame_events.append(
                    {
                        "track_id": track_id,
                        "frame": int(frame_id),
                        "timestamp": timestamp,
                        "violation_type": "Rash Driving",
                        "signal_state": signal_state,
                        "vehicle_type": track.get("vehicle_type", "unknown"),
                        "speed": float(speed),
                        "acceleration": float(acceleration),
                        "angle_change": float(heading_change),
                        "zig_zag": bool(is_zigzag),
                    }
                )
                self._rash_cooldown[track_id] = int(fps * 0.5)

            self._prev_center[track_id] = center
            self._prev_speed[track_id] = speed
            self._prev_side[track_id] = current_side

        return signal_state, frame_events
