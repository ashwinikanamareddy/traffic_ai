import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _detect_emergency_colors(frame, bbox, min_ratio=0.02):
    """Analyze a vehicle bounding box for red/blue emergency siren colors using HSV."""
    x, y, w, h = bbox
    if w < 10 or h < 10:
        return False, 0.0, ""

    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)

    # Analyze the top 40% of the bounding box (where sirens/lights typically are)
    siren_y2 = y1 + int((y2 - y1) * 0.4)
    roi = frame[y1:siren_y2, x1:x2]
    if roi.size == 0:
        return False, 0.0, ""

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total_pixels = max(1, roi.shape[0] * roi.shape[1])

    # Red color ranges (wraps around hue 0/180)
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_pixels = int(cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2))

    # Blue color range
    blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([130, 255, 255]))
    blue_pixels = int(cv2.countNonZero(blue_mask))

    # White color range (ambulance body)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    white_pixels = int(cv2.countNonZero(white_mask))

    red_ratio = red_pixels / total_pixels
    blue_ratio = blue_pixels / total_pixels
    white_ratio = white_pixels / total_pixels

    emergency_score = (red_ratio * 2.0 + blue_ratio * 2.0 + white_ratio * 0.3)

    # Determine emergency type
    if red_ratio >= min_ratio and blue_ratio >= min_ratio:
        return True, float(min(1.0, emergency_score)), "ambulance"
    elif red_ratio >= min_ratio * 1.5:
        return True, float(min(1.0, emergency_score)), "fire_truck"
    elif blue_ratio >= min_ratio * 1.5:
        return True, float(min(1.0, emergency_score)), "police"

    return False, float(min(1.0, emergency_score)), ""


class _VehicleDetector:
    # COCO classes: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
    VEHICLE_CLASS_MAP = {
        1: "bike",
        2: "car",
        3: "bike",
        5: "bus",
        7: "truck",
    }
    AUTO_LABEL_ALIASES = {
        "auto",
        "auto rickshaw",
        "auto-rickshaw",
        "autorickshaw",
        "rickshaw",
        "three wheeler",
        "three-wheeler",
        "three_wheeler",
        "tuk tuk",
        "tuktuk",
    }
    AUX_CLASS_ALIASES = {
        "person": {"person", "rider"},
        "cell_phone": {"cell phone", "mobile phone", "phone", "cellphone", "mobile"},
        "helmet": {"helmet", "hardhat", "hard hat"},
        "no_helmet": {"no helmet", "without helmet", "no-helmet", "helmetless", "nohelmet"},
        "heavy_load": {"heavy load", "overloaded", "overload", "overloaded vehicle", "heavy vehicle load"},
    }

    def __init__(self):
        self.model = None
        self.model_ready = False
        self.runtime_class_map = dict(self.VEHICLE_CLASS_MAP)
        self.aux_runtime_class_map = {k: set() for k in self.AUX_CLASS_ALIASES.keys()}
        self._load_model()

    def _build_runtime_class_map(self):
        class_map = dict(self.VEHICLE_CLASS_MAP)
        aux_map = {k: set() for k in self.AUX_CLASS_ALIASES.keys()}
        if self.model is None:
            self.runtime_class_map = class_map
            self.aux_runtime_class_map = aux_map
            return
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        for cls_id, cls_name in names.items():
            name = str(cls_name).lower().replace("_", " ").strip()
            if name in self.AUTO_LABEL_ALIASES:
                class_map[int(cls_id)] = "auto"
            for aux_name, aliases in self.AUX_CLASS_ALIASES.items():
                if name in aliases:
                    aux_map[aux_name].add(int(cls_id))
        self.runtime_class_map = class_map
        self.aux_runtime_class_map = aux_map

    def _load_model(self):
        if YOLO is None:
            return
        try:
            self.model = YOLO("yolov8n.pt")
            self._build_runtime_class_map()
            self.model_ready = True
        except Exception:
            self.model = None
            self.model_ready = False
            self.runtime_class_map = dict(self.VEHICLE_CLASS_MAP)
            self.aux_runtime_class_map = {k: set() for k in self.AUX_CLASS_ALIASES.keys()}

    def detect(self, frame, conf_threshold=0.35, imgsz=480, include_aux=False):
        counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "autos": 0}
        aux_detections = {k: [] for k in self.AUX_CLASS_ALIASES.keys()}
        emergency_vehicles = []
        if frame is None:
            if include_aux:
                return [], counts, 0, aux_detections, emergency_vehicles
            return [], counts, 0

        h, _ = frame.shape[:2]
        queue_zone_y = int(h * 0.68)
        detections = []

        if not self.model_ready:
            if include_aux:
                return detections, counts, queue_zone_y, aux_detections, emergency_vehicles
            return detections, counts, queue_zone_y

        target_classes = set(self.runtime_class_map.keys())
        if include_aux:
            for ids in self.aux_runtime_class_map.values():
                target_classes.update(ids)

        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.5,
            verbose=False,
            imgsz=imgsz,
            classes=sorted(list(target_classes)),
        )

        if not results:
            if include_aux:
                return detections, counts, queue_zone_y, aux_detections, emergency_vehicles
            return detections, counts, queue_zone_y

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            if include_aux:
                return detections, counts, queue_zone_y, aux_detections, emergency_vehicles
            return detections, counts, queue_zone_y

        key_map = {
            "car": "cars",
            "bike": "bikes",
            "bus": "buses",
            "truck": "trucks",
            "auto": "autos",
        }

        for b in boxes:
            cls_id = int(b.cls.item())
            vehicle_type = self.runtime_class_map.get(cls_id)

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            area = bw * bh
            confidence = float(b.conf.item())

            if vehicle_type is None:
                if include_aux:
                    for aux_name, ids in self.aux_runtime_class_map.items():
                        if cls_id in ids:
                            aux_detections[aux_name].append(
                                {
                                    "bbox": (x1, y1, bw, bh),
                                    "xyxy": (x1, y1, x2, y2),
                                    "area": int(area),
                                    "confidence": confidence,
                                }
                            )
                            break
                continue

            counts[key_map[vehicle_type]] += 1

            det = {
                "bbox": (x1, y1, bw, bh),
                "type": vehicle_type,
                "area": int(area),
                "confidence": confidence,
                "in_queue": y2 >= queue_zone_y,
            }
            detections.append(det)

            # Emergency vehicle detection via color heuristics
            if include_aux and vehicle_type in ("car", "bus", "truck"):
                is_emergency, em_score, em_type = _detect_emergency_colors(
                    frame, (x1, y1, bw, bh)
                )
                if is_emergency and em_score > 0.03:
                    emergency_vehicles.append({
                        "bbox": (x1, y1, bw, bh),
                        "type": em_type,
                        "vehicle_base_type": vehicle_type,
                        "confidence": confidence,
                        "emergency_score": em_score,
                        "in_queue": y2 >= queue_zone_y,
                    })

        if include_aux:
            return detections, counts, queue_zone_y, aux_detections, emergency_vehicles
        return detections, counts, queue_zone_y


_DETECTOR = _VehicleDetector()


def detect_vehicles(frame, conf_threshold=0.35, imgsz=480, include_aux=False):
    return _DETECTOR.detect(frame, conf_threshold=conf_threshold, imgsz=imgsz, include_aux=include_aux)

