import cv2

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


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

    def __init__(self):
        self.model = None
        self.model_ready = False
        self.runtime_class_map = dict(self.VEHICLE_CLASS_MAP)
        self._load_model()

    def _build_runtime_class_map(self):
        class_map = dict(self.VEHICLE_CLASS_MAP)
        if self.model is None:
            self.runtime_class_map = class_map
            return
        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        for cls_id, cls_name in names.items():
            name = str(cls_name).lower().replace("_", " ").strip()
            if name in self.AUTO_LABEL_ALIASES:
                class_map[int(cls_id)] = "auto"
        self.runtime_class_map = class_map

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

    def detect(self, frame, conf_threshold=0.35, imgsz=480):
        counts = {"cars": 0, "bikes": 0, "buses": 0, "trucks": 0, "autos": 0}
        if frame is None:
            return [], counts, 0

        h, _ = frame.shape[:2]
        queue_zone_y = int(h * 0.68)
        detections = []

        if not self.model_ready:
            return detections, counts, queue_zone_y

        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=0.5,
            verbose=False,
            imgsz=imgsz,
            classes=list(self.runtime_class_map.keys()),
        )

        if not results:
            return detections, counts, queue_zone_y

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
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
            if vehicle_type is None:
                continue

            x1, y1, x2, y2 = b.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            area = bw * bh

            counts[key_map[vehicle_type]] += 1

            detections.append(
                {
                    "bbox": (x1, y1, bw, bh),
                    "type": vehicle_type,
                    "area": int(area),
                    "confidence": float(b.conf.item()),
                    "in_queue": y2 >= queue_zone_y,
                }
            )

        return detections, counts, queue_zone_y


_DETECTOR = _VehicleDetector()


def detect_vehicles(frame, conf_threshold=0.35, imgsz=480):
    return _DETECTOR.detect(frame, conf_threshold=conf_threshold, imgsz=imgsz)
