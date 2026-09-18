import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO


class SonarAnomalyEngine:
    def __init__(self, model_path="models/best.pt", conf_thresh=0.40):
        # Always resolve model path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))

        abs_model_path = model_path
        if not os.path.isabs(model_path):
            abs_model_path = os.path.join(base_dir, model_path)

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(
                f"Model not found at: {abs_model_path}\n"
                f"Put best.pt inside: {os.path.join(base_dir, 'models')}"
            )

        # Load trained YOLO model
        self.model = YOLO(abs_model_path)

        self.conf = conf_thresh
        self.class_names = self.model.names

        print(f"[OK] Model loaded: {abs_model_path}")
        print(f"[OK] Classes: {list(self.class_names.values())}")

    # ---------------------------------------------------------
    # HAZARD CLASSIFICATION
    # ---------------------------------------------------------
    def estimate_hazard_level(self, cls_name, conf):
        """
        Application-level hazard classification based on
        the five classes present in the trained model.

        This is NOT a model accuracy metric.
        """

        # Highest-priority detected object
        if cls_name == "mine_cylinder":
            return "CRITICAL"

        # Potentially significant underwater structures
        if cls_name in ["shipwreck", "submarine_pipeline"]:
            return "HIGH"

        # Marine entanglement / fishing equipment
        if cls_name in ["ghost_net", "crab_pot"]:
            return "MEDIUM"

        return "LOW"

    # ---------------------------------------------------------
    # SONAR IMAGE DENOISING
    # ---------------------------------------------------------
    def apply_noise_filter(self, img):
        return cv2.bilateralFilter(
            img,
            d=5,
            sigmaColor=50,
            sigmaSpace=50
        )

    # ---------------------------------------------------------
    # PROCESS SONAR IMAGE
    # ---------------------------------------------------------
    def process_sonar_image(
        self,
        img_path,
        output_path,
        base_lat=18.9438,
        base_lon=72.8360,
        range_meters=50.0,
        conf_override=None,
    ):

        # Read image
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(
                f"Could not read image: {img_path}"
            )

        h_img, w_img = img.shape[:2]

        # Apply sonar noise reduction
        denoised = self.apply_noise_filter(img)

        # Confidence threshold
        conf_val = (
            float(conf_override)
            if conf_override is not None
            else float(self.conf)
        )

        # YOLO inference
        results = self.model(
            denoised,
            conf=conf_val,
            verbose=False
        )[0]

        detections = []

        # Original image for annotation
        annotated = img.copy()

        # -----------------------------------------------------
        # APPROXIMATE SCALE
        # -----------------------------------------------------
        meters_per_pixel = (
            float(range_meters) /
            max(h_img, w_img)
        )

        # Hazard colors in BGR format
        hazard_colors = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 100, 255),
            "MEDIUM": (0, 200, 255),
            "LOW": (0, 255, 100),
        }

        # -----------------------------------------------------
        # PROCESS EACH DETECTION
        # -----------------------------------------------------
        for idx, box in enumerate(results.boxes):

            cls_id = int(box.cls[0])

            # Safety check for unknown class
            if cls_id not in self.class_names:
                cls_name = f"class_{cls_id}"
            else:
                cls_name = self.class_names[cls_id]

            # YOLO confidence -> percentage
            conf = float(box.conf[0]) * 100.0

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # -------------------------------------------------
            # OBJECT DIMENSIONS
            # -------------------------------------------------
            dim_w_m = round(
                (x2 - x1) * meters_per_pixel,
                2
            )

            dim_h_m = round(
                (y2 - y1) * meters_per_pixel,
                2
            )

            # -------------------------------------------------
            # OBJECT CENTER
            # -------------------------------------------------
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # -------------------------------------------------
            # APPROXIMATE GEOLOCATION
            # -------------------------------------------------
            lat_offset = (
                ((h_img / 2.0) - center_y)
                * (meters_per_pixel / 111000.0)
            )

            lon_offset = (
                (center_x - (w_img / 2.0))
                * (
                    meters_per_pixel
                    / (
                        111000.0
                        * np.cos(np.radians(base_lat))
                    )
                )
            )

            anomaly_lat = round(
                base_lat + lat_offset,
                6
            )

            anomaly_lon = round(
                base_lon + lon_offset,
                6
            )

            # -------------------------------------------------
            # HAZARD
            # -------------------------------------------------
            hazard = self.estimate_hazard_level(
                cls_name,
                conf
            )

            color = hazard_colors.get(
                hazard,
                (0, 255, 0)
            )

            # -------------------------------------------------
            # DRAW BOUNDING BOX
            # -------------------------------------------------
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Label
            label = (
                f"{cls_name.upper()} "
                f"{conf:.0f}%"
            )

            (lw, lh), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2
            )

            # Label background
            cv2.rectangle(
                annotated,
                (
                    x1,
                    max(0, y1 - lh - 10)
                ),
                (
                    x1 + lw + 8,
                    y1
                ),
                color,
                -1
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (
                    x1 + 4,
                    max(15, y1 - 5)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

            # -------------------------------------------------
            # DETECTION RECORD
            # -------------------------------------------------
            detections.append(
                {
                    "anomaly_id": f"ANO-{idx + 1:03d}",
                    "classification": cls_name,
                    "confidence": round(conf, 2),
                    "hazard_level": hazard,
                    "latitude": anomaly_lat,
                    "longitude": anomaly_lon,
                    "width_m": dim_w_m,
                    "length_m": dim_h_m,
                    "area_sq_m": round(
                        dim_w_m * dim_h_m,
                        2
                    ),
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2
                    ],
                }
            )

        # -----------------------------------------------------
        # SAVE RESULT IMAGE
        # -----------------------------------------------------
        output_dir = os.path.dirname(output_path)

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True
            )

        ok = cv2.imwrite(
            output_path,
            annotated
        )

        if not ok:
            raise RuntimeError(
                f"Failed to write result image: {output_path}"
            )

        return output_path, detections

    # ---------------------------------------------------------
    # GENERATE SUMMARY
    # ---------------------------------------------------------
    def generate_summary(self, detections):

        if not detections:
            return {
                "total": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
                "avg_confidence": 0,
                "total_area": 0,
                "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }

        return {
            "total": len(detections),

            "critical": len([
                d for d in detections
                if d["hazard_level"] == "CRITICAL"
            ]),

            "high": len([
                d for d in detections
                if d["hazard_level"] == "HIGH"
            ]),

            "medium": len([
                d for d in detections
                if d["hazard_level"] == "MEDIUM"
            ]),

            "low": len([
                d for d in detections
                if d["hazard_level"] == "LOW"
            ]),

            "avg_confidence": round(
                float(
                    np.mean([
                        d["confidence"]
                        for d in detections
                    ])
                ),
                2
            ),

            "total_area": round(
                sum(
                    d["area_sq_m"]
                    for d in detections
                ),
                2
            ),

            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }