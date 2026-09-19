import os
import cv2
import torch
import numpy as np
from datetime import datetime


class SonarAnomalyEngine:
    def __init__(self, model_path="models/best.torchscript", conf_thresh=0.40):

        # ---------------------------------------------------------
        # MODEL PATH
        # ---------------------------------------------------------
        base_dir = os.path.dirname(os.path.abspath(__file__))

        abs_model_path = model_path
        if not os.path.isabs(model_path):
            abs_model_path = os.path.join(base_dir, model_path)

        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(
                f"Model not found at: {abs_model_path}\n"
                f"Put best.torchscript inside: "
                f"{os.path.join(base_dir, 'models')}"
            )

        # ---------------------------------------------------------
        # LOAD TORCHSCRIPT MODEL DIRECTLY
        # ---------------------------------------------------------
        print(f"[INIT] Loading TorchScript model from: {abs_model_path}")

        self.model = torch.jit.load(
            abs_model_path,
            map_location="cpu"
        ).eval()

        # Keep CPU execution predictable on Render
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self.conf = conf_thresh

        # Your trained DRISHTI classes
        self.class_names = {
            0: "crab_pot",
            1: "submarine_pipeline",
            2: "shipwreck",
            3: "ghost_net",
            4: "mine_cylinder"
        }

        self.input_size = 320
        self.iou_threshold = 0.45

        print(f"[OK] Model loaded: {abs_model_path}")
        print(f"[OK] Classes: {list(self.class_names.values())}")
        print("[OK] Direct TorchScript inference enabled")

    # ---------------------------------------------------------
    # HAZARD CLASSIFICATION
    # ---------------------------------------------------------
    def estimate_hazard_level(self, cls_name, conf):

        """
        Application-level hazard classification based on
        the five classes present in the trained model.

        This is NOT a model accuracy metric.
        """

        if cls_name == "mine_cylinder":
            return "CRITICAL"

        if cls_name in ["shipwreck", "submarine_pipeline"]:
            return "HIGH"

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
    # LETTERBOX PREPROCESSING
    # ---------------------------------------------------------
    def letterbox(self, image, new_size=320):

        original_h, original_w = image.shape[:2]

        # Scale while preserving aspect ratio
        scale = min(
            new_size / original_w,
            new_size / original_h
        )

        new_w = int(round(original_w * scale))
        new_h = int(round(original_h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Create 320x320 canvas
        canvas = np.full(
            (new_size, new_size, 3),
            114,
            dtype=np.uint8
        )

        pad_x = (new_size - new_w) / 2
        pad_y = (new_size - new_h) / 2

        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))

        canvas[
            top:top + new_h,
            left:left + new_w
        ] = resized

        return canvas, scale, pad_x, pad_y

    # ---------------------------------------------------------
    # IOU
    # ---------------------------------------------------------
    def box_iou(self, box, boxes):

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])

        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection_w = np.maximum(0, x2 - x1)
        intersection_h = np.maximum(0, y2 - y1)

        intersection = (
            intersection_w *
            intersection_h
        )

        box_area = (
            max(0, box[2] - box[0]) *
            max(0, box[3] - box[1])
        )

        boxes_area = (
            np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
            np.maximum(0, boxes[:, 3] - boxes[:, 1])
        )

        union = (
            box_area +
            boxes_area -
            intersection +
            1e-6
        )

        return intersection / union

    # ---------------------------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # ---------------------------------------------------------
    def nms(self, boxes, scores, iou_threshold=0.45):

        if len(boxes) == 0:
            return []

        order = scores.argsort()[::-1]

        keep = []

        while len(order) > 0:

            current = order[0]

            keep.append(current)

            if len(order) == 1:
                break

            remaining = order[1:]

            ious = self.box_iou(
                boxes[current],
                boxes[remaining]
            )

            order = remaining[
                ious < iou_threshold
            ]

        return keep

    # ---------------------------------------------------------
    # DECODE TORCHSCRIPT OUTPUT
    # ---------------------------------------------------------
    def decode_predictions(
        self,
        output,
        original_shape,
        scale,
        pad_x,
        pad_y,
        conf_threshold
    ):

        # -----------------------------------------------------
        # Expected TorchScript output:
        #
        # [1, 9, 2100]
        #
        # 4 bbox values + 5 class scores
        # -----------------------------------------------------

        if isinstance(output, (tuple, list)):
            output = output[0]

        prediction = output.detach().cpu().numpy()

        # [1, 9, 2100] -> [2100, 9]
        prediction = prediction[0].transpose(1, 0)

        boxes_xywh = prediction[:, :4]
        class_scores = prediction[:, 4:]

        # Best class for every prediction
        class_ids = np.argmax(
            class_scores,
            axis=1
        )

        confidences = np.max(
            class_scores,
            axis=1
        )

        # Confidence filtering
        mask = confidences >= conf_threshold

        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # -----------------------------------------------------
        # Convert XYWH -> XYXY
        # -----------------------------------------------------

        boxes = np.zeros_like(boxes_xywh)

        boxes[:, 0] = (
            boxes_xywh[:, 0] -
            boxes_xywh[:, 2] / 2
        )

        boxes[:, 1] = (
            boxes_xywh[:, 1] -
            boxes_xywh[:, 3] / 2
        )

        boxes[:, 2] = (
            boxes_xywh[:, 0] +
            boxes_xywh[:, 2] / 2
        )

        boxes[:, 3] = (
            boxes_xywh[:, 1] +
            boxes_xywh[:, 3] / 2
        )

        # -----------------------------------------------------
        # Undo letterbox padding
        # -----------------------------------------------------

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y

        boxes /= scale

        # -----------------------------------------------------
        # Clip boxes to original image
        # -----------------------------------------------------

        original_h, original_w = original_shape[:2]

        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            original_w - 1
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            original_h - 1
        )

        # -----------------------------------------------------
        # CLASS-WISE NMS
        # -----------------------------------------------------

        final_indices = []

        for cls_id in np.unique(class_ids):

            cls_indices = np.where(
                class_ids == cls_id
            )[0]

            cls_boxes = boxes[cls_indices]
            cls_scores = confidences[cls_indices]

            keep = self.nms(
                cls_boxes,
                cls_scores,
                self.iou_threshold
            )

            final_indices.extend(
                cls_indices[keep].tolist()
            )

        # Highest confidence first
        final_indices.sort(
            key=lambda i: confidences[i],
            reverse=True
        )

        detections = []

        for i in final_indices:

            detections.append({
                "class_id": int(class_ids[i]),
                "confidence": float(confidences[i]),
                "box": boxes[i].tolist()
            })

        return detections

    # ---------------------------------------------------------
    # DIRECT TORCHSCRIPT INFERENCE
    # ---------------------------------------------------------
    def run_inference(
        self,
        image,
        conf_threshold
    ):

        # -----------------------------------------------------
        # Letterbox image
        # -----------------------------------------------------
        processed, scale, pad_x, pad_y = self.letterbox(
            image,
            self.input_size
        )

        # -----------------------------------------------------
        # BGR -> RGB
        # -----------------------------------------------------
        rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        )

        # -----------------------------------------------------
        # NumPy -> Torch Tensor
        # -----------------------------------------------------
        tensor = torch.from_numpy(
            rgb
        ).permute(
            2,
            0,
            1
        ).float()

        tensor = tensor.unsqueeze(0) / 255.0

        # -----------------------------------------------------
        # Inference
        # -----------------------------------------------------
        with torch.inference_mode():

            output = self.model(tensor)

        # -----------------------------------------------------
        # Decode predictions
        # -----------------------------------------------------
        detections = self.decode_predictions(
            output,
            image.shape,
            scale,
            pad_x,
            pad_y,
            conf_threshold
        )

        return detections

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

        # -----------------------------------------------------
        # READ IMAGE
        # -----------------------------------------------------
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(
                f"Could not read image: {img_path}"
            )

        h_img, w_img = img.shape[:2]

        # -----------------------------------------------------
        # APPLY SONAR NOISE REDUCTION
        # -----------------------------------------------------
        denoised = self.apply_noise_filter(img)

        # -----------------------------------------------------
        # CONFIDENCE THRESHOLD
        # -----------------------------------------------------
        conf_val = (
            float(conf_override)
            if conf_override is not None
            else float(self.conf)
        )

        # -----------------------------------------------------
        # DIRECT TORCHSCRIPT INFERENCE
        # -----------------------------------------------------
        raw_detections = self.run_inference(
            denoised,
            conf_val
        )

        detections = []

        # -----------------------------------------------------
        # ORIGINAL IMAGE FOR ANNOTATION
        # -----------------------------------------------------
        annotated = img.copy()

        # -----------------------------------------------------
        # APPROXIMATE SCALE
        # -----------------------------------------------------
        meters_per_pixel = (
            float(range_meters) /
            max(h_img, w_img)
        )

        # -----------------------------------------------------
        # HAZARD COLORS IN BGR
        # -----------------------------------------------------
        hazard_colors = {
            "CRITICAL": (0, 0, 255),
            "HIGH": (0, 100, 255),
            "MEDIUM": (0, 200, 255),
            "LOW": (0, 255, 100),
        }

        # -----------------------------------------------------
        # PROCESS EACH DETECTION
        # -----------------------------------------------------
        for idx, detection in enumerate(raw_detections):

            cls_id = detection["class_id"]

            # Safety check
            if cls_id not in self.class_names:
                cls_name = f"class_{cls_id}"
            else:
                cls_name = self.class_names[cls_id]

            # Confidence -> percentage
            conf = (
                detection["confidence"] *
                100.0
            )

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                detection["box"]
            )

            # -------------------------------------------------
            # OBJECT DIMENSIONS
            # -------------------------------------------------
            dim_w_m = round(
                (x2 - x1) *
                meters_per_pixel,
                2
            )

            dim_h_m = round(
                (y2 - y1) *
                meters_per_pixel,
                2
            )

            # -------------------------------------------------
            # OBJECT CENTER
            # -------------------------------------------------
            center_x = (
                x1 + x2
            ) / 2.0

            center_y = (
                y1 + y2
            ) / 2.0

            # -------------------------------------------------
            # APPROXIMATE GEOLOCATION
            # -------------------------------------------------
            lat_offset = (
                (
                    (h_img / 2.0) -
                    center_y
                )
                *
                (
                    meters_per_pixel /
                    111000.0
                )
            )

            lon_offset = (
                (
                    center_x -
                    (w_img / 2.0)
                )
                *
                (
                    meters_per_pixel /
                    (
                        111000.0 *
                        np.cos(
                            np.radians(base_lat)
                        )
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

            # -------------------------------------------------
            # LABEL
            # -------------------------------------------------
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

            # -------------------------------------------------
            # LABEL BACKGROUND
            # -------------------------------------------------
            label_y1 = max(
                0,
                y1 - lh - 10
            )

            label_y2 = y1

            cv2.rectangle(
                annotated,
                (
                    x1,
                    label_y1
                ),
                (
                    x1 + lw + 8,
                    label_y2
                ),
                color,
                -1
            )

            # -------------------------------------------------
            # LABEL TEXT
            # -------------------------------------------------
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
                    "anomaly_id": (
                        f"ANO-{idx + 1:03d}"
                    ),
                    "classification": cls_name,
                    "confidence": round(
                        conf,
                        2
                    ),
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
        output_dir = os.path.dirname(
            output_path
        )

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
                f"Failed to write result image: "
                f"{output_path}"
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