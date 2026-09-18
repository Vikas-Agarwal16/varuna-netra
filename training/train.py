import os
from ultralytics import YOLO

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset configuration
DATASET_CONFIG = os.path.join(
    BASE_DIR,
    "dataset_config",
    "data.yaml"
)

print("=" * 60)
print("MARINE DEBRIS YOLO TRAINING")
print("=" * 60)
print("Dataset config:", DATASET_CONFIG)
print("Config exists:", os.path.exists(DATASET_CONFIG))

# Create YOLO model from scratch
model = YOLO("yolo11n.yaml")

# Train model
results = model.train(
    data=DATASET_CONFIG,
    epochs=100,
    imgsz=640,
    batch=16,
    project=os.path.join(BASE_DIR, "runs"),
    name="marine_debris"
)

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)