from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('/workspace/PHA_YOLO/runs/obb/       /weights/best.pt')
metrics = model.val(
  data='/workspace/PHA_YOLO/ultralytics/cfg/datasets/DOTA.yaml',
  split='test'
)
