# vision.py
import cv2
import numpy as np
from ultralytics import YOLO

yolo = YOLO("models/yolov8n.pt")

def detect_people(frame):
    results = yolo(frame, conf=0.3, classes=[0])
    count = len(results[0].boxes)
    centers = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = box
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    return count, centers

def optical_flow(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    return flow