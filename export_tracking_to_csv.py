import cv2
import csv
from ultralytics import YOLO

# Load the trained model
model = YOLO("weights/best.pt")

# Tracking parameters
tracker = "botsort.yaml"
conf_threshold = 0.25
iou_threshold = 0.5

# Open video
cap = cv2.VideoCapture("videos/broadcast.mp4")
frame_idx = 0

# Output CSV file
csv_file = open("outputs/broadcast_tracking.csv", mode="w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "track_id", "class_id", "x1", "y1", "x2", "y2", "confidence"])

while True:
    success, frame = cap.read()
    if not success:
        break

    results = model.track(frame, tracker=tracker, conf=conf_threshold, iou=iou_threshold, persist=True)[0]
    boxes = results.boxes

    if boxes.id is not None:
        for i in range(len(boxes.id)):
            track_id = int(boxes.id[i].cpu().item())
            class_id = int(boxes.cls[i].cpu().item())
            conf = float(boxes.conf[i].cpu().item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().tolist()

            writer.writerow([frame_idx, track_id, class_id, x1, y1, x2, y2, conf])

    frame_idx += 1

cap.release()
csv_file.close()
print("✅ CSV saved as outputs/broadcast_tracking.csv")
