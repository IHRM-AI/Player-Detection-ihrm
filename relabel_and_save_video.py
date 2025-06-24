import cv2
import json
from ultralytics import YOLO

#  Load player ID mapping (tacticam ID → broadcast ID) 
with open("outputs/player_id_mapping.json", "r") as f:
    id_mapping = json.load(f)

#  Load YOLO model 
model = YOLO("weights/best.pt")

#  Setup video input/output 
video_path = "videos/tacticam.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_path = "outputs/tacticam_mapped_ids.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

# Process frame-by-frame
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        source=frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.25,
        iou=0.5,
        stream=False,
        verbose=False
    )

    if not results:
        out.write(frame)
        continue

    result = results[0]  # Single frame result
    boxes = result.boxes

    if boxes is None or boxes.id is None:
        out.write(frame)
        continue

    ids = boxes.id.cpu().numpy().astype(int)
    xyxy = boxes.xyxy.cpu().numpy()

    for box, tid in zip(xyxy, ids):
        x1, y1, x2, y2 = map(int, box)
        orig_id = f"id_{tid}"
        new_id = id_mapping.get(orig_id, orig_id)

        label = f"Player {new_id}"
        color = (0, 255, 0) if new_id != orig_id else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"✅ Saved remapped video to: {out_path}")
