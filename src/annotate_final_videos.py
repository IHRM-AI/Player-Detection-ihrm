import cv2
import json
from ultralytics import YOLO

def annotate_video(video_path, out_path, model_path, tracker_cfg, id_map=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    model = YOLO(model_path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            conf=0.25,
            iou=0.5,
            stream=False,
            verbose=False
        )

        if not results:
            out.write(frame)
            continue

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            out.write(frame)
            continue

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        for box, tid in zip(xyxy, ids):
            x1, y1, x2, y2 = map(int, box)

            # Map tacticam ID to broadcast ID if mapping provided
            orig_id = f"id_{tid}"
            final_id = id_map.get(orig_id, orig_id) if id_map else orig_id

            label = f"{final_id}"
            color = (0, 255, 0) if id_map else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Saved: {out_path}")

#  Load mapping
with open("outputs/player_id_mapping.json", "r") as f:
    id_mapping = json.load(f)

#  Broadcast video (original IDs)
annotate_video(
    video_path="videos/broadcast.mp4",
    out_path="outputs/broadcast_final.mp4",
    model_path="weights/best.pt",
    tracker_cfg="botsort.yaml",
    id_map=None
)

#  Tacticam video (mapped IDs)
annotate_video(
    video_path="videos/tacticam.mp4",
    out_path="outputs/tacticam_final.mp4",
    model_path="weights/best.pt",
    tracker_cfg="botsort.yaml",
    id_map=id_mapping
)