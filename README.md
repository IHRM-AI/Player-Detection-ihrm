
Player Re-Identification Across Multiple Camera Views

📌 Overview
This project tackles the problem of player re-identification in sports videos — matching and tracking the same players across two different camera views: a broadcast video and a tacticam video. We use a combination of deep learning models for detection, tracking, and appearance-based re-identification, followed by ID matching using cosine similarity and the Hungarian algorithm. The result is two consistently annotated videos with unified player IDs.

🎯 Objectives
Detect and track players using YOLOv8 and BoT-SORT
Extract per-player appearance features using a ReID model (OSNet)
Match player identities across camera views using cosine similarity
Relabel IDs in the tacticam video to match the broadcast IDs
Annotate and save final videos with consistent global IDs
Prototype enhancements like OCR-based jersey number detection and feature fusion

🧱 Project Structure
player-reid-liatal/
├── detect.py                 # (Optional) Detection-only mode
├── track.py                  # Tracks players using YOLOv8 + BoT-SORT
├── reid.py                   # Extracts appearance features (ReID embeddings)
├── match.py                  # Matches players using cosine similarity + Hungarian algorithm
├── relabel.py                # Rewrites tacticam tracking IDs using broadcast mapping
├── annotate.py               # Draws final annotations on both videos
├── extensions/
│   ├── ocr_jersey.py         # OCR-based jersey number recognition (EasyOCR)
│   └── ensemble_feature.py   # Feature fusion: OSNet + ResNet (prototype)
├── weights/
│   └── best.pt               # Trained YOLOv8 detection model
├── videos/
│   ├── broadcast.mp4         # Broadcast angle video
│   └── tacticam.mp4          # Tacticam video
├── outputs/
│   ├── player_id_mapping.json
│   ├── tacticam_final.mp4
│   └── broadcast_final.mp4
├── reid_embeddings/          # Stores crops and features
├── botsort.yaml              # Tracker config
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── report.md                 # Final report / project summary

⚙️ Setup Instructions

📥 Dependencies
Ensure Python 3.10+ is installed. Then install all dependencies:
pip install -r requirements.txt

🧪 Requirements.txt Contents
torch>=2.0.0
ultralytics>=8.1.0
opencv-python
scikit-learn
scipy
torchreid
easyocr

🚀 Step-by-Step Pipeline

✅ Step 1: Track Players
Track players in both videos using YOLOv8 + BoT-SORT.

python track.py --video videos/broadcast.mp4 --out reid_embeddings/broadcast/crops
python track.py --video videos/tacticam.mp4 --out reid_embeddings/tacticam/crops

✅ Step 2: Crop and Save Per-Player Images
Player crops are automatically saved during tracking in the above step (via BoT-SORT tracking IDs).

✅ Step 3: Extract ReID Embeddings
Use Torchreid to extract 2048-dim OSNet embeddings from each player's image crop folder.

python reid.py --crop-dir reid_embeddings/broadcast/crops --out reid_embeddings/broadcast/features/features.npy
python reid.py --crop-dir reid_embeddings/tacticam/crops --out reid_embeddings/tacticam/features/features.npy

✅ Step 4: Match Player IDs Across Cameras
Use cosine distance + Hungarian algorithm to compute the optimal one-to-one mapping.

python match.py

Output: outputs/tacticam_mapped_ids.mp4

✅ Step 6: Annotate Final Videos
Draw bounding boxes and player IDs on both videos. This ensures consistency.

python annotate.py

Outputs:
outputs/broadcast_final.mp4
outputs/tacticam_final.mp4

🧠 Step 7: Extensions / Enhancements
🔢 Jersey Number OCR (EasyOCR)
Use OCR on player image crops to recognize jersey numbers and improve ID confidence.

python extensions/ocr_jersey.py

🧬 Feature Fusion (ReID + ResNet)
Prototype: combine appearance features from OSNet and ResNet50.

python extensions/ensemble_feature.py

✨ Why This Project Is Unique
Multi-modal fusion: Combines object detection, ReID, and OCR.
Robust across views: Matches players across camera angles despite changes in lighting, orientation, and scale.
End-to-end automation: Each step is modular and can be run independently.
Real-world ready: Mimics real broadcast-tactical sports pipelines.

🔚 Contact
Developed by ISHAN MISHRA. For queries, feel free to reach out via GitHub[https://github.com/IHRM-AI] , email [ihrm.aiml@gmail.com] or Linkedin[https://www.linkedin.com/in/ihrm-ishan/]

Thank you for reviewing this project! We hope it demonstrates practical skill, thoughtful design, and potential for real-world sports AI systems.
=======
# Player-Detection-ihrm
Multi-camera player re-identification and tracking using YOLOv8, BoT-SORT, and Deep Re-ID (OSNet) with optional OCR and feature fusion. Tracks players across broadcast and tacticam sports videos with globally consistent IDs.
>>>>>>> ef9d11215e6b9bc6476321dbbdf1951c057b8aa5
