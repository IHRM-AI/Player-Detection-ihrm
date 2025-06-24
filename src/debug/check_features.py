import numpy as np

broadcast_feats = np.load("reid_embeddings/broadcast/features/features.npy", allow_pickle=True).item()
tacticam_feats  = np.load("reid_embeddings/tacticam/features/features.npy", allow_pickle=True).item()

print("Broadcast feature IDs:", list(broadcast_feats.keys()))
print("Tacticam feature IDs:", list(tacticam_feats.keys()))
print("Broadcast total:", len(broadcast_feats))
print("Tacticam total:", len(tacticam_feats))
