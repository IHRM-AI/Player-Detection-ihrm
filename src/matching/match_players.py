import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment

#  Load saved ReID features
broadcast_feats = np.load("reid_embeddings/broadcast/features/features.npy", allow_pickle=True).item()
tacticam_feats  = np.load("reid_embeddings/tacticam/features/features.npy", allow_pickle=True).item()

# Extract ID lists and feature vectors
broadcast_ids = list(broadcast_feats.keys())
tacticam_ids  = list(tacticam_feats.keys())

broadcast_vectors = np.array([broadcast_feats[i] for i in broadcast_ids])
tacticam_vectors  = np.array([tacticam_feats[i] for i in tacticam_ids])

#  Compute cosine distance matrix
distance_matrix = cosine_distances(broadcast_vectors, tacticam_vectors)

#  Apply Hungarian Matching (Optimal one-to-one ID match)
row_inds, col_inds = linear_sum_assignment(distance_matrix)

#  Map tacticam ID → broadcast ID
final_mapping = {}
for r, c in zip(row_inds, col_inds):
    b_id = broadcast_ids[r]
    t_id = tacticam_ids[c]
    final_mapping[t_id] = b_id

# Save the mapping as JSON
os.makedirs("outputs", exist_ok=True)
with open("outputs/player_id_mapping.json", "w") as f:
    json.dump(final_mapping, f, indent=4)

#  Print Result
print("\n📊 Final Mapping (tacticam ID → broadcast ID):")
for t_id, b_id in final_mapping.items():
    print(f"{t_id} → {b_id}")
