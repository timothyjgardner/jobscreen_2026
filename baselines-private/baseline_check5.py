"""
Private feasibility check #5 — change-point segmentation + segment clustering.

1. Window features (plane projector + angular speed) as in check #4.
2. Detect change points from feature-distance spikes between adjacent windows.
3. Recompute one feature vector per detected segment (long, clean support).
4. k-means over segments, broadcast labels back to time steps.
"""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

npz = np.load(Path(__file__).resolve().parent.parent / 'data' / 'data.npz')
X = npz['X']
states = npz['states']
T = len(X)

Z = PCA(n_components=4).fit_transform(X - X.mean(0))
K = 7
kern = np.ones(K) / K
Zs = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, Z)


def window_feature(seg):
    segc = seg - seg.mean(0)
    U, S, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T
    P = (B @ B.T).ravel()
    xy = segc @ B
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(len(seg)), ang, 1)[0])
    return P, omega


W, hop = 100, 10
starts = np.arange(0, T - W + 1, hop)
planes = []
omegas = []
for s in starts:
    P, om = window_feature(Zs[s:s + W])
    planes.append(P)
    omegas.append(om)
planes = np.array(planes)
omegas = np.array(omegas)

# change score: projector distance between windows one step apart
d = np.linalg.norm(planes[2:] - planes[:-2], axis=1)
thr = np.percentile(d, 80)
cand = np.where(d > thr)[0] + 1

# collapse candidate runs into single boundaries
bounds = []
for c in cand:
    if not bounds or c - bounds[-1] > 10:   # windows, i.e. >100 steps apart
        bounds.append(c)
bound_steps = [starts[b] + W // 2 for b in bounds]
bound_steps = [0] + bound_steps + [T]

segs = [(bound_steps[i], bound_steps[i + 1])
        for i in range(len(bound_steps) - 1)
        if bound_steps[i + 1] - bound_steps[i] > 30]

print(f"{len(segs)} segments detected "
      f"(true segment count: {np.count_nonzero(np.diff(states)) + 1})")

feats = []
for a, b in segs:
    P, om = window_feature(Zs[a:b])
    feats.append(np.concatenate([3.0 * P, [1.5 * np.log(om + 1e-9)]]))
F = np.array(feats)

km = KMeans(n_clusters=10, n_init=50, random_state=0).fit(F)

labels = np.zeros(T, dtype=int)
for (a, b), l in zip(segs, km.labels_):
    labels[a:b] = l
# fill any gaps (dropped tiny segments) by nearest neighbour fill
covered = np.zeros(T, dtype=bool)
for a, b in segs:
    covered[a:b] = True
if not covered.all():
    idx_cov = np.where(covered)[0]
    idx_gap = np.where(~covered)[0]
    nearest = idx_cov[np.searchsorted(idx_cov, idx_gap).clip(0, len(idx_cov) - 1)]
    labels[idx_gap] = labels[nearest]

print(f"segment-cluster ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")
