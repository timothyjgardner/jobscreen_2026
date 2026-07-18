"""
Private feasibility check #6 — refined segmentation pipeline.

Like check #5 plus: boundary trimming when computing segment features,
label-based merging of adjacent segments, and a second clustering pass.
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


def seg_feature(a, b, trim=40):
    a2, b2 = a + trim, b - trim
    if b2 - a2 < 60:
        a2, b2 = a, b
    seg = Zs[a2:b2]
    segc = seg - seg.mean(0)
    U, S, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T
    P = (B @ B.T).ravel()
    xy = segc @ B
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(len(seg)), ang, 1)[0])
    return np.concatenate([3.0 * P, [2.0 * np.log(omega + 1e-9)]])


# --- change detection on short-window plane projectors ---
W, hop = 100, 10
starts = np.arange(0, T - W + 1, hop)
planes = []
for s in starts:
    seg = Zs[s:s + W]
    segc = seg - seg.mean(0)
    _, _, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T
    planes.append((B @ B.T).ravel())
planes = np.array(planes)

d = np.linalg.norm(planes[2:] - planes[:-2], axis=1)
thr = np.percentile(d, 75)
cand = np.where(d > thr)[0] + 1
bounds = []
for c in cand:
    if not bounds or c - bounds[-1] > 8:
        bounds.append(c)
bound_steps = [0] + [starts[b] + W // 2 for b in bounds] + [T]
segs = [(bound_steps[i], bound_steps[i + 1])
        for i in range(len(bound_steps) - 1)
        if bound_steps[i + 1] - bound_steps[i] > 30]

for it in range(3):
    F = np.array([seg_feature(a, b) for a, b in segs])
    km = KMeans(n_clusters=10, n_init=50, random_state=0).fit(F)
    lab = km.labels_
    # merge adjacent segments with identical labels
    merged = [list(segs[0]) + [lab[0]]]
    for (a, b), l in zip(segs[1:], lab[1:]):
        if l == merged[-1][2] and a - merged[-1][1] < 60:
            merged[-1][1] = b
        else:
            merged.append([a, b, l])
    new_segs = [(a, b) for a, b, _ in merged]
    if len(new_segs) == len(segs):
        segs = new_segs
        break
    segs = new_segs

F = np.array([seg_feature(a, b) for a, b in segs])
km = KMeans(n_clusters=10, n_init=50, random_state=0).fit(F)

labels = np.zeros(T, dtype=int)
for (a, b), l in zip(segs, km.labels_):
    labels[a:b] = l
covered = np.zeros(T, dtype=bool)
for a, b in segs:
    covered[a:b] = True
if not covered.all():
    idx_cov = np.where(covered)[0]
    idx_gap = np.where(~covered)[0]
    nearest = idx_cov[np.searchsorted(idx_cov, idx_gap).clip(0, len(idx_cov) - 1)]
    labels[idx_gap] = labels[nearest]

true_n = np.count_nonzero(np.diff(states)) + 1
print(f"{len(segs)} segments after merge (true: {true_n})")
print(f"refined ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")
