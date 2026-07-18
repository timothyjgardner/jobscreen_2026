"""
Private feasibility check #7 — window features + sticky Viterbi decode.

W=200 plane projector + angular speed, k-means centers, then a dynamic
program over windows: cost = distance to center, plus a penalty per
state switch. Decoded labels are then upsampled to time steps.
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

W, hop = 200, 10
starts = np.arange(0, T - W + 1, hop)
feats = []
for s in starts:
    seg = Zs[s:s + W]
    segc = seg - seg.mean(0)
    _, _, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T
    P = (B @ B.T).ravel()
    xy = segc @ B
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(W), ang, 1)[0])
    feats.append(np.concatenate([3.0 * P, [1.5 * np.log(omega + 1e-9)]]))
F = np.array(feats)

km = KMeans(n_clusters=10, n_init=20, random_state=0).fit(F)
C = km.cluster_centers_

# cost matrix: squared distance to each center
D = ((F[:, None, :] - C[None, :, :]) ** 2).sum(-1)

# Viterbi with constant switch penalty
lam = np.median(D) * 3.0
n, k = D.shape
V = D[0].copy()
back = np.zeros((n, k), dtype=int)
for i in range(1, n):
    stay = V
    best_prev = V.min()
    switch = best_prev + lam
    choose_stay = stay <= switch
    V_new = np.where(choose_stay, stay, switch) + D[i]
    back[i] = np.where(choose_stay, np.arange(k), V.argmin())
    V = V_new
path = np.zeros(n, dtype=int)
path[-1] = V.argmin()
for i in range(n - 1, 0, -1):
    path[i - 1] = back[i, path[i]]

centers_t = starts + W // 2
t = np.arange(T)
idx = np.clip(np.searchsorted(centers_t, t), 0, n - 1)
left = np.clip(idx - 1, 0, n - 1)
use_left = np.abs(t - centers_t[left]) < np.abs(t - centers_t[idx])
idx[use_left] = left[use_left]
labels = path[idx]

n_runs = np.count_nonzero(np.diff(labels)) + 1
true_runs = np.count_nonzero(np.diff(states)) + 1
print(f"decoded runs {n_runs} (true {true_runs})")
print(f"viterbi ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

# second pass: per-run refit over full support, re-cluster, re-decode
runs = []
prev = 0
for i in range(1, T):
    if labels[i] != labels[i - 1]:
        runs.append((prev, i))
        prev = i
runs.append((prev, T))

def run_feature(a, b):
    seg = Zs[a:b]
    segc = seg - seg.mean(0)
    _, _, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T
    P = (B @ B.T).ravel()
    xy = segc @ B
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(len(seg)), ang, 1)[0])
    return np.concatenate([3.0 * P, [1.5 * np.log(omega + 1e-9)]])

RF = np.array([run_feature(a, b) for a, b in runs])
km2 = KMeans(n_clusters=10, n_init=50, random_state=0).fit(RF)
labels2 = np.zeros(T, dtype=int)
for (a, b), l in zip(runs, km2.labels_):
    labels2[a:b] = l

print(f"refit   ARI {adjusted_rand_score(states, labels2):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels2):.3f}")
