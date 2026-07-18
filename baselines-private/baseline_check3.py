"""
Private feasibility check #3 — local linear dynamics features.

Fit x_{t+L} ≈ A x_t per sliding window in the 4D PCA space; the operator A
captures both the circle's plane and its angular speed. Cluster vec(A).
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

# light smoothing to tame per-step noise (short enough to keep period-40)
K = 9
kern = np.ones(K) / K
Zs = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, Z)

W, hop, L = 200, 25, 5
starts = np.arange(0, T - W - L, hop)
feats = []
ridge = 1e-3
for s in starts:
    X0 = Zs[s:s + W].T                # (4, W)
    X1 = Zs[s + L:s + W + L].T        # (4, W)
    G = X0 @ X0.T + ridge * np.eye(4)
    A = (X1 @ X0.T) @ np.linalg.inv(G)
    feats.append(A.ravel())
F = np.array(feats)

km = KMeans(n_clusters=10, n_init=20, random_state=0).fit(F)
win_labels = km.labels_

centers = starts + W // 2
t = np.arange(T)
idx = np.clip(np.searchsorted(centers, t), 0, len(centers) - 1)
left = np.clip(idx - 1, 0, len(centers) - 1)
use_left = np.abs(t - centers[left]) < np.abs(t - centers[idx])
idx[use_left] = left[use_left]
labels = win_labels[idx]

print(f"dyn raw       ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

onehot = np.zeros((T, 10), dtype=np.int32)
onehot[t, labels] = 1
cs = np.vstack([np.zeros((1, 10), int), np.cumsum(onehot, axis=0)])
half = 100
lo = np.clip(t - half, 0, T)
hi = np.clip(t + half + 1, 0, T)
smoothed = (cs[hi] - cs[lo]).argmax(axis=1)

print(f"dyn smoothed  ARI {adjusted_rand_score(states, smoothed):.3f}   "
      f"NMI {normalized_mutual_info_score(states, smoothed):.3f}")
