"""
Private feasibility check #2 — ACF period features + change-aware smoothing.
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

W, hop = 400, 20
starts = np.arange(0, T - W + 1, hop)
max_lag = 440

feats = []
for s in starts:
    seg = Z[s:s + W]
    seg = seg - seg.mean(0)
    # summed normalised ACF across the 4 components
    n = seg.shape[0]
    acf = np.zeros(max_lag)
    denom = (seg * seg).sum()
    for l in range(1, max_lag):
        if l < n:
            acf[l] = (seg[:-l] * seg[l:]).sum() / denom
    feats.append(acf)
F = np.array(feats)

km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(F)
win_labels = km.labels_

centers = starts + W // 2
t = np.arange(T)
idx = np.clip(np.searchsorted(centers, t), 0, len(centers) - 1)
left = np.clip(idx - 1, 0, len(centers) - 1)
use_left = np.abs(t - centers[left]) < np.abs(t - centers[idx])
idx[use_left] = left[use_left]
labels = win_labels[idx]

print(f"ACF-kmeans raw       ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

onehot = np.zeros((T, 10), dtype=np.int32)
onehot[t, labels] = 1
cs = np.vstack([np.zeros((1, 10), int), np.cumsum(onehot, axis=0)])
half = 100
lo = np.clip(t - half, 0, T)
hi = np.clip(t + half + 1, 0, T)
smoothed = (cs[hi] - cs[lo]).argmax(axis=1)

print(f"ACF-kmeans smoothed  ARI {adjusted_rand_score(states, smoothed):.3f}   "
      f"NMI {normalized_mutual_info_score(states, smoothed):.3f}")
