"""
Private feasibility check for TASK.md — do NOT ship to candidates.

Baseline: PCA to the 4D signal subspace, sliding-window spectral +
autocorrelation features, k-means into 10 clusters, mode smoothing.
Reports ARI / NMI against ground truth.
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

W, hop = 240, 40
starts = np.arange(0, T - W + 1, hop)
feats = []
for s in starts:
    seg = Z[s:s + W]
    seg = seg - seg.mean(0)
    # summed power spectrum across the 4 components, low-frequency bins
    spec = np.abs(np.fft.rfft(seg, axis=0))[1:40].sum(axis=1)
    spec = spec / spec.sum()
    # normalised autocorrelation at a grid of lags (helps slow circles)
    denom = (seg * seg).sum()
    acf = np.array([(seg[:-l] * seg[l:]).sum() / denom
                    for l in range(10, 201, 10)])
    feats.append(np.concatenate([spec, acf]))
F = np.array(feats)

km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(F)
win_labels = km.labels_

# assign each time step the label of the nearest window centre
centers = starts + W // 2
t = np.arange(T)
idx = np.clip(np.searchsorted(centers, t), 0, len(centers) - 1)
left = np.clip(idx - 1, 0, len(centers) - 1)
use_left = np.abs(t - centers[left]) < np.abs(t - centers[idx])
idx[use_left] = left[use_left]
labels = win_labels[idx]

print(f"raw       ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

# mode smoothing over a ~201-step window
half = 100
smoothed = labels.copy()
counts = np.zeros((T, 10), dtype=np.int32)
onehot = np.zeros((T, 10), dtype=np.int32)
onehot[t, labels] = 1
cs = np.vstack([np.zeros((1, 10), int), np.cumsum(onehot, axis=0)])
lo = np.clip(t - half, 0, T)
hi = np.clip(t + half + 1, 0, T)
window_counts = cs[hi] - cs[lo]
smoothed = window_counts.argmax(axis=1)

print(f"smoothed  ARI {adjusted_rand_score(states, smoothed):.3f}   "
      f"NMI {normalized_mutual_info_score(states, smoothed):.3f}")
