"""
Private feasibility check #4 — short windows, plane + angular-speed features.

Per 100-step window in the 4D PCA space:
  - top-2 local PCA plane -> 4x4 projector (captures which circle plane)
  - angular speed from unwrapped phase slope in that plane
  - RMS radius
Cluster with k-means, then mode-smooth.
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

# mild temporal smoothing (boxcar 7) to reduce noise
K = 7
kern = np.ones(K) / K
Zs = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, Z)

W, hop = 100, 20
starts = np.arange(0, T - W + 1, hop)

feats = []
for s in starts:
    seg = Zs[s:s + W]
    segc = seg - seg.mean(0)
    U, S, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T                      # (4,2) plane basis
    P = B @ B.T                       # projector, plane signature
    xy = segc @ B                     # in-plane coords
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(W), ang, 1)[0])
    r = np.sqrt((xy ** 2).sum(axis=1)).mean()
    feats.append(np.concatenate([
        3.0 * P.ravel(),                          # weight plane signature
        [40.0 * np.log(omega + 1e-6)],            # weight speed
        [0.2 * r],
    ]))
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

print(f"plane+omega raw       ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

onehot = np.zeros((T, 10), dtype=np.int32)
onehot[t, labels] = 1
cs = np.vstack([np.zeros((1, 10), int), np.cumsum(onehot, axis=0)])
half = 100
lo = np.clip(t - half, 0, T)
hi = np.clip(t + half + 1, 0, T)
smoothed = (cs[hi] - cs[lo]).argmax(axis=1)

print(f"plane+omega smoothed  ARI {adjusted_rand_score(states, smoothed):.3f}   "
      f"NMI {normalized_mutual_info_score(states, smoothed):.3f}")

# diagnostic: how well does omega alone identify the true circle?
periods = npz['periods']
true_omega = 2 * np.pi / periods
win_true = states[np.clip(centers, 0, T - 1)]
est_omega = np.exp(F[:, 16] / 40.0)
err = np.abs(est_omega - true_omega[win_true]) / true_omega[win_true]
print(f"omega rel-err median {np.median(err):.3f}")
