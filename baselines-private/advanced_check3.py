"""
Advanced check #3 — UMAP + GMM, applied to dynamics features.

The point of this variant: UMAP + clustering is NOT doomed on this dataset —
it only fails when applied to raw time points. Applied to window-level
dynamics features (local plane projector + angular speed), UMAP separates
the syllables cleanly and a full-covariance GMM + sticky Viterbi decode
scores competitively.

Pipeline:
  1. PCA 20D -> 4D, windowed plane-projector + angular-speed features.
  2. UMAP (10 neighbours) of the ~10k window features -> 5D embedding.
  3. Full-covariance GMM, K=10, on the embedding.
  4. Sticky Viterbi on per-component negative log-likelihoods.
  5. Segment refit: mean embedding per decoded run, re-cluster.
"""

from pathlib import Path

import numpy as np
import umap
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

npz = np.load(Path(__file__).resolve().parent.parent / 'data' / 'data.npz')
X = npz['X']
states = npz['states']
T = len(X)
K = 10

Z = PCA(n_components=4).fit_transform(X - X.mean(0))
box = 7
kern = np.ones(box) / box
Zs = np.apply_along_axis(lambda v: np.convolve(v, kern, mode='same'), 0, Z)

W, hop = 200, 10
starts = np.arange(0, T - W + 1, hop)
feats = []
for s in starts:
    seg = Zs[s:s + W]
    segc = seg - seg.mean(0)
    _, _, Vt = np.linalg.svd(segc, full_matrices=False)
    B = Vt[:2].T
    Pmat = (B @ B.T).ravel()
    xy = segc @ B
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(W), ang, 1)[0])
    feats.append(np.concatenate([3.0 * Pmat, [1.5 * np.log(omega + 1e-9)]]))
F = np.array(feats)
n = len(F)

print(f"UMAP on {n} window feature vectors...")
reducer = umap.UMAP(n_components=5, n_neighbors=50, min_dist=0.0,
                    random_state=42)
E = reducer.fit_transform(F)

gmm = GaussianMixture(n_components=K, covariance_type='full', n_init=5,
                      random_state=0).fit(E)

# per-component negative log-likelihood as Viterbi cost.  Raw NLLs from a
# tight GMM can be astronomically large for wrong components, which makes any
# switch penalty either irrelevant or prohibitive — so use per-window
# *relative* costs, capped.
D = np.empty((n, K))
for k in range(K):
    D[:, k] = -multivariate_normal.logpdf(E, gmm.means_[k],
                                          gmm.covariances_[k],
                                          allow_singular=True)
D = np.clip(D - D.min(axis=1, keepdims=True), 0.0, 20.0)

lam = 60.0
V = D[0].copy()
back = np.zeros((n, K), dtype=int)
for i in range(1, n):
    switch = V.min() + lam
    choose_stay = V <= switch
    back[i] = np.where(choose_stay, np.arange(K), V.argmin())
    V = np.where(choose_stay, V, switch) + D[i]
path = np.zeros(n, dtype=int)
path[-1] = V.argmin()
for i in range(n - 1, 0, -1):
    path[i - 1] = back[i, path[i]]

centers_t = starts + W // 2
t_all = np.arange(T)
idx = np.clip(np.searchsorted(centers_t, t_all), 0, n - 1)
left = np.clip(idx - 1, 0, n - 1)
use_left = np.abs(t_all - centers_t[left]) < np.abs(t_all - centers_t[idx])
idx[use_left] = left[use_left]
labels = path[idx]

print(f"umap+gmm+viterbi  ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

# segment refit on mean embeddings
runs = []
prev = 0
for i in range(1, T):
    if labels[i] != labels[i - 1]:
        runs.append((prev, i))
        prev = i
runs.append((prev, T))

RF = []
for a, b in runs:
    wa, wb = idx[a], idx[b - 1] + 1
    RF.append(E[wa:wb].mean(axis=0))
RF = np.array(RF)

km2 = KMeans(n_clusters=K, n_init=50, random_state=0).fit(RF)
labels2 = np.empty(T, dtype=int)
for (a, b), l in zip(runs, km2.labels_):
    labels2[a:b] = l

n_runs = np.count_nonzero(np.diff(labels2)) + 1
true_runs = np.count_nonzero(np.diff(states)) + 1
print(f"umap+gmm refit    ARI {adjusted_rand_score(states, labels2):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels2):.3f}   "
      f"runs {n_runs} (true {true_runs})")
