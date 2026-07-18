"""
Advanced check #2 — self-supervised contrastive window embedding (PyTorch).

Modern deep-representation route: learn a window encoder with InfoNCE where
the positive pair is the same trajectory a short time offset later (temporal
proximity => usually the same syllable, since dwells are ~400 steps). Then
k-means the embeddings, decode with a sticky Viterbi pass, and re-cluster
mean embeddings over decoded segments.

CPU-only, ~1 minute of training.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

torch.manual_seed(0)
rng = np.random.default_rng(0)

npz = np.load(Path(__file__).resolve().parent.parent / 'data' / 'data.npz')
X = npz['X']
states = npz['states']
T = len(X)
K = 10

Z = PCA(n_components=4).fit_transform(X - X.mean(0)).astype(np.float32)
Z /= Z.std()

W, hop = 128, 10
MAX_SHIFT = 60
starts = np.arange(0, T - W - MAX_SHIFT, hop)
n_win = len(starts)


class Encoder(nn.Module):
    def __init__(self, in_ch=4, emb=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(64, emb)

    def forward(self, x):            # x: (B, C, W)
        h = self.net(x).mean(dim=2)
        z = self.head(h)
        return Fn.normalize(z, dim=1)


enc = Encoder()
opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
TAU = 0.1
EPOCHS = 25
BATCH = 256

Zt = torch.from_numpy(Z)


def get_windows(idx_starts):
    # gather (B, W, 4) then transpose to (B, 4, W)
    w = torch.stack([Zt[s:s + W] for s in idx_starts])
    return w.transpose(1, 2)


for ep in range(EPOCHS):
    perm = rng.permutation(n_win)
    tot, nb = 0.0, 0
    for i in range(0, n_win, BATCH):
        b = perm[i:i + BATCH]
        if len(b) < 8:
            continue
        s0 = starts[b]
        s1 = s0 + rng.integers(10, MAX_SHIFT + 1, size=len(b))
        za = enc(get_windows(s0))
        zb = enc(get_windows(s1))
        logits = (za @ zb.T) / TAU
        target = torch.arange(len(b))
        loss = 0.5 * (Fn.cross_entropy(logits, target) +
                      Fn.cross_entropy(logits.T, target))
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    print(f"epoch {ep:2d}  loss {tot / nb:.3f}")

# ---------------------------------------------------------------------------
# Embed all windows, cluster, sticky Viterbi over the window sequence
# ---------------------------------------------------------------------------
enc.eval()
with torch.no_grad():
    E = torch.cat([enc(get_windows(starts[i:i + 512]))
                   for i in range(0, n_win, 512)]).numpy()

km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(E)
C = km.cluster_centers_
D = ((E[:, None, :] - C[None, :, :]) ** 2).sum(-1)

lam = np.median(D) * 3.0
n = D.shape[0]
V = D[0].copy()
back = np.zeros((n, K), dtype=int)
for i in range(1, n):
    best_prev = V.min()
    switch = best_prev + lam
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

print(f"\ncontrastive+viterbi  ARI {adjusted_rand_score(states, labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels):.3f}")

# ---------------------------------------------------------------------------
# Segment refit: mean embedding over each decoded run, re-cluster
# ---------------------------------------------------------------------------
runs = []
prev = 0
for i in range(1, T):
    if labels[i] != labels[i - 1]:
        runs.append((prev, i))
        prev = i
runs.append((prev, T))

win_of_step = idx  # nearest window per step
RF = []
for a, b in runs:
    wa, wb = win_of_step[a], win_of_step[b - 1] + 1
    RF.append(E[wa:wb].mean(axis=0))
RF = np.array(RF)

km2 = KMeans(n_clusters=K, n_init=50, random_state=0).fit(RF)
labels2 = np.empty(T, dtype=int)
for (a, b), l in zip(runs, km2.labels_):
    labels2[a:b] = l

n_runs = np.count_nonzero(np.diff(labels2)) + 1
true_runs = np.count_nonzero(np.diff(states)) + 1
print(f"contrastive refit    ARI {adjusted_rand_score(states, labels2):.3f}   "
      f"NMI {normalized_mutual_info_score(states, labels2):.3f}   "
      f"runs {n_runs} (true {true_runs})")
