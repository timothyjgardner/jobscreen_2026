"""
Advanced check #1 — switching AR(2) HMM (ARHMM), fit by EM, Viterbi decode.

This is the model class actually matched to the generative process: within a
syllable the signal is a noisy sinusoid in a fixed 2D plane, which an AR(2)
vector autoregression captures per state; switches are rare and discrete,
which the sticky HMM backbone captures. Forward-backward integrates evidence
over whole dwells, and Viterbi localises boundaries to a few steps.

Pipeline:
  1. PCA 20D -> 4D (the shared signal subspace).
  2. Init: k-means on windowed plane-projector + angular-speed features
     (same features as baseline_check7).
  3. EM over a 10-state AR(2) HMM with a sticky transition prior.
  4. Viterbi decode, evaluate ARI / NMI / boundary accuracy.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

rng = np.random.default_rng(0)

# optional argument: directory containing data.npz (defaults to repo data/,
# i.e. the shipped subspace-4 dataset); used for robustness tests on variants
data_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
            else Path(__file__).resolve().parent.parent / 'data')
npz = np.load(data_dir / 'data.npz')
X = npz['X']
states = npz['states']
T = len(X)
K = 10
P_ORDER = 2

Z = PCA(n_components=4).fit_transform(X - X.mean(0))

# ---------------------------------------------------------------------------
# Init labels from windowed plane+speed features (as in baseline_check7)
# ---------------------------------------------------------------------------
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
km = KMeans(n_clusters=K, n_init=20, random_state=0).fit(F)

centers_t = starts + W // 2
t_all = np.arange(T)
idx = np.clip(np.searchsorted(centers_t, t_all), 0, len(centers_t) - 1)
left = np.clip(idx - 1, 0, len(centers_t) - 1)
use_left = np.abs(t_all - centers_t[left]) < np.abs(t_all - centers_t[idx])
idx[use_left] = left[use_left]
init_labels = km.labels_[idx]

print(f"init (window kmeans broadcast)  "
      f"ARI {adjusted_rand_score(states, init_labels):.3f}   "
      f"NMI {normalized_mutual_info_score(states, init_labels):.3f}")

# ---------------------------------------------------------------------------
# ARHMM: emissions x_t | x_{t-1}, x_{t-2}, z_t=k  ~  N(B_k y_t, Sigma_k)
# ---------------------------------------------------------------------------
N = T - P_ORDER
Xtgt = Z[P_ORDER:]                                  # (N, 4)
Yreg = np.hstack([Z[1:T - 1], Z[0:T - 2],
                  np.ones((N, 1))])                 # (N, 9)
d = Xtgt.shape[1]
q = Yreg.shape[1]

gamma = np.zeros((N, K))
gamma[np.arange(N), init_labels[P_ORDER:]] = 1.0

# sticky transition prior (pseudo-counts).  True mean dwell is ~400 steps,
# so the diagonal should sit near 0.9975; a strong prior keeps EM from
# drifting toward flickery transitions.
PRIOR_DIAG = 20000.0
PRIOR_OFF = 0.5

Ptrans = np.full((K, K), 0.0025 / (K - 1))
np.fill_diagonal(Ptrans, 0.9975)
pi0 = np.full(K, 1.0 / K)

RIDGE = 1e-3
SIG_FLOOR = 1e-4


def m_step(gamma):
    Bs, Sigmas, chols, logdets = [], [], [], []
    for k in range(K):
        w = gamma[:, k]
        sw = w.sum()
        Yw = Yreg * w[:, None]
        G = Yw.T @ Yreg + RIDGE * np.eye(q)
        Bk = np.linalg.solve(G, Yw.T @ Xtgt)         # (q, d)
        resid = Xtgt - Yreg @ Bk
        Sk = (resid * w[:, None]).T @ resid / max(sw, 1.0)
        Sk += SIG_FLOOR * np.eye(d)
        Bs.append(Bk)
        Sigmas.append(Sk)
        c = np.linalg.cholesky(Sk)
        chols.append(c)
        logdets.append(2.0 * np.log(np.diag(c)).sum())
    return Bs, Sigmas, chols, logdets


def log_emissions(Bs, chols, logdets):
    loge = np.empty((N, K))
    for k in range(K):
        resid = Xtgt - Yreg @ Bs[k]
        # solve L u = resid^T  ->  quad = sum(u^2)
        u = np.linalg.solve(chols[k], resid.T)
        quad = (u * u).sum(axis=0)
        loge[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdets[k] + quad)
    return loge


def forward_backward(loge, Ptrans, pi0):
    m = loge.max(axis=1, keepdims=True)
    e = np.exp(loge - m)
    alpha = np.empty((N, K))
    c = np.empty(N)
    a = pi0 * e[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, N):
        a = (alpha[t - 1] @ Ptrans) * e[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty((N, K))
    beta[-1] = 1.0
    xi_sum = np.zeros((K, K))
    for t in range(N - 2, -1, -1):
        eb = e[t + 1] * beta[t + 1]
        xi_sum += np.outer(alpha[t], eb / c[t + 1]) * Ptrans
        beta[t] = (Ptrans @ eb) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    loglik = np.log(c).sum() + m.sum()
    return gamma, xi_sum, loglik


prev_ll = -np.inf
for it in range(15):
    Bs, Sigmas, chols, logdets = m_step(gamma)
    loge = log_emissions(Bs, chols, logdets)
    gamma, xi_sum, ll = forward_backward(loge, Ptrans, pi0)
    # transition update with sticky prior
    counts = xi_sum + PRIOR_OFF
    counts[np.diag_indices(K)] += PRIOR_DIAG - PRIOR_OFF
    Ptrans = counts / counts.sum(axis=1, keepdims=True)
    pi0 = gamma[0] + 1e-6
    pi0 /= pi0.sum()
    print(f"EM iter {it:2d}  loglik {ll:.1f}")
    if ll - prev_ll < 1.0 and it > 3:
        break
    prev_ll = ll

# ---------------------------------------------------------------------------
# Viterbi decode
# ---------------------------------------------------------------------------
logP = np.log(Ptrans)
V = np.log(pi0) + loge[0]
back = np.zeros((N, K), dtype=np.int32)
for t in range(1, N):
    scores = V[:, None] + logP
    back[t] = scores.argmax(axis=0)
    V = scores.max(axis=0) + loge[t]
path = np.empty(N, dtype=int)
path[-1] = V.argmax()
for t in range(N - 1, 0, -1):
    path[t - 1] = back[t, path[t]]

labels = np.empty(T, dtype=int)
labels[P_ORDER:] = path
labels[:P_ORDER] = path[0]

ari = adjusted_rand_score(states, labels)
nmi = normalized_mutual_info_score(states, labels)
n_runs = np.count_nonzero(np.diff(labels)) + 1
true_runs = np.count_nonzero(np.diff(states)) + 1
print(f"\nARHMM raw      ARI {ari:.3f}   NMI {nmi:.3f}   "
      f"runs {n_runs} (true {true_runs})")

# ---------------------------------------------------------------------------
# Minimum-dwell cleanup: no true visit is shorter than one revolution of the
# fastest circle (40 steps).  Absorb shorter runs into their neighbours.
# ---------------------------------------------------------------------------
MIN_DWELL = 40
changed = True
while changed:
    changed = False
    runs = []
    prev = 0
    for i in range(1, T):
        if labels[i] != labels[i - 1]:
            runs.append([prev, i])
            prev = i
    runs.append([prev, T])
    for j, (a, b) in enumerate(runs):
        if b - a < MIN_DWELL:
            left_len = runs[j - 1][1] - runs[j - 1][0] if j > 0 else -1
            right_len = runs[j + 1][1] - runs[j + 1][0] if j < len(runs) - 1 else -1
            if left_len >= right_len and j > 0:
                labels[a:b] = labels[a - 1]
            elif j < len(runs) - 1:
                labels[a:b] = labels[b]
            changed = True
            break

ari = adjusted_rand_score(states, labels)
nmi = normalized_mutual_info_score(states, labels)
n_runs = np.count_nonzero(np.diff(labels)) + 1
print(f"ARHMM cleaned  ARI {ari:.3f}   NMI {nmi:.3f}   "
      f"runs {n_runs} (true {true_runs})")

# ---------------------------------------------------------------------------
# Segment-identity re-clustering.  A VAR(2) state in 4D can host two rotation
# frequencies at once, so EM may merge two circles into one state (and waste
# a state elsewhere).  The decoded *boundaries* are near-exact, though, so we
# recompute plane + angular-speed features over each full decoded segment
# (long, clean support -> accurate omega even for the slowest circle) and
# re-cluster the segments into K groups.
# ---------------------------------------------------------------------------
runs = []
prev = 0
for i in range(1, T):
    if labels[i] != labels[i - 1]:
        runs.append((prev, i))
        prev = i
runs.append((prev, T))


def run_feature(a, b, trim=20):
    a2, b2 = a + trim, b - trim
    if b2 - a2 < 80:
        a2, b2 = a, b
    seg = Zs[a2:b2]
    segc = seg - seg.mean(0)
    _, _, Vt = np.linalg.svd(segc, full_matrices=False)
    Bp = Vt[:2].T
    Pmat = (Bp @ Bp.T).ravel()
    xy = segc @ Bp
    ang = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    omega = abs(np.polyfit(np.arange(len(seg)), ang, 1)[0])
    return np.concatenate([3.0 * Pmat, [1.5 * np.log(omega + 1e-9)]])


RF = np.array([run_feature(a, b) for a, b in runs])
km2 = KMeans(n_clusters=K, n_init=50, random_state=0).fit(RF)
labels_final = np.empty(T, dtype=int)
for (a, b), l in zip(runs, km2.labels_):
    labels_final[a:b] = l
labels = labels_final

ari = adjusted_rand_score(states, labels)
nmi = normalized_mutual_info_score(states, labels)
n_runs = np.count_nonzero(np.diff(labels)) + 1
print(f"ARHMM refit    ARI {ari:.3f}   NMI {nmi:.3f}   "
      f"runs {n_runs} (true {true_runs})")

# boundary accuracy: distance from each true boundary to nearest predicted
tb = np.where(np.diff(states) != 0)[0]
pb = np.where(np.diff(labels) != 0)[0]
if len(pb):
    dists = np.abs(tb[:, None] - pb[None, :]).min(axis=1)
    print(f"boundary error: median {np.median(dists):.0f} steps, "
          f"90th pct {np.percentile(dists, 90):.0f} steps")

# per-step accuracy after Hungarian matching
from scipy.optimize import linear_sum_assignment
conf = np.zeros((K, K), dtype=int)
for a, b in zip(states, labels):
    conf[a, b] += 1
r, cidx = linear_sum_assignment(-conf)
acc = conf[r, cidx].sum() / T
print(f"matched per-step accuracy: {acc:.4f}")

np.set_printoptions(linewidth=200)
print("\nconfusion (rows=true circle, cols=matched model state, kilosteps):")
print(np.round(conf[:, cidx] / 1000, 1))
