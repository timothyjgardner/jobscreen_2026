# Baseline results — INTERNAL, DO NOT SEND TO CANDIDATES

This folder is the answer key for `TASK.md` (unsupervised syllable labeling of
the subspace-dim-4 dataset). It documents seven baseline attempts, their
scores, and how to use them when grading. **None of this should reach a
candidate before or during the screen.**

## Setup

All scripts assume:

- `data/` (repo root) contains the dataset generated with
  `python markov_circles_timeseries.py --subspace-dim 4` (the repo ships with
  exactly this; seed 42 makes regeneration byte-identical)
- `numpy` and `scikit-learn` are installed

The scripts resolve the data path relative to their own location, so they run
from anywhere, e.g. `python baselines-private/baseline_check7.py` from the
repo root. Each runs in a few seconds on CPU and prints ARI/NMI against the
ground-truth `states` array.

## Results (100,000 steps, 10 syllables, true segment count 251)

| Script | Approach | ARI | NMI |
|---|---|---|---|
| `baseline_check.py` | PCA→4D, windowed power spectrum + ACF features, k-means, mode smoothing | 0.46 | 0.66 |
| `baseline_check2.py` | Long-window (400) autocorrelation features, k-means, smoothing | 0.45 | 0.64 |
| `baseline_check3.py` | Local linear dynamics operator (fit x_{t+5} ≈ A x_t per window), cluster vec(A) | 0.35 | 0.53 |
| `baseline_check4.py` | Per-window circle-plane projector (local 2D PCA) + angular speed, k-means, smoothing | 0.72 | 0.83 |
| `baseline_check5.py` | Change-point detection on plane projectors, then per-segment clustering | 0.73 | 0.77 |
| `baseline_check6.py` | check5 + boundary trimming and label-based segment merging | 0.71 | 0.75 |
| `baseline_check7.py` | Window features + sticky Viterbi decode (switch penalty), then per-run refit and re-cluster | **0.86** | **0.88** |

## What the numbers mean

- **ARI ≲ 0.5 — naive tier.** Generic window featurization (spectra, ACFs,
  dynamics operators) plus clustering. This is what "featurize and cluster"
  produces without thinking about the specific structure of the data.
  Frequency features alone underperform because slow circles (period up to
  400) don't complete a cycle inside a short window, and per-window frequency
  estimates are noisy at SNR ≈ 2.5.
- **ARI 0.6–0.75 — insight tier.** Requires identifying the discriminative
  structure: each syllable lives in a specific 2D plane (the local PCA
  projector is a strong signature) and has a fixed angular speed. Windowed
  versions of these features cluster well but lose accuracy at syllable
  boundaries, which caps ARI around 0.75.
- **ARI > 0.8 — strong tier.** Additionally exploits temporal structure:
  dwell times are long (~400 steps) and switches are discrete, so decoding
  with a switching penalty (Viterbi/HMM-style) and refitting features over
  whole decoded segments cleans up the boundaries. `baseline_check7.py`
  reaches 0.86 and recovers 260 segments vs. 251 true. There is headroom
  above this (proper ARHMM/SLDS, learned sequence embeddings), so scores
  near 0.9+ are plausible for excellent candidates.

The calibration line in TASK.md ("naive plateaus at 0.4–0.5, well-designed
methods exceed 0.8") comes directly from this table.

## Debrief prompts

Useful questions regardless of the candidate's score:

- Why does clustering raw time points (or their UMAP) fail at subspace-dim 4
  when it works at 20? What property of the data did your method exploit
  instead?
- How did you choose the window length? What breaks for the slowest circles
  (period 400) if windows are short, and for boundaries if windows are long?
- Where are your errors concentrated? (Expected answer: at syllable
  transitions and between circles with similar planes/speeds.)
- How did you guarantee the ground-truth arrays in `data.npz` never leaked
  into fitting or model selection? (`dataset.py` returns `state` — did they
  notice?)
- If they used an LLM: pick any non-trivial block of their code and ask them
  to explain it, then ask what alternatives they considered and rejected.

## Caveats

- Scores above are for seed-42 data with `n_init` and thresholds as written;
  reruns and minor tweaks move ARI by a few points. Judge the tier, not the
  third decimal.
- A candidate whose method lands at 0.7 with a sharp failure analysis and a
  clear plan for the boundary problem is a stronger signal than one at 0.85
  who can't explain why refitting segments helps.
