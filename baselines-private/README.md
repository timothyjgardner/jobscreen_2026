# Baseline results — INTERNAL, DO NOT SEND TO CANDIDATES

This folder is the answer key for `TASK.md` (unsupervised syllable labeling of
the subspace-dim-4 dataset). It documents two rounds of solution attempts —
seven quick baselines (`baseline_check*.py`) and three stronger methods
(`advanced_check*.py`, best ARI 0.994) — their scores, and how to use them
when grading. **None of this should reach a candidate before or during the
screen.**

## Setup

All scripts assume:

- `data/` (repo root) contains the dataset generated with
  `python markov_circles_timeseries.py --subspace-dim 4` (the repo ships with
  exactly this; seed 42 makes regeneration byte-identical)
- `numpy` and `scikit-learn` are installed; `advanced_check2.py` also needs
  `torch` (CPU is fine) and `advanced_check3.py` needs `umap-learn`

The scripts resolve the data path relative to their own location, so they run
from anywhere, e.g. `python baselines-private/baseline_check7.py` from the
repo root. The `baseline_check*` scripts run in a few seconds on CPU; the
advanced ones take seconds (#1, #3) to ~5 minutes (#2, training). All print
ARI/NMI against the ground-truth `states` array.

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
  reaches 0.86 and recovers 260 segments vs. 251 true.

The calibration line in TASK.md ("naive plateaus at 0.4–0.5, well-designed
methods exceed 0.8") comes directly from this table.

## Round 2 — stronger unsupervised methods (`advanced_check*.py`)

A second pass with heavier machinery, to establish the ceiling and to check
whether the deep-learning route is competitive.

| Script | Approach | ARI | NMI |
|---|---|---|---|
| `advanced_check1.py` | **Switching AR(2) HMM** in the 4D PCA space: EM with a sticky transition prior, Viterbi decode, minimum-dwell cleanup, then per-segment plane/speed re-clustering | **0.994** | **0.991** |
| `advanced_check2.py` | **Deep contrastive embedding**: conv encoder on 128-step windows, InfoNCE with temporal-proximity positives (25 epochs, CPU), k-means + sticky Viterbi + segment refit | 0.904 | 0.902 |
| `advanced_check3.py` | **UMAP + GMM done right**: UMAP of window-level dynamics features (not raw points!) → full-covariance GMM → sticky Viterbi | 0.810 | 0.831 |

Notes per method:

- **ARHMM (`advanced_check1.py`) essentially solves the task.** It is the
  model class matched to the generative process: within a syllable the signal
  is a noisy sinusoid in a fixed plane (captured exactly by a vector AR(2)),
  and switches are rare and discrete (captured by the sticky HMM backbone).
  Initialised from the window-feature k-means (ARI 0.76), EM converges in ~7
  iterations and localises **boundaries to median 0 / 90th-pct 2 steps**.
  Two post-passes matter: (1) minimum-dwell cleanup (no true visit is shorter
  than 40 steps) fixes segment count to exactly 251/251; (2) EM sometimes
  lets one AR state host two circles — re-clustering plane+speed features
  over the decoded segments fixes the identities, taking ARI 0.91 → 0.994
  (per-step matched accuracy 99.7%, perfectly diagonal confusion matrix).
  Runtime ~8 s. This is within reach of a strong candidate using an LLM to
  draft the EM machinery.
- **Contrastive embedding (`advanced_check2.py`)** shows the generic deep
  route works with zero domain features: positives are simply the same
  trajectory 10–60 steps later. It cleanly beats every classic baseline
  (0.90 vs 0.86) and recovers 251/251 segments, but its boundary resolution
  is limited by the 128-step window, so it cannot catch the ARHMM. More
  epochs helped (15 → 25 epochs: 0.85 → 0.90); further training/architecture
  tuning would likely add a little more.
- **UMAP + GMM (`advanced_check3.py`)** is included deliberately: TASK.md
  tells candidates UMAP-on-raw-points fails, and this shows the fix is the
  representation, not the algorithm — the same UMAP+clustering applied to
  window dynamics features scores 0.81. Two practical details: larger
  `n_neighbors` (50) matters, and raw GMM log-likelihoods must be converted
  to capped relative costs before Viterbi or the switch penalty is
  meaningless.

Updated headroom statement for grading: the task ceiling is ARI ≈ 0.99+
(ARHMM), not ~0.86 as Round 1 suggested. A candidate reporting 0.95+ with a
switching-dynamics model has genuinely nailed it; 0.85–0.95 indicates a
strong solution with imperfect boundaries or state identities; the TASK.md
tier boundaries (0.5 / 0.8) remain valid.

### Robustness test: subspace_dim = 3 (heavier overlap)

We regenerated the dataset with `--subspace-dim 3 --no-umap` (10 circle
planes packed into 3D — every pair of planes intersects along a line) and
ran `advanced_check1.py` on it **unchanged**, no retuning:

| Stage | subspace_dim=4 | subspace_dim=3 |
|---|---|---|
| init (window k-means) | 0.76 | 0.72 |
| ARHMM Viterbi (cleaned) | 0.91 | 0.90 |
| + segment refit | **0.994** | **0.90** |
| boundary error (median / 90th pct) | 0 / 2 steps | 0 / 20 steps |
| matched per-step accuracy | 99.7% | 92.4% |

Degradation is graceful and highly localised: the confusion matrix stays
perfectly diagonal for **9 of 10 circles**, and essentially all the lost ARI
comes from circle 4 (period 200) being absorbed into circle 3's state
(period 160). We verified the cause geometrically: in the seed-42 subspace-3
draw, planes 3 and 4 are the closest pair in the whole set — largest
principal angle only 16° (next-closest pair: 24°) — while also being
adjacent in frequency (period ratio 1.25). Both identity channels are weak
at once for exactly this pair. Segment count stays
almost exact (259 vs. 251 true) and boundaries stay sharp, so it is a state-
*identity* failure, not a segmentation failure. Re-weighting frequency in
the segment-refit features did not separate the pair (0.901), so fixing it
would need a genuinely better identity model (e.g. splitting candidate
merged states and testing likelihood improvement).

Implications: the task is still very solvable at subspace_dim 3 (ARI 0.90
with an untuned pipeline), which supports offering it as the harder stretch
variant in TASK.md. `advanced_check1.py` now accepts an optional data-dir
argument for exactly this kind of variant testing:
`python baselines-private/advanced_check1.py /path/to/variant/data`.

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
  reruns and minor tweaks move ARI by a few points (the torch run is also
  hardware-dependent). Judge the tier, not the third decimal.
- A candidate whose method lands at 0.7 with a sharp failure analysis and a
  clear plan for the boundary problem is a stronger signal than one at 0.85
  who can't explain why refitting segments helps.
