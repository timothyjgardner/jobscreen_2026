# Screening Task: Unsupervised Syllable Labeling

**Time budget: ~5 hours.** We respect your time — when you hit 5 hours, stop, and note in your writeup what you would do next.

## Background

This repo generates a synthetic "song": a 20-dimensional time series that switches between 10 latent states ("syllables"). Each syllable traces a circle in its own 2D plane with a fixed angular velocity, and syllables follow each other according to a sparse Markov transition matrix. See the [README](README.md) for details.

In this task the data is generated with `--subspace-dim 4`: all 10 circle planes are packed into a shared 4-dimensional subspace, so the circles overlap heavily in space. As the README's UMAP figure shows, at this setting the trajectories from different syllables intermingle — clustering raw time points does not recover the syllables. Your method will need to exploit something other than instantaneous position.

## The task

1. Generate the overlapping dataset:

   ```bash
   python markov_circles_timeseries.py --subspace-dim 4 --no-umap
   ```

   This writes `data/data.npz` and `data/config.json`.

2. Build an **unsupervised** method that assigns a syllable label to every one of the 100,000 time steps, using **only the observations `X` and their time order**.

   Everything else in `data.npz` (`states`, `thetas`, `transition_matrix`, `radii`, `entry_angles`, `periods`) is ground truth. You may load it **only** in your final evaluation script — never for fitting, tuning, or model selection. (Note that `dataset.py` returns `state` alongside each window; if you use it, ignore that output.) We will ask you how you kept evaluation and training separate.

   You may assume there are 10 syllables.

3. Evaluate your labels against the ground truth. Labels are unordered, so use permutation-invariant metrics: report **ARI** and **NMI**, plus a confusion matrix after optimal (Hungarian) matching, and a plot comparing predicted vs. true label sequences over a few thousand-step windows.

## Deliverables

- Working code (fork, zip, or repo link) that reproduces your result end to end
- `labels.npy` — your length-100,000 integer label array
- A short writeup (~1 page): your approach and why it fits this data, what you tried that didn't work, your metrics, and known failure modes (e.g. what happens near syllable transitions)

## Ground rules

- Any language and libraries; Python with numpy/scipy/scikit-learn is plenty. No GPU is needed — deep learning is permitted but not required, and simple methods can score very well here.
- **AI assistants (ChatGPT, Claude, Copilot, Cursor, etc.) are allowed and encouraged.** The requirement is that you can explain and defend every part of the solution — the method, the code, and the results — in a follow-up conversation. Letting an LLM write code is fine; submitting a method you can't explain is disqualifying.
- As a rough calibration: straightforward window-feature clustering tends to plateau around ARI 0.4–0.5; a well-designed method should exceed ARI 0.8. An imperfect method with an honest analysis of where and why it fails beats an inflated number.

## Optional stretch goals

Only if you have time left — these are differentiators, not requirements:

- Estimate the number of syllables from the data instead of assuming 10
- Recover the transition matrix and dwell-time statistics; compare to ground truth
- Characterize robustness: how does your method degrade at `--subspace-dim 3` or `2`, or with higher `noise_std`?

## What we look at

Roughly in order of weight: (1) whether the method is well-matched to the structure of this data, (2) honest, correct evaluation, (3) clarity of the writeup and the follow-up discussion, (4) code quality and reproducibility.
