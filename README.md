# Animal Close-Encounter Risk Prediction

| | |
| --- | --- |
| Final rank | 2nd |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-08-07 |

## Problem statement

### Pre-Occlusion Intervention Windows

### Overview

Multi-animal trackers often discover an identity problem only after animals crowd together. At that point, recovering the correct identity may require expensive manual review. This challenge asks an earlier operational question: given a target animal that is currently separated from its neighbours, how likely is it to enter an identity-ambiguous close encounter during the next observation window?

Each example contains two target-centred crops from a real behavioral video. The left panel is the previous observation and the right panel is the current observation. A cyan ring identifies the tracked animal at the centre of each panel. The target is positive when its human-annotated trajectory comes within 30 pixels of another animal during the next three annotation intervals, approximately eight seconds. Examples begin only when all neighbours are at least 40 pixels away.

Useful models can combine current spatial context with motion evidence. The hidden data use complete unseen recordings and acquisition conditions, so memorizing adjacent frames is insufficient.

### Dataset

### File descriptions

- `train.csv`: labeled development examples.
- `test.csv`: unlabeled held-out examples.
- `sample_submission.csv`: correctly formatted random probability predictions.
- `train/`: JPEG temporal panels referenced by `train.csv`.
- `test/`: JPEG temporal panels referenced by `test.csv`.

### Column descriptions

- `sample_id`: anonymized identifier for one target-time example.
- `image_path`: path to its temporal JPEG panel relative to the public dataset directory.
- `encounter_risk`: training-only binary target; `1` means the close-encounter event occurs during the future window and `0` means it does not.

### Evaluation

Submissions are scored with **Encounter Alert Utility**. Average precision rewards models that rank the limited intervention budget toward true upcoming encounters. One minus Brier loss rewards probabilities that an operator can use as calibrated risk estimates. Both components are computed separately for each hidden recording regime and macro-averaged, preventing the larger recording from dominating.

```
regime_utility = (

    0.70 * average_precision(y_true, probability)

    + 0.30 * (1.0 - brier_loss(y_true, probability))

)

score = mean(regime_utility for each hidden recording regime)
```

The score is bounded between 0 and 1, and higher is better.

### Submission

Submit one CSV containing:

- `sample_id`: every hashed identifier from `test.csv`, exactly once.
- `encounter_probability`: a finite probability between 0 and 1.

Example:

```
sample_id,encounter_probability

000543f069f2648e,0.317482

000854e3e9ad895c,0.781205
```

### Requirements

- Column names and order must match the example exactly.
- Row order may differ, but identifiers must match `test.csv`.
- Probabilities must be finite values in the closed interval `[0, 1]`.
- Do not include an index column or any additional columns.

### Prohibited methods

- Do not recover or use the source videos, source annotations, hidden trajectories, or external labels.
- Do not reverse-match public panels to an external copy of the source dataset or manually relabel test examples.
- Do not use `sample_id`, file names, file order, row order, or submission history as prediction signals.
