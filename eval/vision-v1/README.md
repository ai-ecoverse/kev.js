# vision-v1: image decision questions for Kev behind Qwen3.5's vision tower

41 images and 106 questions (50 `noul`, 35 `choice`, 21 `score`; 25 of the 50 `noul` answers are yes) with known
answers. `questions.json` holds each item's image, a neutral `context` that every condition shows, a `caption` with
everything the questions need (the text-only upper bound) and the questions in `kev.serve`'s request shape plus a
`label` (the index of the correct option: noul 0 = no, 1 = yes; choice in criteria order; score = level index).

Built by `export/kev_web_export/vision_set.py` (seed 20260923, Pillow 12.3's bundled default font):

```bash
cd export && uv run --group vision python -m kev_web_export.vision_set --photos <skimage-data> --out ../eval/vision-v1
```

| Family | Images | Questions | Source |
|---|---|---|---|
| shapes | 8 | 24 | generated: count, majority color, is there a triangle |
| chart | 6 | 18 | generated bar charts: top fruit, A more than B, range of the top value |
| screen | 10 | 28 | generated app screens: payment result, settings switches, to-do list, sign-in error |
| receipt | 4 | 8 | generated receipts: total range, paid in cash |
| sign | 2 | 2 | generated OPEN / CLOSED door signs |
| traffic | 4 | 8 | generated traffic lights: which light, should a car stop |
| photo | 7 | 18 | scikit-image sample data, questions and captions written by hand |

## Licensing

- Generated images (`shapes-*`, `chart-*`, `screen-*`, `receipt-*`, `sign-*`, `traffic-*`): Apache-2.0, like this repo.
- Photos, from scikit-image v0.25.2 `skimage/data` (the notes in `skimage/data/_fetchers.py`):
  - `astronaut.png`: NASA, no known copyright restrictions, public domain.
  - `chelsea.png`: CC0 by the photographer, Stefan van der Walt.
  - `coffee.png`: CC0 by the photographer, Rachel Michetti.
  - `rocket.jpg`: SpaceX, released into the public domain.
  - `coins.png`: Brooklyn Museum Collection, no known copyright restrictions.
  - `hubble_deep_field.jpg`: NASA / HubbleSite, public domain.
  - `horse.png`: CC0 by the owner, Andreas Preuss (marauder).

## Results (Phase 1, PyTorch fp32)

`export/kev_web_export/vision_eval.py` runs each question in four conditions: `image` (the image, then the context),
`caption` (context + caption, no image), `omitted` (context only) and `shuffled` (another image of the same family).
The vision tower and merger are the unmodified ones of each checkpoint's own base; nothing was trained. Accuracy with
a 95% interval from resampling images; Brier is summed over options (uniform = 0.640 on this set, chance accuracy
0.360); flatness is mean entropy / log K (1 = uniform). Raw logits (T = 1) unless marked @T (the serving temperature).

| Model (pinned) | Condition | Accuracy | Brier | Brier @T | NLL | Mean top p | Flatness | Flatness @T |
|---|---|---|---|---|---|---|---|---|
| Kev-4B `kev-4b@139fdd9` (T = 2.41) | image | 0.991 [0.971, 1.000] | 0.010 | 0.023 | 0.020 | 0.99 | 0.02 | 0.15 |
| | caption | 1.000 | 0.002 | 0.012 | 0.007 | 0.99 | 0.02 | 0.17 |
| | omitted | 0.292 [0.231, 0.359] | 0.817 | 0.693 | 1.442 | 0.61 | 0.78 | 0.94 |
| | shuffled | 0.208 [0.131, 0.294] | 1.519 | 1.405 | 8.622 | 0.96 | 0.09 | 0.24 |
| Kev-0.8B `kev-0.8b@9a45d25` (T = 2.35) | image | 0.906 [0.845, 0.961] | 0.125 | 0.159 | 0.243 | 0.89 | 0.22 | 0.50 |
| | caption | 0.906 [0.833, 0.964] | 0.123 | 0.180 | 0.224 | 0.90 | 0.24 | 0.56 |
| | omitted | 0.321 [0.230, 0.410] | 0.812 | 0.699 | 1.466 | 0.61 | 0.74 | 0.91 |
| | shuffled | 0.236 [0.162, 0.314] | 1.337 | 1.107 | 4.063 | 0.86 | 0.27 | 0.56 |

Pinned runs (Kev 1.0): `jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101` on `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`,
`jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e` on `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
Per-question logits are in `results/<model>/rows.json`, the summary with per-type and per-family accuracy in
`results/<model>/report.json`.

What this set does and does not show: the questions are easy for Qwen3.5's own vision (the stock base describes the
chart, sign-in error and traffic light correctly through the same splice), so it tests whether Kev's text-only
fine-tune still reads the image features, not how well it reasons over hard images. Kev-4B's one remaining error is a
counting `score` question (shapes-5, one short); Kev-0.8B's are mostly counting (shapes 0.71, `score` 0.57 overall).
