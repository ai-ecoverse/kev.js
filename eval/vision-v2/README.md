# vision-v2: harder image decision questions for Kev behind Qwen3.5's vision tower

44 images and 128 questions (48 `noul`, 32 `choice`, 48 `score`) with answers known by construction. vision-v1
showed that Kev still reads the stock tower's image features; this set is built to find where that breaks: real
screen sizes that Qwen's processor has to downscale to kev.js's 589,824-pixel cap, dense charts, counting, and small
print. The file format is vision-v1's: each item has an image, a neutral `context` shown in every condition, a
`caption` with everything the questions need (the text-only upper bound) and the questions in `kev.serve`'s request
shape plus a `label` (the index of the correct option).

Built by `export/kev_web_export/vision_set_v2.py` (seed 20260924). The 36 screen, chart and small-text images are HTML
pages (`pages/`) screenshotted by Playwright's Chromium; the dots are drawn with Pillow:

```bash
cd export && uv run --group vision python -m kev_web_export.vision_set_v2 --out ../eval/vision-v2
cd .. && node scripts/render-vision-set.mjs eval/vision-v2
```

| Family | Images | Questions | Size | What the answer needs |
|---|---|---|---|---|
| inbox | 4 | 12 | 1280 × 800 | who sent the top email, is a sender/subject pair present, how many are unread |
| shop | 4 | 12 | 1280 × 900 | the cheapest of four products, one product's stock state, how many cost over $50 |
| dashboard | 4 | 16 | 1280 × 800 | KPI cards: direction of revenue and users, churn band, churn vs target |
| orders | 4 | 12 | 1280 × 800 | a 12-row table: largest order, how many shipped, is an order refunded |
| line | 4 | 12 | 960 × 640 | a 12-month line chart: the peak month, values read off the axis |
| bars | 4 | 12 | 960 × 640 | 10 bars: the second tallest, two bars a few percent apart, how many above 60 |
| dots | 8 | 24 | 640 × 640 | 8-30 dots: total, red ones, which color is more |
| departures | 4 | 8 | 1280 × 800 | a departures board: a flight's gate, is it delayed |
| label | 4 | 12 | 1024 × 700 | a shipping label in small print: weight band, country, signature required |
| terms | 4 | 8 | 1280 × 800 | fine print at 11-13 px: auto-renewal, refund window |

Everything here is generated (Apache-2.0, like this repo); no third-party images.

## Results (PyTorch fp32, MPS)

`export/kev_web_export/vision_eval.py`, the same four conditions as vision-v1. Chance accuracy on this set is 0.320
and a uniform answer has Brier 0.680. Accuracy with a 95% interval from resampling images, raw logits (T = 1) unless
marked @T.

| Model (pinned) | Condition | Accuracy | Brier | Brier @T | NLL | Mean top p |
|---|---|---|---|---|---|---|
| Kev-4B `kev-4b@139fdd9` (T = 2.41) | image | 0.898 [0.847, 0.945] | 0.159 | 0.181 | 0.390 | 0.91 |
| | caption | 0.914 [0.865, 0.960] | 0.121 | 0.135 | 0.260 | 0.95 |
| | omitted | 0.305 [0.240, 0.376] | 0.779 | 0.705 | 1.493 | 0.49 |
| | shuffled | 0.242 [0.178, 0.312] | 1.324 | 1.160 | 6.165 | 0.87 |
| Kev-0.8B `kev-0.8b@9a45d25` (T = 2.35) | image | 0.648 [0.551, 0.738] | 0.433 | 0.453 | 0.906 | 0.71 |
| | caption | 0.750 [0.669, 0.831] | 0.343 | 0.403 | 0.689 | 0.74 |
| | omitted | 0.344 [0.272, 0.412] | 0.724 | 0.681 | 1.376 | 0.49 |
| | shuffled | 0.258 [0.197, 0.326] | 1.069 | 0.871 | 2.514 | 0.70 |

Accuracy per family, `image` condition (`caption` in parentheses):

| Model | inbox | shop | dashboard | orders | line | bars | dots | departures | label | terms |
|---|---|---|---|---|---|---|---|---|---|---|
| Kev-4B | 0.83 (0.75) | 0.83 (0.75) | 1.00 (1.00) | 0.83 (0.83) | 1.00 (1.00) | 1.00 (0.75) | 0.71 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) |
| Kev-0.8B | 0.42 (0.58) | 0.92 (0.75) | 0.69 (0.81) | 0.42 (0.42) | 1.00 (0.58) | 0.58 (0.50) | 0.42 (0.92) | 0.75 (0.88) | 0.92 (1.00) | 0.62 (1.00) |

By type, `image`: Kev-4B `noul` 0.979, `choice` 0.969, `score` 0.771; Kev-0.8B 0.729, 0.875, 0.417.

What breaks: counting. 11 of Kev-4B's 13 errors are `score` questions that count (red dots, usually one or two short;
products over $50, one short; unread emails, twice answered too many); the other two read one row of the orders
table. Kev-4B with the image is as good as with the text caption everywhere except the
dots, where the caption states the count. Where the caption is a long list (inbox, shop, orders, bars), the image does
as well as or better than the caption: the text version asks Kev to count over a long list too. Downscaling full
screens to the pixel cap costs Kev-4B nothing visible here, including the 11-13 px fine print. Kev-0.8B loses more
on the image than on the caption for dots, terms, dashboard and inbox, and its `score` accuracy is near chance.

Per-question logits are in `results/<model>/rows.json`, the summary with per-type and per-family accuracy in
`results/<model>/report.json`. Browser (onnxruntime-web, WebGPU) numbers for both sets are in the repository README
(Images).
