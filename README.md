# @ai-ecoverse/kev.js

[Kev](https://github.com/jaredpalmer/kev) decision models running in the browser on WebGPU (or WASM), with the same
TypeSafe System One request and response shapes as `kev.serve`. No server: the model, tokenizer and pointer head
run in a Web Worker.

**[Homepage and live demo](https://ai-ecoverse.github.io/kev.js/)** · **[Weights](https://huggingface.co/ai-ecoverse/kev.js)** · `npm install @ai-ecoverse/kev.js onnxruntime-web`

![kev.js demo](docs/demo.png)

```ts
import * as ort from "onnxruntime-web/webgpu";
import { loadKev } from "@ai-ecoverse/kev.js";

const kev = await loadKev("https://huggingface.co/ai-ecoverse/kev.js/resolve/main/kev-0.8b", { ort, variant: "q8f32" });
const res = await kev.systemOne({
  state: "I was charged twice. Please fix this ASAP.",
  questions: {
    billing: { type: "noul", instructions: "Is this ticket about billing?" },
    urgency: { type: "score", instructions: "How urgent is this ticket?", criteria: ["can wait", "this week", "today"] },
  },
});
// Kev-0.8B (kev-0.8b@2256796), as served (T ≈ 2.41):
res.answers.billing.noul;    // 0.9102 probability of yes, rounded to 4 places like kev.serve
res.answers.urgency.score;   // 1.482  expected level (0 "can wait" … 2 "today"); "today" has 0.5724
```

The pointer head applies the checkpoint's fitted temperature by default (0.8B 2.41, 4B 2.14, 9B 2.30), matching `kev.serve`. It never changes the argmax. Pass `{ temperature: 1 }` to `loadKev` for the raw logits. `{ dateFacts: true }` appends day counts between absolute dates in the state (`KEV_DATE_FACTS=1`).

## Results

Measured in Chrome on an Apple M4 Max (WebGPU on Metal), against the fp32 PyTorch model (`kev.checkpoint.load`) at the
same commit. Each fp32 ONNX export matches PyTorch to 1e-4 or better, so the 300-record comparisons use it as the
reference. The 300 records are held-out transfer-v4 development data (sources Kev was not trained on).

| Model (pinned checkpoint) | Download | Accuracy, browser / reference | Brier, browser / reference | Mean / max \|Δp\| (300 records) | Answers changed | 3-question request |
|---|---|---|---|---|---|---|
| Kev-0.8B `q8f32` (`kev-0.8b@2256796`) | 822 MB | 0.660 / 0.657 | 0.475 / 0.472 | 0.0063 / 0.085 | 3 / 300 | 108 ms |
| Kev-0.8B `q8` (int8, fp16 activations) | 788 MB | 0.663 / 0.657 | 0.476 / 0.472 | 0.0073 / 0.087 | 4 / 300 | 115 ms |
| Kev-4B `q8f32` (`kev-4b@4bc64c6`) | 4.7 GB | 0.773 / 0.773 | 0.324 / 0.324 | 0.0028 / 0.075 | 0 / 300 | 360 ms |
| Kev-9B `q8f32` (`kev-9b@442e597`) | 8.8 GB | 0.800 / 0.800 | 0.318 / 0.318 | 0.0033 / 0.160 | 0 / 300 | 570 ms |

Kev's own PyTorch server takes 329 ms (Kev-0.8B), 779 ms (Kev-4B) and about 2 s (Kev-9B) for a comparable request
on an M5 with MPS, which has no fast DeltaNet kernels. Kev-9B needs a GPU with roughly 9 GB free for its weights,
and some browsers cap one origin's Cache Storage below that; the loader then runs uncached and downloads again next
time. The WASM (CPU) fallback is roughly 5× slower than WebGPU. The largest
deviations come from near-ties, where the reference itself is split close to 50/50. Accuracy and Brier in the table
are the raw logits (T = 1), which is what the ONNX graph and the fixtures compare; serving applies the checkpoint
temperature on top and does not change any answer.

`q8f32` (int8 weights, fp32 activations) is the default everywhere. int4 is not usable: at 0.8B, round-to-nearest,
k-quant, block size 16, and int8 for the linear-attention layers all moved probabilities by 0.24–0.78 and flipped
1–8 of 49 answers, and at 4B int4 moved them by 0.31–0.37 with 3 of 29 flipped. For a model whose output is a
calibrated probability that is disqualifying. `export/build.sh` lists the variants that were tried.

### Checkpoint revisions

Kev's checkpoints are republished under the same Hub ids (all three were updated on 2026-09-21), so everything
here is pinned to a commit. `kev_web_export.pin` resolves `jaredpalmer/kev-4b` to `jaredpalmer/kev-4b@<sha>`. The
merge, the fixtures and the reference sets record it, and `package.py` refuses to combine fixtures and weights
from different commits. Each bundle's `manifest.json` names its revision (`run`), and its files live under
`r-<sha>/`. Publishing a new checkpoint therefore never overwrites a file an older manifest points at: the switch is
the single commit that replaces `manifest.json`, and browsers key their cache by that revision.

The published bundles were exported from the night-2 LoRA (`kev-0.8b@2256796`, `kev-4b@4bc64c6`, `kev-9b@442e597`).
Hub `main` later added the fitted temperature to `head.pt` without changing the adapter. The runtime applies those
temperatures for those revisions even when an older `manifest.json` does not yet name `temperature`; re-running
`build_model.sh` writes it into the manifest from `head.pt`.

## Images

Kev was fine-tuned on text only, but its Qwen3.5 bases are vision-language models, and the LoRA leaves the vision
tower and its merger alone. A vision bundle (`kev-4b-vision`, `kev-0.8b-vision`) adds that stock tower, unchanged and
untrained, in front of the same decoder. A request may then carry an image as RGBA pixels; it goes in front of the
state, as Qwen's chat template places it. Without an image the request is exactly the text request.

```ts
const kev = await loadKev(".../kev-4b-vision", { ort, variant: "q8f32" });
const bitmap = await createImageBitmap(file, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
const ctx = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d")!;
ctx.drawImage(bitmap, 0, 0);
const res = await kev.systemOne({
  state: "A screenshot of an online shop.",
  image: ctx.getImageData(0, 0, bitmap.width, bitmap.height),   // any { width, height, data: RGBA bytes }
  questions: { stock: { type: "noul", instructions: "Is the coffee grinder in stock?" } },
});
kev.timing;   // { preprocess, vision, state, branches, imageTokens, cached } of the last request, in ms
```

The vision bundles are built locally and are not published yet. The demo lists `kev-4b-vision` and has image presets,
a file picker and drag and drop.

Two eval sets measure it, each in four conditions: the image, a text caption holding the same facts (the text upper
bound), the context alone, and another image of the same family. [vision-v1](eval/vision-v1) (41 images, 106
questions) has easy generated images and public-domain photos. [vision-v2](eval/vision-v2) (44 images, 128
questions) has full screens downscaled to the pixel cap, dense charts, counting and small print. Accuracy with a 95%
interval from resampling images; PyTorch fp32 on MPS; raw logits.

| Model | Set | Image | Caption | Context only | Other image | Chance |
|---|---|---|---|---|---|---|
| Kev-4B `kev-4b@4bc64c6` | v1 | 0.981 [0.954, 1.000] | 1.000 | 0.425 | 0.208 | 0.360 |
| | v2 | 0.875 [0.822, 0.927] | 0.883 | 0.320 | 0.234 | 0.320 |
| Kev-0.8B `kev-0.8b@2256796` | v1 | 0.906 [0.844, 0.955] | 0.915 | 0.330 | 0.245 | 0.360 |
| | v2 | 0.672 [0.583, 0.756] | 0.742 | 0.320 | 0.242 | 0.320 |

Kev-4B reads an image about as well as the caption that describes it. Swapping in another image makes the answers
worse than chance, so they follow the image. What fails is counting: 13 of Kev-4B's 16 v2 errors are counting
questions. Each eval set's README breaks the results down by type and family.

In the browser (Chrome, WebGPU on an Apple M4 Max, `q8f32` decoder and the tower with fp16 weights computing in fp32),
against the fp32 PyTorch reference (`fixtures/<model>-vision.json`, raw logits, CPU):

| Model | Set | Accuracy, browser / PyTorch | Mean / max \|Δp\| | Answers changed (reference margin) | Vision / decoder, median |
|---|---|---|---|---|---|
| Kev-4B | v1 | 0.972 / 0.981 | 0.0009 / 0.035 | 1 / 106 (0.007) | 142 / 574 ms |
| | v2 | 0.875 / 0.875 | 0.0036 / 0.059 | 0 / 128 | 549 / 1083 ms |
| Kev-0.8B | v1 | 0.906 / 0.906 | 0.0022 / 0.020 | 0 / 106 | 49 / 150 ms |
| | v2 | 0.664 / 0.672 | 0.0043 / 0.025 | 1 / 128 (0.006) | 217 / 273 ms |

v1 images become 108-550 image tokens (median 196) and v2 images 400-560 (median 540). The decoder time covers the
state with the image tokens plus one to four questions. Both answers that changed were near-ties in the reference. The same
bundles on onnxruntime CPU (`kev_web_export.vision_parity`): Kev-4B v1 0.0024 / 0.075 with 2 near-tie changes
(margins 0.007 and 0.044), v2 0.0053 / 0.142 with none; Kev-0.8B v1 0.0035 / 0.043, v2 0.0081 / 0.065 with 2 near-tie
changes (0.021, 0.006). The fp32 ONNX graphs match PyTorch to 1.4e-4 (4B v1), 1.9e-4 (4B v2), 3.5e-5 and 1.1e-4
(0.8B).

The text path does not move. The decoder of a vision bundle is the published `q8f32` graph with one extra input,
`image_embeds`, spliced in (`kev_web_export.splice`). It uses the same weight files, hard-linked, and a text request
feeds it one zero row that no token selects. The text fixtures give the published figures on both bundles: 4B max
\|Δp\| 0.0625 with 1 near-tie change out of 29, 0.8B 0.0967 with none out of 49. On the fp32 graphs, spliced and
unspliced both match PyTorch to 1.3e-4 (4B).

The tower adds 668 MB to Kev-4B's download and 202 MB to Kev-0.8B's. Images are resized the way Qwen's processor does
it, to multiples of 32 pixels and at most 768 × 768 = 589,824 pixels (`max_pixels`, about 576 image tokens); the JS
port matches Qwen's `pixel_values` to a mean difference of 2.3e-5 per value (one uint8 step is 7.8e-3). One image per request. Images need a float32-io variant
(`q8f32`), because the tower's output is fed to the decoder as is. Nothing was trained for this, so images carry
Kev's text calibration: Kev-4B's v1 Brier is 0.021, and 0.173 on v2.

Build a vision bundle from a merged build directory (`build_model.sh` output) and its published `q8f32` graph:

```bash
cd export
uv run --group vision python -m kev_web_export.vision_onnx --run jaredpalmer/kev-4b@<sha> --out build/kev-4b-vision
uv run python -m kev_web_export.splice --src ../public/models/kev-4b/r-<sha>/q8f32 --out build/kev-4b/q8f32-vision
uv run --group vision python -m kev_web_export.vision_fixtures --run jaredpalmer/kev-4b@<sha> \
    --set ../eval/vision-v1 --set ../eval/vision-v2 --out ../fixtures/kev-4b-vision.json        # CPU, about an hour for 4B
uv run --group vision python -m kev_web_export.vision_parity --decoder build/kev-4b/q8f32-vision/model.onnx \
    --vision build/kev-4b-vision/web/model.onnx --build build/kev-4b --vision-json build/kev-4b-vision/vision.json \
    --fixtures ../fixtures/kev-4b-vision.json --text-fixtures ../fixtures/kev-4b.json --out build/kev-4b-vision/parity-q8f32.json
uv run python -m kev_web_export.package --build build/kev-4b --out ../public/models/kev-4b-vision \
    --variant q8f32=q8f32-vision --fixtures ../fixtures/kev-4b.json \
    --vision build/kev-4b-vision --vision-parity q8f32=build/kev-4b-vision/parity-q8f32.json
```

`kev_web_export.vision_eval` gives the PyTorch four-condition tables above (`--set ../eval/vision-v2 --out
../eval/vision-v2/results/kev-4b`).

## How It Works

Kev is a Qwen3.5 base, a rank-16 LoRA and a small pointer head. The head scores each option's `</opt>` hidden state
against the question's `<decide>` hidden state, then divides the logits by a temperature fitted on that checkpoint's
in-distribution development rows (about 2.1–2.4, stored in `head.pt` and written into `manifest.json` on export).
Nothing is generated, so one forward pass is the whole job.

Qwen3.5 mixes Gated DeltaNet (recurrent) layers with full attention, so Kev serves hybrid models in rows. The state
runs once, and each question runs as a continuation of the state's cache
(`kev.model._branch_rows_from_prefix`). That maps directly onto an ordinary exported decoder with a KV cache,
with no custom attention mask:

1. **Merge** (`kev_web_export.merge`): fold the LoRA into the base in fp32 (exact), write a
   `Qwen3_5ForConditionalGeneration` checkpoint, the head (`head.safetensors`) and `kev.json`.
2. **Build** (`export/build.sh`): the onnxruntime-genai model builder with `exclude_lm_head=true` outputs
   `hidden_states` after the final norm. It emits `LinearAttention`, `CausalConvWithState`,
   `LinearAttentionGate`, `GatedRMSNorm`, `MRotaryEmbedding` and `GroupQueryAttention`, all of which have WebGPU
   kernels in onnxruntime-web 1.30 (`onnxruntime-web/webgpu`, the native WebGPU EP). 667 of 675 nodes run on
   WebGPU; the other 8 are shape ops.
3. **Post-process** (`kev_web_export.postprocess`): the builder can only quantize embeddings to int4. The
   embedding table (0.5 GB fp16 for 0.8B, 2.5 GB fp32 for 4B) becomes int8 with one scale per row, using plain
   `Gather`/`Cast`/`Mul`. The rotary caches are trimmed from 262k positions to 8,192, which is `kev.serve`'s limit.
   External weights are split into files of at most 32 MB (`--shard-mb`): proxies and CDNs cap response bodies (bb
   connect cuts one at 34.5 MiB), a failed file is cheap to retry, and browsers cap a single buffer near 2 GB. A
   single tensor larger than that keeps its own file — Kev-9B's MLP matrices are 50 MB each — which Hugging Face
   serves fine but a capped proxy would not. An
   embedding table is one tensor of hundreds of MB and an ONNX initializer cannot span files, so it is stored as
   column slices that are gathered separately and concatenated — bit-identical, and every file stays small.
4. **Package** (`kev_web_export.package`): `manifest.json` (I/O names, empty-cache shapes, parity), tokenizer,
   head and one directory per variant.

The runtime (`src/`) is a port of `kev/api.py` and `kev/model.py`. `render`, `to_record`, `to_answers`,
`encode` and the delimiter escaping are reproduced exactly, including Python's `str()` formatting (`True`,
`1e-07`). On WebGPU the state's recurrent, conv and KV caches stay on the GPU (`gpu-buffer`) and are reused by
every question, and by later requests with the same state (LRU of 4, as in `kev.serve`).

## Quick Start

Using the library needs nothing but `npm install @ai-ecoverse/kev.js onnxruntime-web` and the published weights.
Building the weights yourself needs Python 3.12 with [uv](https://docs.astral.sh/uv/) and a lot of disk (Kev-0.8B
about 20 GB, Kev-4B about 45 GB, Kev-9B about 85 GB at peak):

```bash
git clone --recursive https://github.com/ai-ecoverse/kev.js && cd kev.js
cd export && uv sync
./build_model.sh kev-0.8b                  # or kev-4b, kev-9b; optionally a pinned run: jaredpalmer/kev-4b@<sha>
cd .. && npm install && npm run dev        # http://127.0.0.1:5173, serving public/models
```

`build_model.sh` pins the checkpoint, merges the LoRA in fp32, builds the int8 WebGPU graph, shards it, builds the
fp32 reference, writes the PyTorch fixtures and the 300-record reference set, measures parity and packages the bundle
into `public/models/<name>`. On macOS the onnxruntime-genai builder can abort with `recursive_mutex lock failed`
after it has written everything, so `build.sh` checks for `genai_config.json` instead of the exit code.

### Publishing

```bash
HF_TOKEN=... uv run python -m kev_web_export.upload_hf --repo ai-ecoverse/kev.js   # weights + model card
npm run build:lib                                                                 # dist/ for npm (tsc)
npm run build:demo                                                                # homepage (PAGES_BASE=/kev.js/ in CI)
```

`.github/workflows` builds the homepage to GitHub Pages on every push to `main`, runs the tests that need no
weights, and publishes the npm package from `main` with [semantic-release](https://semantic-release.org/)
(trusted publishing, OIDC, no token). Commits that follow
[Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `BREAKING CHANGE:`) cut the next
patch / minor / major; `v0.2.0` is the last hand-tagged release. The demo loads
weights from Hugging Face unless `VITE_MODEL_BASE` or `?models=<url>` says otherwise; `npm run dev` serves
`public/models` instead. GitHub Pages cannot set COOP/COEP headers, so `crossOriginIsolated` is false there and the
WASM fallback runs single-threaded; WebGPU is unaffected.

`kev.systemOne(req, { onAnswer })` reports each question as it finishes, so a UI can fill answers in as they land.
`kev.systemOneSeparate()` answers each question in its own pass. `kev.probs(record)` returns probabilities after the
checkpoint temperature. `kev.systemOne(req, { dateFacts: true })` is `KEV_DATE_FACTS=1`.
Weight files are cached in Cache Storage (`kev-web-v1`). In the demo page, `window.kev.systemOne(...)` works from
the console, and `?verbose` logs where onnxruntime placed each node.

`loadKev` also reads a bundle that is already on disk, in place: pass a directory handle that holds it (OPFS, or
`showDirectoryPicker()`) or a function that reads a file by its bundle path, instead of a URL. Nothing is fetched and
nothing is copied into Cache Storage, so a 9 GB model is stored once. `modelFiles(manifest, variant)` lists the files a
variant loads with their sizes, for a download or resume step; a local file of the wrong size fails the load by name.

```ts
const opfs = await (await navigator.storage.getDirectory()).getDirectoryHandle("kev-9b");
const kev = await loadKev(opfs, { ort, variant: "q8f32" });
// or through a virtual file system: loadKev((path) => fs.readFileBinary(`/models/kev-9b/${path}`), { ort })
```

## Testing

```bash
npm test                                   # rendering, tokenization, encoding, date_facts, temperature; full runtime on onnxruntime-node
KEV_VARIANTS=fp32 npm test                 # just the exact variant
KEV_MODEL=kev-4b npm test                  # fixtures/kev-4b.json against public/models/kev-4b
npm run fetch-model -- kev-0.8b q8f32      # published bundle -> public/models/kev-0.8b (resumes; keeps files at size)
npx playwright install chromium && npm run test:browser   # Chromium: loadKev from the real OPFS, the model on WASM and WebGPU
cd export && uv run python -m kev_web_export.parity --model build/kev-0.8b/web-q8f32/model.onnx \
    --head build/kev-0.8b/head.safetensors --fixtures ../fixtures/kev-0.8b.json
```

The browser tests (`test/browser`, Playwright) run their cases in a module worker, as the demo does. Three write a
synthetic bundle into the browser's OPFS and load it through a directory handle and a read function. Two copy the
Kev-0.8B bundle into OPFS, load it from there on WASM and on WebGPU, and compare with the PyTorch fixtures; they also
check that no model file was fetched and nothing went to Cache Storage. Without a bundle, or without a WebGPU adapter,
those skip; `KEV_REQUIRE_MODEL=1` and `KEV_REQUIRE_WEBGPU=1` make that a failure. CI downloads the bundle (cached per
published manifest) and runs the Node and browser suites against it. Its runners have no GPU, so WebGPU runs on
SwiftShader (a CPU Vulkan): that checks the WebGPU kernels' results, not GPU speed, at about 150 s per fixture, so CI
runs two (`KEV_WEBGPU_FIXTURES`). `KEV_WEBGPU_SWIFTSHADER=1` forces that adapter locally.

Image requests (`test/vision.test.ts`, and the browser case `a Kev vision bundle matches the PyTorch image fixtures`)
check the preprocessing and mRoPE positions without weights. With a vision bundle in `public/models/kev-4b-vision`
(`KEV_VISION_MODEL=kev-0.8b-vision` for the other) they run every eval image through the tower and the spliced
decoder and compare per set with `fixtures/<model>-vision.json`, under the text rule's bounds against the
manifest's `vision.parity`. Node has no JPEG decoder here, so it skips v1's two JPEG photos; the browser decodes all
85 images itself. `KEV_VISION_EP=wasm` runs the browser case on WASM, and `KEV_VISION_REPORT=<file>` saves its per-set
report. On macOS, Playwright's headless Chromium reports a Metal adapter but stalls for minutes on Kev-4B's first
pass, so the WebGPU numbers above come from `KEV_BROWSER_CHANNEL=chrome` (the installed Chrome).
`KEV_HARNESS_PORT` moves the harness off 5174 when another dev server holds that port.

`scripts/cdp.mjs` drives a page in a Chrome started with `--remote-debugging-port=9222`, for checking the demo in a
real browser: `node scripts/cdp.mjs '<expression>'` evaluates in the tab (`MATCH=` picks it by URL), and
`node scripts/cdp.mjs --shot out.png` screenshots it.

- `fixtures/kev-0.8b.json`: 43 records (3 hand-written, including structured state and delimiter injection, plus
  40 from transfer-v4 dev) with the rendered record, Kev's token encoding and the PyTorch probabilities.
  Regenerate with `kev_web_export.fixtures`.
- `fixtures/kev-4b.json`: the same 3 hand-written records plus 20 dev records, for Kev-4B.
- `fixtures/kev-4b-vision.json`, `fixtures/kev-0.8b-vision.json`: every vision-v1 and vision-v2 image with its
  request, Qwen's patch grid, statistics of `pixel_values` and the fp32 CPU PyTorch probabilities
  (`kev_web_export.vision_fixtures`).
- `fixtures/kev-*-transfer-v4-dev300.json`: 300 labelled records with fp32 reference probabilities
  (`kev_web_export.evalset`), used for the browser accuracy and Brier comparisons above.

## Limitations

- JSON parsing loses two things Kev's Python server keeps. `1.0` arrives as `1` and is rendered `1`, where Python
  renders `1.0`. Object keys that look like integers (`"10"`, `"2"`) are iterated in numeric order, which can
  reorder Choice options with numeric names.
- The first load downloads 822 MB for Kev-0.8B, or 4.7 GB for Kev-4B, and the files stay in Cache Storage. Serve
  them from a fast origin: over a tunnel at ~1.2 MB/s, Kev-0.8B takes 11 minutes. More parallel requests do not
  help on a bandwidth-limited link (measured: 1.2 MB/s with one stream, 0.75 MB/s across six), so the loader
  fetches 2 files at a time.
  Kev-4B needs a GPU with enough memory for about 4.5 GB of weights. Only Chrome was tested; Safari and Firefox
  WebGPU are untested.
- onnxruntime-node 1.30 on Node 24+ reads 0 bytes from `Float16Array` float16 tensors, so the tests hide
  `Float16Array` (`test/no-float16.ts`). Browsers are not affected.
- Requests run one at a time per model instance.

## License

Apache-2.0, like Kev and the Qwen3.5 weights. Kev itself is vendored as a pinned submodule (`vendor/kev`).
