// The image side of Kev: Qwen2-VL's image processor, the size-dependent inputs of the exported vision tower
// (export/kev_web_export/vision_onnx.py), and Qwen3.5's 3D (mRoPE) positions for a state that starts with an image.
// Ports of transformers' Qwen2VLImageProcessorFast, vision_utils.get_vision_interpolation_indices_and_weights and
// get_vision_position_ids, Qwen3_5VisionRotaryEmbedding and Qwen3_5Model.get_rope_index. Adapted from cua-s1.js
// (src/four-b-vision.ts).

/** RGBA pixels, as canvas getImageData, OffscreenCanvas and ImageData produce them. */
export interface ImageLike { width: number; height: number; data: Uint8ClampedArray | Uint8Array }

/** manifest.vision.config: the tower's shape, Qwen's normalization, the pixel budget and the image token ids. */
export interface VisionConfig {
  patch_size: number;            // 16
  merge_size: number;            // 2
  temporal_patch_size: number;   // 2
  hidden_size: number;           // 1024: the ViT's width
  out_hidden_size: number;       // the text model's hidden size
  num_heads: number;             // 16
  num_grid_per_side: number;     // 48: the learned position table is 48 x 48
  rope_theta: number;
  image_mean: number[];
  image_std: number[];
  min_pixels: number;
  max_pixels: number;
  image_token: number;           // <|image_pad|>
  vision_start: number;          // <|vision_start|>
  vision_end: number;            // <|vision_end|>
}

/** Python's round(): halves go to the even neighbour. */
const pyRound = (x: number) => { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f; };

/** qwen2_vl smart_resize: both sides a multiple of patch * merge, the area within [min_pixels, max_pixels]. */
export function smartResize(height: number, width: number, c: VisionConfig): [number, number] {
  const f = c.patch_size * c.merge_size;
  if (Math.max(height, width) / Math.min(height, width) > 200) throw new RangeError("aspect ratio must be below 200");
  let h = Math.max(f, pyRound(height / f) * f), w = Math.max(f, pyRound(width / f) * f);
  if (h * w > c.max_pixels) {
    const beta = Math.sqrt((height * width) / c.max_pixels);
    h = Math.max(f, Math.floor(height / beta / f) * f); w = Math.max(f, Math.floor(width / beta / f) * f);
  } else if (h * w < c.min_pixels) {
    const beta = Math.sqrt(c.min_pixels / (height * width));
    h = Math.ceil((height * beta) / f) * f; w = Math.ceil((width * beta) / f) * f;
  }
  return [h, w];
}

// torch's antialiased bicubic (which torchvision uses for uint8 images) takes Pillow's kernel, a = -0.5, not the
// -0.75 of plain bicubic. With it, about 0.1% of pixels still differ from torchvision's, by at most 2 levels: torch
// rounds its weights to fixed point.
const cubic = (x: number, a = -0.5) => {
  x = Math.abs(x);
  return x <= 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? ((a * x - 5 * a) * x + 8 * a) * x - 4 * a : 0;
};

/** One axis of a bicubic resize (align_corners=False, edge clamped, antialiased when shrinking): taps and weights
 * per output index, the scheme torch's _upsample_bicubic2d_aa uses. */
function axis(inSize: number, outSize: number) {
  const scale = inSize / outSize, support = scale > 1 ? 2 * scale : 2, inv = scale > 1 ? 1 / scale : 1;
  const out: { start: number; w: number[] }[] = [];
  for (let i = 0; i < outSize; i++) {
    const center = (i + 0.5) * scale;
    const start = Math.max(0, Math.trunc(center - support + 0.5)), end = Math.min(inSize, Math.trunc(center + support + 0.5));
    const w: number[] = [];
    let sum = 0;
    for (let j = start; j < end; j++) { const k = cubic((j - center + 0.5) * inv); w.push(k); sum += k; }
    out.push({ start, w: w.map((k) => (sum ? k / sum : 0)) });
  }
  return out;
}

/** Bicubic resize of an RGBA image to three RGB planes, rounded and clamped to uint8 after each pass as torchvision
 * does for uint8 images. The horizontal pass reads the bytes directly, so nothing is allocated at the source's full
 * size: a 48-megapixel photo would otherwise take ~576 MB of float planes before being cut down to max_pixels. */
function resizeRGBA(data: ImageLike["data"], h: number, w: number, H: number, W: number): Float32Array[] {
  const ax = axis(w, W), ay = axis(h, H);
  return [0, 1, 2].map((ch) => {
    const tmp = new Float32Array(h * W);
    for (let y = 0; y < h; y++) for (let x = 0; x < W; x++) {
      const { start, w: k } = ax[x]; let s = 0;
      for (let t = 0; t < k.length; t++) s += data[(y * w + start + t) * 4 + ch] * k[t];
      tmp[y * W + x] = Math.min(255, Math.max(0, Math.round(s)));
    }
    const out = new Float32Array(H * W);
    for (let y = 0; y < H; y++) {
      const { start, w: k } = ay[y];
      for (let x = 0; x < W; x++) {
        let s = 0;
        for (let t = 0; t < k.length; t++) s += tmp[(start + t) * W + x] * k[t];
        out[y * W + x] = Math.min(255, Math.max(0, Math.round(s)));
      }
    }
    return out;
  });
}

export interface Patches {
  /** [gridH * gridW, 3 * temporal * patch * patch], rows in spatial-merge-block order */
  data: Float32Array;
  gridH: number;
  gridW: number;
}

/** Qwen2VLImageProcessorFast for one image: resize to smartResize's size, rescale and normalize, repeat the frame
 * to the temporal patch size, and cut 16 x 16 patches ordered block by block (2 x 2 blocks, row-major within).
 * Alpha is ignored, as Pillow's convert("RGB") drops it. */
export function preprocess(img: ImageLike, c: VisionConfig): Patches {
  const { width: w, height: h } = img;
  if (img.data.length !== w * h * 4) throw new RangeError(`image data has ${img.data.length} bytes, expected ${w} x ${h} x 4 (RGBA)`);
  const [H, W] = smartResize(h, w, c);
  const planes = H !== h || W !== w ? resizeRGBA(img.data, h, w, H, W)
    : [0, 1, 2].map((ch) => { const p = new Float32Array(w * h); for (let i = 0; i < w * h; i++) p[i] = img.data[i * 4 + ch]; return p; });
  const ps = c.patch_size, m = c.merge_size, tps = c.temporal_patch_size;
  const gridH = H / ps, gridW = W / ps, dim = 3 * tps * ps * ps;
  const data = new Float32Array(gridH * gridW * dim);
  const norm = [0, 1, 2].map((ch) => [1 / 255 / c.image_std[ch], c.image_mean[ch] / c.image_std[ch]]);
  let row = 0;
  for (let bh = 0; bh < gridH / m; bh++) for (let bw = 0; bw < gridW / m; bw++) for (let mh = 0; mh < m; mh++) for (let mw = 0; mw < m; mw++) {
    const y0 = (bh * m + mh) * ps, x0 = (bw * m + mw) * ps;
    let o = row++ * dim;
    for (let ch = 0; ch < 3; ch++) {
      const [s, b] = norm[ch], plane = planes[ch];
      for (let t = 0; t < tps; t++) for (let y = 0; y < ps; y++) for (let x = 0; x < ps; x++) data[o++] = plane[(y0 + y) * W + x0 + x] * s - b;
    }
  }
  return { data, gridH, gridW };
}

/** Patch (row, col) of the i-th patch in spatial-merge-block order. */
const blockRC = (i: number, gridW: number, m: number) => {
  const bw = gridW / m;
  return [Math.floor(i / (m * m * bw)) * m + (Math.floor(i / m) % m), (Math.floor(i / (m * m)) % bw) * m + (i % m)];
};

export interface VisionInputs { posIdx: BigInt64Array; posW: Float32Array; cos: Float32Array; sin: Float32Array }

/** The vision graph's size-dependent inputs: bilinear taps into the learned position table (align_corners) and the
 * 2D rotary angles, one row per patch in block order. */
export function visionInputs(gridH: number, gridW: number, c: VisionConfig): VisionInputs {
  const P = gridH * gridW, side = c.num_grid_per_side, m = c.merge_size;
  const headDim = c.hidden_size / c.num_heads, half = headDim / 2;
  const invFreq = Float32Array.from({ length: half / 2 }, (_, i) => 1 / c.rope_theta ** ((2 * i) / half));
  const posIdx = new BigInt64Array(P * 4), posW = new Float32Array(P * 4), cos = new Float32Array(P * headDim), sin = new Float32Array(P * headDim);
  const taps = (i: number, size: number) => {
    const src = Math.fround((Math.fround(i) * (side - 1)) / Math.max(size - 1, 1)), f = Math.floor(src);
    return [[Math.min(f, side - 1), 1 - Math.abs(src - f)], [Math.min(f + 1, side - 1), Math.max(0, 1 - Math.abs(src - f - 1))]];
  };
  for (let i = 0; i < P; i++) {
    const [r, col] = blockRC(i, gridW, m);
    const th = taps(r, gridH), tw = taps(col, gridW);
    let k = 0;
    for (const [hi, hw] of th) for (const [wi, ww] of tw) { posIdx[i * 4 + k] = BigInt(hi * side + wi); posW[i * 4 + k] = hw * ww; k++; }
    for (let j = 0; j < invFreq.length; j++) {
      const fh = Math.fround(r * invFreq[j]), fw = Math.fround(col * invFreq[j]);
      for (const [pos, f] of [[j, fh], [invFreq.length + j, fw]] as const) {
        cos[i * headDim + pos] = cos[i * headDim + half + pos] = Math.cos(f);
        sin[i * headDim + pos] = sin[i * headDim + half + pos] = Math.sin(f);
      }
    }
  }
  return { posIdx, posW, cos, sin };
}

/** [3][S] mRoPE positions for token ids with at most one image run (Qwen3_5Model.get_rope_index): text counts up on
 * all three axes; the image's tokens get (start, start + row, start + column) over the merged grid, and the text
 * after it resumes at start + max(rows, columns). */
export function ropePositions(ids: number[], gridH: number, gridW: number, imageToken: number, m = 2): number[][] {
  const t: number[] = [], h: number[] = [], w: number[] = [];
  let cur = 0;
  for (let i = 0; i < ids.length;) {
    if (ids[i] !== imageToken) { t.push(cur); h.push(cur); w.push(cur); cur++; i++; continue; }
    const gh = gridH / m, gw = gridW / m;
    for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) { t.push(cur); h.push(cur + r); w.push(cur + c); }
    i += gh * gw; cur += Math.max(gh, gw);
  }
  return [t, h, w];
}

/** FNV-1a over an image's size and pixels: part of the state cache key, so the same text with another image misses. */
export function imageKey(img: ImageLike): string {
  let a = 0x811c9dc5, b = 0x01000193 ^ img.width ^ (img.height << 16);
  const d = img.data;
  for (let i = 0; i < d.length; i++) { a = Math.imul(a ^ d[i], 0x01000193); if ((i & 3) === 3) b = Math.imul(b ^ a, 0x01000193); }
  return `${img.width}x${img.height}:${(a >>> 0).toString(16)}${(b >>> 0).toString(16)}`;
}
