// Ministral 3 / Pixtral image preprocessing (port of transformers' PixtralImageProcessor + PixtralProcessor):
//   target size: scale down to fit longest_edge 1540 if needed, then round each side up to a multiple of 28 px (14 px
//   patches x the 2 x 2 merge, as transformers' processor does: 90 x 90 -> 112 x 112, 1024 x 576 -> 1036 x 588); bicubic resize (Pillow's kernel, a = -0.5, antialiased, uint8 after each pass: jev-omni.js / cua-s1.js);
//   rescale 1/255; normalize with the CLIP mean/std; CHW float32.
//   tokens: one [IMG] per 2 x 2 merged patch, row by row, [IMG_BREAK] after each row but the last, which ends in
//   [IMG_END].

export interface ImageLike { width: number; height: number; data: Uint8ClampedArray | Uint8Array }   // RGBA

const PATCH = 14, MERGE = 2, LONGEST = 1540;
const MEAN = [0.48145466, 0.4578275, 0.40821073], STD = [0.26862954, 0.26130258, 0.27577711];

export function targetSize(h: number, w: number): [number, number] {
  const ratio = Math.max(h / LONGEST, w / LONGEST);
  if (ratio > 1) { h = Math.floor(h / ratio); w = Math.floor(w / ratio); }
  const unit = PATCH * MERGE;   // an odd patch count would not merge 2 x 2: features and [IMG] tokens would disagree
  return [Math.ceil(h / unit) * unit, Math.ceil(w / unit) * unit];
}

const cubic = (x: number, a = -0.5) => {
  x = Math.abs(x);
  return x <= 1 ? ((a + 2) * x - (a + 3)) * x * x + 1 : x < 2 ? ((a * x - 5 * a) * x + 8 * a) * x - 4 * a : 0;
};

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

function resizeRGB(src: Uint8Array, h: number, w: number, H: number, W: number): Uint8Array {
  const ax = axis(w, W), ay = axis(h, H);
  const tmp = new Uint8Array(h * W * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < W; x++) {
    const { start, w: k } = ax[x];
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let t = 0; t < k.length; t++) s += src[(y * w + start + t) * 3 + c] * k[t];
      tmp[(y * W + x) * 3 + c] = Math.min(255, Math.max(0, Math.round(s)));
    }
  }
  const out = new Uint8Array(H * W * 3);
  for (let y = 0; y < H; y++) {
    const { start, w: k } = ay[y];
    for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let t = 0; t < k.length; t++) s += tmp[((start + t) * W + x) * 3 + c] * k[t];
      out[(y * W + x) * 3 + c] = Math.min(255, Math.max(0, Math.round(s)));
    }
  }
  return out;
}

export interface Pixels { data: Float32Array; height: number; width: number; rows: number; cols: number }

export function preprocess(img: ImageLike): Pixels {
  const { width: w, height: h } = img;
  let rgb: Uint8Array = new Uint8Array(w * h * 3);
  for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) rgb[i * 3 + c] = img.data[i * 4 + c];
  const [H, W] = targetSize(h, w);
  if (H !== h || W !== w) rgb = resizeRGB(rgb, h, w, H, W);
  const data = new Float32Array(3 * H * W);
  for (let c = 0; c < 3; c++) for (let i = 0; i < H * W; i++) data[c * H * W + i] = (rgb[i * 3 + c] / 255 - MEAN[c]) / STD[c];
  return { data, height: H, width: W, rows: H / PATCH / MERGE, cols: W / PATCH / MERGE };
}

export function imageTokens(p: Pixels, ids: { img: number; brk: number; end: number }): number[] {
  const out: number[] = [];
  for (let r = 0; r < p.rows; r++) {
    for (let c = 0; c < p.cols; c++) out.push(ids.img);
    out.push(r === p.rows - 1 ? ids.end : ids.brk);
  }
  return out;
}

/** RGBA pixels of an image URL as the browser decodes it, without color management (PIL reads raw values). */
export async function loadImage(src: string | Blob): Promise<ImageLike> {
  const blob = typeof src === "string" ? await (await fetch(src)).blob() : src;
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const cv = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height);
  return { width: d.width, height: d.height, data: d.data };
}
