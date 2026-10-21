// The image side of kev-ministral: Pixtral's preprocessing as Ministral 3's PixtralProcessor does it (port of
// transformers' PixtralImageProcessor and PixtralProcessor), and the image's token block.
//
//   size: scale down to fit the longest edge (1540) if needed, then round each side up to a multiple of 28 px (14 px
//   patches x the 2 x 2 merge: 90 x 90 -> 112 x 112, 1024 x 576 -> 1036 x 588); Pillow's bicubic (a = -0.5,
//   antialiased, uint8 after each pass); rescale 1/255; CLIP mean and std; CHW float32. Alpha is dropped, as
//   transformers' convert_rgb is not given a background.
//   tokens: one [IMG] per 2 x 2 merged patch, row by row, [IMG_BREAK] after each row but the last, which ends in
//   [IMG_END].

import { resizeRGBA, type ImageLike } from "./vision.ts";

export const PIXTRAL = {
  patch: 14,
  merge: 2,
  longestEdge: 1540,
  mean: [0.48145466, 0.4578275, 0.40821073],
  std: [0.26862954, 0.26130258, 0.27577711],
} as const;

/** The [height, width] Pixtral resizes an image to. */
export function pixtralSize(height: number, width: number, longestEdge: number = PIXTRAL.longestEdge): [number, number] {
  const ratio = Math.max(height / longestEdge, width / longestEdge);
  if (ratio > 1) { height = Math.floor(height / ratio); width = Math.floor(width / ratio); }
  // an odd patch count would not merge 2 x 2: the features and the [IMG] tokens would disagree
  const unit = PIXTRAL.patch * PIXTRAL.merge;
  return [Math.max(unit, Math.ceil(height / unit) * unit), Math.max(unit, Math.ceil(width / unit) * unit)];
}

/** pixel_values for the vision encoder ([1, 3, height, width]) and the merged grid: rows x cols [IMG] tokens. */
export interface PixtralPixels { data: Float32Array; height: number; width: number; rows: number; cols: number }

export function pixtralPreprocess(img: ImageLike, longestEdge?: number): PixtralPixels {
  const { width: w, height: h } = img;
  if (!Number.isInteger(w) || !Number.isInteger(h) || w < 1 || h < 1 || img.data.length !== w * h * 4)
    throw new RangeError(`image data has ${img.data.length} bytes, expected ${w} x ${h} x 4 (RGBA)`);
  const [H, W] = pixtralSize(h, w, longestEdge);
  const planes = H !== h || W !== w ? resizeRGBA(img.data, h, w, H, W)
    : [0, 1, 2].map((ch) => { const p = new Float32Array(w * h); for (let i = 0; i < w * h; i++) p[i] = img.data[i * 4 + ch]; return p; });
  const data = new Float32Array(3 * H * W);
  for (let ch = 0; ch < 3; ch++) {
    const plane = planes[ch], mean = PIXTRAL.mean[ch], std = PIXTRAL.std[ch], o = ch * H * W;
    for (let i = 0; i < H * W; i++) data[o + i] = (plane[i] / 255 - mean) / std;
  }
  const unit = PIXTRAL.patch * PIXTRAL.merge;
  return { data, height: H, width: W, rows: H / unit, cols: W / unit };
}

/** The image's tokens, in the order the encoder's features fill the [IMG] positions. */
export function pixtralTokens(p: { rows: number; cols: number }, ids: { img: number; brk: number; end: number }): number[] {
  const out: number[] = [];
  for (let r = 0; r < p.rows; r++) {
    for (let c = 0; c < p.cols; c++) out.push(ids.img);
    out.push(r === p.rows - 1 ? ids.end : ids.brk);
  }
  return out;
}
