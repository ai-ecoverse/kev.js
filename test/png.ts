// A minimal PNG decoder for the tests: 8-bit grey, grey + alpha, RGB or RGBA, not interlaced (what Pillow and Chromium
// write for the eval images). Adapted from cua-s1.js (test/png.ts). Output is RGBA, as canvas getImageData gives it.
import { inflateSync } from "node:zlib";

const CHANNELS: Record<number, number> = { 0: 1, 2: 3, 4: 2, 6: 4 };

export function decodePng(buf: Uint8Array): { width: number; height: number; data: Uint8Array } {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 8, width = 0, height = 0, channels = 0;
  const idat: Uint8Array[] = [];
  while (o < buf.length) {
    const len = dv.getUint32(o), type = String.fromCharCode(...buf.subarray(o + 4, o + 8)), body = buf.subarray(o + 8, o + 8 + len);
    if (type === "IHDR") {
      width = dv.getUint32(o + 8); height = dv.getUint32(o + 12);
      const depth = body[8], color = body[9], interlace = body[12];
      if (depth !== 8 || interlace || !CHANNELS[color]) throw new Error(`unsupported PNG: depth ${depth}, color ${color}, interlace ${interlace}`);
      channels = CHANNELS[color];
    } else if (type === "IDAT") idat.push(body);
    else if (type === "IEND") break;
    o += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat)), stride = width * channels, px = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)], line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? px[y * stride + x - channels] : 0, b = y ? px[(y - 1) * stride + x] : 0, c = x >= channels && y ? px[(y - 1) * stride + x - channels] : 0;
      const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
      const pred = [0, a, b, (a + b) >> 1, pa <= pb && pa <= pc ? a : pb <= pc ? b : c][f];
      px[y * stride + x] = (line[x] + pred) & 255;
    }
  }
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const s = px.subarray(i * channels, (i + 1) * channels);
    const grey = channels <= 2;
    for (let ch = 0; ch < 3; ch++) data[i * 4 + ch] = grey ? s[0] : s[ch];
    data[i * 4 + 3] = channels === 2 ? s[1] : channels === 4 ? s[3] : 255;
  }
  return { width, height, data };
}
