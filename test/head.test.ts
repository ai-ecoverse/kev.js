import { test } from "node:test";
import assert from "node:assert/strict";
import { PointerHead, type F32Tensor } from "../src/head.ts";
import { temperatureFor } from "../src/model.ts";

function tensor(shape: number[], data: number[]): F32Tensor {
  return { shape, data: Float32Array.from(data) };
}

/** Identity 2×2 head: logits are just the option hidden states dotted with the decide state. */
function identityHead(temperature = 1) {
  const I = [1, 0, 0, 1];
  return new PointerHead({
    "q.weight": tensor([2, 2], I), "q.bias": tensor([2], [0, 0]),
    "k.weight": tensor([2, 2], I), "k.bias": tensor([2], [0, 0]),
  }, temperature);
}

test("softmax(z/T) is the serving temperature; argmax is unchanged", () => {
  const h = identityHead(1);
  const decide = Float32Array.from([1, 0]);
  const opts = [Float32Array.from([2, 0]), Float32Array.from([0, 0])];   // logits 2/√2 and 0
  const raw = h.probs(decide, opts, 1);
  const cal = h.probs(decide, opts, 2);
  assert.equal(raw.indexOf(Math.max(...raw)), cal.indexOf(Math.max(...cal)));
  assert.ok(cal[0] < raw[0]);   // mass moves toward uniform
  assert.ok(Math.abs(cal.reduce((s, x) => s + x, 0) - 1) < 1e-12);
  // p^(1/T) renormalised equals softmax(z/T)
  const T = 2;
  const lifted = raw.map((p) => p ** (1 / T));
  const z = lifted.reduce((s, x) => s + x, 0);
  lifted.forEach((p, i) => assert.ok(Math.abs(p / z - cal[i]) < 1e-12));
});

test("PointerHead.temperature is the default for probs()", () => {
  const h = identityHead(2);
  const decide = Float32Array.from([1, 0]);
  const opts = [Float32Array.from([2, 0]), Float32Array.from([0, 0])];
  assert.deepEqual(h.probs(decide, opts), h.probs(decide, opts, 2));
});

test("temperatureFor: manifest wins, else night-2 SHA, else 1", () => {
  assert.equal(temperatureFor("jaredpalmer/kev-4b@4bc64c6", 1.5), 1.5);
  assert.equal(temperatureFor("jaredpalmer/kev-0.8b@2256796"), 2.406050072164233);
  assert.equal(temperatureFor("jaredpalmer/kev-4b@4bc64c6b4c4881148661ffb823ce21fcfdc79a0e"), 2.1435469250725863);
  assert.equal(temperatureFor("jaredpalmer/kev-9b@442e597"), 2.2973967099940698);
  assert.equal(temperatureFor("jaredpalmer/kev-4b@qwen3"), 1);
  assert.equal(temperatureFor("jaredpalmer/kev-0.8b@v7-base"), 1);
  assert.equal(temperatureFor("runs/local"), 1);
});
