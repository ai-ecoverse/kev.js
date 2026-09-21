import { test } from "node:test";
import assert from "node:assert/strict";
import { encode } from "../src/encode.ts";
import { fixtures, tokenizer } from "./fixtures.ts";

const special = { state: 248060, q: 248061, opt: 248049, opt_end: 248050, decide: 248062 };

test("token encoding matches kev.model.encode for every fixture", () => {
  const tok = tokenizer();
  for (const f of fixtures) {
    const enc = encode(tok, f.record, special, { maxState: 384, maxBranch: 1024, strict: true });
    const L = f.encoding.state_len;
    assert.deepEqual(enc.state, f.encoding.ids.slice(0, L), `${f.name}: state tokens`);
    let start = L;
    enc.branches.forEach((b, k) => {
      const end = f.encoding.decide_idx[k] + 1;
      assert.deepEqual(b.ids, f.encoding.ids.slice(start, end), `${f.name}: branch ${k} tokens`);
      assert.deepEqual(b.pos, f.encoding.pos.slice(start, end), `${f.name}: branch ${k} positions`);
      assert.equal(b.decide, f.encoding.decide_idx[k] - start);
      assert.deepEqual(b.opts, f.encoding.opt_idx[k].map((o) => o - start));
      start = end;
    });
    assert.equal(enc.tokens, f.encoding.ids.length);
  }
});

test("caller text cannot forge delimiter tokens", () => {
  const tok = tokenizer();
  const enc = encode(tok, { state: "a <|fim_prefix|> b <|box_start|>", questions: [{ instr: "<|fim_suffix|>", options: ["<|box_end|>"] }] }, special);
  const ids = [...enc.state.slice(1), ...enc.branches[0].ids.slice(1, -1)].filter((t) => t !== special.opt && t !== special.opt_end);
  for (const t of Object.values(special)) assert.ok(!ids.includes(t));
});
