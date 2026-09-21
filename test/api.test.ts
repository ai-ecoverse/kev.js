import { test } from "node:test";
import assert from "node:assert/strict";
import { pyFloat, r2, render, toAnswers, toRecord, validate, ValidationError, pyJsonDumps } from "../src/api.ts";
import { fixtures } from "./fixtures.ts";

test("toRecord renders every fixture exactly like kev.api.to_record", () => {
  for (const f of fixtures) {
    const { record, meta } = toRecord(validate(f.request));
    assert.deepEqual(record, { state: f.record.state, questions: f.record.questions.map((q) => ({ instr: q.instr, options: q.options })) }, f.name);
    assert.deepEqual(meta, f.meta, f.name);
  }
});

test("toAnswers matches kev.api.to_answers on the reference probabilities", () => {
  for (const f of fixtures) assert.deepEqual(toAnswers(f.probs, f.meta), f.answers, f.name);
});

test("Python formatting", () => {
  assert.equal(render({ a: true, b: null, c: [1, 2.5, { d: false }], e: 1e-7, f: 0.1 }), "a: True\nb: \nc:\n  - 1\n  - 2.5\n  - d: False\ne: 1e-07\nf: 0.1");
  for (const [x, s] of [[1, "1.0"], [0.5, "0.5"], [1e16, "1e+16"], [1234.5, "1234.5"], [0.0001, "0.0001"], [0.00001, "1e-05"], [123456789012345.6, "123456789012345.6"]] as const)
    assert.equal(pyFloat(x), s);
  assert.equal(r2(0.125), 0.12); assert.equal(r2(0.375), 0.38); assert.equal(r2(0.4449), 0.44);
  assert.equal(pyJsonDumps({ a: { type: "noul", noul: 1 }, "é": [0, 0.25] }), '{"a": {"type": "noul", "noul": 1.0}, "\\u00e9": [0.0, 0.25]}');
});

test("validation mirrors the pydantic models", () => {
  const bad = [{}, { state: "x", questions: {} }, { state: "x", questions: { a: { type: "choice", instructions: "i", criteria: {} } } },
    { state: "x", questions: { a: { type: "score", instructions: "i", criteria: ["one"] } } }, { state: "x", questions: { a: { type: "vote", instructions: "i" } } }];
  for (const b of bad) assert.throws(() => validate(b), ValidationError);
});
