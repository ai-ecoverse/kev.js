import { test } from "node:test";
import assert from "node:assert/strict";
import { choiceConfidence, dateFacts, pyFloat, pyRound, render, roundProb, scoreConfidence, toAnswers, toRecord, validate, ValidationError, pyJsonDumps, withDateFacts } from "../src/api.ts";
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
  assert.equal(pyJsonDumps({ a: { type: "noul", noul: 1 }, "é": [0, 0.25] }), '{"a": {"type": "noul", "noul": 1.0}, "\\u00e9": [0.0, 0.25]}');
});

test("pyRound matches Python's round(x, 2) and round(x, 4)", () => {
  // [x, round(x, 2), round(x, 4)] from CPython. Exact ties (odd / 2^(digits+1)) go to even; 0.015 is below its tie.
  const cases: [number, number, number][] = [
    [0.015, 0.01, 0.015], [0.125, 0.12, 0.125], [0.375, 0.38, 0.375], [0.4449, 0.44, 0.4449],
    [0.03125, 0.03, 0.0312], [0.09375, 0.09, 0.0938], [0.15625, 0.16, 0.1562], [0.21875, 0.22, 0.2188],
    [0.28125, 0.28, 0.2812], [0.34375, 0.34, 0.3438], [0.71875, 0.72, 0.7188], [0.00015, 0.0, 0.0001],
    [1.00005, 1.0, 1.0001], [0.99995, 1.0, 1.0], [4.9999e-05, 0.0, 0.0], [0.12345, 0.12, 0.1235],
    [0.5358820043066892, 0.54, 0.5359], [0.36568891691258554, 0.37, 0.3657], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0],
  ];
  for (const [x, two, four] of cases) {
    assert.equal(pyRound(x, 2), two, `round(${x}, 2)`);
    assert.equal(pyRound(x, 4), four, `round(${x}, 4)`);
    assert.equal(roundProb(x), four);
  }
});

test("dateFacts matches kev.api.date_facts", () => {
  assert.equal(dateFacts("The report was received on September 27, 2026. The filing deadline for Priya's report was September 8, 2026."),
    "September 8, 2026 is 19 days before September 27, 2026.");
  assert.equal(dateFacts("June 26, 2026 and July 4, 2026"), "July 4, 2026 is 8 days after June 26, 2026.");
  assert.equal(dateFacts("Meet on 2026-01-01 then 2026-01-02"), "2026-01-02 is 1 day after 2026-01-01.");
  assert.equal(dateFacts("July 22, 2026 then August 3, 2026"), "August 3, 2026 is 12 days after July 22, 2026.");
  assert.equal(dateFacts("January 1, 2026 then January 3, 2026 then January 2, 2026"),
    "January 3, 2026 is 2 days after January 1, 2026. January 2, 2026 is 1 day after January 1, 2026. January 2, 2026 is 1 day before January 3, 2026.");
  assert.equal(dateFacts("2026-09-08 and September 8, 2026"), "September 8, 2026 is the same day as 2026-09-08.");
  assert.equal(dateFacts("March 1, 2026 and March 2, 2026"), "March 2, 2026 is 1 day after March 1, 2026.");
  assert.equal(dateFacts("Same: January 1, 2026 and January 1, 2026"), "");
  assert.equal(dateFacts("only one: March 3, 2026"), "");
  assert.equal(dateFacts("February 30, 2026 and March 1, 2026"), "");
  assert.equal(dateFacts("0099-01-01 and 0099-01-02"), "0099-01-02 is 1 day after 0099-01-01.");
  assert.equal(dateFacts("January 1, 0001 and January 2, 0001"), "January 2, 0001 is 1 day after January 1, 0001.");
  assert.equal(dateFacts("0000-01-01 and 0000-01-02"), "");
});

test("withDateFacts matches kev.api.with_date_facts", () => {
  const s = "June 26, 2026 and July 4, 2026";
  const facts = "July 4, 2026 is 8 days after June 26, 2026.";
  assert.equal(withDateFacts(s), `${s}\n\ndate_facts: ${facts}`);
  assert.deepEqual(withDateFacts({ policy: s, x: 1 }), { policy: s, x: 1, date_facts: facts });
  assert.deepEqual(withDateFacts([s, "other"]), [s, "other", { date_facts: facts }]);
  assert.equal(withDateFacts("only one: March 3, 2026"), "only one: March 3, 2026");
});

test("validation mirrors the pydantic models", () => {
  const bad = [{}, { state: "x", questions: {} }, { state: "x", questions: { a: { type: "choice", instructions: "i", criteria: {} } } },
    { state: "x", questions: { a: { type: "score", instructions: "i", criteria: [] } } }, { state: "x", questions: { a: { type: "vote", instructions: "i" } } }];
  for (const b of bad) assert.throws(() => validate(b), ValidationError);
  // instructions are optional and a score may have a single level (kev.api since the TypeSafe-contract update)
  const ok = { state: "x", questions: { a: { type: "noul" }, b: { type: "score", criteria: ["only"] }, c: { type: "choice", criteria: { x: null } } } };
  const { record, meta } = toRecord(validate(ok));
  assert.deepEqual(record.questions.map((q) => q.instr), ["", "", ""]);
  const answers = toAnswers([[0.25, 0.75], [1], [1]], meta);
  assert.deepEqual(answers.b, { type: "score", score: 0, legend: { 0: "only" }, probabilities: { 0: 1 }, confidence: 1 });
  assert.equal(scoreConfidence([1]), 1);
});

test("toAnswers confidence matches kev.api formulas (uniform-MAD Score)", () => {
  const { meta } = toRecord(validate({
    state: "s",
    questions: {
      n: { type: "noul" },
      c: { type: "choice", criteria: { a: null, b: null, c: null } },
      s: { type: "score", criteria: ["lo", "mid", "hi"] },
    },
  }));
  const ans = toAnswers([[0.3, 0.7], [0.8, 0.15, 0.05], [0.1, 0.3, 0.6]], meta);
  assert.deepEqual(ans.n, { type: "noul", noul: 0.7 });
  assert.equal(ans.c.type, "choice");
  assert.equal(ans.c.choice, "a");
  assert.deepEqual(ans.c.probabilities, { a: 0.8, b: 0.15, c: 0.05 });
  assert.equal(ans.c.confidence, roundProb((0.8 - 1 / 3) / (1 - 1 / 3)));
  assert.equal(ans.s.type, "score");
  assert.equal(ans.s.score, 1.5);
  assert.deepEqual(ans.s.probabilities, { 0: 0.1, 1: 0.3, 2: 0.6 });
  assert.deepEqual(ans.s.legend, { 0: "lo", 1: "mid", 2: "hi" });
  // 1 - (0.1*2 + 0.3*1) / (2/3) = 0.25 under the TypeSafe uniform-MAD normaliser
  assert.equal(ans.s.confidence, 0.25);
});

test("confidence edge cases match kev.api (134d170)", () => {
  assert.equal(choiceConfidence([1.0]), 1.0);
  assert.equal(scoreConfidence([1.0]), 1.0);
  assert.equal(choiceConfidence([0.5, 0.5]), 0.0);
  assert.ok(Math.abs(choiceConfidence([1.0, 0.0, 0.0]) - 1.0) < 1e-12);
  assert.equal(scoreConfidence([0.0, 1.0, 0.0]), 1.0);
  assert.equal(scoreConfidence([0.0, 0.0, 0.0, 1.0]), 1.0);
  for (let L = 2; L < 11; L++) assert.equal(scoreConfidence(Array(L).fill(1 / L)), 0.0);
  assert.equal(scoreConfidence([0.5, 0.0, 0.5]), 0.0);
  assert.equal(scoreConfidence([2.0, 6.0, 0.0]), scoreConfidence([0.25, 0.75, 0.0]));
  assert.equal(choiceConfidence([0.0, 0.0]), 0.0);
  assert.equal(scoreConfidence([0.0, 0.0, 0.0]), 0.0);
});

test("scoreConfidence matches TypeSafe docs examples", () => {
  // docs.typesafe.ai/primitives/score — display probs/confidence at two decimals
  const cases: [number[], number][] = [
    [[0.0, 0.57, 0.43], 0.35],
    [[0.0, 0.14, 0.86, 0.0, 0.0], 0.89],
    [[0.0, 0.0, 0.48, 0.52], 0.52],
    [[0.0, 0.74, 0.26], 0.61],
    [[0.0, 0.0, 0.0, 1.0], 1.0],
  ];
  for (const [p, want] of cases) {
    assert.ok(Math.abs(Number(scoreConfidence(p).toFixed(2)) - want) < 0.011, `${JSON.stringify(p)} -> ${scoreConfidence(p)}`);
  }
});
