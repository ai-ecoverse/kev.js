import { test } from "node:test";
import assert from "node:assert/strict";
import { dateFacts, pyFloat, r2, render, toAnswers, toRecord, validate, ValidationError, pyJsonDumps, withDateFacts } from "../src/api.ts";
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
    { state: "x", questions: { a: { type: "score", instructions: "i", criteria: ["one"] } } }, { state: "x", questions: { a: { type: "vote", instructions: "i" } } }];
  for (const b of bad) assert.throws(() => validate(b), ValidationError);
});
