// TypeSafe System One request/response shapes mapped onto Kev's pointer primitive (port of kev/api.py).
//
// Noul   -> 2 options [no, yes];                answer = p(yes)
// Choice -> options 'name' or 'name: desc';     answer = argmax, probabilities by name, confidence
// Score  -> options = ordered level descriptions; answer = expected level, legend, probabilities by index
//
// Kev renders JSON content with Python's str(), so the text the model sees depends on Python formatting rules
// (True/False, float repr). Those are reproduced here so browser and server feed the model identical text.

export type JSONContent = string | number | boolean | null | JSONContent[] | { [key: string]: JSONContent };

export interface NoulQuestion { type: "noul"; instructions?: JSONContent; criteria?: { true?: JSONContent; false?: JSONContent } | null }
export interface ChoiceQuestion { type: "choice"; instructions?: JSONContent; criteria: Record<string, JSONContent> }
export interface ScoreQuestion { type: "score"; instructions?: JSONContent; criteria: JSONContent[] }
export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export interface SystemOneRequest { state: JSONContent; model?: string; questions: Record<string, Question> }

export interface NoulAnswer { type: "noul"; noul: number }
export interface ChoiceAnswer { type: "choice"; choice: string; confidence: number; probabilities: Record<string, number> }
export interface ScoreAnswer { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number }
export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface SystemOneResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
  latency_ms: number;
}

/** Internal record consumed by encode(): rendered state and, per question, instruction + option texts. */
export interface DecisionRecord { state: string; questions: { instr: string; options: string[] }[] }

/** Per-question metadata to map probabilities back: `keys` are the names probabilities are reported under, in option
 * order (criteria names for choice, ["false", "true"] for noul, level indices for score). */
export type QuestionMeta =
  | { id: string; type: "noul"; keys: string[] }
  | { id: string; type: "choice"; keys: string[] }
  | { id: string; type: "score"; keys: string[]; legend: Record<string, string> };

export const MAX_OPTIONS = 255;

export class ValidationError extends Error {
  status = 422;
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

export function validate(req: unknown): SystemOneRequest {
  if (!isObject(req)) throw new ValidationError("request must be an object");
  if (!("state" in req)) throw new ValidationError("state is required");
  if (!isObject(req.questions) || Object.keys(req.questions).length < 1) throw new ValidationError("questions must have at least 1 entry");
  for (const [id, q] of Object.entries(req.questions)) {
    if (!isObject(q)) throw new ValidationError(`questions.${id} must be an object`);
    if (q.type === "noul") {
      if (q.criteria != null && !isObject(q.criteria)) throw new ValidationError(`questions.${id}.criteria must be an object`);
    } else if (q.type === "choice") {
      if (!isObject(q.criteria)) throw new ValidationError(`questions.${id}.criteria must be an object`);
      const n = Object.keys(q.criteria).length;
      if (n < 1 || n > MAX_OPTIONS) throw new ValidationError(`criteria must have 1..${MAX_OPTIONS} options`);
    } else if (q.type === "score") {
      if (!Array.isArray(q.criteria) || q.criteria.length < 1 || q.criteria.length > MAX_OPTIONS)
        throw new ValidationError(`questions.${id}.criteria must be a list of 1..${MAX_OPTIONS} levels`);
    } else {
      throw new ValidationError(`questions.${id}.type must be noul, choice or score`);
    }
  }
  return req as unknown as SystemOneRequest;
}

/** Python's repr(float): shortest round-trip digits, exponent form below 1e-4 and from 1e16. */
export function pyFloat(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const sign = x < 0 ? "-" : "";
  const [mant, expS] = Math.abs(x).toExponential().split("e");
  const exp = parseInt(expS, 10);
  const digits = mant.replace(".", "");
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    return `${sign}${m}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  }
  if (exp >= 0) return `${sign}${digits.slice(0, exp + 1).padEnd(exp + 1, "0")}.${digits.slice(exp + 1) || "0"}`;
  return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
}

/** Python's str() for JSON scalars. JSON cannot tell 1 from 1.0 once parsed, so integral numbers print as ints. */
function pyStr(v: string | number | boolean): string {
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "number") return Number.isInteger(v) && Math.abs(v) < 1e21 ? String(v) : pyFloat(v);
  return v;
}

/** Flatten str | object | array into the text the model sees. Field names are kept as labels. */
export function render(v: JSONContent | undefined, indent = 0): string {
  const pad = "  ".repeat(indent);
  if (v === null || v === undefined) return "";
  if (typeof v !== "object") return pyStr(v);
  if (Array.isArray(v)) return v.map((x) => `${pad}- ${render(x, indent + 1).trimStart()}`).join("\n");
  return Object.entries(v)
    .map(([k, x]) => (typeof x === "object" && x !== null ? `${pad}${k}:\n${render(x, indent + 1)}` : `${pad}${k}: ${render(x)}`))
    .join("\n");
}

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"] as const;
const DATE_RE = new RegExp(String.raw`\b(?:${MONTHS.join("|")}) \d{1,2}, \d{4}\b|\b\d{4}-\d{2}-\d{2}\b`, "g");

/** UTC day number for an absolute date, or null if the text is not a real calendar day (e.g. February 30). */
function utcDay(raw: string): number | null {
  let y: number, month: number, d: number;
  if (raw.includes(",")) {
    const m = raw.match(/^([A-Za-z]+) (\d{1,2}), (\d{4})$/);
    if (!m) return null;
    month = (MONTHS as readonly string[]).indexOf(m[1]);
    if (month < 0) return null;
    d = Number(m[2]); y = Number(m[3]);
  } else {
    const m = raw.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!m) return null;
    y = Number(m[1]); month = Number(m[2]) - 1; d = Number(m[3]);
  }
  if (y < 1) return null;   // datetime.datetime rejects year 0; Date.UTC remaps 0–99 to 1900–1999
  const dt = new Date(Date.UTC(y, month, d));
  dt.setUTCFullYear(y);     // undo the 0–99 → 1900+ remap so 0099-01-01 is year 99, matching strptime
  if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== month || dt.getUTCDate() !== d) return null;
  return dt.getTime() / 86_400_000;
}

/** Port of kev.api.date_facts: one sentence per pair of absolute dates, in order of first appearance. */
export function dateFacts(text: string): string {
  const found: { raw: string; day: number }[] = [];
  for (const m of text.matchAll(DATE_RE)) {
    const raw = m[0];
    if (found.some((x) => x.raw === raw)) continue;
    const day = utcDay(raw);
    if (day != null) found.push({ raw, day });
  }
  const facts: string[] = [];
  for (let i = 0; i < found.length; i++) {
    for (let j = i + 1; j < found.length; j++) {
      const n = found[j].day - found[i].day;
      facts.push(n === 0
        ? `${found[j].raw} is the same day as ${found[i].raw}.`
        : `${found[j].raw} is ${Math.abs(n)} day${Math.abs(n) !== 1 ? "s" : ""} ${n > 0 ? "after" : "before"} ${found[i].raw}.`);
    }
  }
  return facts.join(" ");
}

/** Port of kev.api.with_date_facts / KEV_DATE_FACTS=1: append day counts when two or more absolute dates appear. */
export function withDateFacts(state: JSONContent): JSONContent {
  const facts = dateFacts(render(state));
  if (!facts) return state;
  if (isObject(state)) return { ...state, date_facts: facts };
  if (Array.isArray(state)) return [...state, { date_facts: facts }];
  return `${state}\n\ndate_facts: ${facts}`;
}

export function optionText(name: string, desc: JSONContent | undefined): string {
  return desc === null || desc === undefined || desc === "" ? name : `${name}: ${render(desc)}`;
}

/** Port of kev.api.question_keys: the keys a question's probabilities are reported under, in option order. */
export function questionKeys(q: Question): string[] {
  if (q.type === "choice") return Object.keys(q.criteria);
  if (q.type === "noul") return ["false", "true"];
  return q.criteria.map((_, i) => String(i));
}

export function toRecord(req: SystemOneRequest): { record: DecisionRecord; meta: QuestionMeta[] } {
  const questions: DecisionRecord["questions"] = [];
  const meta: QuestionMeta[] = [];
  for (const [id, q] of Object.entries(req.questions)) {
    const keys = questionKeys(q);
    let options: string[];
    if (q.type === "noul") {
      const c = q.criteria ?? {};
      options = [optionText("no", c.false), optionText("yes", c.true)];
      meta.push({ id, type: "noul", keys });
    } else if (q.type === "choice") {
      options = Object.entries(q.criteria).map(([k, v]) => optionText(k, v));
      meta.push({ id, type: "choice", keys });
    } else {
      options = q.criteria.map((x) => render(x));
      meta.push({ id, type: "score", keys, legend: Object.fromEntries(keys.map((k, i) => [k, options[i]])) });
    }
    questions.push({ instr: render(q.instructions), options });
  }
  return { record: { state: render(req.state), questions }, meta };
}

const argmax = (p: number[]) => p.reduce((best, v, i) => (v > p[best] ? i : best), 0);

export function choiceConfidence(p: number[]): number {
  const K = p.length;
  return K === 1 ? 1 : (Math.max(...p) - 1 / K) / (1 - 1 / K);
}

/** Approximation of TypeSafe's 'distance from the modal level' statistic: 1 - E|level - mode| / (L - 1). */
export function scoreConfidence(p: number[]): number {
  if (p.length === 1) return 1;
  const mode = argmax(p);
  return 1 - p.reduce((s, pi, i) => s + pi * Math.abs(i - mode), 0) / (p.length - 1);
}

/** Python round(x, digits), which rounds the exact binary value: toFixed does the same, except that it breaks exact
 * ties upward where Python breaks them to even. An exact tie is odd / 2^(digits+1) (0.125 at 2 digits, 0.03125 at 4);
 * 0.015 is not one, since its double lies below it. */
export function pyRound(x: number, digits: number): number {
  const half = x * 2 ** (digits + 1);   // exact: scaling by a power of two
  if (Number.isInteger(half) && half % 2 !== 0) {
    const f = Math.floor(x * 10 ** digits);   // exact too: x * 10^digits is odd * 5^digits / 2
    return (f % 2 === 0 ? f : f + 1) / 10 ** digits;
  }
  return Number(x.toFixed(digits));
}

/** kev.api.round_prob: 4 decimals keep a rounded 255-option distribution within TypeSafe's |sum - 1| < 0.02. */
export const roundProb = (x: number): number => pyRound(x, 4);

/** @deprecated kev.serve rounds to 4 decimals now (roundProb); this is Python's round(x, 2). */
export const r2 = (x: number): number => pyRound(x, 2);

export function toAnswers(probs: number[][], meta: QuestionMeta[]): Record<string, Answer> {
  const out: Record<string, Answer> = {};
  probs.forEach((p, i) => {
    const m = meta[i];
    if (m.type === "noul") out[m.id] = { type: "noul", noul: roundProb(p[1]) };
    else if (m.type === "choice")
      out[m.id] = {
        type: "choice", choice: m.keys[argmax(p)], confidence: roundProb(choiceConfidence(p)),
        probabilities: Object.fromEntries(m.keys.map((k, j) => [k, roundProb(p[j])])),
      };
    else
      out[m.id] = {
        type: "score", score: roundProb(p.reduce((s, pi, j) => s + j * pi, 0)), legend: m.legend,
        probabilities: Object.fromEntries(p.map((v, j) => [String(j), roundProb(v)])), confidence: roundProb(scoreConfidence(p)),
      };
  });
  return out;
}

/** Python json.dumps (default separators, ensure_ascii). Every number in an answer is a Python float. */
export function pyJsonDumps(v: unknown): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") return pyFloat(v);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return JSON.stringify(v).replace(/[\u0080-￿]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, "0")}`);
  if (Array.isArray(v)) return `[${v.map(pyJsonDumps).join(", ")}]`;
  return `{${Object.entries(v as object).map(([k, x]) => `${pyJsonDumps(k)}: ${pyJsonDumps(x)}`).join(", ")}}`;
}
