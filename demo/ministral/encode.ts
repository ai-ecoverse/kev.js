// Token layout of one decision request (kev.js src/encode.ts with Ministral's tokens; port of kev/model.py encode +
// rows_of): the state runs once, each question as a continuation of the state's cache.
//
//   <s> <state> [image block] ...state...
//   <q> instructions <opt> option 1 </opt> ... <decide>       (one row per question)

import type { DecisionRecord } from "../../src/api.ts";

export interface SpecialTokens { state: number; q: number; opt: number; opt_end: number; decide: number }
export interface TokenizerLike { encode(text: string, options?: { add_special_tokens?: boolean }): { ids: number[] } }
export interface Branch { ids: number[]; decide: number; opts: number[] }
export interface Encoding { state: number[]; branches: Branch[] }

/** Caller text can never produce a control token: Mistral's tokenizer parses names such as `[IMG]` or `<SPECIAL_20>`
 * out of plain text, so a word joiner goes after the first character of every special token name (the Python rule in
 * train_ministral.install_tokenization, longest names first). */
export function makeUserTokens(tok: TokenizerLike, specials: string[]) {
  const names = [...specials].filter((s) => s.length > 1).sort((a, b) => b.length - a.length);
  return (text: string): number[] => {
    for (const s of names) if (text.includes(s)) text = text.split(s).join(s[0] + "⁠" + s.slice(1));
    return tok.encode(text, { add_special_tokens: false }).ids;
  };
}

export function encode(user: (t: string) => number[], rec: DecisionRecord, sp: SpecialTokens): Encoding {
  const state = [sp.state, ...user(rec.state)];
  const branches = rec.questions.map((q) => {
    const instr = [sp.q, ...user(q.instr)];
    const spans = q.options.map((opt) => [sp.opt, ...user(opt), sp.opt_end]);
    const ids = [...instr, ...spans.flat(), sp.decide];
    const opts: number[] = [];
    let cursor = instr.length;
    for (const s of spans) { cursor += s.length; opts.push(cursor - 1); }
    return { ids, decide: ids.length - 1, opts };
  });
  return { state, branches };
}
