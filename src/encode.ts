// Token layout of one request (port of kev/model.py encode + rows_of, without option isolation, which the hybrid
// Qwen3.5 backbones do not support):
//
//   <state> ...state...
//   <q> instructions <opt> option 1 </opt> <opt> option 2 </opt> ... <decide>     (one branch per question)
//
// Every branch continues the state's positions, so running the state once and then each branch as a continuation
// of the state's cache is exactly Kev's hybrid serving path (kev.model._branch_rows_from_prefix).

import type { DecisionRecord } from "./api.ts";

export interface SpecialTokens { state: number; q: number; opt: number; opt_end: number; decide: number }

export interface TokenizerLike { encode(text: string, options?: { add_special_tokens?: boolean }): { ids: number[] } }

export interface Branch {
  ids: number[];
  pos: number[];
  /** offset of <decide> within the branch */
  decide: number;
  /** offsets of each option's </opt> within the branch */
  opts: number[];
}

export interface Encoding {
  state: number[];
  branches: Branch[];
  stateTruncated: boolean;
  /** total tokens across the packed sequence (state + all branches), as Kev reports usage.input_tokens */
  tokens: number;
}

const SPECIAL_RE = /<\|([A-Za-z0-9_]+)\|>/g;

/** Tokenize caller-supplied text so it can never produce delimiter/control tokens: `<|name|>` becomes `<¦name¦>`. */
export function userTokens(tok: TokenizerLike, text: string): number[] {
  return tok.encode(text.replace(SPECIAL_RE, "<¦$1¦>"), { add_special_tokens: false }).ids;
}

export interface EncodeOptions {
  /** state tokens kept, including <state>. Defaults match kev.serve (SERVE_MAX_STATE = 65,536). Training still
   * defaults to 384. The graph's rotary tables must cover this many positions (`manifest.max_positions`). */
  maxState?: number;
  /** max tokens of state + one branch (default SERVE_MAX_BRANCH = 73,728) */
  maxBranch?: number;
  /** throw instead of truncating an over-long state */
  strict?: boolean;
}

export function encode(tok: TokenizerLike, rec: DecisionRecord, sp: SpecialTokens, o: EncodeOptions = {}): Encoding {
  const maxState = o.maxState ?? 65536, maxBranch = o.maxBranch ?? 73728;
  const stateTokens = userTokens(tok, rec.state);
  if (o.strict && stateTokens.length + 1 > maxState) throw new RangeError(`state exceeds ${maxState} tokens: ${stateTokens.length + 1}`);
  const state = [sp.state, ...stateTokens.slice(0, maxState - 1)];
  const L = state.length;
  let tokens = L;
  const branches = rec.questions.map((q) => {
    const instr = [sp.q, ...userTokens(tok, q.instr)];
    const spans = q.options.map((opt) => [sp.opt, ...userTokens(tok, opt), sp.opt_end]);
    const ids = [...instr, ...spans.flat(), sp.decide];
    if (ids.length > maxBranch - L) throw new RangeError(`branch too long: ${ids.length}`);
    const opts: number[] = [];
    let cursor = instr.length;
    for (const s of spans) { cursor += s.length; opts.push(cursor - 1); }
    tokens += ids.length;
    return { ids, pos: ids.map((_, i) => L + i), decide: ids.length - 1, opts };
  });
  return { state, branches, stateTruncated: stateTokens.length + 1 > maxState, tokens };
}
