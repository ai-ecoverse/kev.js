// Serving admission (kev.model.admit, kev.serve since Kev 1.0): a state over maxState tokens is refused, not cut,
// unless the Kev was loaded with truncateStates, which then marks every response. Runs on a stub session.
import { test } from "node:test";
import assert from "node:assert/strict";
import { ContextOverflow, Kev, PointerHead, type KevManifest, type KevOptions, type OrtModule } from "../src/index.ts";
import { tokenizer } from "./fixtures.ts";
import { headFile } from "./synthetic.ts";

const manifest = {
  name: "kev-test", run: "jaredpalmer/kev-test@abc", hidden_size: 3,
  special: { state: 248060, q: 248061, opt: 248049, opt_end: 248050, decide: 248062 },
  variants: { v: { model: "", data: [], bytes: 0, io_dtype: "float32", inputs: [], outputs: [] } },
} as unknown as KevManifest;

async function stubKev(options: KevOptions): Promise<{ kev: Kev; runs: number[] }> {
  const runs: number[] = [];   // tokens per session run
  const session = {
    run: async (feeds: Record<string, { dims: number[] }>) => {
      const S = feeds.input_ids.dims[1];
      runs.push(S);
      return { hidden_states: { data: new Float32Array(S * 3) } };
    },
    release: async () => {},
  };
  const ortStub = { Tensor: class { constructor(readonly type: string, readonly data: unknown, readonly dims: number[]) {} } } as unknown as OrtModule;
  const kev = new Kev({ ort: ortStub, session: session as never, head: PointerHead.fromSafetensors(headFile().buffer as ArrayBuffer),
    tokenizer: await tokenizer(), manifest, variant: "v", options });
  return { kev, runs };
}

const request = {
  state: "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen",
  questions: { a: { type: "noul", instructions: "Is it long?" }, b: { type: "noul", instructions: "Is it short?" } },
};

test("a state over maxState is refused before inference, with its length and the limit", async () => {
  const { kev, runs } = await stubKev({ maxState: 8 });
  const n = (await tokenizer()).encode(request.state, { add_special_tokens: false }).ids.length + 1;
  assert.ok(n > 8);
  await assert.rejects(kev.systemOne(request), (e: unknown) => e instanceof ContextOverflow && e.stateTokens === n && e.maxState === 8
    && e.message.startsWith(`state is ${n} tokens, over the 8-token limit (the <state> token included): shorten the document`));
  await assert.rejects(kev.systemOneSeparate(request), ContextOverflow);
  assert.deepEqual(runs, [], "no session ran");
  const fits = await kev.systemOne({ ...request, state: "one two" });
  assert.equal(fits.truncated, undefined, "a Kev that never truncates adds no marks");
  assert.deepEqual(Object.keys(fits.usage), ["input_tokens", "output_tokens"]);
});

test("truncateStates reads the first maxState tokens and marks every response", async () => {
  const { kev, runs } = await stubKev({ maxState: 8, truncateStates: true });
  const n = (await tokenizer()).encode(request.state, { add_special_tokens: false }).ids.length + 1;
  const r = await kev.systemOne(request);
  assert.equal(runs[0], 8, "the state pass reads 8 tokens");
  assert.equal(r.truncated, true);
  assert.equal(r.usage.state_tokens, n);
  assert.equal(r.usage.state_tokens_used, 8);
  const s = await kev.systemOneSeparate(request);
  assert.equal(s.truncated, true);
  assert.equal(s.usage.state_tokens, n);
  assert.equal(s.usage.state_tokens_used, 8);
  const short = await kev.systemOne({ ...request, state: "one two" });
  assert.equal(short.truncated, false);
  assert.equal(short.usage.state_tokens, short.usage.state_tokens_used);
});
