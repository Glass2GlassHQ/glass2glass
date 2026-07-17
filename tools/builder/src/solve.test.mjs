// Unit tests for solve.js's pure applyValidation, with mocked validate results
// (the wasm solver is not loaded here). Covers the node-index -> builder-id
// mapping via topoOrder for a linear chain and a tee fan-out, on both the ok
// (caps onto edges) and empty-link (color the right edge) paths. No test runner
// is configured, so run it with plain node:
//   node src/solve.test.mjs

import assert from "node:assert";

import { applyValidation } from "./solve.js";

const node = (id, element, props = {}) => ({ id, data: { element, props } });
const edge = (source, target) => ({ id: `${source}->${target}`, source, target });

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
};

check("ok result labels each linear edge with negotiated caps", () => {
  const nodes = [node("videotestsrc0", "videotestsrc"), node("videoconvert0", "videoconvert"), node("fakesink0", "fakesink")];
  const edges = [edge("videotestsrc0", "videoconvert0"), edge("videoconvert0", "fakesink0")];
  // Indices 0/1/2 match topoOrder (src, convert, sink).
  const result = {
    ok: true,
    edges: [
      { from: 0, to: 1, caps: "video/x-raw, format=(string)I420, width=(int)320" },
      { from: 1, to: 2, caps: "video/x-raw, format=(string)RGBA" },
    ],
  };
  const out = applyValidation(nodes, edges, result);
  assert.strictEqual(out[0].label, "video/x-raw");
  assert.strictEqual(out[0].data.solverCaps, "video/x-raw, format=(string)I420, width=(int)320");
  assert.strictEqual(out[0].data.capsWarn, false);
  assert.strictEqual(out[1].data.solverCaps, "video/x-raw, format=(string)RGBA");
});

check("empty-link failure colors the offending linear edge only", () => {
  const nodes = [node("videotestsrc0", "videotestsrc"), node("capsfilter0", "capsfilter", { caps: "video/x-raw,format=NV12" }), node("fakesink0", "fakesink")];
  const edges = [edge("videotestsrc0", "capsfilter0"), edge("capsfilter0", "fakesink0")];
  // Conflict on src -> capsfilter (indices 0 -> 1).
  const result = { ok: false, stage: "negotiate", failure: { kind: "empty-link", upstream: 0, downstream: 1 } };
  const out = applyValidation(nodes, edges, result);
  assert.strictEqual(out[0].style.stroke, "#e5484d");
  assert.strictEqual(out[0].label, "no caps overlap");
  assert.strictEqual(out[0].data.capsWarn, true);
  // The other edge is cleared, not colored.
  assert.notStrictEqual(out[1].style?.stroke, "#e5484d");
  assert.strictEqual(out[1].data.capsWarn, false);
});

check("ok result maps caps onto a fan-out through the synthetic auto-tee", () => {
  // The builder has NO tee element: a node with two out-edges makes g2g's parser
  // splice an implicit tee, appended at index n (=3 here). So the negotiated
  // edges are src(0) -> tee(3), tee(3) -> s0(1), tee(3) -> s1(2). applyValidation
  // must resolve each builder edge (src->s0, src->s1) one hop through the tee.
  const nodes = [node("videotestsrc0", "videotestsrc"), node("fakesink0", "fakesink"), node("fakesink1", "fakesink")];
  const edges = [edge("videotestsrc0", "fakesink0"), edge("videotestsrc0", "fakesink1")];
  const result = {
    ok: true,
    edges: [
      { from: 0, to: 3, caps: "video/x-raw,format=RGBA" },
      { from: 3, to: 1, caps: "video/x-raw,format=RGBA" },
      { from: 3, to: 2, caps: "video/x-raw,format=RGBA" },
    ],
  };
  const out = applyValidation(nodes, edges, result);
  for (const e of out) {
    assert.strictEqual(e.data.solverCaps, "video/x-raw,format=RGBA");
    assert.strictEqual(e.label, "video/x-raw");
  }
});

check("fan-out failure through the synthetic tee colors the right branch", () => {
  // Conflict on the tee(3) -> s0(1) branch: upstream is the synthetic tee, so it
  // resolves to null and the real downstream (s0) pins the offending builder edge
  // (src -> s0), leaving the other branch (src -> s1) clean.
  const nodes = [node("videotestsrc0", "videotestsrc"), node("fakesink0", "fakesink"), node("fakesink1", "fakesink")];
  const edges = [edge("videotestsrc0", "fakesink0"), edge("videotestsrc0", "fakesink1")];
  const result = { ok: false, stage: "negotiate", failure: { kind: "empty-link", upstream: 3, downstream: 1 } };
  const out = applyValidation(nodes, edges, result);
  const bad = out.find((e) => e.target === "fakesink0");
  const other = out.find((e) => e.target === "fakesink1");
  assert.strictEqual(bad.style.stroke, "#e5484d");
  assert.strictEqual(bad.label, "no caps overlap");
  assert.notStrictEqual(other.style?.stroke, "#e5484d");
});

check("parse error leaves edges unchanged", () => {
  const nodes = [node("videotestsrc0", "videotestsrc"), node("fakesink0", "fakesink")];
  const edges = [edge("videotestsrc0", "fakesink0")];
  const out = applyValidation(nodes, edges, { ok: false, stage: "parse", error: "boom" });
  assert.strictEqual(out, edges);
});

console.log(`\n${passed} checks passed`);
