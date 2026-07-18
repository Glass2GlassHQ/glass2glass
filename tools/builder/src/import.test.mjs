// Round-trip self-check for the importers: build a builder-shaped graph, export
// it with export.js, re-import with import.js, re-export, and assert the two
// exports match. Plus a couple of parse-only checks (caps shorthand, caps
// heuristic). No test runner is configured, so run it with plain node:
//   node src/import.test.mjs

import assert from "node:assert";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { toLaunch, toJSON } from "./export.js";
import { fromLaunch, fromJSON, capsCompat } from "./import.js";

const here = dirname(fileURLToPath(import.meta.url));
const registry = JSON.parse(readFileSync(join(here, "../public/registry.json"), "utf8"));
const elements = Object.fromEntries(registry.elements.map((e) => [e.name, e]));

// A builder-shaped node: only id + data.{element,props} matter to the exporters.
const node = (id, element, props = {}) => ({ id, data: { element, props } });
const edge = (source, target) => ({ source, target });

let passed = 0;
const check = (name, fn) => {
  fn();
  passed += 1;
  console.log(`ok - ${name}`);
};

// A gst round-trip: export -> import -> re-export must be identical.
function gstRoundTrip(nodes, edges) {
  const str = toLaunch(nodes, edges);
  const g = fromLaunch(str, elements);
  const str2 = toLaunch(g.nodes, g.edges);
  assert.strictEqual(str2, str, `\n--- expected ---\n${str}\n--- got ---\n${str2}\n`);
  return g;
}

// A JSON round-trip.
function jsonRoundTrip(nodes, edges) {
  const str = toJSON(nodes, edges);
  const g = fromJSON(str, elements);
  const str2 = toJSON(g.nodes, g.edges);
  assert.strictEqual(str2, str, `\n--- expected ---\n${str}\n--- got ---\n${str2}\n`);
  return g;
}

check("gst linear chain round-trips", () => {
  const nodes = [
    node("videotestsrc0", "videotestsrc", { "num-buffers": "3" }),
    node("videoconvert0", "videoconvert"),
    node("fakesink0", "fakesink"),
  ];
  const edges = [edge("videotestsrc0", "videoconvert0"), edge("videoconvert0", "fakesink0")];
  const g = gstRoundTrip(nodes, edges);
  assert.strictEqual(g.nodes.length, 3);
  assert.strictEqual(g.edges.length, 2);
});

check("gst tee fan-out (branched form) round-trips", () => {
  const nodes = [
    node("videotestsrc0", "videotestsrc"),
    node("fakesink0", "fakesink"),
    node("fakesink1", "fakesink"),
  ];
  const edges = [edge("videotestsrc0", "fakesink0"), edge("videotestsrc0", "fakesink1")];
  const g = gstRoundTrip(nodes, edges);
  assert.strictEqual(g.nodes.length, 3);
  assert.strictEqual(g.edges.length, 2, "both fan-out edges resolved");
});

check("gst capsfilter (caps= form) round-trips", () => {
  const nodes = [
    node("videotestsrc0", "videotestsrc"),
    node("capsfilter0", "capsfilter", { caps: "video/x-raw,format=NV12" }),
    node("fakesink0", "fakesink"),
  ];
  const edges = [edge("videotestsrc0", "capsfilter0"), edge("capsfilter0", "fakesink0")];
  const g = gstRoundTrip(nodes, edges);
  const cf = g.nodes.find((n) => n.data.element === "capsfilter");
  assert.ok(cf, "capsfilter node present");
  assert.strictEqual(cf.data.props.caps, "video/x-raw,format=NV12");
});

check("gst bare caps-shorthand token becomes a capsfilter", () => {
  const g = fromLaunch("videotestsrc ! video/x-raw,format=NV12 ! fakesink", elements);
  assert.strictEqual(g.nodes.length, 3);
  const cf = g.nodes.find((n) => n.data.element === "capsfilter");
  assert.ok(cf, "shorthand mapped to capsfilter");
  assert.strictEqual(cf.data.props.caps, "video/x-raw,format=NV12");
  assert.strictEqual(g.edges.length, 2);
});

check("json linear round-trips", () => {
  const nodes = [
    node("videotestsrc0", "videotestsrc", { "num-buffers": "3" }),
    node("videoconvert0", "videoconvert"),
    node("fakesink0", "fakesink"),
  ];
  const edges = [edge("videotestsrc0", "videoconvert0"), edge("videoconvert0", "fakesink0")];
  jsonRoundTrip(nodes, edges);
});

check("json fan-out round-trips", () => {
  const nodes = [
    node("videotestsrc0", "videotestsrc"),
    node("fakesink0", "fakesink"),
    node("fakesink1", "fakesink"),
  ];
  const edges = [edge("videotestsrc0", "fakesink0"), edge("videotestsrc0", "fakesink1")];
  jsonRoundTrip(nodes, edges);
});

check("json caps shorthand (no element) maps to capsfilter", () => {
  const doc = { nodes: [{ id: "cf", caps: "video/x-raw,format=NV12" }], edges: [] };
  const g = fromJSON(JSON.stringify(doc), elements);
  const cf = g.nodes[0];
  assert.strictEqual(cf.data.element, "capsfilter");
  assert.strictEqual(cf.data.props.caps, "video/x-raw,format=NV12");
});

check("json pipeline escape hatch defers to gst importer", () => {
  const g = fromJSON('{ "pipeline": "videotestsrc ! fakesink" }', elements);
  assert.strictEqual(g.nodes.length, 2);
  assert.strictEqual(g.edges.length, 1);
});

check("unknown element becomes an unresolved placeholder", () => {
  const g = fromLaunch("videotestsrc ! nosuchelem ! fakesink", elements);
  const ph = g.nodes.find((n) => n.data.element === "nosuchelem");
  assert.ok(ph, "placeholder created, not dropped");
  assert.strictEqual(ph.data.unresolved, true);
  assert.strictEqual(g.edges.length, 2);
});

check("caps heuristic flags a disjoint pairing", () => {
  // videotestsrc produces RawVideo; audioconvert accepts Audio -> disjoint.
  assert.strictEqual(capsCompat(elements.videotestsrc, elements.audioconvert), "incompatible");
});

check("caps heuristic passes a compatible pairing", () => {
  assert.strictEqual(capsCompat(elements.videotestsrc, elements.videoconvert), "ok");
});

check("caps heuristic stays unknown when a family is undeterminable", () => {
  // identity exposes no caps; must not false-positive.
  assert.strictEqual(capsCompat(elements.videotestsrc, elements.identity), "unknown");
  // ByteStream is a wildcard: filesrc -> h264parse must not be flagged.
  assert.strictEqual(capsCompat(elements.filesrc, elements.h264parse), "unknown");
});

check("imported disjoint edge is decorated as a caps warning", () => {
  const g = fromLaunch("videotestsrc ! audioconvert", elements);
  assert.strictEqual(g.edges[0].data?.capsWarn, true);
  assert.strictEqual(g.edges[0].style.stroke, "#e5484d");
});

console.log(`\n${passed} checks passed`);
