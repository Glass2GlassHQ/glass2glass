// Pure importers, the inverse of export.js: turn a gst-launch line or a
// declarative JSON document back into the builder's node/edge shape. Pure ESM,
// no React, so node can run the round-trip self-check (import.test.mjs).

import { topoOrder, roleClass, isSink, nodeData } from "./export.js";

// ---- caps-family heuristic --------------------------------------------------
// A cheap, honest stand-in for the Rust caps solver: classify each element's
// accepted (SINK) and produced (SRC) caps into a coarse FAMILY, and flag an edge
// only when both ends are known and disjoint. It never claims full validation;
// the authoritative solver is Rust-only and runs at graph build time.

// The leading variant token of a Rust Debug caps string names its family, e.g.
// "RawVideo { format: Rgba8, ... }" -> RawVideo. ByteStream is deliberately left
// out of the strict set: it is a raw byte container that legitimately feeds any
// parser (filesrc ! h264parse), so treating it as a family would false-positive.
export const CAPS_FAMILIES = ["RawVideo", "CompressedVideo", "Audio", "Text"];
const ALL_FAMILY_TOKENS = [...CAPS_FAMILIES, "ByteStream"];

// Families named in a caps blob. Matched against the allowlist by word boundary
// so a nested field (`width: Range { .. }`) is never mistaken for a variant.
function familiesIn(text) {
  const found = new Set();
  if (!text) return found;
  for (const f of ALL_FAMILY_TOKENS) {
    if (new RegExp(`\\b${f}\\b`).test(text)) found.add(f);
  }
  return found;
}

// { in, out }: the families an element accepts and produces. Pads carry both
// (`"SINK: [RawVideo { .. }]"`, `"SRC: [..]"`); a pad-less source falls back to
// its top-level `caps` string for the output side.
export function capsFamilies(doc) {
  const inFam = new Set();
  const outFam = new Set();
  for (const pad of doc?.pads || []) {
    const colon = pad.indexOf(":");
    if (colon < 0) continue;
    const which = pad.slice(0, colon).trim();
    const set = which === "SINK" ? inFam : outFam;
    familiesIn(pad.slice(colon + 1)).forEach((f) => set.add(f));
  }
  if (outFam.size === 0) familiesIn(doc?.caps).forEach((f) => outFam.add(f));
  return { in: inFam, out: outFam };
}

// "ok" | "incompatible" | "unknown". Compares the source's produced families to
// the target's accepted families over the strict set only (ByteStream is a
// wildcard). Unknown whenever either side has no strict family, so a valid graph
// is never flagged.
export function capsCompat(srcDoc, tgtDoc) {
  const out = [...capsFamilies(srcDoc).out].filter((f) => CAPS_FAMILIES.includes(f));
  const inn = [...capsFamilies(tgtDoc).in].filter((f) => CAPS_FAMILIES.includes(f));
  if (!out.length || !inn.length) return "unknown";
  return out.some((f) => inn.includes(f)) ? "ok" : "incompatible";
}

export const CAPS_WARN_LABEL = "caps families differ";
export const CAPS_WARN_TITLE =
  "caps families look incompatible (heuristic, not the full solver; run to check)";

// An edge with the warning styling applied when the two ends' caps families are
// known-disjoint. `edge` is the base React Flow edge; docs are looked up by id.
export function decorateEdge(edge, docById) {
  const srcDoc = docById[edge.source];
  const tgtDoc = docById[edge.target];
  if (srcDoc && tgtDoc && capsCompat(srcDoc, tgtDoc) === "incompatible") {
    return {
      ...edge,
      style: { ...(edge.style || {}), stroke: "#e5484d" },
      label: CAPS_WARN_LABEL,
      labelStyle: { fill: "#e5484d", fontSize: 10 },
      data: { ...(edge.data || {}), capsWarn: true },
    };
  }
  return edge;
}

// ---- shared node building ---------------------------------------------------

// A minimal doc for an element name the registry does not know, so an imported
// line that references an unavailable (feature-gated / hand-written) element
// still loads instead of being silently dropped. `unresolved` flags it visually.
function placeholderDoc(name) {
  return {
    name,
    role: "element",
    klass: "",
    long_name: name,
    description: "unresolved: not in the loaded registry",
    caps: "",
    pads: [],
    properties: [],
    unresolved: true,
  };
}

// Build a node from a resolved doc (known element) or a placeholder. A
// placeholder's role and handles follow link degree, since it has no declared
// role: no inbound edge = source, no outbound = sink, else a transform.
function makeNode(id, element, doc, props, indeg, outdeg) {
  if (!doc.unresolved) {
    return { id, type: "g2g", position: { x: 0, y: 0 }, data: nodeData(id, element, doc, props) };
  }
  const role = indeg === 0 ? "source" : "element";
  return {
    id,
    type: "g2g",
    position: { x: 0, y: 0 },
    data: {
      name: id,
      element,
      role,
      roleClass: roleClass(role),
      doc,
      props,
      hasIn: indeg > 0,
      hasOut: outdeg > 0 || indeg === 0,
      unresolved: true,
    },
  };
}

// Left-to-right layered layout: column = longest path from a source, nodes in a
// column stacked vertically. Mutates node.position in place.
function layout(nodes, edges) {
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const order = topoOrder(nodes, edges);
  const depth = Object.fromEntries(nodes.map((n) => [n.id, 0]));
  order.forEach((n) => {
    edges
      .filter((e) => e.source === n.id)
      .forEach((e) => {
        if (byId[e.target]) depth[e.target] = Math.max(depth[e.target], depth[n.id] + 1);
      });
  });
  const rowInCol = {};
  order.forEach((n) => {
    const col = depth[n.id];
    const row = rowInCol[col] || 0;
    rowInCol[col] = row + 1;
    n.position = { x: 60 + col * 220, y: 40 + row * 110 };
  });
}

// Assemble nodes+edges from resolved node records and directed links, applying
// degree-based placeholder roles, layout, and the caps heuristic to edges.
function assemble(records, links, elements) {
  const indeg = {};
  const outdeg = {};
  links.forEach(([s, d]) => {
    outdeg[s] = (outdeg[s] || 0) + 1;
    indeg[d] = (indeg[d] || 0) + 1;
  });
  const nodes = records.map((r) => {
    const doc = elements[r.element] || placeholderDoc(r.element);
    return makeNode(r.id, r.element, doc, r.props, indeg[r.id] || 0, outdeg[r.id] || 0);
  });
  const docById = Object.fromEntries(nodes.map((n) => [n.id, n.data.doc]));
  const edges = links.map(([s, d], i) =>
    decorateEdge({ id: `e${i}-${s}-${d}`, source: s, target: d, animated: true }, docById),
  );
  layout(nodes, edges);
  return { nodes, edges };
}

// Rebuild the App's per-element id counters so a later addNode does not collide
// with an imported id (ids shaped `element<N>`).
export function seedCounters(nodes) {
  const counters = {};
  nodes.forEach((n) => {
    const m = /^(.*?)(\d+)$/.exec(n.id);
    if (m && m[1] === n.data.element) {
      counters[m[1]] = Math.max(counters[m[1]] || 0, Number(m[2]) + 1);
    }
  });
  return counters;
}

// ---- gst-launch parsing (ported from g2g-core/src/runtime/launch.rs) --------

// A media-type caps token: a `/` before any `=` (or a `/` with no `=`). A
// property value's `/` (a path or fraction) comes after its `=`.
function isCapsToken(tok) {
  const slash = tok.indexOf("/");
  const eq = tok.indexOf("=");
  if (slash < 0) return false;
  if (eq < 0) return true;
  return slash < eq;
}

// A pad reference (`t.`, `d.video_0`): a name, a `.`, and no `=` / `/`.
function splitPadRef(tok) {
  if (tok.includes("=") || tok.includes("/") || !tok.includes(".")) return null;
  const dot = tok.indexOf(".");
  const name = tok.slice(0, dot);
  return name ? { name, pad: tok.slice(dot + 1) } : null;
}

function stripQuotes(v) {
  if (v.length >= 2 && (v[0] === '"' || v[0] === "'") && v[v.length - 1] === v[0]) {
    return v.slice(1, -1);
  }
  return v;
}

// Split a pipeline string into tokens, honoring quoted values, `#` comments, and
// the standalone `!` link separator.
function tokenize(s) {
  const tokens = [];
  let cur = "";
  let quote = null;
  let inComment = false;
  for (const c of s) {
    if (inComment) {
      if (c === "\n") inComment = false;
      continue;
    }
    if (quote === null && (c === '"' || c === "'")) {
      quote = c;
      cur += c;
    } else if (c === quote) {
      quote = null;
      cur += c;
    } else if (c === "#" && quote === null && cur === "") {
      inComment = true;
    } else if (c === "!" && quote === null) {
      if (cur) {
        tokens.push(cur);
        cur = "";
      }
      tokens.push("!");
    } else if (/\s/.test(c) && quote === null) {
      if (cur) {
        tokens.push(cur);
        cur = "";
      }
    } else {
      cur += c;
    }
  }
  if (cur) tokens.push(cur);
  return tokens;
}

// Consume an element's `key=value` props until the next `!`, caps token, pad
// ref, or a bare (no `=`) token. `name=` sets the instance handle, not a
// property. A bare token starts the next element: the builder's branched export
// separates its `name=` decls by whitespace, not `!`, so unlike the Rust launch
// parser (which errors on a bare token) this treats it as an element boundary.
function consumeElement(name, toks, pos) {
  const spec = { kind: "element", name, props: {}, instance: null };
  let i = pos;
  while (i < toks.length) {
    const tok = toks[i];
    if (tok === "!" || isCapsToken(tok) || splitPadRef(tok)) break;
    const eq = tok.indexOf("=");
    if (eq < 0) break;
    const key = tok.slice(0, eq);
    const value = stripQuotes(tok.slice(eq + 1));
    if (key === "name") spec.instance = value;
    else spec.props[key] = value;
    i += 1;
  }
  return { spec, next: i };
}

// Split tokens into chains: runs of items linked by `!`, branches as separate
// chains joined through `name=` / `t.` references. Mirrors parse_chains.
function parseChains(toks) {
  const chains = [];
  let cur = [];
  let i = 0;
  let st = "start"; // start | afterBang | afterNode
  const pushChain = () => {
    if (cur.length) chains.push(cur);
    cur = [];
  };
  for (;;) {
    if (st === "start" || st === "afterBang") {
      const afterBang = st === "afterBang";
      if (i >= toks.length) {
        if (afterBang) throw new Error("empty node after trailing '!'");
        break;
      }
      const tok = toks[i];
      if (tok === "!") throw new Error("empty node between '!' separators");
      if (isCapsToken(tok)) {
        cur.push({ kind: "element", name: "capsfilter", props: { caps: tok }, instance: null });
        i += 1;
        st = "afterNode";
      } else {
        const ref = splitPadRef(tok);
        if (ref) {
          cur.push({ kind: "ref", name: ref.name, pad: ref.pad });
          i += 1;
          if (afterBang) {
            pushChain();
            st = "start";
          } else {
            st = "afterNode";
          }
        } else {
          const { spec, next } = consumeElement(tok, toks, i + 1);
          cur.push(spec);
          i = next;
          st = "afterNode";
        }
      }
    } else {
      // afterNode
      if (i >= toks.length) break;
      if (toks[i] === "!") {
        i += 1;
        st = "afterBang";
      } else {
        pushChain();
        st = "start";
      }
    }
  }
  pushChain();
  return chains;
}

export function fromLaunch(text, elements = {}) {
  const chains = parseChains(tokenize(text));

  // Assign a node id to every element item; refs resolve to instance ids.
  const records = [];
  const byInstance = {};
  const counters = {};
  const endpoints = []; // per chain: array of { type:'node'|'ref', ... }
  for (const chain of chains) {
    const eps = [];
    for (const item of chain) {
      if (item.kind === "ref") {
        eps.push({ type: "ref", name: item.name });
        continue;
      }
      let id = item.instance;
      if (!id) {
        const n = counters[item.name] || 0;
        counters[item.name] = n + 1;
        id = item.name + n;
      }
      const rec = { id, element: item.name, props: item.props };
      records.push(rec);
      if (item.instance) byInstance[item.instance] = id;
      eps.push({ type: "node", id });
    }
    endpoints.push(eps);
  }

  // Resolve refs and collect directed links between consecutive endpoints.
  const links = [];
  for (const eps of endpoints) {
    const ids = eps.map((ep) => (ep.type === "node" ? ep.id : byInstance[ep.name]));
    for (let w = 0; w < ids.length - 1; w += 1) {
      if (ids[w] && ids[w + 1]) links.push([ids[w], ids[w + 1]]);
    }
  }

  return assemble(records, links, elements);
}

// ---- declarative JSON parsing (GraphSpec, declarative.rs) -------------------

export function fromJSON(text, elements = {}) {
  const spec = typeof text === "string" ? JSON.parse(text) : text;
  if (spec.pipeline) return fromLaunch(spec.pipeline, elements);

  const records = (spec.nodes || []).map((n) => {
    // A node with `caps` and no `element` is a capsfilter; fold caps into props.
    const element = n.element || (n.caps != null ? "capsfilter" : n.element);
    const props = { ...(n.props || {}) };
    if (n.caps != null) props.caps = n.caps;
    return { id: n.id, element, props };
  });
  const known = new Set(records.map((r) => r.id));
  const links = (spec.edges || [])
    .filter((e) => known.has(e.from) && known.has(e.to))
    .map((e) => [e.from, e.to]);
  return assemble(records, links, elements);
}
