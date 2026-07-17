// Authoritative caps validation via g2g's real solver, compiled to wasm. This
// module is the pure, node-testable half: applyValidation maps a validate result
// back onto the builder's edges. Loading the blob lives in solver-load.js (a
// browser-only virtual import), so this file stays importable by the node test.
// When the solver is unavailable the caller keeps import.js's family heuristic.

import { topoOrder } from "./export.js";

const SOLVER_RED = "#e5484d";
const CAPS_COLOR = "#7a8699";

// Human reason per NegotiationFailure kind that names a link (toolingjson
// failure_json). The unnamed kinds (degenerate / cyclic / ...) can't be pinned to
// one edge, so they are not colored here.
const LINK_FAILURE_REASON = {
  "empty-link": "no caps overlap",
  unfixable: "caps cannot be fixed",
};

// The media-type head of a gst caps string ("video/x-raw, format=..." ->
// "video/x-raw"), for a compact edge label; the full string rides in edge.data.
function capsSummary(caps) {
  if (!caps) return "";
  const comma = caps.indexOf(",");
  return (comma < 0 ? caps : caps.slice(0, comma)).trim();
}

// An edge carrying the solver's negotiated caps: clears any heuristic warning,
// labels it with the media type, keeps the full caps for a tooltip.
function capsEdge(edge, caps) {
  const rest = { ...edge };
  delete rest.style;
  delete rest.labelStyle;
  return {
    ...rest,
    ...(caps
      ? {
          label: capsSummary(caps),
          labelStyle: { fill: CAPS_COLOR, fontSize: 10 },
        }
      : { label: undefined }),
    data: { ...(edge.data || {}), capsWarn: false, solverCaps: caps || null },
  };
}

// An edge the solver could not negotiate: red, labeled with the reason.
function conflictEdge(edge, reason) {
  return {
    ...edge,
    style: { ...(edge.style || {}), stroke: SOLVER_RED },
    label: reason,
    labelStyle: { fill: SOLVER_RED, fontSize: 10 },
    data: { ...(edge.data || {}), capsWarn: true, solverCaps: null },
  };
}

// Pure: given a parsed validate result and the current nodes/edges, return new
// edges. The solver reports node INDICES in parse spec order; the builder's
// toLaunch emits one element per node in topoOrder, so a real node's index is its
// topoOrder position. A fan-out (multiple edges from one node) makes g2g's parser
// splice an implicit tee, appended AFTER the builder's own nodes, so any index
// >= node count is that synthetic tee (fan-in uses an explicit muxer node, so it
// keeps its index; g2g never auto-inserts a converter).
//
// - ok: label each builder edge with its negotiated caps, resolving one hop
//   through a synthetic tee (u -> tee -> d), clear warnings.
// - negotiate failure naming a link: color that builder edge red with the reason,
//   resolving a synthetic endpoint to the real builder edge it stands for.
// - parse / setup error, or an unnamed failure: leave edges unchanged (the caller
//   surfaces the message in a banner).
export function applyValidation(nodes, edges, result) {
  if (!result) return edges;
  const order = topoOrder(nodes, edges);
  const n = order.length;
  const pos = {};
  order.forEach((node, i) => (pos[node.id] = i));
  const realId = (idx) => (idx != null && idx < n ? order[idx].id : null); // idx >= n is a synthetic tee

  if (result.ok === true) {
    const res = result.edges || [];
    // A builder edge's caps: a direct negotiated edge, or one hop u -> tee -> d
    // when a fan-out spliced a tee between them.
    const capsFor = (u, d) => {
      const direct = res.find((e) => e.from === u && e.to === d);
      if (direct) return direct.caps;
      const viaTee = res.find(
        (e) => e.from === u && e.to >= n && res.some((e2) => e2.from === e.to && e2.to === d),
      );
      return viaTee ? viaTee.caps : null;
    };
    return edges.map((edge) => {
      const u = pos[edge.source];
      const d = pos[edge.target];
      return capsEdge(edge, u != null && d != null ? capsFor(u, d) : null);
    });
  }

  if (result.ok === false && result.stage === "negotiate" && result.failure) {
    const f = result.failure;
    if (f.upstream == null && f.downstream == null) return edges; // node-only failure
    const reason = LINK_FAILURE_REASON[f.kind] || f.kind;
    const up = realId(f.upstream);
    const down = realId(f.downstream);
    // Match the builder edge the (possibly synthetic) endpoints stand for: both
    // real -> that exact edge; a synthetic tee upstream -> the single edge into
    // the real downstream; a synthetic tee downstream -> edges out of the real
    // upstream.
    const isBad = (edge) => {
      if (up && down) return edge.source === up && edge.target === down;
      if (down) return edge.target === down;
      if (up) return edge.source === up;
      return false;
    };
    return edges.map((edge) => (isBad(edge) ? conflictEdge(edge, reason) : capsEdge(edge, null)));
  }

  return edges; // parse / setup error: banner handled by the caller
}
