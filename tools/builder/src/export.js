// Pure export helpers over React Flow's node/edge shape (node.id,
// node.data.element, node.data.props; edge.source, edge.target). Every output
// loads back into g2g: the gst-launch line via the text parser, the JSON and the
// YAML via the declarative graph loader (one schema, two spellings).

// Shared node-shape helpers, used by both the palette (App.jsx) and the
// importers (import.js), so the two build identical node data.
export const isSink = (doc) => /Sink/i.test(doc.klass || "");
export const roleClass = (role) =>
  role === "source" ? "source" : role.startsWith("muxer") ? "muxer" : "element";
export function nodeData(id, element, doc, props = {}) {
  return {
    name: id,
    element,
    role: doc.role,
    roleClass: roleClass(doc.role),
    doc,
    props,
    hasIn: doc.role !== "source",
    hasOut: !isSink(doc),
  };
}

export function topoOrder(nodes, edges) {
  const indeg = Object.fromEntries(nodes.map((n) => [n.id, 0]));
  edges.forEach((e) => {
    if (e.target in indeg) indeg[e.target] += 1;
  });
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const queue = nodes.filter((n) => indeg[n.id] === 0).map((n) => n.id);
  const out = [];
  while (queue.length) {
    const id = queue.shift();
    out.push(byId[id]);
    edges
      .filter((e) => e.source === id)
      .forEach((e) => {
        if ((indeg[e.target] -= 1) === 0) queue.push(e.target);
      });
  }
  return out.length === nodes.length ? out : nodes; // cycle: fall back to input order
}

function propStr(node) {
  return Object.entries(node.data.props || {})
    .filter(([, v]) => v !== "" && v != null)
    .map(([k, v]) => ` ${k}=${v}`)
    .join("");
}

// Linear chains use the `!` form; any fan-out / fan-in switches to the `name=` +
// `elem.` form so branches are unambiguous.
export function toLaunch(nodes, edges) {
  if (!nodes.length) return "";
  const order = topoOrder(nodes, edges);
  const outdeg = {};
  const indeg = {};
  edges.forEach((e) => {
    outdeg[e.source] = (outdeg[e.source] || 0) + 1;
    indeg[e.target] = (indeg[e.target] || 0) + 1;
  });
  const branched =
    edges.length > 0 &&
    (Object.values(outdeg).some((d) => d > 1) || Object.values(indeg).some((d) => d > 1));
  if (!branched) return order.map((n) => n.data.element + propStr(n)).join(" ! ");
  // g2g's launch parser only starts a new chain at a `name.` reference, never by
  // juxtaposing two elements, so a plain decls-then-links dump does not parse.
  // Emit each node as its own named definition chain and each edge as a
  // `src. ! dst.` link chain, ordering them so every definition is immediately
  // followed by a link chain (which begins with a ref, the only safe boundary).
  // Verbose but always parses and runs; linear chains keep the clean `!` form.
  const defChain = (n) => `${n.data.element}${propStr(n)} name=${n.id}`;
  const linkChain = (e) => `${e.source}. ! ${e.target}.`;
  const out = [];
  order.forEach((n, i) => {
    out.push(defChain(n));
    if (i < edges.length) out.push(linkChain(edges[i]));
  });
  for (let j = order.length; j < edges.length; j += 1) out.push(linkChain(edges[j]));
  return out.join("\n");
}

// The declarative document behind both structured exports: nodes in topological
// order with their props, then the edges.
function graphDoc(nodes, edges) {
  return {
    nodes: topoOrder(nodes, edges).map((n) => ({
      id: n.id,
      element: n.data.element,
      ...(Object.keys(n.data.props || {}).length ? { props: n.data.props } : {}),
    })),
    edges: edges.map((e) => ({ from: e.source, to: e.target })),
  };
}

export function toJSON(nodes, edges) {
  return JSON.stringify(graphDoc(nodes, edges), null, 2);
}

// Every scalar is emitted double-quoted: props hold UI strings, and the loader
// types them from the element's PropertySpec, so "3" and 3 mean the same thing.
// Quoting also keeps a caps value (commas, slashes) from needing flow-style rules.
function yamlStr(v) {
  return `"${String(v).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

export function toYAML(nodes, edges) {
  const doc = graphDoc(nodes, edges);
  const out = [];
  out.push(doc.nodes.length ? "nodes:" : "nodes: []");
  doc.nodes.forEach((n) => {
    out.push(`  - id: ${yamlStr(n.id)}`);
    out.push(`    element: ${yamlStr(n.element)}`);
    if (n.props) {
      out.push("    props:");
      Object.entries(n.props).forEach(([k, v]) => out.push(`      ${k}: ${yamlStr(v)}`));
    }
  });
  out.push(doc.edges.length ? "edges:" : "edges: []");
  doc.edges.forEach((e) => {
    out.push(`  - from: ${yamlStr(e.from)}`);
    out.push(`    to: ${yamlStr(e.to)}`);
  });
  return out.join("\n") + "\n";
}
