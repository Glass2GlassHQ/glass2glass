// Pure export helpers over React Flow's node/edge shape (node.id,
// node.data.element, node.data.props; edge.source, edge.target). Both outputs
// load back into g2g: the gst-launch line via the text parser, the JSON via the
// declarative graph loader.

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
  const decls = order.map((n) => `${n.data.element}${propStr(n)} name=${n.id}`).join("\n");
  const links = edges.map((e) => `${e.source}. ! ${e.target}.`).join("\n");
  return `${decls}\n${links}`;
}

export function toJSON(nodes, edges) {
  return JSON.stringify(
    {
      nodes: topoOrder(nodes, edges).map((n) => ({
        id: n.id,
        element: n.data.element,
        ...(Object.keys(n.data.props || {}).length ? { props: n.data.props } : {}),
      })),
      edges: edges.map((e) => ({ from: e.source, to: e.target })),
    },
    null,
    2,
  );
}
