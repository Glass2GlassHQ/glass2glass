import { Handle, Position } from "@xyflow/react";

// One pipeline element. Input handle unless it's a source; output handle unless
// it's a sink, so React Flow only allows valid out -> in links (and lets you
// drag from either end). The body shows the set properties, else the element
// name.
export function G2gNode({ data, selected }) {
  const set = Object.entries(data.props || {}).filter(([, v]) => v !== "" && v != null);
  const summary = set.length ? set.map(([k, v]) => `${k}=${v}`).join(" ") : data.element;
  return (
    <div
      className={`g2gnode role-${data.roleClass}${selected ? " sel" : ""}${data.unresolved ? " unresolved" : ""}`}
      title={data.unresolved ? "unresolved: not in the loaded registry" : undefined}
    >
      {data.hasIn && <Handle type="target" position={Position.Left} />}
      <div className="g2gnode-title">
        {data.name}
        {data.unresolved ? <span className="tag warn">?</span> : null}
      </div>
      <div className="g2gnode-body">{summary}</div>
      {data.hasOut && <Handle type="source" position={Position.Right} />}
    </div>
  );
}
