import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
} from "@xyflow/react";
import { G2gNode } from "./nodes.jsx";
import { toLaunch, toJSON } from "./export.js";

const nodeTypes = { g2g: G2gNode };
const isSink = (doc) => /Sink/i.test(doc.klass || "");
const roleClass = (role) =>
  role === "source" ? "source" : role.startsWith("muxer") ? "muxer" : "element";

export default function App() {
  const [registry, setRegistry] = useState({});
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("");
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [exportMode, setExportMode] = useState("gst");
  const counters = useRef({});

  useEffect(() => {
    const load = (d) => setRegistry(Object.fromEntries(d.elements.map((e) => [e.name, e])));
    // A published/self-contained build can inline the registry; otherwise fetch
    // the sibling snapshot the dev server / static host serves.
    if (window.__G2G_REGISTRY__) {
      load(window.__G2G_REGISTRY__);
      return;
    }
    fetch("./registry.json")
      .then((r) => r.json())
      .then(load)
      .catch(() =>
        setError("registry.json not found - run: g2g-inspect --json > tools/builder/public/registry.json"),
      );
  }, []);

  const addNode = useCallback(
    (element) => {
      const doc = registry[element];
      const idx = counters.current[element] || 0;
      counters.current[element] = idx + 1;
      const id = element + idx;
      const node = {
        id,
        type: "g2g",
        position: { x: 80 + (Object.keys(counters.current).length % 3) * 60, y: 60 + idx * 90 },
        data: {
          name: id,
          element,
          role: doc.role,
          roleClass: roleClass(doc.role),
          doc,
          props: {},
          hasIn: doc.role !== "source",
          hasOut: !isSink(doc),
        },
      };
      setNodes((nds) => nds.concat(node));
      setSelectedId(id);
    },
    [registry, setNodes],
  );

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge({ ...params, animated: true }, eds)),
    [setEdges],
  );

  const setProp = useCallback(
    (id, key, value) => {
      setNodes((nds) =>
        nds.map((n) => {
          if (n.id !== id) return n;
          const props = { ...n.data.props };
          if (value === "") delete props[key];
          else props[key] = value;
          return { ...n, data: { ...n.data, props } };
        }),
      );
    },
    [setNodes],
  );

  const selected = nodes.find((n) => n.id === selectedId) || null;

  // Palette grouped by role, filtered by the search box.
  const groups = useMemo(() => {
    const g = { source: [], element: [], "muxer (fan-in)": [] };
    Object.values(registry)
      .filter((e) => e.name.includes(filter))
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach((e) => (g[e.role] || g.element).push(e));
    return g;
  }, [registry, filter]);

  const exportText = useMemo(
    () => (exportMode === "gst" ? toLaunch(nodes, edges) : toJSON(nodes, edges)),
    [exportMode, nodes, edges],
  );

  return (
    <div className="app">
      <aside className="palette">
        <input
          className="search"
          placeholder="filter elements…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <div className="plist">
          {error && <div className="err">{error}</div>}
          {Object.entries(groups).map(([role, items]) =>
            items.length ? (
              <div key={role}>
                <div className="pgroup">{role}</div>
                {items.map((e) => (
                  <div
                    key={e.name}
                    className="pitem"
                    title={e.description || ""}
                    onClick={() => addNode(e.name)}
                  >
                    {e.name}
                    {isSink(e) ? <span className="tag">sink</span> : null}
                  </div>
                ))}
              </div>
            ) : null,
          )}
        </div>
      </aside>

      <div className="canvas">
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_, n) => setSelectedId(n.id)}
            onPaneClick={() => setSelectedId(null)}
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={22} color="#1b1f27" />
            <Controls />
            <MiniMap pannable zoomable nodeColor={miniColor} maskColor="rgba(0,0,0,.5)" />
          </ReactFlow>
        </ReactFlowProvider>
      </div>

      <aside className="side">
        <h2>{selected ? selected.data.name : "properties"}</h2>
        <div className="props">
          {!selected && <span className="hint">select a node</span>}
          {selected && !selected.data.doc.properties.length && (
            <span className="hint">no properties</span>
          )}
          {selected &&
            selected.data.doc.properties
              .filter((p) => p.writable)
              .map((p) => (
                <div className="prow" key={p.name}>
                  <label title={p.blurb}>
                    {p.name} <span className="ptype">{p.type}</span>
                  </label>
                  <input
                    value={selected.data.props[p.name] ?? ""}
                    placeholder={p.default != null ? `default ${p.default}` : p.type}
                    onChange={(e) => setProp(selected.id, p.name, e.target.value.trim())}
                  />
                </div>
              ))}
        </div>

        <div className="export">
          <div className="exhdr">
            <span className="hint">export</span>
            <button className={exportMode === "gst" ? "on" : ""} onClick={() => setExportMode("gst")}>
              gst-launch
            </button>
            <button className={exportMode === "json" ? "on" : ""} onClick={() => setExportMode("json")}>
              JSON
            </button>
            <button
              className="copy"
              onClick={() => navigator.clipboard && navigator.clipboard.writeText(exportText)}
            >
              copy
            </button>
          </div>
          <pre className="exout">{exportText}</pre>
        </div>
      </aside>
    </div>
  );
}

function miniColor(n) {
  return n.data?.roleClass === "source"
    ? "#46c07a"
    : n.data?.roleClass === "muxer"
      ? "#e0a54d"
      : "#4da3ff";
}
