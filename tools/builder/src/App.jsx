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

// Parse-time autoplug macros (uridecodebin/decodebin), not registered elements,
// so g2g-inspect omits them. Shaped like a registry ElementDoc so the palette,
// node handles, and property panel treat them uniformly. They only round-trip
// through the gst-launch export, not the declarative JSON loader.
const DYNAMIC = [
  {
    name: "uridecodebin",
    role: "source",
    klass: "Source/Dynamic",
    long_name: "URI decode bin",
    description: "Auto-plugs a source + demux + decode chain for a URI (parse-time macro)",
    caps: "",
    pads: [],
    dynamic: true,
    properties: [
      {
        name: "uri",
        type: "String",
        blurb: "media URI (file://, http://, rtsp://, ...)",
        default: null,
        enum_values: null,
        range: null,
        readable: true,
        writable: true,
      },
    ],
  },
  {
    name: "decodebin",
    role: "element",
    klass: "Codec/Decoder/Dynamic",
    long_name: "Decode bin",
    description: "Auto-plugs a demux + decode chain for its input (parse-time macro)",
    caps: "",
    pads: [],
    dynamic: true,
    properties: [],
  },
];
const dynamicMap = Object.fromEntries(DYNAMIC.map((e) => [e.name, e]));

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

  // Registered elements plus the dynamic autoplug macros, keyed by name.
  const elements = useMemo(() => ({ ...dynamicMap, ...registry }), [registry]);

  const addNode = useCallback(
    (element) => {
      const doc = elements[element];
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
    [elements, setNodes],
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
    const g = { source: [], element: [], "muxer (fan-in)": [], dynamic: [] };
    Object.values(elements)
      .filter((e) => e.name.includes(filter))
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach((e) => (e.dynamic ? g.dynamic : g[e.role] || g.element).push(e));
    return g;
  }, [elements, filter]);

  const exportText = useMemo(
    () => (exportMode === "gst" ? toLaunch(nodes, edges) : toJSON(nodes, edges)),
    [exportMode, nodes, edges],
  );

  const hasDynamic = nodes.some((n) => n.data.doc?.dynamic);

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
        <h2>
          {selected ? selected.data.name : "properties"}
          {selected ? (
            <span className="pcount">{selected.data.doc.properties.length} props</span>
          ) : null}
        </h2>
        <div className="props">
          {!selected && <span className="hint">select a node</span>}
          {selected && !selected.data.doc.properties.length && (
            <span className="hint">this element exposes no properties</span>
          )}
          {selected &&
            selected.data.doc.properties.map((p) => (
              <PropRow
                key={p.name}
                prop={p}
                value={selected.data.props[p.name] ?? ""}
                onChange={(v) => setProp(selected.id, p.name, v)}
              />
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
          {hasDynamic && exportMode === "json" && (
            <div className="err">
              uridecodebin/decodebin are parse-time macros; load via the gst-launch export, not --graph
            </div>
          )}
          <pre className="exout">{exportText}</pre>
        </div>
      </aside>
    </div>
  );
}

// One property row: an enum dropdown when the property has named choices, else a
// typed text input. Shows the type, default, accepted range, and description so
// every knob is discoverable and settable. Read-only properties are disabled.
function PropRow({ prop, value, onChange }) {
  const choices = prop.enum_values ? prop.enum_values.split("|").map((s) => s.trim()) : null;
  const placeholder =
    prop.default != null
      ? `default ${prop.default}`
      : prop.range
        ? `${prop.range[0]} .. ${prop.range[1]}`
        : prop.type;
  return (
    <div className="prow">
      <label>
        {prop.name} <span className="ptype">{prop.type}</span>
        {!prop.writable && <span className="ro">read-only</span>}
      </label>
      {choices ? (
        <select value={value} disabled={!prop.writable} onChange={(e) => onChange(e.target.value)}>
          <option value="">{prop.default != null ? `default (${prop.default})` : "(default)"}</option>
          {choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : (
        <input
          value={value}
          disabled={!prop.writable}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value.trim())}
        />
      )}
      {prop.blurb && <div className="pblurb">{prop.blurb}</div>}
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
