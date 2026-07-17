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
import { toLaunch, toJSON, isSink, nodeData } from "./export.js";
import { fromLaunch, fromJSON, seedCounters, decorateEdge, CAPS_WARN_TITLE } from "./import.js";
import { loadSolver, applyValidation } from "./solve.js";

const nodeTypes = { g2g: G2gNode };

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
        data: nodeData(id, element, doc, {}),
      };
      setNodes((nds) => nds.concat(node));
      setSelectedId(id);
    },
    [elements, setNodes],
  );

  // Latest nodes for onConnect's caps lookup, without rebuilding the callback
  // (or running a side effect inside a state updater) on every node change.
  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;
  const onConnect = useCallback(
    (params) => {
      const docById = Object.fromEntries(nodesRef.current.map((n) => [n.id, n.data.doc]));
      setEdges((eds) => addEdge(decorateEdge({ ...params, animated: true }, docById), eds));
    },
    [setEdges],
  );

  // Authoritative caps validation via the real solver compiled to wasm, loaded
  // best-effort once. When present it supersedes the family heuristic; when not
  // (missing blob, or the CSP-restricted single-file artifact) the heuristic on
  // onConnect stays the feedback.
  const solverRef = useRef(null);
  const [solverActive, setSolverActive] = useState(false);
  const [capsError, setCapsError] = useState("");
  useEffect(() => {
    let alive = true;
    loadSolver().then((fn) => {
      if (!alive) return;
      solverRef.current = fn;
      setSolverActive(!!fn);
    });
    return () => {
      alive = false;
    };
  }, []);

  // Re-validate only when the graph STRUCTURE changes (elements, props, links),
  // not when we restyle edges with the result (which would retrigger the effect).
  // The effect body reads live nodes/edges through refs, so structure is the only
  // trigger; edgesRef mirrors edges the way nodesRef mirrors nodes above.
  const edgesRef = useRef(edges);
  edgesRef.current = edges;
  const graphSig = useMemo(
    () =>
      JSON.stringify([
        nodes.map((n) => [n.id, n.data.element, n.data.props || {}]),
        edges.map((e) => [e.source, e.target]),
      ]),
    [nodes, edges],
  );
  useEffect(() => {
    if (!solverActive || !solverRef.current || !nodesRef.current.length) return undefined;
    const handle = setTimeout(async () => {
      const live = nodesRef.current;
      const line = toLaunch(live, edgesRef.current);
      if (!line.trim()) return;
      let result;
      try {
        result = JSON.parse(await solverRef.current(line));
      } catch {
        return; // solver unavailable mid-session: leave the last styling
      }
      const parseOrSetup = result.ok === false && (result.stage === "parse" || result.stage === "setup");
      setCapsError(parseOrSetup ? `caps: ${result.error}` : "");
      setEdges((eds) => applyValidation(live, eds, result));
    }, 300);
    return () => clearTimeout(handle);
  }, [graphSig, solverActive, setEdges]);

  const [importText, setImportText] = useState("");
  const [importFmt, setImportFmt] = useState("auto"); // auto | gst | json
  const [importErr, setImportErr] = useState("");

  const loadImport = useCallback(() => {
    const text = importText.trim();
    if (!text) return;
    let asJson = importFmt === "json";
    if (importFmt === "auto") {
      try {
        JSON.parse(text);
        asJson = true;
      } catch {
        asJson = false;
      }
    }
    try {
      const { nodes: ns, edges: es } = asJson
        ? fromJSON(text, elements)
        : fromLaunch(text, elements);
      setNodes(ns);
      setEdges(es);
      counters.current = seedCounters(ns);
      setSelectedId(null);
      setImportErr("");
    } catch (e) {
      setImportErr(String(e.message || e));
    }
  }, [importText, importFmt, elements, setNodes, setEdges]);

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

        <div className="import">
          <div className="exhdr">
            <span className="hint">import</span>
            <button className={importFmt === "auto" ? "on" : ""} onClick={() => setImportFmt("auto")}>
              auto
            </button>
            <button className={importFmt === "gst" ? "on" : ""} onClick={() => setImportFmt("gst")}>
              gst
            </button>
            <button className={importFmt === "json" ? "on" : ""} onClick={() => setImportFmt("json")}>
              JSON
            </button>
            <button className="copy" onClick={loadImport}>
              load
            </button>
          </div>
          <textarea
            className="imin"
            placeholder="paste a gst-launch line or declarative JSON, then load (replaces the canvas)"
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
          />
          {importErr && <div className="err">{importErr}</div>}
          {capsError && <div className="err">{capsError}</div>}
          <div className="hint" title={CAPS_WARN_TITLE}>
            {solverActive
              ? "live caps: real solver. red links = negotiation fails; labels = negotiated caps"
              : "live caps: family heuristic. red links = caps families look incompatible (not the full solver)"}
          </div>
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
