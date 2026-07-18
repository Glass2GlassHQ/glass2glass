//! Turn a validated graph into monomorphized `no_std` Rust: the same shape the
//! hand-written flagship pipeline (`noalloc-pipeline::audio`) has, but with the
//! ring sizes computed from the graph's frame geometry rather than hard-coded.
//! The emitted function is generic over the peripheral seams (the capture
//! grabbers and the RTP packet sender), so a board supplies its HAL impls and
//! the proof harness supplies mocks, exactly as the reference graph does.
//!
//! Supported topologies match what the static runners actually provide: a
//! linear `source -> transforms -> sink` chain, or two source branches joined
//! by one fan-in (the mixer) then a linear tail to the sink. Anything else is
//! rejected with a diagnostic before emission.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::catalog::{Geometry, Kind, Role};
use crate::model::GraphDoc;
use crate::CompileError;

/// The compiler's output: the generated Rust plus the ring-memory budget the
/// bin reports.
#[derive(Debug)]
pub struct Compiled {
    /// The generated `no_std` Rust module source.
    pub source: String,
    /// Every lent ring, as (variable name, bytes), in emission order.
    pub rings: Vec<(String, usize)>,
    /// Total ring bytes (the graph's static buffer budget).
    pub ring_bytes_total: usize,
    /// The generated entry function's name.
    pub entry: String,
    /// The capture-grabber parameters, in the order the entry takes them.
    pub grabber_params: Vec<String>,
    /// The sink's output seams, in the order the entry takes them after the
    /// grabbers (an RTP `sender`, or an SPI display's `spi`, `dc`, `delay`).
    pub sink_params: Vec<String>,
}

/// A node resolved for emission: its kind, output geometry, and the generated
/// variable / ring names.
struct Wired {
    kind: Kind,
    /// `None` for the sink (no output link).
    out_geom: Option<Geometry>,
    var: String,
    ring: String,
    ring_bytes: usize,
}

fn sanitize(id: &str) -> String {
    let mut s: String = id
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();
    if s.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        s.insert(0, 'n');
    }
    s
}

/// Compile a document into generated Rust. `doc` must already be name-valid.
pub(crate) fn compile(doc: &GraphDoc) -> Result<Compiled, CompileError> {
    if doc.nodes.is_empty() {
        return Err(CompileError::Empty);
    }
    // Resolve kinds and index nodes.
    let mut kinds = Vec::with_capacity(doc.nodes.len());
    let mut id_to_idx = BTreeMap::new();
    for (i, node) in doc.nodes.iter().enumerate() {
        if id_to_idx.insert(node.id.clone(), i).is_some() {
            return Err(CompileError::DuplicateId(node.id.clone()));
        }
        kinds.push(crate::catalog::resolve(node)?);
    }

    // Adjacency.
    let n = doc.nodes.len();
    let mut out: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut inn: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in &doc.edges {
        let from = *id_to_idx
            .get(&edge.from)
            .ok_or_else(|| CompileError::UnknownNode(edge.from.clone()))?;
        let to = *id_to_idx
            .get(&edge.to)
            .ok_or_else(|| CompileError::UnknownNode(edge.to.clone()))?;
        out[from].push(to);
        inn[to].push(from);
    }

    let role = |i: usize| kinds[i].role();
    let id = |i: usize| doc.nodes[i].id.as_str();

    // Sinks and sources.
    let sinks: Vec<usize> = (0..n).filter(|&i| role(i) == Role::Sink).collect();
    if sinks.len() != 1 {
        return Err(CompileError::Topology(format!(
            "expected exactly one sink, found {}",
            sinks.len()
        )));
    }
    let sink = sinks[0];
    if !out[sink].is_empty() {
        return Err(CompileError::Topology(format!(
            "sink `{}` has an outgoing edge",
            id(sink)
        )));
    }
    let sources: Vec<usize> = (0..n).filter(|&i| role(i) == Role::Source).collect();
    for &s in &sources {
        if !inn[s].is_empty() {
            return Err(CompileError::Topology(format!(
                "source `{}` has an incoming edge",
                id(s)
            )));
        }
    }
    let fanins: Vec<usize> = (0..n).filter(|&i| role(i) == Role::FanIn).collect();
    if fanins.len() > 1 {
        return Err(CompileError::Topology(
            "more than one fan-in is not supported".into(),
        ));
    }

    // The single successor of `node`, erroring unless there is exactly one.
    let succ = |node: usize| -> Result<usize, CompileError> {
        match out[node].as_slice() {
            [only] => Ok(*only),
            other => Err(CompileError::Topology(format!(
                "node `{}` must have exactly one output, has {}",
                id(node),
                other.len()
            ))),
        }
    };

    // Walk a source branch up to (excluding) `stop`, requiring interior nodes
    // to be transforms.
    let branch_to = |src: usize, stop: usize| -> Result<Vec<usize>, CompileError> {
        let mut chain = vec![src];
        let mut cur = src;
        loop {
            let nxt = succ(cur)?;
            if nxt == stop {
                break;
            }
            if role(nxt) != Role::Transform {
                return Err(CompileError::Topology(format!(
                    "node `{}` on a branch is not a transform",
                    id(nxt)
                )));
            }
            chain.push(nxt);
            cur = nxt;
        }
        Ok(chain)
    };

    // Assemble the topology into (branch_a, branch_b, fanin, tail-with-sink).
    // A linear graph is the degenerate case: one branch is the whole chain,
    // there is no fan-in, and the "tail" is empty.
    let (branches, fanin, tail): (Vec<Vec<usize>>, Option<usize>, Vec<usize>) =
        match fanins.as_slice() {
            [] => {
                if sources.len() != 1 {
                    return Err(CompileError::Topology(format!(
                        "a linear graph needs exactly one source, found {}",
                        sources.len()
                    )));
                }
                // Walk source -> ... -> sink; the sink terminates the tail.
                let mut chain = vec![sources[0]];
                let mut cur = sources[0];
                while cur != sink {
                    let nxt = succ(cur)?;
                    if nxt != sink && role(nxt) != Role::Transform {
                        return Err(CompileError::Topology(format!(
                            "node `{}` in a linear graph is not a transform",
                            id(nxt)
                        )));
                    }
                    chain.push(nxt);
                    cur = nxt;
                }
                // Source is the branch; the rest (transforms + sink) is the tail.
                let source_node = chain.remove(0);
                (vec![vec![source_node]], None, chain)
            }
            [f] => {
                let f = *f;
                if inn[f].len() != 2 {
                    return Err(CompileError::Topology(format!(
                        "fan-in `{}` needs exactly two inputs, has {}",
                        id(f),
                        inn[f].len()
                    )));
                }
                if sources.len() != 2 {
                    return Err(CompileError::Topology(format!(
                        "a fan-in graph needs exactly two sources, found {}",
                        sources.len()
                    )));
                }
                // Deterministic branch order: by source id.
                let mut srcs = sources.clone();
                srcs.sort_by(|&a, &b| id(a).cmp(id(b)));
                let branch_a = branch_to(srcs[0], f)?;
                let branch_b = branch_to(srcs[1], f)?;
                // Tail: fan-in's successor chain to the sink.
                let mut tail = Vec::new();
                let mut cur = succ(f)?;
                loop {
                    if cur != sink && role(cur) != Role::Transform {
                        return Err(CompileError::Topology(format!(
                            "node `{}` after the fan-in is not a transform",
                            id(cur)
                        )));
                    }
                    tail.push(cur);
                    if cur == sink {
                        break;
                    }
                    cur = succ(cur)?;
                }
                (vec![branch_a, branch_b], Some(f), tail)
            }
            _ => unreachable!("fanins.len() > 1 handled above"),
        };

    // Geometry pass + ring assignment for every node, in emission order.
    let mut wired: BTreeMap<usize, Wired> = BTreeMap::new();
    let frame_ns = doc.frame_ns;
    let mut flow =
        |nodes: &[usize], mut geom: Option<Geometry>| -> Result<Option<Geometry>, CompileError> {
            for &node in nodes {
                let out_geom = kinds[node].output_geometry(id(node), geom)?;
                let ring_bytes = match (kinds[node].role(), out_geom) {
                    (Role::Sink, _) => 0,
                    (_, Some(g)) => g.ring_bytes(frame_ns)?,
                    (_, None) => 0,
                };
                let var = sanitize(id(node));
                wired.insert(
                    node,
                    Wired {
                        kind: kinds[node].clone(),
                        out_geom,
                        ring: format!("ring_{var}"),
                        var,
                        ring_bytes,
                    },
                );
                geom = out_geom;
            }
            Ok(geom)
        };

    // Branch geometries feed the fan-in; the tail flows from the fan-in out.
    let branch_geoms: Vec<Option<Geometry>> = branches
        .iter()
        .map(|b| flow(b, None))
        .collect::<Result<_, _>>()?;
    let fanin_in_geom = if let Some(f) = fanin {
        let (Some(ga), Some(gb)) = (branch_geoms[0], branch_geoms[1]) else {
            return Err(CompileError::Topology(
                "a fan-in branch produced no frames".into(),
            ));
        };
        if ga != gb {
            return Err(CompileError::MixerInputMismatch {
                a: format!("{ga:?}"),
                b: format!("{gb:?}"),
            });
        }
        flow(&[f], Some(ga))?
    } else {
        branch_geoms[0]
    };
    flow(&tail, fanin_in_geom)?;

    // Grabber params, one per source, in the branch order emitted.
    let grabber_params: Vec<String> = branches
        .iter()
        .filter_map(|b| b.first())
        .map(|&s| format!("grab_{}", sanitize(id(s))))
        .collect();

    let sink_params: Vec<String> = sink_plan(&wired[&sink].kind, &wired[&sink].var)
        .seams
        .iter()
        .map(|s| s.param.clone())
        .collect();

    let compiled_source = emit(doc, &wired, &branches, fanin, &tail, sink, &grabber_params)?;

    let mut rings: Vec<(String, usize)> = Vec::new();
    let mut total = 0usize;
    for w in wired.values() {
        if w.ring_bytes > 0 {
            rings.push((w.ring.clone(), w.ring_bytes));
            total += w.ring_bytes;
        }
    }
    rings.sort();

    Ok(Compiled {
        source: compiled_source,
        rings,
        ring_bytes_total: total,
        entry: format!("run_{}_with", sanitize(&doc.name)),
        grabber_params,
        sink_params,
    })
}

/// A right-nested `Chain(a, Chain(b, c))` of transform vars, or the single var.
fn chain_expr(vars: &[&str]) -> String {
    match vars {
        [] => String::new(),
        [only] => (*only).to_string(),
        [head, rest @ ..] => format!("Chain({}, {})", head, chain_expr(rest)),
    }
}

/// One peripheral seam the generated entry takes: the parameter name, its type
/// parameter, and the trait bound (a `g2g_mcu` trait or a fully-qualified
/// `embedded_hal` path). Sources contribute a `FrameGrabber` seam each; the
/// sink contributes its own (an RTP sender, or an SPI bus + D/C pin + delay).
/// `mutable` marks a seam the body borrows `&mut` (the display's delay), so its
/// parameter is bound `mut`.
struct Seam {
    param: String,
    tparam: String,
    bound: String,
    mutable: bool,
}

/// How a sink is codegen'd: its seams, constructor, optional pre-run setup
/// (e.g. a display `init`), return type, and return expression. This is the
/// audio-vs-video generality: an RTP sink returns its sender for the checksum,
/// a display sink drives an externally-owned bus and returns nothing.
struct SinkPlan {
    seams: Vec<Seam>,
    ctor: String,
    setup: Option<String>,
    ret_type: String,
    ret_expr: Option<String>,
    core_imports: Vec<&'static str>,
    mcu_imports: Vec<&'static str>,
}

fn sink_plan(kind: &Kind, var: &str) -> SinkPlan {
    match kind {
        Kind::RtpSink { clock_rate, payload_type, ssrc, sequence } => SinkPlan {
            seams: vec![Seam {
                param: "sender".into(),
                tparam: "S".into(),
                bound: "PacketSender".into(),
                mutable: false,
            }],
            ctor: format!(
                "    let mut {var} = RtpSink::new(sender, MediaClock::audio({clock_rate}), {payload_type}, {ssrc:#010x}, {sequence});"
            ),
            setup: None,
            ret_type: "S".into(),
            ret_expr: Some(format!("    {var}.free()")),
            core_imports: vec!["MediaClock"],
            mcu_imports: vec!["PacketSender", "RtpSink"],
        },
        Kind::SpiDisplaySink { driver, width_px, height_px } => SinkPlan {
            seams: vec![
                Seam { param: "spi".into(), tparam: "SPI".into(), bound: "embedded_hal::spi::SpiDevice".into(), mutable: false },
                Seam { param: "dc".into(), tparam: "DC".into(), bound: "embedded_hal::digital::OutputPin".into(), mutable: false },
                Seam { param: "delay".into(), tparam: "DLY".into(), bound: "embedded_hal::delay::DelayNs".into(), mutable: true },
            ],
            ctor: format!(
                "    let mut {var} = SpiDisplaySink::{}(spi, dc, {width_px}, {height_px});",
                driver.ctor()
            ),
            setup: Some(format!("    if {var}.init(&mut delay).is_err() {{\n        return;\n    }}")),
            ret_type: "()".into(),
            ret_expr: None,
            core_imports: vec![],
            mcu_imports: vec!["SpiDisplaySink"],
        },
        _ => unreachable!("sink_plan on a non-sink kind"),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit(
    doc: &GraphDoc,
    wired: &BTreeMap<usize, Wired>,
    branches: &[Vec<usize>],
    fanin: Option<usize>,
    tail: &[usize],
    sink: usize,
    grabber_params: &[String],
) -> Result<String, CompileError> {
    let w = |i: usize| &wired[&i];
    let plan = sink_plan(&w(sink).kind, &w(sink).var);
    let mut body = String::new();

    // Which composition helpers this graph needs, to import exactly them.
    let is_fanin = fanin.is_some();
    let branch_has_transforms = branches.iter().any(|b| b.len() > 1);
    let tail_transforms = tail.iter().filter(|&&t| t != sink).count();
    let uses_chain = branches.iter().any(|b| b.len() > 2) || tail_transforms > 1;
    let uses_source_chain = is_fanin && branch_has_transforms;
    let uses_sink_chain = tail_transforms > 0;
    // Linear graphs put all middle transforms in the transform slot.
    let linear_transforms: usize = if is_fanin { 0 } else { tail_transforms };

    // Rings, then elements, in emission order.
    let emit_order: Vec<usize> = branches
        .iter()
        .flatten()
        .chain(fanin.iter())
        .chain(tail.iter())
        .copied()
        .collect();

    for &node in &emit_order {
        let e = w(node);
        if e.ring_bytes > 0 {
            let _ = writeln!(
                body,
                "    let {}: StaticLendRing<1, {}> = StaticLendRing::new();",
                e.ring, e.ring_bytes
            );
        }
    }
    let _ = writeln!(body);
    let _ = writeln!(
        body,
        "    // SAFETY: every ring above outlives the graph; the runner drains the"
    );
    let _ = writeln!(
        body,
        "    // pipeline (each lent frame dropped in its iteration) before this future"
    );
    let _ = writeln!(body, "    // completes and drops the rings.");

    let mut grab_iter = grabber_params.iter();
    for &node in &emit_order {
        let e = w(node);
        let line = if node == sink {
            plan.ctor.clone()
        } else {
            match &e.kind {
                Kind::GrabberSrc { .. } => {
                    let g = grab_iter.next().ok_or(CompileError::Internal("grabber param underflow"))?;
                    format!(
                        "    let {} = unsafe {{ GrabberSrc::with_ring({}, &{}, FRAME_NS) }}.with_frame_limit(FRAMES);",
                        e.var, g, e.ring
                    )
                }
                Kind::PcmConvert => {
                    format!("    let {} = unsafe {{ PcmConvert::with_ring(&{}) }};", e.var, e.ring)
                }
                Kind::Resample { from, to } => format!(
                    "    let {} = unsafe {{ Resampler::with_ring(SampleRate::Hz{}, SampleRate::Hz{}, &{}) }};",
                    e.var, from, to, e.ring
                ),
                Kind::Mixer { gain_a, gain_b } => format!(
                    "    let {} = unsafe {{ Mixer::with_ring({}, {}, &{}) }};",
                    e.var, gain_a, gain_b, e.ring
                ),
                Kind::G711Enc { law } => {
                    format!("    let {} = unsafe {{ G711Enc::with_ring({}, &{}) }};", e.var, law.variant(), e.ring)
                }
                Kind::RtpSink { .. } | Kind::SpiDisplaySink { .. } => {
                    return Err(CompileError::Internal("sink kind in mid-graph"))
                }
            }
        };
        let _ = writeln!(body, "{line}");
    }
    let _ = writeln!(body);

    // Wiring expressions.
    let sink_var = &w(sink).var;
    let tail_transform_vars: Vec<&str> = tail
        .iter()
        .filter(|&&t| t != sink)
        .map(|&t| w(t).var.as_str())
        .collect();
    let sink_expr = if tail_transform_vars.is_empty() {
        format!("&mut {sink_var}")
    } else {
        format!(
            "SinkChain({}, &mut {})",
            chain_expr(&tail_transform_vars),
            sink_var
        )
    };

    let run_line = if let Some(f) = fanin {
        let branch_expr = |b: &Vec<usize>| -> String {
            let src = &w(b[0]).var;
            let ts: Vec<&str> = b[1..].iter().map(|&t| w(t).var.as_str()).collect();
            if ts.is_empty() {
                src.clone()
            } else {
                format!("SourceChain({}, {})", src, chain_expr(&ts))
            }
        };
        format!(
            "    let _ = run_sources_fanin_sink({}, {}, {}, {}).await;",
            branch_expr(&branches[0]),
            branch_expr(&branches[1]),
            w(f).var,
            sink_expr
        )
    } else {
        let src = &w(branches[0][0]).var;
        let ts: Vec<&str> = tail
            .iter()
            .filter(|&&t| t != sink)
            .map(|&t| w(t).var.as_str())
            .collect();
        if ts.is_empty() {
            format!("    let _ = run_source_sink({src}, {sink_expr}).await;")
        } else {
            format!(
                "    let _ = run_source_transform_sink({}, {}, {}).await;",
                src,
                chain_expr(&ts),
                sink_expr
            )
        }
    };

    // Optional fan-in caps negotiation (keeps `Caps::intersect` in the archive
    // for the proofs, like the hand-written graph). Only audio graphs fan in.
    let mut negotiate = String::new();
    if let Some(f) = fanin {
        if let Some(Geometry::Audio {
            sample_rate,
            channels,
            ..
        }) = w(f).out_geom
        {
            let _ = writeln!(
                negotiate,
                "/// Negotiate the fan-in link's `Caps::Audio` (both sides black-boxed so"
            );
            let _ = writeln!(
                negotiate,
                "/// the audio arm of `Caps::intersect` stays in the archive for the proofs)."
            );
            let _ = writeln!(negotiate, "fn negotiate_fanin_link() -> Option<Caps> {{");
            let _ = writeln!(
                negotiate,
                "    let mk = || Caps::Audio {{ format: AudioFormat::PcmS16Le, channels: {channels}, sample_rate: {sample_rate} }};"
            );
            let _ = writeln!(
                negotiate,
                "    black_box(mk()).intersect(&black_box(mk())).ok()"
            );
            let _ = writeln!(negotiate, "}}\n");
        }
    }

    // Assemble the module.
    let entry = format!("run_{}_with", sanitize(&doc.name));
    let (type_params, where_clause, params) = signature(grabber_params, &plan.seams);
    let mut src = String::new();
    let _ = writeln!(src, "{}", header(doc));
    let _ = write!(
        src,
        "{}",
        imports(
            is_fanin,
            uses_chain,
            uses_source_chain,
            uses_sink_chain,
            &emit_order,
            wired,
            linear_transforms,
            &plan
        )
    );
    let _ = writeln!(src, "pub const FRAME_NS: u64 = {};", doc.frame_ns);
    let _ = writeln!(src, "pub const FRAMES: u32 = {};", doc.frames);
    let total: usize = wired.values().map(|x| x.ring_bytes).sum();
    let _ = writeln!(src, "/// Total ring memory this graph statically owns.");
    let _ = writeln!(src, "pub const RING_BYTES_TOTAL: usize = {total};\n");
    if !negotiate.is_empty() {
        let _ = write!(src, "{negotiate}");
    }
    let _ = writeln!(
        src,
        "/// Run the graph. Peripheral seams (the capture grabbers, then the sink's"
    );
    let _ = writeln!(
        src,
        "/// output seam) are taken in the documented order; a board supplies HAL"
    );
    let _ = writeln!(src, "/// impls, a proof harness supplies mocks.");
    // Omit the arrow for a unit return (a display sink) so no `-> ()` is emitted.
    let ret_arrow = if plan.ret_type == "()" {
        String::new()
    } else {
        format!(" -> {}", plan.ret_type)
    };
    let _ = writeln!(
        src,
        "pub async fn {entry}{type_params}({params}){ret_arrow}{where_clause} {{"
    );
    if fanin.is_some() {
        let _ = writeln!(src, "    if negotiate_fanin_link().is_none() {{");
        let _ = writeln!(src, "        return sender;");
        let _ = writeln!(src, "    }}");
    }
    let _ = write!(src, "{body}");
    if let Some(setup) = &plan.setup {
        let _ = writeln!(src, "{setup}");
    }
    let _ = writeln!(src, "{run_line}");
    if let Some(ret) = &plan.ret_expr {
        let _ = writeln!(src, "{ret}");
    }
    let _ = writeln!(src, "}}");
    Ok(src)
}

/// The entry signature: type params, params, and where clause, from the source
/// grabber seams (one per source) followed by the sink's seams.
fn signature(grabber_params: &[String], sink_seams: &[Seam]) -> (String, String, String) {
    let mut type_params = Vec::new();
    let mut where_bounds = Vec::new();
    let mut params = Vec::new();
    for (i, g) in grabber_params.iter().enumerate() {
        let t = format!("G{i}");
        where_bounds.push(format!("{t}: FrameGrabber"));
        params.push(format!("{g}: {t}"));
        type_params.push(t);
    }
    for seam in sink_seams {
        where_bounds.push(format!("{}: {}", seam.tparam, seam.bound));
        // A seam the body borrows `&mut` (the display's delay) needs a `mut`
        // binding; one moved into a constructor (sender, spi, dc) does not.
        let bind = if seam.mutable { "mut " } else { "" };
        params.push(format!("{bind}{}: {}", seam.param, seam.tparam));
        type_params.push(seam.tparam.clone());
    }
    (
        format!("<{}>", type_params.join(", ")),
        format!("\nwhere\n    {},", where_bounds.join(",\n    ")),
        params.join(", "),
    )
}

fn header(doc: &GraphDoc) -> String {
    let mut h = String::new();
    let _ = writeln!(
        h,
        "//! GENERATED by g2g-mcugen from graph `{}`. Do not edit by hand;",
        doc.name
    );
    let _ = writeln!(
        h,
        "//! regenerate with `g2g-mcugen <graph>.yaml`. This is the monomorphized"
    );
    let _ = writeln!(
        h,
        "//! static MCU pipeline: heap-free, no `dyn`, ring sizes computed from the"
    );
    let _ = writeln!(h, "//! graph's frame geometry.");
    h
}

#[allow(clippy::too_many_arguments)]
fn imports(
    is_fanin: bool,
    uses_chain: bool,
    uses_source_chain: bool,
    uses_sink_chain: bool,
    emit_order: &[usize],
    wired: &BTreeMap<usize, Wired>,
    linear_transforms: usize,
    plan: &SinkPlan,
) -> String {
    // Items imported from the `g2g_core` crate root, deduped and sorted.
    let mut root: Vec<String> = Vec::new();
    if uses_chain {
        root.push("Chain".into());
    }
    root.push(
        if is_fanin {
            "run_sources_fanin_sink"
        } else if linear_transforms > 0 {
            "run_source_transform_sink"
        } else {
            "run_source_sink"
        }
        .into(),
    );
    if uses_sink_chain {
        root.push("SinkChain".into());
    }
    if uses_source_chain {
        root.push("SourceChain".into());
    }
    for item in &plan.core_imports {
        root.push((*item).into());
    }
    if is_fanin {
        root.push("AudioFormat".into());
        root.push("Caps".into());
    }
    root.sort();
    root.dedup();

    // Items from `g2g_mcu`, by the kinds present (sink items from the plan).
    let mut mcu: Vec<&str> = Vec::new();
    for &node in emit_order {
        match &wired[&node].kind {
            Kind::GrabberSrc { .. } => mcu.extend(["FrameGrabber", "GrabberSrc"]),
            Kind::PcmConvert => mcu.push("PcmConvert"),
            Kind::Resample { .. } => mcu.extend(["Resampler", "SampleRate"]),
            Kind::Mixer { .. } => mcu.push("Mixer"),
            Kind::G711Enc { .. } => mcu.extend(["G711Enc", "Law"]),
            Kind::RtpSink { .. } | Kind::SpiDisplaySink { .. } => {}
        }
    }
    mcu.extend(plan.mcu_imports.iter().copied());
    mcu.sort_unstable();
    mcu.dedup();

    let mut s = String::new();
    if is_fanin {
        let _ = writeln!(s, "use core::hint::black_box;");
    }
    let _ = writeln!(s, "use g2g_core::staticpool::StaticLendRing;");
    let _ = writeln!(s, "use g2g_core::{{{}}};", root.join(", "));
    let _ = writeln!(s, "use g2g_mcu::{{{}}};", mcu.join(", "));
    let _ = writeln!(s);
    s
}
