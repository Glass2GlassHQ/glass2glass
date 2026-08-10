//! In-terminal live pipeline telemetry (`g2g-launch --tui`, M1013).
//!
//! The terminal sibling of the browser dashboard: it reads the same
//! [`Observer`](g2g_core::runtime::Observer) tap in-process and draws it with
//! ratatui, so a box with no browser still watches per-element latency, per-edge
//! traffic, the pipeline's shape, and the single-frame journey while the run is
//! in flight. No JSON and no server: the snapshot structs render directly.
//!
//! Raw mode and the alternate screen are entered by [`PipelineTui::start`] and
//! left again by its `Drop` (ratatui's init also installs a panic hook that
//! restores before the panic prints). g2g's log sink writes to stderr, which
//! would tear the drawing apart, so `start` swaps it for an in-memory ring the
//! TUI renders as a log pane, and `Drop` puts the stderr sink back.

use std::io;
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::{format, vec};

use g2g_core::log::{self, LogLevel, OwnedLogRecord, RingSink, StderrSink};
use g2g_core::runtime::{NodeRole, TelemetrySnapshot};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Row, Table};
use ratatui::{DefaultTerminal, Frame};

/// Log records kept while the TUI owns the terminal. Only the newest
/// [`LOG_LINES`] are drawn; the rest are headroom for a burst.
const LOG_CAPACITY: usize = 64;
/// Log lines shown in the bottom pane.
const LOG_LINES: usize = 5;
/// Columns the graph pane scrolls per arrow keypress.
const GRAPH_SCROLL_STEP: usize = 8;

/// Which pane the screen is showing. `g` toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum View {
    #[default]
    Tables,
    Graph,
}

/// What a keypress asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    Quit,
    ToggleView,
    ScrollLeft,
    ScrollRight,
    Ignore,
}

/// The terminal, the diverted log ring, and the view state. Constructing one
/// takes over the terminal; dropping it gives the terminal and the stderr log
/// sink back.
pub struct PipelineTui {
    terminal: DefaultTerminal,
    logs: RingSink,
    view: View,
    graph_scroll: usize,
}

impl core::fmt::Debug for PipelineTui {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PipelineTui")
            .field("view", &self.view)
            .field("graph_scroll", &self.graph_scroll)
            .finish_non_exhaustive()
    }
}

impl PipelineTui {
    /// Enter raw mode + the alternate screen and divert g2g logging into the
    /// in-memory ring. Fails when stdout is not a terminal.
    pub fn start() -> io::Result<Self> {
        let terminal = ratatui::try_init()?;
        let logs = RingSink::new(LOG_CAPACITY);
        log::set_sink(Box::new(logs.clone()));
        Ok(Self {
            terminal,
            logs,
            view: View::default(),
            graph_scroll: 0,
        })
    }

    /// Redraw the screen from one telemetry read.
    pub fn draw(&mut self, snap: &TelemetrySnapshot) -> io::Result<()> {
        let records = self.logs.snapshot();
        let logs: Vec<&OwnedLogRecord> = records.iter().rev().take(LOG_LINES).rev().collect();
        let view = self.view;
        let scroll = self.graph_scroll;
        self.terminal
            .draw(|frame| render(frame, view, scroll, snap, &logs))?;
        Ok(())
    }

    /// Drain pending keypresses without blocking. `true` means the user asked
    /// to quit.
    pub fn handle_input(&mut self) -> io::Result<bool> {
        while event::poll(Duration::ZERO)? {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            match action_for(key) {
                Action::Quit => return Ok(true),
                Action::ToggleView => {
                    self.view = match self.view {
                        View::Tables => View::Graph,
                        View::Graph => View::Tables,
                    };
                }
                Action::ScrollLeft => {
                    self.graph_scroll = self.graph_scroll.saturating_sub(GRAPH_SCROLL_STEP);
                }
                Action::ScrollRight => {
                    self.graph_scroll = self.graph_scroll.saturating_add(GRAPH_SCROLL_STEP);
                }
                Action::Ignore => {}
            }
        }
        Ok(false)
    }
}

impl Drop for PipelineTui {
    fn drop(&mut self) {
        ratatui::restore();
        log::set_sink(Box::new(StderrSink));
    }
}

/// Ctrl-C reaches us as a keypress, not a signal, because raw mode is on.
fn action_for(key: KeyEvent) -> Action {
    if key.kind == KeyEventKind::Release {
        return Action::Ignore;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return match key.code {
            KeyCode::Char('c') | KeyCode::Char('C') => Action::Quit,
            _ => Action::Ignore,
        };
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => Action::Quit,
        KeyCode::Char('g') | KeyCode::Char('G') => Action::ToggleView,
        KeyCode::Left => Action::ScrollLeft,
        KeyCode::Right => Action::ScrollRight,
        _ => Action::Ignore,
    }
}

// ---------------------------------------------------------------- derivation

/// One row of the nodes table: the fields the dashboard derives from
/// `ElementLatency`, with the same "absent means never measured" rules.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeRow {
    name: String,
    role: &'static str,
    proc_p50_ns: Option<u64>,
    proc_p99_ns: Option<u64>,
    transit_p50_ns: Option<u64>,
    push_wait_p50_ns: Option<u64>,
    fill_mean_pct: u8,
    fill_max_pct: u8,
}

fn node_rows(snap: &TelemetrySnapshot) -> Vec<NodeRow> {
    snap.nodes
        .iter()
        .map(|n| {
            let measured = n.latency.as_ref().filter(|l| l.proc.count > 0);
            NodeRow {
                name: display_name(snap, n.id),
                role: role_label(n.role),
                proc_p50_ns: measured.map(|l| l.proc.p50_ns),
                proc_p99_ns: measured.map(|l| l.proc.p99_ns),
                transit_p50_ns: n
                    .latency
                    .as_ref()
                    .filter(|l| l.transit.count > 0)
                    .map(|l| l.transit.p50_ns),
                push_wait_p50_ns: n
                    .latency
                    .as_ref()
                    .filter(|l| l.push_wait.max_ns > 0)
                    .map(|l| l.push_wait.p50_ns),
                fill_mean_pct: n.latency.as_ref().map_or(0, |l| l.fill_mean_pct),
                fill_max_pct: n.latency.as_ref().map_or(0, |l| l.fill_max_pct),
            }
        })
        .collect()
}

/// One row of the edges table. `caps` prefers the caps observed crossing the
/// link over the negotiated ones, so a stream that refines mid-run reads true.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EdgeRow {
    from: String,
    to: String,
    caps: String,
    packets: u64,
    bytes: u64,
    drops: u64,
}

fn edge_rows(snap: &TelemetrySnapshot) -> Vec<EdgeRow> {
    snap.edges
        .iter()
        .map(|e| EdgeRow {
            from: display_name(snap, e.from),
            to: display_name(snap, e.to),
            caps: e
                .observed_caps
                .clone()
                .or_else(|| e.caps.clone())
                .unwrap_or_else(|| "-".to_string()),
            packets: e.counts.packets,
            bytes: e.counts.bytes,
            drops: e.counts.drops,
        })
        .collect()
}

fn display_name(snap: &TelemetrySnapshot, id: usize) -> String {
    match snap.nodes.get(id) {
        Some(n) if !n.name.is_empty() => n.name.clone(),
        _ => format!("n{id}"),
    }
}

fn role_label(role: NodeRole) -> &'static str {
    match role {
        NodeRole::Source => "source",
        NodeRole::Transform => "transform",
        NodeRole::Sink => "sink",
        NodeRole::Tee => "tee",
        NodeRole::Muxer => "muxer",
    }
}

/// Packets that reached a sink, the closest thing to "frames out" the tap can
/// state honestly, plus every link's drops.
fn delivered_and_dropped(snap: &TelemetrySnapshot) -> (u64, u64) {
    let delivered = snap
        .edges
        .iter()
        .filter(|e| {
            snap.nodes
                .get(e.to)
                .is_some_and(|n| n.role == NodeRole::Sink)
        })
        .map(|e| e.counts.packets)
        .sum();
    let dropped = snap.edges.iter().map(|e| e.counts.drops).sum();
    (delivered, dropped)
}

fn header_line(snap: &TelemetrySnapshot, view: View) -> String {
    let (delivered, dropped) = delivered_and_dropped(snap);
    let view = match view {
        View::Tables => "tables",
        View::Graph => "graph",
    };
    format!(
        "uptime {}   nodes {}   edges {}   delivered {delivered}   drops {dropped}   view {view}",
        format_uptime(snap.uptime_ns),
        snap.nodes.len(),
        snap.edges.len(),
    )
}

/// Split `width` columns across one journey stage's wait / work / blocked in
/// proportion to `scale_ns` (the widest stage's total), so stages compare by
/// eye. A non-zero segment always gets at least one column, and the three
/// together never exceed `width`.
fn bar_widths(
    wait_ns: u64,
    work_ns: u64,
    blocked_ns: u64,
    scale_ns: u64,
    width: usize,
) -> [usize; 3] {
    if scale_ns == 0 || width == 0 {
        return [0, 0, 0];
    }
    let mut out = [0usize; 3];
    let mut used = 0usize;
    for (slot, value) in out.iter_mut().zip([wait_ns, work_ns, blocked_ns]) {
        if value == 0 || used >= width {
            continue;
        }
        let scaled = (value as u128 * width as u128 / scale_ns as u128) as usize;
        let cells = scaled.max(1).min(width - used);
        *slot = cells;
        used += cells;
    }
    out
}

fn format_ns(ns: u64) -> String {
    if ns < 1_000 {
        format!("{ns} ns")
    } else if ns < 1_000_000 {
        format!("{:.1} us", ns as f64 / 1e3)
    } else if ns < 1_000_000_000 {
        format!("{:.1} ms", ns as f64 / 1e6)
    } else {
        format!("{:.2} s", ns as f64 / 1e9)
    }
}

fn format_opt_ns(ns: Option<u64>) -> String {
    ns.map_or_else(|| "-".to_string(), format_ns)
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn format_uptime(ns: u64) -> String {
    let seconds = ns / 1_000_000_000;
    if seconds < 60 {
        format!("{:.1} s", ns as f64 / 1e9)
    } else if seconds < 3_600 {
        format!("{}m {:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {:02}m", seconds / 3_600, (seconds % 3_600) / 60)
    }
}

// -------------------------------------------------------------- graph layout

/// One horizontal run of the ASCII graph, tagged with the node whose box it
/// belongs to (`None` for connectors and blank space) so the renderer can color
/// it by that node's health.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphSpan {
    text: String,
    node: Option<usize>,
}

/// Box interior height: one line for the name, one for the live stat.
const BOX_CONTENT_LINES: usize = 2;
/// Box height including its two borders.
const BOX_HEIGHT: usize = BOX_CONTENT_LINES + 2;
/// Blank line between stacked boxes.
const ROW_GAP: usize = 1;
/// Longest element name drawn in a box before it is cut.
const MAX_NAME: usize = 16;

/// A character canvas plus, per cell, the node that owns it.
#[derive(Debug)]
struct Canvas {
    width: usize,
    cells: Vec<Vec<char>>,
    owner: Vec<Vec<Option<usize>>>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            cells: vec![vec![' '; width]; height],
            owner: vec![vec![None; width]; height],
        }
    }

    fn get(&self, x: usize, y: usize) -> Option<char> {
        self.cells.get(y).and_then(|row| row.get(x)).copied()
    }

    fn owned(&self, x: usize, y: usize) -> bool {
        self.owner
            .get(y)
            .and_then(|row| row.get(x))
            .is_some_and(Option::is_some)
    }

    fn set(&mut self, x: usize, y: usize, ch: char, node: Option<usize>) {
        if x >= self.width || y >= self.cells.len() {
            return;
        }
        self.cells[y][x] = ch;
        self.owner[y][x] = node;
    }

    /// Draw a connector cell, leaving a node's box intact where the route runs
    /// over one, and crossing an existing line rather than erasing it.
    fn connector(&mut self, x: usize, y: usize, ch: char) {
        if self.owned(x, y) {
            return;
        }
        let merged = self.get(x, y).map_or(ch, |old| merge_line(old, ch));
        self.set(x, y, merged, None);
    }

    fn write(&mut self, x: usize, y: usize, text: &str, node: Option<usize>) {
        for (i, ch) in text.chars().enumerate() {
            self.set(x + i, y, ch, node);
        }
    }

    /// Group each line's cells into runs of one owner, dropping the blank tail.
    fn into_spans(self) -> Vec<Vec<GraphSpan>> {
        self.cells
            .into_iter()
            .zip(self.owner)
            .map(|(chars, owners)| {
                let mut spans: Vec<GraphSpan> = Vec::new();
                for (ch, node) in chars.into_iter().zip(owners) {
                    match spans.last_mut() {
                        Some(last) if last.node == node => last.text.push(ch),
                        _ => spans.push(GraphSpan {
                            text: String::from(ch),
                            node,
                        }),
                    }
                }
                if let Some(last) = spans.last_mut().filter(|s| s.node.is_none()) {
                    last.text.truncate(last.text.trim_end().len());
                }
                if spans.last().is_some_and(|s| s.text.is_empty()) {
                    spans.pop();
                }
                spans
            })
            .collect()
    }
}

/// Where a box-drawing character connects: up, down, left, right.
const LINE_PIECES: [(char, u8); 11] = [
    ('─', 0b0011),
    ('│', 0b1100),
    ('┐', 0b0101),
    ('┘', 0b1001),
    ('└', 0b1010),
    ('┌', 0b0110),
    ('┬', 0b0111),
    ('┴', 0b1011),
    ('├', 0b1110),
    ('┤', 0b1101),
    ('┼', 0b1111),
];

/// Combine two crossing connector characters into the piece that keeps both
/// their arms, so a branch turning out of a run that continues past it reads as
/// a tee rather than cutting the run. Anything that is not a line piece (an
/// arrow head, an edge label) stays put.
fn merge_line(existing: char, new: char) -> char {
    let mask = |ch: char| {
        LINE_PIECES
            .iter()
            .find(|(c, _)| *c == ch)
            .map(|(_, bits)| *bits)
    };
    let (Some(old_bits), Some(new_bits)) = (mask(existing), mask(new)) else {
        return if existing == ' ' { new } else { existing };
    };
    let combined = old_bits | new_bits;
    LINE_PIECES
        .iter()
        .find(|(_, bits)| *bits == combined)
        .map_or(new, |(c, _)| *c)
}

/// Longest-path column per node: a node sits one column right of its furthest
/// upstream. Bounded by the node count, so a malformed cyclic topology still
/// terminates.
fn columns_of(snap: &TelemetrySnapshot) -> Vec<usize> {
    let count = snap.nodes.len();
    let mut column = vec![0usize; count];
    for _ in 0..count {
        let mut changed = false;
        for e in &snap.edges {
            if e.from >= count || e.to >= count {
                continue;
            }
            if column[e.to] < column[e.from] + 1 {
                column[e.to] = column[e.from] + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    column
}

/// The two lines inside a node's box: its name and one live stat (measured
/// `process()` p99, or the node's role when nothing times it).
fn box_text(snap: &TelemetrySnapshot, id: usize) -> (String, String) {
    let name = display_name(snap, id);
    let name = if name.chars().count() > MAX_NAME {
        name.chars().take(MAX_NAME).collect()
    } else {
        name
    };
    let node = snap.nodes.get(id);
    let stat = match node.and_then(|n| n.latency.as_ref()) {
        Some(l) if l.proc.count > 0 => format!("p99 {}", format_ns(l.proc.p99_ns)),
        Some(_) => "idle".to_string(),
        None => node.map_or("?", |n| role_label(n.role)).to_string(),
    };
    (name, stat)
}

/// Live decoration for one edge: packets crossed, and `-N` when the link
/// dropped N under a leaky policy.
fn edge_label(snap: &TelemetrySnapshot, index: usize) -> String {
    let Some(edge) = snap.edges.get(index) else {
        return String::new();
    };
    if edge.counts.drops > 0 {
        format!("{} -{}", edge.counts.packets, edge.counts.drops)
    } else {
        format!("{}", edge.counts.packets)
    }
}

/// Draw the pipeline as boxes and connectors: columns left to right in
/// topological order, one row per node sharing a column (so a tee's branches
/// and a muxer's inputs stack), each edge labelled with its live packet count.
fn graph_spans(snap: &TelemetrySnapshot) -> Vec<Vec<GraphSpan>> {
    if snap.nodes.is_empty() {
        return Vec::new();
    }
    let column = columns_of(snap);
    let column_count = column.iter().max().copied().unwrap_or(0) + 1;

    // Row within a column, in node order, plus each column's box width.
    let mut row_of = vec![0usize; snap.nodes.len()];
    let mut rows_used = vec![0usize; column_count];
    let mut box_width = vec![0usize; column_count];
    for id in 0..snap.nodes.len() {
        let col = column[id];
        row_of[id] = rows_used[col];
        rows_used[col] += 1;
        let (name, stat) = box_text(snap, id);
        let width = name.chars().count().max(stat.chars().count()) + 4;
        box_width[col] = box_width[col].max(width);
    }

    // Each gap reserves room for the widest label crossing it, so every turn in
    // that gap lines up: "──<label>" then the turn column, one dash, the arrow.
    let mut label_width = vec![0usize; column_count];
    for (index, e) in snap.edges.iter().enumerate() {
        if e.from >= snap.nodes.len() || e.to >= snap.nodes.len() {
            continue;
        }
        let gap = column[e.from];
        label_width[gap] = label_width[gap].max(edge_label(snap, index).chars().count());
    }
    let gap_width: Vec<usize> = label_width.iter().map(|w| w + 5).collect();

    let mut column_x = vec![0usize; column_count];
    for col in 1..column_count {
        column_x[col] = column_x[col - 1] + box_width[col - 1] + gap_width[col - 1];
    }
    let width = column_x[column_count - 1] + box_width[column_count - 1];
    let row_count = rows_used.iter().max().copied().unwrap_or(1).max(1);
    let height = row_count * (BOX_HEIGHT + ROW_GAP) - ROW_GAP;
    let mut canvas = Canvas::new(width, height);

    for id in 0..snap.nodes.len() {
        let col = column[id];
        draw_box(
            &mut canvas,
            snap,
            id,
            column_x[col],
            row_top(row_of[id]),
            box_width[col],
        );
    }

    let routes: Vec<([usize; 3], [usize; 2], String)> = snap
        .edges
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.from < snap.nodes.len() && e.to < snap.nodes.len() && column[e.to] > column[e.from]
        })
        .map(|(index, e)| {
            let gap = column[e.from];
            let from_x = column_x[gap] + box_width[gap];
            (
                [
                    from_x,
                    from_x + 2 + label_width[gap],
                    column_x[column[e.to]] - 1,
                ],
                [center_y(row_of[e.from]), center_y(row_of[e.to])],
                edge_label(snap, index),
            )
        })
        .collect();
    // Routes first, labels after: a sibling branch's line would otherwise be
    // drawn over a label already placed on the run they share.
    for (xs, ys, _) in &routes {
        draw_route(&mut canvas, *xs, *ys);
    }
    for (xs, ys, label) in &routes {
        place_label(&mut canvas, xs[0] + 2, ys[0], label);
    }

    canvas.into_spans()
}

fn row_top(row: usize) -> usize {
    row * (BOX_HEIGHT + ROW_GAP)
}

fn center_y(row: usize) -> usize {
    row_top(row) + 2
}

fn draw_box(
    canvas: &mut Canvas,
    snap: &TelemetrySnapshot,
    id: usize,
    x: usize,
    y: usize,
    width: usize,
) {
    let (name, stat) = box_text(snap, id);
    let inner = width.saturating_sub(2);
    let bar: String = core::iter::repeat_n('─', inner).collect();
    canvas.write(x, y, &format!("┌{bar}┐"), Some(id));
    canvas.write(
        x,
        y + 1,
        &format!("│ {name:<w$} │", w = inner - 2),
        Some(id),
    );
    canvas.write(
        x,
        y + 2,
        &format!("│ {stat:<w$} │", w = inner - 2),
        Some(id),
    );
    canvas.write(x, y + 3, &format!("└{bar}┘"), Some(id));
}

/// Route one edge: out of the source box, down or up in the turn column, then
/// into the destination box with an arrow head. `xs` is
/// `[source edge, turn column, arrow]`, `ys` is `[source row, dest row]`.
fn draw_route(canvas: &mut Canvas, xs: [usize; 3], ys: [usize; 2]) {
    let [from_x, turn_x, arrow_x] = xs;
    let [from_y, to_y] = ys;
    // A turning edge stops one short: the turn column takes a corner, and a
    // dash written there first would merge into a tee that connects nothing.
    let straight_to = if from_y == to_y { turn_x } else { turn_x - 1 };
    for x in from_x..=straight_to.min(arrow_x) {
        canvas.connector(x, from_y, '─');
    }
    if from_y != to_y {
        let (top, bottom) = (from_y.min(to_y), from_y.max(to_y));
        for y in top + 1..bottom {
            canvas.connector(turn_x, y, '│');
        }
        canvas.connector(turn_x, from_y, if to_y > from_y { '┐' } else { '┘' });
        canvas.connector(turn_x, to_y, if to_y > from_y { '└' } else { '┌' });
    }
    for x in turn_x + 1..arrow_x {
        canvas.connector(x, to_y, '─');
    }
    canvas.connector(arrow_x, to_y, '►');
}

/// Put the edge's packet count on its connector, sliding off the line when a
/// sibling edge already took that spot (a tee's branches share the same run).
fn place_label(canvas: &mut Canvas, x: usize, y: usize, label: &str) {
    let len = label.chars().count();
    if len == 0 {
        return;
    }
    for candidate in [y, y + 1, y.wrapping_sub(1), y + 2] {
        let free = (x..x + len).all(|cx| {
            !canvas.owned(cx, candidate)
                && matches!(canvas.get(cx, candidate), Some(' ') | Some('─'))
        });
        if free {
            canvas.write(x, candidate, label, None);
            return;
        }
    }
}

// ----------------------------------------------------------------- rendering

fn render(
    frame: &mut Frame,
    view: View,
    scroll: usize,
    snap: &TelemetrySnapshot,
    logs: &[&OwnedLogRecord],
) {
    let journey_rows = snap.journey.as_ref().map_or(0, |j| j.stages.len());
    let mut constraints = vec![Constraint::Length(3)];
    match view {
        View::Graph => constraints.push(Constraint::Min(6)),
        View::Tables => {
            // Each pane hugs its rows and the slack collects in the spacer, so a
            // three-element pipeline does not leave a table stretched over half
            // the screen.
            constraints.push(Constraint::Length(pane_height(snap.nodes.len())));
            constraints.push(Constraint::Length(pane_height(snap.edges.len())));
            if journey_rows > 0 {
                constraints.push(Constraint::Length(pane_height(journey_rows) - 1));
            }
            constraints.push(Constraint::Min(0));
        }
    }
    if !logs.is_empty() {
        constraints.push(Constraint::Length(logs.len() as u16 + 2));
    }
    constraints.push(Constraint::Length(1));

    let areas = Layout::vertical(constraints).split(frame.area());
    let mut next = areas.iter().copied();
    let mut take = || next.next().unwrap_or(Rect::ZERO);

    frame.render_widget(
        Paragraph::new(header_line(snap, view)).block(Block::bordered().title(" g2g pipeline ")),
        take(),
    );
    match view {
        View::Graph => render_graph(frame, take(), scroll, snap),
        View::Tables => {
            render_nodes(frame, take(), snap);
            render_edges(frame, take(), snap);
            if journey_rows > 0 {
                render_journey(frame, take(), snap);
            }
            take(); // spacer
        }
    }
    if !logs.is_empty() {
        render_logs(frame, take(), logs);
    }
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " q quit   g graph/tables   <-/-> scroll graph ",
            Style::new().fg(Color::DarkGray),
        ))),
        take(),
    );
}

/// Rows plus the two border lines and the header line.
fn pane_height(rows: usize) -> u16 {
    u16::try_from(rows).unwrap_or(u16::MAX).saturating_add(3)
}

fn header_style() -> Style {
    Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
}

fn render_nodes(frame: &mut Frame, area: Rect, snap: &TelemetrySnapshot) {
    let rows = node_rows(snap).into_iter().map(|n| {
        Row::new(vec![
            n.name,
            n.role.to_string(),
            format_opt_ns(n.proc_p50_ns),
            format_opt_ns(n.proc_p99_ns),
            format_opt_ns(n.transit_p50_ns),
            format_opt_ns(n.push_wait_p50_ns),
            format!("{}/{}%", n.fill_mean_pct, n.fill_max_pct),
        ])
        .style(Style::new().fg(fill_color(n.fill_max_pct)))
    });
    // The trailing filler keeps the columns beside each other instead of the
    // name column stretching across the terminal.
    let widths = [
        Constraint::Length(24),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(9),
        Constraint::Min(0),
    ];
    let table = Table::new(rows, widths)
        .header(
            Row::new(vec![
                "element", "role", "proc p50", "proc p99", "wait p50", "push p50", "fill a/m",
            ])
            .style(header_style()),
        )
        .block(Block::bordered().title(" elements "));
    frame.render_widget(table, area);
}

fn render_edges(frame: &mut Frame, area: Rect, snap: &TelemetrySnapshot) {
    let rows = edge_rows(snap).into_iter().map(|e| {
        let drops = if e.drops > 0 {
            format!("{}", e.drops)
        } else {
            "-".to_string()
        };
        Row::new(vec![
            format!("{} -> {}", e.from, e.to),
            e.caps,
            format!("{}", e.packets),
            format_bytes(e.bytes),
            drops,
        ])
        .style(Style::new().fg(if e.drops > 0 {
            Color::Red
        } else {
            Color::Reset
        }))
    });
    let widths = [
        Constraint::Length(28),
        Constraint::Min(20),
        Constraint::Length(10),
        Constraint::Length(10),
        Constraint::Length(6),
    ];
    let table = Table::new(rows, widths)
        .header(Row::new(vec!["link", "caps", "packets", "bytes", "drops"]).style(header_style()))
        .block(Block::bordered().title(" links "));
    frame.render_widget(table, area);
}

fn render_journey(frame: &mut Frame, area: Rect, snap: &TelemetrySnapshot) {
    let Some(journey) = snap.journey.as_ref() else {
        return;
    };
    let scale = journey
        .stages
        .iter()
        .map(|s| s.wait_ns + s.work_ns + s.blocked_ns)
        .max()
        .unwrap_or(0);
    let bar_columns = usize::from(area.width.saturating_sub(46)).min(40);
    let lines: Vec<Line> = journey
        .stages
        .iter()
        .map(|s| {
            let [wait, work, blocked] =
                bar_widths(s.wait_ns, s.work_ns, s.blocked_ns, scale, bar_columns);
            Line::from(vec![
                Span::raw(format!(" {:<14} ", truncate(&s.name, 14))),
                Span::styled(bar(wait), Style::new().fg(Color::Blue)),
                Span::styled(bar(work), Style::new().fg(Color::Green)),
                Span::styled(bar(blocked), Style::new().fg(Color::Red)),
                Span::raw(format!(
                    " wait {} work {} blocked {}",
                    format_ns(s.wait_ns),
                    format_ns(s.work_ns),
                    format_ns(s.blocked_ns)
                )),
            ])
        })
        .collect();
    let title = format!(
        " frame {} : total {} vs {} queueing floor (capacity {}){} ",
        journey.sequence,
        format_ns(journey.total_ns),
        format_ns(journey.floor_ns),
        journey.capacity,
        if journey.truncated { ", partial" } else { "" },
    );
    frame.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(title)),
        area,
    );
}

fn render_graph(frame: &mut Frame, area: Rect, scroll: usize, snap: &TelemetrySnapshot) {
    let health: Vec<Color> = snap
        .nodes
        .iter()
        .map(|n| fill_color(n.latency.as_ref().map_or(0, |l| l.fill_max_pct)))
        .collect();
    let lines: Vec<Line> = graph_spans(snap)
        .into_iter()
        .map(|spans| {
            Line::from(
                scrolled(&spans, scroll)
                    .into_iter()
                    .map(|s| {
                        let style = match s.node.and_then(|id| health.get(id)) {
                            Some(color) => Style::new().fg(*color),
                            None => Style::new().fg(Color::DarkGray),
                        };
                        Span::styled(s.text, style)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" topology ")),
        area,
    );
}

fn render_logs(frame: &mut Frame, area: Rect, logs: &[&OwnedLogRecord]) {
    let lines: Vec<Line> = logs
        .iter()
        .map(|r| {
            let instance = r.instance.as_deref().unwrap_or(&r.category);
            Line::from(vec![
                Span::styled(
                    format!("{:<5} ", r.level.as_str()),
                    Style::new().fg(level_color(r.level)),
                ),
                Span::raw(format!("{instance} {}", r.message)),
            ])
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(Block::bordered().title(" log ")),
        area,
    );
}

/// Drop the leftmost `skip` columns of a graph line, so a pipeline wider than
/// the terminal can be panned.
fn scrolled(spans: &[GraphSpan], skip: usize) -> Vec<GraphSpan> {
    let mut left = skip;
    let mut out = Vec::new();
    for span in spans {
        let len = span.text.chars().count();
        if left >= len {
            left -= len;
            continue;
        }
        out.push(GraphSpan {
            text: span.text.chars().skip(left).collect(),
            node: span.node,
        });
        left = 0;
    }
    out
}

fn bar(cells: usize) -> String {
    core::iter::repeat_n('█', cells).collect()
}

fn truncate(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

fn fill_color(fill_max_pct: u8) -> Color {
    match fill_max_pct {
        90..=u8::MAX => Color::Red,
        60..=89 => Color::Yellow,
        _ => Color::Green,
    }
}

fn level_color(level: LogLevel) -> Color {
    match level {
        LogLevel::Error => Color::Red,
        LogLevel::Warn | LogLevel::Fixme => Color::Yellow,
        _ => Color::DarkGray,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::metrics::LatencySnapshot;
    use g2g_core::runtime::{
        EdgeCounts, EdgeInfo, ElementLatency, FrameJourney, JourneyStage, NodeTelemetry,
    };

    fn latency(name: &str, proc_p50: u64, proc_p99: u64, fill_max: u8) -> ElementLatency {
        ElementLatency {
            name: name.to_string(),
            proc: stats(2, proc_p50, proc_p99),
            push_wait: stats(0, 0, 0),
            transit: stats(2, 500, 900),
            fill_mean_pct: fill_max / 2,
            fill_max_pct: fill_max,
            presentation: None,
        }
    }

    fn stats(count: u64, p50_ns: u64, p99_ns: u64) -> LatencySnapshot {
        LatencySnapshot {
            count,
            mean_ns: p50_ns,
            max_ns: p99_ns,
            p50_ns,
            p95_ns: p99_ns,
            p99_ns,
        }
    }

    fn node(
        id: usize,
        name: &str,
        role: NodeRole,
        latency: Option<ElementLatency>,
    ) -> NodeTelemetry {
        NodeTelemetry {
            id,
            name: name.to_string(),
            role,
            latency,
        }
    }

    fn edge(from: usize, to: usize, packets: u64, drops: u64) -> EdgeInfo {
        EdgeInfo {
            from,
            to,
            caps: Some("video/x-raw".to_string()),
            observed_caps: None,
            counts: EdgeCounts {
                packets,
                bytes: packets * 1024,
                drops,
                blocked_ns: 0,
            },
        }
    }

    /// src -> scale -> sink, the shape almost every pipeline starts as.
    fn linear_snapshot() -> TelemetrySnapshot {
        TelemetrySnapshot {
            uptime_ns: 2_500_000_000,
            nodes: vec![
                node(0, "videotestsrc0", NodeRole::Source, None),
                node(
                    1,
                    "videoscale0",
                    NodeRole::Transform,
                    Some(latency("videoscale0", 1_200_000, 3_400_000, 75)),
                ),
                node(
                    2,
                    "fakesink0",
                    NodeRole::Sink,
                    Some(latency("fakesink0", 900, 1_500, 20)),
                ),
            ],
            edges: vec![edge(0, 1, 42, 0), edge(1, 2, 41, 3)],
            journey: None,
        }
    }

    /// A tee fanning into two sinks: the branch column stacks two rows.
    fn branched_snapshot() -> TelemetrySnapshot {
        TelemetrySnapshot {
            uptime_ns: 1_000_000_000,
            nodes: vec![
                node(0, "videotestsrc0", NodeRole::Source, None),
                node(1, "tee0", NodeRole::Tee, None),
                node(
                    2,
                    "fakesink0",
                    NodeRole::Sink,
                    Some(latency("fakesink0", 800, 1_000, 10)),
                ),
                node(
                    3,
                    "fakesink1",
                    NodeRole::Sink,
                    Some(latency("fakesink1", 800, 1_000, 95)),
                ),
            ],
            edges: vec![edge(0, 1, 10, 0), edge(1, 2, 10, 0), edge(1, 3, 9, 1)],
            journey: None,
        }
    }

    fn text_of(lines: &[Vec<GraphSpan>]) -> Vec<String> {
        lines
            .iter()
            .map(|spans| spans.iter().map(|s| s.text.as_str()).collect())
            .collect()
    }

    #[test]
    fn node_rows_carry_measured_values_only() {
        let rows = node_rows(&linear_snapshot());
        assert_eq!(rows[0].name, "videotestsrc0");
        assert_eq!(rows[0].role, "source");
        assert_eq!(rows[0].proc_p50_ns, None, "no probe on a source");
        assert_eq!(rows[1].proc_p50_ns, Some(1_200_000));
        assert_eq!(rows[1].proc_p99_ns, Some(3_400_000));
        assert_eq!(rows[1].transit_p50_ns, Some(500));
        assert_eq!(rows[1].push_wait_p50_ns, None, "push_wait never recorded");
        assert_eq!(rows[1].fill_max_pct, 75);
    }

    #[test]
    fn edge_rows_name_endpoints_and_prefer_observed_caps() {
        let mut snap = linear_snapshot();
        snap.edges[1].observed_caps = Some("video/x-raw, width=(int)320".to_string());
        let rows = edge_rows(&snap);
        assert_eq!(rows[0].from, "videotestsrc0");
        assert_eq!(rows[0].to, "videoscale0");
        assert_eq!(rows[0].caps, "video/x-raw");
        assert_eq!(rows[1].caps, "video/x-raw, width=(int)320");
        assert_eq!((rows[1].packets, rows[1].drops), (41, 3));
    }

    #[test]
    fn header_counts_sink_deliveries_and_drops() {
        let line = header_line(&linear_snapshot(), View::Tables);
        assert!(line.contains("delivered 41"), "{line}");
        assert!(line.contains("drops 3"), "{line}");
        assert!(line.contains("uptime 2.5 s"), "{line}");
    }

    #[test]
    fn bar_widths_split_by_share_and_never_overflow() {
        assert_eq!(bar_widths(500, 500, 0, 1_000, 20), [10, 10, 0]);
        // A tiny segment still gets one cell so it stays visible.
        assert_eq!(bar_widths(1, 999, 0, 1_000, 10), [1, 9, 0]);
        let widths = bar_widths(400, 400, 400, 1_200, 7);
        assert!(widths.iter().sum::<usize>() <= 7, "{widths:?}");
        assert_eq!(bar_widths(1, 1, 1, 0, 10), [0, 0, 0]);
    }

    #[test]
    fn durations_and_sizes_read_in_scaled_units() {
        assert_eq!(format_ns(900), "900 ns");
        assert_eq!(format_ns(1_500), "1.5 us");
        assert_eq!(format_ns(2_500_000), "2.5 ms");
        assert_eq!(format_ns(3_000_000_000), "3.00 s");
        assert_eq!(format_opt_ns(None), "-");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_uptime(90_000_000_000), "1m 30s");
    }

    #[test]
    fn quit_keys_and_view_toggle() {
        let press = |code, modifiers| action_for(KeyEvent::new(code, modifiers));
        assert_eq!(press(KeyCode::Char('q'), KeyModifiers::NONE), Action::Quit);
        assert_eq!(press(KeyCode::Esc, KeyModifiers::NONE), Action::Quit);
        assert_eq!(
            press(KeyCode::Char('c'), KeyModifiers::CONTROL),
            Action::Quit
        );
        assert_eq!(
            press(KeyCode::Char('g'), KeyModifiers::NONE),
            Action::ToggleView
        );
        assert_eq!(
            press(KeyCode::Char('x'), KeyModifiers::NONE),
            Action::Ignore
        );
    }

    #[test]
    fn graph_draws_a_linear_chain_in_columns() {
        let snap = linear_snapshot();
        let lines = graph_spans(&snap);
        let text = text_of(&lines);
        let joined = text.join("\n");
        assert!(joined.contains("videotestsrc0"), "{joined}");
        assert!(joined.contains("videoscale0"), "{joined}");
        assert!(joined.contains("fakesink0"), "{joined}");
        assert!(joined.contains("p99 3.4 ms"), "stat line: {joined}");
        // One row, so every arrow lands on the boxes' center line.
        let center = &text[center_y(0)];
        assert_eq!(center.matches('►').count(), 2, "{center}");
        assert!(center.contains("42"), "packet count on the link: {center}");
        assert_eq!(text.len(), BOX_HEIGHT, "a single row of boxes");
        // The columns march right: each name starts further along than the last.
        let name_line = &text[1];
        let src = name_line.find("videotestsrc0").unwrap();
        let scale = name_line.find("videoscale0").unwrap();
        let sink = name_line.find("fakesink0").unwrap();
        assert!(src < scale && scale < sink, "{name_line}");
    }

    #[test]
    fn graph_stacks_a_tee_branch_on_its_own_row() {
        let snap = branched_snapshot();
        let lines = graph_spans(&snap);
        let text = text_of(&lines);
        let joined = text.join("\n");
        assert_eq!(text.len(), 2 * (BOX_HEIGHT + ROW_GAP) - ROW_GAP);
        // The two sinks share a column but not a row.
        assert!(text[1].contains("fakesink0"), "{joined}");
        assert!(
            text[BOX_HEIGHT + ROW_GAP + 1].contains("fakesink1"),
            "{joined}"
        );
        // The branch turns down out of the run the straight-through link keeps
        // using, so that turn is a tee piece, and lands on the lower row.
        assert!(
            text[center_y(0)].contains('┬'),
            "turn out of the tee: {joined}"
        );
        assert!(
            text[center_y(1)].contains("└─►"),
            "turn into the branch: {joined}"
        );
        assert_eq!(joined.matches('►').count(), 3, "one arrow per link");
        // The leaky branch shows its drops beside the packet count.
        assert!(joined.contains("9 -1"), "drop decoration: {joined}");
        // Boxes own their cells; connectors do not.
        let owner = lines[1]
            .iter()
            .find(|s| s.text.contains("videotestsrc0"))
            .and_then(|s| s.node);
        assert_eq!(owner, Some(0));
    }

    /// Two sources into a muxer: the lower one has to turn back up, and its
    /// corner must connect only the arms it uses.
    #[test]
    fn graph_routes_a_fan_in_upwards() {
        let snap = TelemetrySnapshot {
            uptime_ns: 1_000_000_000,
            nodes: vec![
                node(0, "audiotestsrc0", NodeRole::Source, None),
                node(1, "videotestsrc0", NodeRole::Source, None),
                node(
                    2,
                    "mp4mux0",
                    NodeRole::Muxer,
                    Some(latency("mp4mux0", 700, 900, 30)),
                ),
            ],
            edges: vec![edge(0, 2, 5, 0), edge(1, 2, 6, 0)],
            journey: None,
        };
        let text = text_of(&graph_spans(&snap));
        let joined = text.join("\n");
        // The straight link keeps its run through the turn column, so that cell
        // is a tee; the branch's own corner turns up out of the lower row.
        assert!(text[center_y(0)].contains('┬'), "{joined}");
        assert!(text[center_y(1)].contains('┘'), "{joined}");
        assert!(!joined.contains('┴'), "no dangling arm: {joined}");
        assert_eq!(joined.matches('►').count(), 1, "both links share one arrow");
    }

    #[test]
    fn crossing_connectors_keep_both_arms() {
        assert_eq!(merge_line('─', '┐'), '┬');
        assert_eq!(merge_line('─', '│'), '┼');
        assert_eq!(merge_line('│', '─'), '┼');
        assert_eq!(merge_line('─', '└'), '┴');
        assert_eq!(merge_line(' ', '─'), '─');
        assert_eq!(merge_line('4', '─'), '4', "an edge label is not overdrawn");
    }

    #[test]
    fn graph_scroll_drops_leading_columns() {
        let spans = vec![
            GraphSpan {
                text: "┌────┐".to_string(),
                node: Some(0),
            },
            GraphSpan {
                text: "──►".to_string(),
                node: None,
            },
        ];
        let out = scrolled(&spans, 7);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "─►");
        assert_eq!(out[0].node, None);
        assert_eq!(scrolled(&spans, 0), spans);
    }

    #[test]
    fn journey_pane_scales_against_the_widest_stage() {
        let mut snap = linear_snapshot();
        snap.journey = Some(FrameJourney {
            sequence: 7,
            stages: vec![
                JourneyStage {
                    node: 1,
                    name: "videoscale0".to_string(),
                    wait_ns: 1_000,
                    work_ns: 3_000,
                    blocked_ns: 0,
                },
                JourneyStage {
                    node: 2,
                    name: "fakesink0".to_string(),
                    wait_ns: 500,
                    work_ns: 500,
                    blocked_ns: 0,
                },
            ],
            total_ns: 5_000,
            frame_period_ns: 33_000,
            capacity: 4,
            floor_ns: 264_000,
            truncated: false,
        });
        let journey = snap.journey.as_ref().unwrap();
        let scale = journey
            .stages
            .iter()
            .map(|s| s.wait_ns + s.work_ns + s.blocked_ns)
            .max()
            .unwrap();
        let wide = bar_widths(1_000, 3_000, 0, scale, 40);
        let narrow = bar_widths(500, 500, 0, scale, 40);
        assert_eq!(wide.iter().sum::<usize>(), 40, "the widest stage fills it");
        assert_eq!(narrow.iter().sum::<usize>(), 10, "a quarter of the time");
    }
}
