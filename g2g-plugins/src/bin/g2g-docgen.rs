//! `g2g-docgen`: generate the searchable element reference (`docs/elements.html`)
//! from the standard registry, the web counterpart of `g2g-inspect`. Every card
//! is built from [`g2g_core::runtime::Registry::describe_all`], so the page is
//! the same source of truth as the CLI dump and never drifts by hand.
//!
//! Usage:
//!   g2g-docgen [out.html]     # default: docs/elements.html
//!
//! Run it with a broad feature set so the catalog is complete, e.g.
//!   cargo run -p g2g-plugins --features linux-full --bin g2g-docgen
//! The listing reflects the elements compiled into the build (a feature-gated
//! element absent from the build is absent from the page), exactly like
//! `g2g-inspect`; platform-only elements (Media Foundation, VideoToolbox,
//! MediaCodec) appear when generated on that platform.

use std::fmt::Write as _;
use std::process;

use g2g_core::runtime::ElementDoc;
use g2g_plugins::registry::default_registry;

/// Minimal HTML-text escape for content interpolated into the page.
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// A lowercased haystack for the client-side filter: name, long name, klass,
/// description, and every property name, so any of them matches a query.
fn search_key(d: &ElementDoc) -> String {
    let mut k = String::new();
    k.push_str(&d.name);
    k.push(' ');
    k.push_str(&d.long_name);
    k.push(' ');
    k.push_str(&d.klass);
    k.push(' ');
    k.push_str(&d.description);
    for p in &d.properties {
        k.push(' ');
        k.push_str(&p.name);
    }
    k.to_lowercase()
}

/// The role badge's CSS modifier class, so sources / elements / muxers colour
/// distinctly.
fn role_class(role: &str) -> &'static str {
    if role.starts_with("source") {
        "role-src"
    } else if role.starts_with("muxer") {
        "role-mux"
    } else {
        "role-el"
    }
}

fn render_property(out: &mut String, p: &g2g_core::runtime::PropertyDoc) {
    let flags = match (p.readable, p.writable) {
        (true, true) => "read / write",
        (true, false) => "read-only",
        (false, true) => "write-only",
        (false, false) => "",
    };
    let _ = write!(
        out,
        "<div class=\"prop\"><code class=\"pn\">{}</code> <span class=\"pt\">{}</span>",
        esc(&p.name),
        esc(&p.type_label)
    );
    if !flags.is_empty() {
        let _ = write!(out, " <span class=\"pf\">{flags}</span>");
    }
    let _ = write!(out, "<div class=\"pb\">{}</div>", esc(&p.blurb));
    let mut facts: Vec<String> = Vec::new();
    if let Some((min, max)) = &p.range {
        facts.push(format!("range {} – {}", esc(min), esc(max)));
    }
    if let Some(vals) = &p.enum_values {
        facts.push(format!("values: {}", esc(vals)));
    }
    if let Some(def) = &p.default {
        facts.push(format!("default: <code>{}</code>", esc(def)));
    }
    if !facts.is_empty() {
        let _ = write!(
            out,
            "<div class=\"pfacts\">{}</div>",
            facts.join(" &middot; ")
        );
    }
    out.push_str("</div>");
}

fn render_card(out: &mut String, d: &ElementDoc) {
    let _ = write!(
        out,
        "<article class=\"el-card\" data-search=\"{}\" id=\"el-{}\">",
        esc(&search_key(d)),
        esc(&d.name)
    );
    let _ = write!(
        out,
        "<div class=\"el-head\"><h3><a href=\"#el-{}\">{}</a></h3>\
         <span class=\"badge {}\">{}</span></div>",
        esc(&d.name),
        esc(&d.name),
        role_class(&d.role),
        esc(&d.role)
    );
    if !d.klass.is_empty() {
        let _ = write!(out, "<div class=\"klass\">{}</div>", esc(&d.klass));
    }
    if !d.description.is_empty() {
        let _ = write!(out, "<p class=\"desc\">{}</p>", esc(&d.description));
    }
    // Caps (sources / muxers) or pad templates (transforms / sinks).
    if let Some(caps) = &d.caps {
        let _ = write!(
            out,
            "<div class=\"seg\"><span class=\"seg-l\">Caps</span><pre>{}</pre></div>",
            esc(caps)
        );
    }
    if !d.pads.is_empty() {
        out.push_str("<div class=\"seg\"><span class=\"seg-l\">Pads</span><pre>");
        for (i, pad) in d.pads.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&esc(pad));
        }
        out.push_str("</pre></div>");
    }
    // Properties.
    if d.properties.is_empty() {
        out.push_str("<div class=\"seg\"><span class=\"seg-l\">Properties</span><span class=\"none\">none</span></div>");
    } else {
        out.push_str(
            "<div class=\"seg\"><span class=\"seg-l\">Properties</span><div class=\"props\">",
        );
        for p in &d.properties {
            render_property(out, p);
        }
        out.push_str("</div></div>");
    }
    out.push_str("</article>");
}

fn render(docs: &[ElementDoc]) -> String {
    let mut cards = String::new();
    for d in docs {
        render_card(&mut cards, d);
    }
    // Self-contained page: inline CSS + JS, no external requests (GitHub Pages
    // serves it directly). Palette matches the landing page.
    format!(
        r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>glass2glass — Element Reference</title>
<meta name="description" content="Searchable reference for every glass2glass element: role, caps, pad templates, and properties. Generated from the registry.">
<style>
:root {{
  --bg:#0A0E14; --bg2:#11161E; --bg3:#161C26; --card:#1A2230; --border:#232D3D;
  --border2:#38445A; --accent:#22D3EE; --purple:#A78BFA; --green:#34D399; --amber:#FBBF24;
  --t1:#F1F5F9; --t2:#94A3B8; --t3:#64748B; --mut:#475569;
}}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:var(--bg); color:var(--t2); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; line-height:1.5; }}
code, pre {{ font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace; }}
a {{ color:var(--accent); text-decoration:none; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 24px; }}
header.top {{ border-bottom:1px solid var(--border); background:var(--bg2); position:sticky; top:0; z-index:10; }}
header.top .wrap {{ display:flex; align-items:center; justify-content:space-between; height:60px; }}
.logo {{ color:var(--t1); font-weight:700; font-size:1.05rem; }}
.logo b {{ color:var(--accent); }}
.hero {{ padding:44px 0 8px; }}
.hero h1 {{ color:var(--t1); font-size:2rem; font-weight:800; letter-spacing:-0.02em; }}
.hero p {{ margin-top:10px; max-width:720px; font-size:0.95rem; }}
.tools {{ margin-top:22px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
#q {{
  flex:1; min-width:240px; background:var(--bg3); border:1px solid var(--border2); color:var(--t1);
  border-radius:10px; padding:12px 14px; font-size:0.95rem; outline:none;
}}
#q:focus {{ border-color:var(--accent); }}
#count {{ color:var(--t3); font-size:0.85rem; white-space:nowrap; }}
.grid {{ margin:26px 0 60px; display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:16px; }}
.el-card {{ background:var(--bg2); border:1px solid var(--border); border-radius:14px; padding:20px 20px 18px; }}
.el-card:target {{ border-color:var(--accent); }}
.el-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:10px; }}
.el-head h3 {{ font-size:1.02rem; }}
.el-head h3 a {{ color:var(--t1); font-family:'JetBrains Mono',monospace; }}
.badge {{ font-size:0.6rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; padding:3px 8px; border-radius:100px; white-space:nowrap; }}
.role-src {{ background:rgba(52,211,153,0.15); color:var(--green); }}
.role-el  {{ background:rgba(34,211,238,0.15); color:var(--accent); }}
.role-mux {{ background:rgba(167,139,250,0.15); color:var(--purple); }}
.klass {{ margin-top:4px; font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--t3); }}
.desc {{ margin-top:10px; font-size:0.85rem; color:var(--t2); }}
.seg {{ margin-top:14px; }}
.seg-l {{ display:block; font-size:0.63rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase; color:var(--t3); margin-bottom:6px; }}
.seg pre {{ background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:9px 11px; font-size:0.7rem; color:var(--t2); overflow-x:auto; white-space:pre; }}
.none {{ font-size:0.8rem; color:var(--mut); font-style:italic; }}
.props {{ display:flex; flex-direction:column; gap:10px; }}
.prop {{ border-left:2px solid var(--border2); padding-left:11px; }}
.pn {{ color:var(--accent); font-size:0.78rem; }}
.pt {{ font-size:0.66rem; color:var(--purple); text-transform:uppercase; letter-spacing:0.04em; margin-left:4px; }}
.pf {{ font-size:0.64rem; color:var(--mut); margin-left:4px; }}
.pb {{ font-size:0.8rem; color:var(--t2); margin-top:3px; }}
.pfacts {{ font-size:0.72rem; color:var(--t3); margin-top:3px; }}
.pfacts code {{ color:var(--amber); }}
.empty {{ grid-column:1/-1; text-align:center; color:var(--t3); padding:60px 0; }}
footer {{ border-top:1px solid var(--border); padding:24px 0; color:var(--t3); font-size:0.8rem; }}
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <span class="logo"><b>g2g</b> · Element Reference</span>
  <nav><a href="index.html">← Home</a></nav>
</div></header>

<div class="wrap">
  <section class="hero">
    <h1>Element Reference</h1>
    <p>Every element in this build &mdash; role, caps, pad templates, and properties. The same data <code>g2g-inspect &lt;name&gt;</code> prints, generated straight from the registry. Search by name, class, description, or property.</p>
    <div class="tools">
      <input id="q" type="search" placeholder="Search elements (e.g. rtsp, encoder, bitrate, videotestsrc)…" autocomplete="off" autofocus>
      <span id="count"></span>
    </div>
  </section>

  <main class="grid" id="grid">
{cards}
    <div class="empty" id="empty" hidden>No element matches that search.</div>
  </main>
</div>

<footer><div class="wrap">Generated by <code>g2g-docgen</code> from the element registry. Reflects the features compiled into this build, like <code>g2g-inspect</code>.</div></footer>

<script>
(function() {{
  var q = document.getElementById('q');
  var cards = Array.prototype.slice.call(document.querySelectorAll('.el-card'));
  var count = document.getElementById('count');
  var empty = document.getElementById('empty');
  var total = cards.length;
  function apply() {{
    var term = q.value.trim().toLowerCase();
    var shown = 0;
    for (var i = 0; i < cards.length; i++) {{
      var hit = !term || cards[i].getAttribute('data-search').indexOf(term) !== -1;
      cards[i].hidden = !hit;
      if (hit) shown++;
    }}
    empty.hidden = shown !== 0;
    count.textContent = term ? (shown + ' of ' + total) : (total + ' elements');
  }}
  // Prefill from ?q=… so a filtered view is a shareable URL, and keep the URL
  // in sync as the user types.
  var initial = new URLSearchParams(location.search).get('q');
  if (initial) q.value = initial;
  q.addEventListener('input', function() {{
    var u = new URL(location);
    if (q.value) u.searchParams.set('q', q.value); else u.searchParams.delete('q');
    history.replaceState(null, '', u);
    apply();
  }});
  apply();
}})();
</script>
</body>
</html>
"##,
        cards = cards
    )
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "docs/elements.html".to_string());
    let reg = default_registry();
    let mut docs = reg.describe_all();
    docs.sort_by(|a, b| a.name.cmp(&b.name));
    let html = render(&docs);
    if let Err(e) = std::fs::write(&out_path, html) {
        eprintln!("g2g-docgen: cannot write {out_path}: {e}");
        process::exit(1);
    }
    println!("g2g-docgen: wrote {out_path} ({} elements)", docs.len());
}
