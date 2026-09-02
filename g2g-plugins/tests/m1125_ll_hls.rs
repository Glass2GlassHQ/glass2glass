//! M1125 low-latency HLS: against a local LL-HLS server (a routing HTTP server
//! that publishes one partial segment every `PART_MS` and holds a playlist
//! request carrying `_HLS_msn` / `_HLS_part` until that part exists), `HlsSrc`
//! reloads with the delivery directives, fetches each `#EXT-X-PART` and emits it
//! as its own `DataFrame`. The same server with `low-latency=false` proves the
//! opposite: no directives, one frame per whole segment.

#![cfg(feature = "hls")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PushOutcome};
use g2g_plugins::hlssrc::HlsSrc;

/// Media model: every part is `PART_BYTES` of its own absolute index, so a whole
/// segment is byte-identical to its parts concatenated and the delivered stream
/// can be checked without knowing where playback joined.
const PARTS_PER_SEGMENT: usize = 5;
const PART_BYTES: usize = 64;
const PART_MS: u64 = 40;
const PART_SECS: f64 = PART_MS as f64 / 1000.0;
const SEGMENT_SECS: f64 = PART_SECS * PARTS_PER_SEGMENT as f64;
/// Parts published before the run starts, and in total: the server publishes one
/// more every `PART_MS` until the total, then closes the playlist with ENDLIST.
const INITIAL_PARTS: usize = PARTS_PER_SEGMENT * 3;
const TOTAL_PARTS: usize = PARTS_PER_SEGMENT * 6;
const INIT_BYTES: usize = 32;
const INIT_PATH: &str = "/init.mp4";
const PLAYLIST_PATH: &str = "/ll.m3u8";

fn part_body(index: usize) -> Vec<u8> {
    vec![(index % 251) as u8; PART_BYTES]
}

fn init_body() -> Vec<u8> {
    vec![0xa5; INIT_BYTES]
}

/// Every part in publication order: the delivered stream is always a suffix of
/// this, whatever part playback joined at.
fn full_media() -> Vec<u8> {
    (0..TOTAL_PARTS).flat_map(part_body).collect()
}

/// The playlist with `published` parts available. Complete segments carry their
/// `#EXTINF` and URI, the run of parts after the last one is the segment still
/// being produced, and the playlist closes once everything is published.
/// `gap_segments` leading `#EXT-X-GAP` entries stand where a live packager pads
/// a freshly started playlist; the parts then belong to the segments after them.
fn playlist(published: usize, gap_segments: usize) -> String {
    let mut text = format!(
        "#EXTM3U\n\
         #EXT-X-VERSION:9\n\
         #EXT-X-TARGETDURATION:1\n\
         #EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK={:.5}\n\
         #EXT-X-PART-INF:PART-TARGET={PART_SECS:.5}\n\
         #EXT-X-MEDIA-SEQUENCE:0\n\
         #EXT-X-MAP:URI=\"{INIT_PATH}\"\n",
        PART_SECS * 3.0
    );
    for gap in 0..gap_segments {
        text.push_str(&format!(
            "#EXT-X-GAP\n#EXTINF:{SEGMENT_SECS:.5},\n/gap{gap}.mp4\n"
        ));
    }
    for index in 0..published {
        let independent = if index % PARTS_PER_SEGMENT == 0 {
            ",INDEPENDENT=YES"
        } else {
            ""
        };
        text.push_str(&format!(
            "#EXT-X-PART:DURATION={PART_SECS:.5},URI=\"/part{index}.mp4\"{independent}\n"
        ));
        if index % PARTS_PER_SEGMENT == PARTS_PER_SEGMENT - 1 {
            let segment = index / PARTS_PER_SEGMENT;
            text.push_str(&format!("#EXTINF:{SEGMENT_SECS:.5},\n/seg{segment}.mp4\n"));
        }
    }
    if published >= TOTAL_PARTS {
        text.push_str("#EXT-X-ENDLIST\n");
    }
    text
}

/// What the server counted while the run played.
#[derive(Debug)]
struct ServerCounts {
    /// Playlist requests carrying the `_HLS_msn` / `_HLS_part` delivery directives.
    blocking: Arc<AtomicUsize>,
    /// Playlist requests without them (the timed-reload path).
    plain: Arc<AtomicUsize>,
    /// Whole-segment fetches (`/segN.mp4`).
    segment_fetches: Arc<AtomicUsize>,
    /// Partial-segment fetches (`/partN.mp4`).
    part_fetches: Arc<AtomicUsize>,
}

/// The absolute part index a `_HLS_msn` / `_HLS_part` pair names, with the
/// playlist's leading gap segments subtracted out of the sequence number.
fn requested_part(query: &str, gap_segments: usize) -> Option<usize> {
    let value = |name: &str| {
        query
            .split('&')
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
            .and_then(|v| v.parse::<usize>().ok())
    };
    Some(value("_HLS_msn")?.saturating_sub(gap_segments) * PARTS_PER_SEGMENT + value("_HLS_part")?)
}

/// A local LL-HLS origin: one part every `PART_MS` from the wall clock, and a
/// playlist request with delivery directives is held until the part it names is
/// published (or the playlist ends). With `honour_directives` off it advertises
/// `CAN-BLOCK-RELOAD` but answers every request at once, the misbehaviour the
/// client has to notice. Returns the playlist URL and the counters.
fn serve_low_latency(honour_directives: bool) -> (String, ServerCounts) {
    serve(honour_directives, 0, INITIAL_PARTS)
}

/// [`serve_low_latency`] with leading gap padding and a chosen number of parts
/// already published at start (small enough and the client joins inside the
/// first real segment, right after the padding, the fresh-packager shape).
fn serve(
    honour_directives: bool,
    gap_segments: usize,
    initial_parts: usize,
) -> (String, ServerCounts) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let counts = ServerCounts {
        blocking: Arc::new(AtomicUsize::new(0)),
        plain: Arc::new(AtomicUsize::new(0)),
        segment_fetches: Arc::new(AtomicUsize::new(0)),
        part_fetches: Arc::new(AtomicUsize::new(0)),
    };
    let (blocking, plain) = (counts.blocking.clone(), counts.plain.clone());
    let (segment_fetches, part_fetches) =
        (counts.segment_fetches.clone(), counts.part_fetches.clone());
    let start = Instant::now();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { break };
            let (blocking, plain) = (blocking.clone(), plain.clone());
            let (segment_fetches, part_fetches) = (segment_fetches.clone(), part_fetches.clone());
            // A held playlist request must not stall a part fetch behind it.
            thread::spawn(move || {
                let mut req = Vec::new();
                let mut byte = [0u8; 1];
                while stream.read(&mut byte).unwrap_or(0) == 1 {
                    req.push(byte[0]);
                    if req.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                let line = String::from_utf8_lossy(&req);
                let target = line.split_whitespace().nth(1).unwrap_or("").to_string();
                let (path, query) = match target.split_once('?') {
                    Some((p, q)) => (p, q),
                    None => (target.as_str(), ""),
                };
                let published = || {
                    (initial_parts + (start.elapsed().as_millis() as u64 / PART_MS) as usize)
                        .min(TOTAL_PARTS)
                };
                let body: Vec<u8> = if path == PLAYLIST_PATH {
                    match requested_part(query, gap_segments) {
                        Some(wanted) => {
                            blocking.fetch_add(1, Ordering::SeqCst);
                            // Hold the response until that part exists.
                            while honour_directives
                                && published() <= wanted
                                && published() < TOTAL_PARTS
                            {
                                thread::sleep(Duration::from_millis(PART_MS / 4));
                            }
                            playlist(published(), gap_segments).into_bytes()
                        }
                        None => {
                            plain.fetch_add(1, Ordering::SeqCst);
                            playlist(published(), gap_segments).into_bytes()
                        }
                    }
                } else if path == INIT_PATH {
                    init_body()
                } else if let Some(index) = path
                    .strip_prefix("/part")
                    .and_then(|p| p.strip_suffix(".mp4"))
                    .and_then(|n| n.parse::<usize>().ok())
                {
                    part_fetches.fetch_add(1, Ordering::SeqCst);
                    part_body(index)
                } else if let Some(segment) = path
                    .strip_prefix("/seg")
                    .and_then(|p| p.strip_suffix(".mp4"))
                    .and_then(|n| n.parse::<usize>().ok())
                {
                    segment_fetches.fetch_add(1, Ordering::SeqCst);
                    (0..PARTS_PER_SEGMENT)
                        .flat_map(|p| part_body(segment * PARTS_PER_SEGMENT + p))
                        .collect()
                } else {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    return;
                };
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&body);
            });
        }
    });
    (format!("http://127.0.0.1:{port}{PLAYLIST_PATH}"), counts)
}

/// Records each emitted payload so delivery granularity (one frame per part) is
/// observable, not just the concatenated bytes.
#[derive(Default)]
struct CaptureSink {
    frames: Vec<Vec<u8>>,
    eos: bool,
}

impl OutputSink for CaptureSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

impl CaptureSink {
    /// Everything after the `#EXT-X-MAP` init segment, concatenated.
    fn media(&self) -> Vec<u8> {
        self.frames
            .iter()
            .skip(1)
            .flat_map(|f| f.iter().copied())
            .collect()
    }
}

async fn play(src: HlsSrc) -> CaptureSink {
    let mut src = src;
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();
    sink
}

#[tokio::test]
async fn parts_are_fetched_and_emitted_one_frame_each() {
    let (url, counts) = serve_low_latency(true);
    // No prebuffer: the run emits each part as it arrives, so frame granularity
    // reflects what was fetched.
    let sink = play(HlsSrc::new(url).with_prebuffer_ms(0)).await;

    assert!(sink.eos, "ENDLIST terminates the run");
    assert_eq!(sink.frames[0], init_body(), "EXT-X-MAP init emitted first");

    // Every published byte from the join point to the end, in order.
    let media = sink.media();
    let all = full_media();
    assert!(
        all.ends_with(&media),
        "delivered {} bytes are the tail of the published stream",
        media.len()
    );
    assert!(
        media.len() >= (TOTAL_PARTS - INITIAL_PARTS) * PART_BYTES,
        "at least everything published after the run started: {} bytes",
        media.len()
    );

    // The live edge is delivered part by part, not segment by segment.
    let part_frames = sink.frames[1..]
        .iter()
        .filter(|f| f.len() == PART_BYTES)
        .count();
    assert!(
        part_frames >= TOTAL_PARTS - INITIAL_PARTS,
        "one DataFrame per part at the live edge, got {part_frames}"
    );
    assert!(
        counts.part_fetches.load(Ordering::SeqCst) >= TOTAL_PARTS - INITIAL_PARTS,
        "parts fetched individually"
    );

    // The reload used the delivery directives rather than polling on a timer:
    // roughly one held request per part, allowing for the ones that come back
    // carrying two (the server publishes while the client is fetching).
    let blocking = counts.blocking.load(Ordering::SeqCst);
    assert!(
        blocking >= (TOTAL_PARTS - INITIAL_PARTS) * 2 / 3,
        "the run is driven by blocking reloads, got {blocking}"
    );
    assert_eq!(
        counts.plain.load(Ordering::SeqCst),
        1,
        "only the initial playlist load is a plain GET"
    );
}

/// A server that advertises `CAN-BLOCK-RELOAD` but answers immediately with the
/// playlist the client already has: the run must notice and go back to timed
/// reloads instead of spinning on held requests that never hold.
#[tokio::test]
async fn a_server_ignoring_the_directives_falls_back_to_timed_reloads() {
    let (url, counts) = serve_low_latency(false);
    let sink = play(
        HlsSrc::new(url)
            .with_prebuffer_ms(0)
            .with_reload_interval_ms(PART_MS),
    )
    .await;

    assert!(sink.eos);
    assert!(
        full_media().ends_with(&sink.media()),
        "byte-exact, in order"
    );
    let (blocking, plain) = (
        counts.blocking.load(Ordering::SeqCst),
        counts.plain.load(Ordering::SeqCst),
    );
    assert!(
        blocking < TOTAL_PARTS - INITIAL_PARTS,
        "blocking reloads stopped rather than one per part, got {blocking}"
    );
    assert!(
        plain > 1,
        "the run reloaded on the timer after giving up on the directives"
    );
}

/// A freshly started packager pads the playlist front with `#EXT-X-GAP`
/// segments and publishes its first real segment part by part. Joining there
/// puts the delivery cursor right after the padding, and the padding must not
/// reset it on later scans: each part is fetched and emitted exactly once
/// (the regression was the gap step-over zeroing `next_part` every reload,
/// replaying the live segment from part 0).
#[tokio::test]
async fn gap_padding_does_not_replay_the_live_segment() {
    let (url, counts) = serve(true, 3, 2);
    let sink = play(HlsSrc::new(url).with_prebuffer_ms(0)).await;

    assert!(sink.eos);
    assert!(
        full_media().ends_with(&sink.media()),
        "byte-exact, in order, no replays"
    );
    let part_fetches = counts.part_fetches.load(Ordering::SeqCst);
    assert!(
        part_fetches <= TOTAL_PARTS,
        "every part at most once, got {part_fetches} fetches for {TOTAL_PARTS} parts"
    );
}

// ---------------------------------------------------------------------------
// Live measurement against a real LL-HLS server (mediamtx).
// ---------------------------------------------------------------------------

const DEFAULT_LIVE_URL: &str = "http://localhost:8888/avpattern/index.m3u8";
const DEFAULT_LIVE_SECS: u64 = 60;
/// How often the measuring poller reloads the playlist to timestamp a newly
/// published part or segment. Well under one part duration, so the recorded
/// publication time is close to the real one.
const POLL_MS: u64 = 20;

/// When each published resource first appeared in the playlist, keyed by the
/// hash of its bytes (which is how an emitted `DataFrame` is matched back to
/// it). The entries already in the playlist when polling started carry no time:
/// they were published before the measurement began.
type Publications = std::collections::HashMap<u64, (Option<Instant>, u32)>;

fn hash_bytes(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Resolve a master playlist down to the media playlist `HlsSrc` will pick.
async fn live_media_url(client: &reqwest::Client, url: &str) -> String {
    let text = client.get(url).send().await.unwrap().text().await.unwrap();
    match g2g_plugins::hls::parse(&text).unwrap() {
        g2g_plugins::hls::Playlist::Media(_) => url.to_string(),
        g2g_plugins::hls::Playlist::Master(master) => {
            let variant = master.select(None).expect("a variant");
            let base = url.rsplit_once('/').map(|(d, _)| d).unwrap_or(url);
            format!("{base}/{}", variant.uri)
        }
    }
}

/// Poll the media playlist and record when each part / segment first appears,
/// with the bytes behind it, until `stop` is set. This is the reference clock
/// the emission latency is measured against.
async fn poll_publications(
    client: reqwest::Client,
    media_url: String,
    published: Arc<std::sync::Mutex<Publications>>,
    stop: Arc<std::sync::atomic::AtomicBool>,
) {
    let mut known: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut priming = true;
    while !stop.load(Ordering::SeqCst) {
        let Ok(resp) = client.get(&media_url).send().await else {
            continue;
        };
        let Ok(text) = resp.text().await else {
            continue;
        };
        let Ok(g2g_plugins::hls::Playlist::Media(playlist)) = g2g_plugins::hls::parse(&text) else {
            break;
        };
        let mut fresh: Vec<(String, u32)> = Vec::new();
        for segment in &playlist.segments {
            for part in &segment.parts {
                if known.insert(part.uri.clone()) {
                    fresh.push((part.uri.clone(), part.duration_ms));
                }
            }
            if !segment.incomplete() && known.insert(segment.uri.clone()) {
                fresh.push((segment.uri.clone(), segment.duration_ms));
            }
        }
        for (uri, duration_ms) in fresh {
            let at = (!priming).then(Instant::now);
            let base = media_url.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
            let Ok(resp) = client.get(format!("{base}/{uri}")).send().await else {
                continue;
            };
            let Ok(bytes) = resp.bytes().await else {
                continue;
            };
            published
                .lock()
                .unwrap()
                .insert(hash_bytes(&bytes), (at, duration_ms));
        }
        priming = false;
        tokio::time::sleep(Duration::from_millis(POLL_MS)).await;
    }
}

/// Each emitted frame by content hash and the moment it was pushed. The match
/// against the publication times happens after the run: a part can reach the
/// sink before the poller has even seen it in the playlist, and looking up at
/// emission time would silently drop exactly those (the fastest) frames.
#[derive(Default)]
struct LatencySink {
    emissions: Vec<(u64, Instant)>,
}

impl OutputSink for LatencySink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            if let Some(bytes) = frame.domain.as_system_slice() {
                self.emissions.push((hash_bytes(bytes), Instant::now()));
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn percentile(sorted: &[i64], pct: usize) -> i64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(sorted.len() - 1) * pct / 100]
}

/// Play `url` for `secs`, returning every emitted frame's content hash and
/// emission instant.
async fn play_live(url: &str, low_latency: bool, secs: u64) -> Vec<(u64, Instant)> {
    let mut src = HlsSrc::new(url);
    if !low_latency {
        src = src.without_low_latency();
    }
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut sink = LatencySink::default();
    let _ = tokio::time::timeout(Duration::from_secs(secs), src.run(&mut sink)).await;
    sink.emissions
}

/// How far behind publication the emitted media ran, in ms, sorted. `delays` is
/// publication to emission, negative where the run beat the measuring poller to
/// the part. `ages` adds the media each frame carries, since the oldest sample in
/// a frame is that much older than its newest, which is what a whole segment
/// costs against a part.
fn latencies(emissions: &[(u64, Instant)], published: &Publications) -> (Vec<i64>, Vec<i64>) {
    let (mut delays, mut ages) = (Vec::new(), Vec::new());
    for (hash, emitted) in emissions {
        let Some((Some(at), duration_ms)) = published.get(hash) else {
            continue;
        };
        let delay = emitted.saturating_duration_since(*at).as_millis() as i64
            - at.saturating_duration_since(*emitted).as_millis() as i64;
        delays.push(delay);
        ages.push(delay + i64::from(*duration_ms));
    }
    delays.sort_unstable();
    ages.sort_unstable();
    (delays, ages)
}

/// Live comparison against a real LL-HLS packager: the same stream played with
/// the low-latency path on and off, reporting how far behind publication each
/// ran. Parts should cut it to about one part duration.
#[tokio::test]
#[ignore = "needs a live LL-HLS server (set G2G_LL_HLS_URL, default local mediamtx)"]
async fn live_low_latency_run_is_nearer_the_edge_than_whole_segments() {
    let url = std::env::var("G2G_LL_HLS_URL").unwrap_or_else(|_| DEFAULT_LIVE_URL.to_string());
    let secs: u64 = std::env::var("G2G_LL_HLS_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_LIVE_SECS);
    let client = reqwest::Client::new();
    let media_url = live_media_url(&client, &url).await;
    eprintln!("measuring {media_url} for {secs}s per mode");

    let published: Arc<std::sync::Mutex<Publications>> = Arc::default();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let poller = tokio::spawn(poll_publications(
        client,
        media_url,
        published.clone(),
        stop.clone(),
    ));
    // Let the poller time a few publications before the first run joins.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let ll_emissions = play_live(&url, true, secs).await;
    let plain_emissions = play_live(&url, false, secs).await;
    stop.store(true, Ordering::SeqCst);
    let _ = poller.await;

    let published = published.lock().unwrap();
    let (ll_frames, plain_frames) = (ll_emissions.len(), plain_emissions.len());
    let (ll_delays, ll_ages) = latencies(&ll_emissions, &published);
    let (plain_delays, plain_ages) = latencies(&plain_emissions, &published);

    for (mode, frames, delays, ages) in [
        ("low-latency", ll_frames, &ll_delays, &ll_ages),
        ("whole-segment", plain_frames, &plain_delays, &plain_ages),
    ] {
        eprintln!(
            "{mode}: {frames} frames, {} timed; publication to emission p50 {} ms p90 {} ms; \
             oldest sample in frame p50 {} ms p90 {} ms",
            delays.len(),
            percentile(delays, 50),
            percentile(delays, 90),
            percentile(ages, 50),
            percentile(ages, 90),
        );
    }

    assert!(
        ll_frames > 0 && plain_frames > 0,
        "frames flowed in both modes"
    );
    assert!(
        !ll_delays.is_empty(),
        "low-latency frames matched publications"
    );
    assert!(
        !plain_delays.is_empty(),
        "whole-segment frames matched publications"
    );
    assert!(
        percentile(&ll_ages, 50) < percentile(&plain_ages, 50),
        "parts run nearer the live edge than whole segments"
    );
    assert!(
        ll_frames > plain_frames,
        "parts arrive as more, smaller frames"
    );
}

/// The same server, with low latency turned off: the parts and the blocking
/// reload are ignored, so this is the shape the M1121 whole-segment path has.
#[tokio::test]
async fn low_latency_off_keeps_whole_segments_and_timed_reloads() {
    let (url, counts) = serve_low_latency(false);
    let sink = play(
        HlsSrc::new(url)
            .without_low_latency()
            .with_prebuffer_ms(0)
            .with_reload_interval_ms(PART_MS),
    )
    .await;

    assert!(sink.eos);
    assert_eq!(
        counts.blocking.load(Ordering::SeqCst),
        0,
        "no delivery directives without low latency"
    );
    assert_eq!(
        counts.part_fetches.load(Ordering::SeqCst),
        0,
        "no partial segments fetched"
    );
    assert!(
        counts.segment_fetches.load(Ordering::SeqCst) > 0,
        "whole segments instead"
    );
    // Whole segments, so no frame is a single part.
    assert!(
        sink.frames[1..]
            .iter()
            .all(|f| f.len() % PART_BYTES == 0 && f.len() >= PART_BYTES * PARTS_PER_SEGMENT),
        "every media frame is a whole segment"
    );
    let media = sink.media();
    assert!(full_media().ends_with(&media), "byte-exact, in order");
}
