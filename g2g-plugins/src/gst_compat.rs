//! GStreamer-to-g2g porting helpers (M200): a `gst`-element-name map and a
//! launch-line linter that turns parse failures into porting guidance.
//!
//! These back `g2g-inspect --gst <name>` and `g2g-launch`'s explain-on-error,
//! and are the programmatic surface a porting tool builds on. They complement
//! [`parse_launch`] (the authoritative parse):
//! the linter runs it and enriches the first error with a gst->g2g suggestion,
//! so porting is fix-and-rerun rather than decode-the-error.

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::runtime::{parse_launch, ParseError, Registry};

/// What a GStreamer element name maps to in g2g.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GstEquivalent {
    /// A registered g2g element (possibly via an alias) or a launch keyword
    /// (`tee`, `queue`, `decodebin`, ...) uses this exact name.
    Available,
    /// g2g has an equivalent under a different name (the suggestion). The target
    /// may be feature-gated, so it is advice, not a guarantee it is compiled in.
    Renamed(&'static str),
    /// No g2g element; the hint explains the closest path.
    Unsupported(&'static str),
    /// g2g has this exact element, but the cargo feature that compiles it (the
    /// payload) is off in this build.
    NotCompiled(&'static str),
    /// Unknown, but close enough to a name this build does have (the payload)
    /// to be a spelling mistake.
    DidYouMean(&'static str),
    /// Unknown to both the registry and the gst-compat table: cannot advise.
    Unknown,
}

/// Launch keywords the parser handles that are not registry elements.
static LAUNCH_KEYWORDS: &[&str] = &[
    "decodebin",
    "encodebin",
    "encodebin2",
    "transcodebin",
    "uridecodebin",
    "playbin",
    "queue",
    "queue2",
    "tee",
];

/// gst element name -> guidance, for names NOT registered under the same name.
/// Registered names (incl. aliases like `avdec_h264` -> `ffmpegdec`) resolve to
/// `Available` before this table is consulted; keep this for the gst names that
/// have no same-name g2g element. Extend freely.
static GST_MAP: &[(&str, GstEquivalent)] = &[
    ("x264enc", GstEquivalent::Unsupported(
        "software H.264 encode (`x264enc`, libx264) needs the `ffmpeg` feature on Linux; \
         otherwise `nvenc` (NVIDIA), `mfencode` (Windows), or encode AV1/VP8/VP9 with `av1enc`/`vpxenc`",
    )),
    ("x265enc", GstEquivalent::Unsupported("no software H.265 encoder; use `nvenc` (NVIDIA HEVC) or `av1enc`")),
    ("theoraenc", GstEquivalent::Unsupported("no Theora encoder; use `vpxenc` (VP8/VP9) or `av1enc`")),
    ("avdec_h264", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_h265", GstEquivalent::Renamed("ffmpegdec")),
    // NVIDIA hardware codecs map to the native NVDEC / NVENC elements (their
    // features are CI-excluded but the names are the direct equivalents, like the
    // VAAPI rows below); `ffmpegdec`'s cuvid backend is the software-feature fallback.
    ("nvh264dec", GstEquivalent::Renamed("nvdec")),
    ("nvh265dec", GstEquivalent::Renamed("nvdec")),
    ("nvh264enc", GstEquivalent::Renamed("nvenc")),
    ("nvh265enc", GstEquivalent::Renamed("nvenc")),
    ("vaapih264dec", GstEquivalent::Renamed("vaapidec")),
    ("vah264dec", GstEquivalent::Renamed("vaapidec")),
    ("vp8enc", GstEquivalent::Renamed("vpxenc")),
    ("vp9enc", GstEquivalent::Renamed("vpxenc")),
    ("jpegenc", GstEquivalent::Renamed("mjpegenc")),
    ("jpegdec", GstEquivalent::Renamed("mjpegdec")),
    // gst's IVF reader is a parser; here the same job is a demuxer, since IVF
    // frames the elementary stream it carries.
    ("ivfparse", GstEquivalent::Renamed("ivfdemux")),
    // `avenc_aac` is a g2g element name itself, so it needs no row; the other
    // gst AAC encoder names point at it.
    ("faac", GstEquivalent::Renamed("avenc_aac")),
    ("souphttpsrc", GstEquivalent::Renamed("httpsrc")),
    // appsrc / appsink are registered elements, so gst_equivalent resolves them
    // to Available before this table; no row is needed (and an Unsupported one
    // would contradict reality).
    ("rtph264depay", GstEquivalent::Unsupported("RTP depayloading is built into `udpsrc` / `rtspsrc`")),
    // The auto-capture aliases only speak when a capture element is compiled in,
    // and they never fall back to a test source.
    ("autovideosrc", GstEquivalent::Unsupported(
        "no capture source is compiled into this build; the alias picks the first of `v4l2src`, \
         `libcamerasrc`, `pipewirevideosrc`, `avfvideosrc`, `mfvideosrc`, `camera2src`, so build \
         one of their features (it never falls back to `videotestsrc`)",
    )),
    ("autoaudiosrc", GstEquivalent::Unsupported(
        "no capture source is compiled into this build; the alias picks the first of `alsasrc`, \
         `pulsesrc`, `pipewiresrc`, `coreaudiosrc`, `wasapisrc`, `avfaudiosrc`, `aaudiosrc`, so \
         build one of their features (it never falls back to `audiotestsrc`)",
    )),
    // The libav / plain decoder names g2g answers with one ffmpeg decoder. Each
    // is also a registry alias (like `avdec_h264`), so the row only speaks when
    // the `ffmpeg` feature is off. Codecs confirmed against `ffmpegdec`'s
    // `VideoCodec` match and `ffmpegaudiodec`'s `AudioFormat` match.
    ("vp8dec", GstEquivalent::Renamed("ffmpegdec")),
    ("vp9dec", GstEquivalent::Renamed("ffmpegdec")),
    ("mpeg2dec", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_vp8", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_vp9", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_mpeg2video", GstEquivalent::Renamed("ffmpegdec")),
    ("avdec_mpeg4", GstEquivalent::Renamed("ffmpegdec")),
    ("mpg123audiodec", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("flacdec", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("a52dec", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("faad", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("fdkaacdec", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("avdec_mp3", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("avdec_aac", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("avdec_ac3", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("avdec_flac", GstEquivalent::Renamed("ffmpegaudiodec")),
    ("theoradec", GstEquivalent::Unsupported(
        "no Theora decoder; `oggdemux` reads the container but only its Opus / Vorbis / \
         FLAC audio decodes, so transcode Theora video to VP9 (`vpxenc`) or AV1 first",
    )),
    ("dtsdec", GstEquivalent::Unsupported(
        "no DTS decoder; `ffmpegaudiodec` covers AAC, MP2, MP3, AC-3 and FLAC only",
    )),
    // Adaptive streaming: g2g's clients fetch the manifest themselves, so there is
    // no separate demuxer to place after an HTTP source.
    ("hlsdemux", GstEquivalent::Unsupported(
        "`hlssrc` fetches the playlist and the segments itself; replace `souphttpsrc ! hlsdemux` \
         with `hlssrc location=<playlist url>`",
    )),
    ("hlsdemux2", GstEquivalent::Unsupported(
        "`hlssrc` fetches the playlist and the segments itself; replace `souphttpsrc ! hlsdemux2` \
         with `hlssrc location=<playlist url>`",
    )),
    ("dashdemux", GstEquivalent::Unsupported(
        "`dashsrc` fetches the MPD and the segments itself; replace `souphttpsrc ! dashdemux` \
         with `dashsrc location=<mpd url>`",
    )),
    ("dashdemux2", GstEquivalent::Unsupported(
        "`dashsrc` fetches the MPD and the segments itself; replace `souphttpsrc ! dashdemux2` \
         with `dashsrc location=<mpd url>`",
    )),
    ("mssdemux", GstEquivalent::Unsupported(
        "no Smooth Streaming client; `hlssrc` (HLS) and `dashsrc` (DASH) are the adaptive sources",
    )),
    ("mssdemux2", GstEquivalent::Unsupported(
        "no Smooth Streaming client; `hlssrc` (HLS) and `dashsrc` (DASH) are the adaptive sources",
    )),
    // SRT: g2g has one source (always a listener) and one sink (always a caller),
    // so the gst client/server split collapses onto the property set.
    ("srtclientsrc", GstEquivalent::Unsupported(
        "`srtsrc` is the only SRT source and always listens; set `address` and `port` for the \
         listen socket (there is no caller-mode receive)",
    )),
    ("srtserversrc", GstEquivalent::Unsupported(
        "use `srtsrc`; `address` and `port` select the listen socket",
    )),
    ("srtclientsink", GstEquivalent::Unsupported(
        "use `srtsink`; `host` and `port` select the listener it calls",
    )),
    ("srtserversink", GstEquivalent::Unsupported(
        "`srtsink` is the only SRT sink and always calls out to a listener (`host` / `port`); \
         there is no listen-mode send",
    )),
    // Launch keywords, not elements: the parser accepts `queue` / `queue2` /
    // `decodebin` / `uridecodebin` / `playbin` and nothing else, so the gst
    // variants name the keyword they should be spelled as.
    ("multiqueue", GstEquivalent::Renamed("queue")),
    ("decodebin3", GstEquivalent::Renamed("decodebin")),
    ("parsebin", GstEquivalent::Renamed("decodebin")),
    ("uridecodebin3", GstEquivalent::Renamed("uridecodebin")),
    ("urisourcebin", GstEquivalent::Renamed("uridecodebin")),
    ("playbin3", GstEquivalent::Renamed("playbin")),
    // `rtmp2sink` is an alias (`location` on both); `rtmpsrc` instead listens for
    // a publisher on `address` / `port`, so the source name only gets a pointer.
    ("rtmp2src", GstEquivalent::Renamed("rtmpsrc")),
    // gst's X11 screen grab; g2g's screen source is the ScreenCaptureKit one and
    // shares none of ximagesrc's region / pointer properties.
    ("ximagesrc", GstEquivalent::Renamed("screencapturesrc")),
    // `equalizer-3bands` / `spectrum` / `clockoverlay` / `splitmuxsink` are
    // registered elements, so gst_equivalent resolves them to Available before this
    // table; only the wider N-band equalizers need a pointer.
    ("equalizer-10bands", GstEquivalent::Renamed("equalizer-3bands")),
    ("equalizer-nbands", GstEquivalent::Renamed("equalizer-3bands")),
    // gst's `ccextractor` passes the video through and puts the captions on a
    // second source pad; `ccextract` consumes the access units, so it sits on a
    // tee branch instead of in line.
    ("ccextractor", GstEquivalent::Unsupported(
        "`ccextract` mines the same CEA-608 / CEA-708 `cc_data`, but it consumes the access \
         units instead of passing the video through: tee the parser output, one branch to the \
         decoder and one to `ccextract`",
    )),
    // The line-21 VBI waveform is not written or sliced anywhere in g2g, so the
    // captions have to leave the raw picture and travel beside it.
    ("line21encoder", GstEquivalent::Unsupported(
        "no line-21 VBI waveform writer; carry the captions beside the video instead, as a \
         `Caps::ClosedCaption` stream `cccombiner` attaches to the frames",
    )),
    ("line21decoder", GstEquivalent::Unsupported(
        "no line-21 VBI waveform slicer; take the captions from the bitstream with `ccextract`, \
         or from a container caption track",
    )),
    // The ONVIF metadata track arrives already depayloaded (retina concatenates
    // the RTP packets to the marker bit), and g2g draws detections with one
    // overlay whatever produced them.
    ("onvifmetadatadepay", GstEquivalent::Unsupported(
        "no standalone depayloader; `rtspsrcn onvif-metadata=true` gives the whole \
         `tt:MetadataStream` document on a pad of its own, gzip already inflated",
    )),
    ("onvifmetadatapay", GstEquivalent::Unsupported(
        "no ONVIF metadata payloader: g2g reads an analytics stream, it does not serve one",
    )),
    ("onvifmetadataoverlay", GstEquivalent::Unsupported(
        "attach the analytics with `onvifmetadatacombiner`, then draw them with \
         `analyticsoverlay`",
    )),
    // WebRTC is one element per role rather than one bin with request pads, and
    // the SRTP / DTLS elements have no standalone counterpart at all.
    ("webrtcbin", GstEquivalent::Unsupported(
        "no WebRTC bin; publish with `webrtcsink` (one stream) or `webrtcsessionsink` (one \
         session, one pad per stream), and receive with `webrtcsrc` (WHEP) or \
         `webrtcwhepsessionsrc` (one pad per track)",
    )),
    ("dtlsenc", GstEquivalent::Unsupported(WEBRTC_SECURITY_HINT)),
    ("dtlsdec", GstEquivalent::Unsupported(WEBRTC_SECURITY_HINT)),
    // Subtitle overlays: g2g splits by cue type (text or bitmap) and takes the
    // decoder as a separate element, so no one name covers gst's bins.
    ("subtitleoverlay", GstEquivalent::Unsupported(
        "pick the overlay the cues need: `textoverlay` for timed text (video on input 0, \
         `Text{Utf8}` on input 1) and `subpictureoverlay` for bitmap subpictures",
    )),
    ("dvbsuboverlay", GstEquivalent::Unsupported(
        "decode the subpictures first: `dvbsubdec ! subpictureoverlay`, with the video on the \
         overlay's other input",
    )),
    ("dvdspu", GstEquivalent::Unsupported(
        "decode the subpictures first: `vobsubdec ! subpictureoverlay`, with the video on the \
         overlay's other input",
    )),
    ("ssaparse", GstEquivalent::Unsupported(
        "`subparse` reads SRT and WebVTT only; convert SSA / ASS cues to SRT first",
    )),
    ("subparse_typefind", GstEquivalent::Unsupported(
        "a gst typefind function, not an element; `typefind` sniffs the stream and `subparse` \
         parses the SRT / WebVTT cues",
    )),
    // Inference lives in `g2g-ml`, which the standard registry does not carry.
    ("onnxinference", GstEquivalent::Unsupported(
        "ONNX inference is the `g2g-ml` crate's `ortinfer` element (its `ort` feature), not a \
         `g2g-plugins` one",
    )),
    ("streamiddemux", GstEquivalent::Unsupported(
        "no stream-id splitter; a demuxer's output pads are already one per stream \
         (`tsdemux name=d  d.video_0 ! ...`), and `output-selector` switches one input between \
         output pads at run time",
    )),
    ("dashsink", GstEquivalent::Unsupported(
        "no DASH packager; `hlssink` is the packaging sink (`... ! tsmux ! hlssink`) and \
         `dashsrc` is the DASH client",
    )),
    ("sdpsrc", GstEquivalent::Unsupported(SDP_HINT)),
    ("sdpdemux", GstEquivalent::Unsupported(SDP_HINT)),
    // `av1dec` is also a registry alias (like `avdec_av1`), so this row only
    // speaks when neither AV1 decoder feature is built.
    ("av1dec", GstEquivalent::Renamed("dav1ddec")),
    ("dynudpsink", GstEquivalent::Unsupported(
        "`udpsink` sends to destinations fixed at configure time (`host` / `port`, or the \
         `clients` list); it reads no per-buffer destination",
    )),
    ("bin", GstEquivalent::Unsupported(BIN_HINT)),
    ("pipeline", GstEquivalent::Unsupported(BIN_HINT)),
];

/// The answer for the gst DTLS element names that carry no media of their own.
const WEBRTC_SECURITY_HINT: &str =
    "there is no bare DTLS tunnel; `dtlssrtpenc` / `dtlssrtpdec` run the handshake and key the \
     RTP and RTCP it protects, the `webrtc*` elements (`webrtcsink`, `webrtcsessionsink`, \
     `webrtcsrc`, `webrtcwhepsessionsrc`) do the same inside a session, and `srtpenc` / \
     `srtpdec` protect packets with a key you already have";

/// The answer for the gst SDP source names.
const SDP_HINT: &str =
    "`udpsrc sdp=<document text or .sdp path>` reads the description and takes its codec, \
     geometry, frame rate and receive port from it";

/// The answer for gst's `bin` / `pipeline` grouping keywords.
const BIN_HINT: &str = "g2g flattens bins, and the parser does not accept the `bin.( ... )` \
                        grouping syntax: write the elements in line";

/// The GStreamer plugins whose element names come in whole families (90 `rtp*pay`
/// / `rtp*depay` names, 48 `gl*` names, one `*tv` name per effectv filter), where
/// one answer serves every member. Each row is `(prefix, suffix, guidance)` and
/// matches a name that starts with `prefix` and ends with `suffix`; an empty
/// half matches anything. First match wins, so a narrower row goes above the
/// wider one it sits inside.
static GST_FAMILY_MAP: &[(&str, &str, GstEquivalent)] = &[
    // RTP: payloading and depayloading are inside the transports, not separate
    // elements, and the session / jitter / RTX / FEC knobs are their properties.
    (
        "rtp",
        "depay",
        GstEquivalent::Unsupported(
            "RTP depayloading is built into `udpsrc`, `rtspsrc` and `webrtcsrc`; \
         drop the depayloader and read from the transport directly",
        ),
    ),
    (
        "rtp",
        "pay",
        GstEquivalent::Unsupported(
            "RTP payloading is built into `udpsink`, `rtspserversink` and `webrtcsink`; \
         drop the payloader and set `payload-type` / `max-payload` on the transport",
        ),
    ),
    (
        "rtp",
        "",
        GstEquivalent::Unsupported(
            "RTP session, jitter buffer, retransmission and FEC are `udpsrc` properties \
         (`jitter-latency`, `jitter-depth`, `rtcp-rr-interval`, `nack`, `rtx-payload-type`, \
         `rtx-apt`, `fec-payload-type`, `flexfec-payload-type`) and `udpsink` properties \
         (`rtcp-sr-interval`, `retransmit`, `retx-capacity`, `rtx-payload-type`, \
         `fec-columns`, `fec-rows`, `fec-payload-type`)",
        ),
    ),
    // GPU: g2g has no OpenGL elements at all; wgpu is the GPU path.
    (
        "gl",
        "",
        GstEquivalent::Unsupported(
            "no OpenGL elements; the GPU path is wgpu: `wgpusink` presents, `wgpucompositor` \
         mixes, and `dmabuftowgpu` / `wgputodmabuf` move frames in and out",
        ),
    ),
    (
        "cuda",
        "",
        GstEquivalent::Unsupported(
            "no generic CUDA filters; `nvdec` / `nvenc` are the CUDA codecs and \
         `localcudasrc` / `localcudasink` share CUDA memory between processes",
        ),
    ),
    (
        "vulkan",
        "",
        GstEquivalent::Unsupported(
            "`vulkanvideodec` is the only Vulkan element (decode); present with `wgpusink`",
        ),
    ),
    ("nv", "dec", GstEquivalent::Renamed("nvdec")),
    ("nv", "enc", GstEquivalent::Renamed("nvenc")),
    // Only the decoders map; a bare `va` prefix would also catch `valve`.
    ("vaapi", "dec", GstEquivalent::Renamed("vaapidec")),
    ("va", "dec", GstEquivalent::Renamed("vaapidec")),
    (
        "vaapi",
        "enc",
        GstEquivalent::Unsupported("no VA-API encoder; use `ffmpegenc`, `nvenc` or `vpxenc`"),
    ),
    (
        "va",
        "enc",
        GstEquivalent::Unsupported("no VA-API encoder; use `ffmpegenc`, `nvenc` or `vpxenc`"),
    ),
    (
        "ladspa",
        "",
        GstEquivalent::Unsupported(
            "no LADSPA host; the built-in audio filters are `volume`, `audiopanorama`, \
         `equalizer-3bands`, `level` and `cutter`",
        ),
    ),
    // effectv: every one of its filters is named `<something>tv`.
    (
        "",
        "tv",
        GstEquivalent::Unsupported("no video-effects plugin"),
    ),
    (
        "qml",
        "",
        GstEquivalent::Unsupported(
            "no toolkit sinks; render with `wgpusink` or pull frames out with `appsink`",
        ),
    ),
    (
        "gtk",
        "",
        GstEquivalent::Unsupported(
            "no toolkit sinks; render with `wgpusink` or pull frames out with `appsink`",
        ),
    ),
    (
        "decklink",
        "",
        GstEquivalent::Unsupported("no DeckLink support"),
    ),
];

/// The geometrictransform filters, whose names share neither a prefix nor a
/// suffix, as full-name [`GST_FAMILY_MAP`] prefixes with one shared answer.
static GST_GEOMETRIC_TRANSFORM_NAMES: &[&str] = &[
    "bulge",
    "circle",
    "diffuse",
    "fisheye",
    "kaleidoscope",
    "marble",
    "mirror",
    "perspective",
    "pinch",
    "rotate",
    "sphere",
    "square",
    "stretch",
    "tunnel",
    "twirl",
    "waterripple",
];

/// The answer for every [`GST_GEOMETRIC_TRANSFORM_NAMES`] name.
const GEOMETRIC_TRANSFORM_HINT: &str =
    "no geometric-distortion filters; `videoflip`, `videocrop`, `videobox` and `videoscale` \
     are the geometry elements";

/// The family guidance for `gst_name`, `None` when no family covers it.
///
/// A name that is itself a g2g element declines every family rule, so a build
/// with `nvdec` or `vaapidec` switched off still answers with its cargo feature
/// instead of a hint pointing back at the same name.
fn family_equivalent(gst_name: &str) -> Option<GstEquivalent> {
    if crate::registry::required_feature(gst_name).is_some() {
        return None;
    }
    if GST_GEOMETRIC_TRANSFORM_NAMES.contains(&gst_name) {
        return Some(GstEquivalent::Unsupported(GEOMETRIC_TRANSFORM_HINT));
    }
    GST_FAMILY_MAP
        .iter()
        .find(|(prefix, suffix, _)| {
            gst_name.len() >= prefix.len() + suffix.len()
                && gst_name.starts_with(prefix)
                && gst_name.ends_with(suffix)
        })
        .map(|(_, _, equivalent)| equivalent.clone())
}

/// Map a GStreamer element name to its g2g equivalent, consulting the live
/// `registry` first (so aliases resolve and feature-gated elements that ARE
/// compiled in show as `Available`), then the launch keywords, then the static
/// guidance table, then the family rules (`GST_FAMILY_MAP`, for the plugins
/// whose names come by the dozen), then the feature catalog (the name is a g2g
/// element this build left out), and finally the nearest known name (a spelling
/// mistake).
///
/// The hand-written table outranks the feature catalog: both know `x264enc`, and
/// the table's entry also lists the alternatives for a platform where the feature
/// cannot be built.
pub fn gst_equivalent(registry: &Registry, gst_name: &str) -> GstEquivalent {
    if registry_has(registry, gst_name) || LAUNCH_KEYWORDS.contains(&gst_name) {
        return GstEquivalent::Available;
    }
    if let Some((_, equivalent)) = GST_MAP.iter().find(|(name, _)| *name == gst_name) {
        return equivalent.clone();
    }
    if let Some(equivalent) = family_equivalent(gst_name) {
        return equivalent;
    }
    if let Some(feature) = crate::registry::required_feature(gst_name) {
        return GstEquivalent::NotCompiled(feature);
    }
    match nearest_known_name(registry, gst_name) {
        Some(near) => GstEquivalent::DidYouMean(near),
        None => GstEquivalent::Unknown,
    }
}

/// How many single-character insertions, deletions, or substitutions turn `left`
/// into `right`, comparing ASCII case-insensitively (element names are lowercase,
/// so `FileSrc` should still read as `filesrc`).
fn edit_distance(left: &str, right: &str) -> usize {
    let right: Vec<u8> = right.bytes().map(|b| b.to_ascii_lowercase()).collect();
    // One row of the edit matrix: `previous[j]` is the distance from the prefix
    // of `left` handled so far to the first `j` bytes of `right`.
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = alloc::vec![0usize; right.len() + 1];
    for (i, l) in left.bytes().map(|b| b.to_ascii_lowercase()).enumerate() {
        current[0] = i + 1;
        for (j, r) in right.iter().enumerate() {
            let substitute = previous[j] + usize::from(l != *r);
            current[j + 1] = substitute.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        core::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

/// A typo suggestion allows one edit per this many characters of the unknown
/// name, so a name under this length gets no suggestion at all.
const TYPO_CHARS_PER_EDIT: usize = 4;

/// The edit allowance cap, so a long garbage token cannot reach a real name.
const TYPO_MAX_EDITS: usize = 2;

/// The name closest to `name` among everything a launch line can reference, when
/// one is close enough to be a typo of it (see [`TYPO_CHARS_PER_EDIT`] /
/// [`TYPO_MAX_EDITS`]), so a garbage token gets no suggestion. Ties go to the
/// earliest candidate, registered elements before keywords before gst names.
fn nearest_known_name(registry: &Registry, name: &str) -> Option<&'static str> {
    let mut best: Option<(usize, &'static str)> = None;
    let candidates = registry
        .element_names()
        .into_iter()
        .chain(LAUNCH_KEYWORDS.iter().copied())
        .chain(GST_MAP.iter().map(|(gst_name, _)| *gst_name));
    for candidate in candidates {
        let distance = edit_distance(name, candidate);
        if best.is_none_or(|(best_distance, _)| distance < best_distance) {
            best = Some((distance, candidate));
        }
    }
    let (distance, candidate) = best?;
    let allowed = (name.len() / TYPO_CHARS_PER_EDIT).min(TYPO_MAX_EDITS);
    (distance <= allowed).then_some(candidate)
}

/// Every GStreamer element name g2g's runtime reports under a different name,
/// as `(gst name, g2g runtime name)`.
///
/// The runtime name is what a graph dump calls the element, its log category,
/// which is the Rust type name and so often not the launch name: gst's
/// `h264parse` is g2g's `NalParse`. A tool comparing the two engines' graphs
/// pairs elements with this; names that already read the same on both sides
/// (`filesrc` against `FileSrc`) are left out, since pairing those needs no
/// table. Backs `g2g-inspect --gst-map`.
pub fn gst_name_synonyms(registry: &Registry) -> Vec<(&'static str, &'static str)> {
    let mut pairs = Vec::new();
    let mut add = |gst_name: &'static str, g2g_name: &str| {
        let Some(runtime) = runtime_name(registry, g2g_name) else {
            return;
        };
        if !same_word(gst_name, runtime) {
            pairs.push((gst_name, runtime));
        }
    };
    for name in registry.element_names() {
        add(name, name);
    }
    for (gst_name, equivalent) in GST_MAP {
        if let GstEquivalent::Renamed(g2g_name) = equivalent {
            add(gst_name, g2g_name);
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

/// What the runtime calls the element registered as `name`: its log category,
/// which the runner suffixes with an instance number to name a graph node.
fn runtime_name(registry: &Registry, name: &str) -> Option<&'static str> {
    if let Some(element) = registry.make_element(name) {
        return Some(element.log_category());
    }
    registry.make_source(name).map(|s| s.log_category())
}

/// Whether two element names are the same word once case and punctuation are
/// dropped, which is how a graph comparison pairs `filesrc0` with `FileSrc0`.
fn same_word(left: &str, right: &str) -> bool {
    let word = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    };
    word(left) == word(right)
}

/// Whether `name` resolves to a registered element of any role (transform/sink,
/// source, muxer, fan-out demuxer, or terminal fan-out source), aliases
/// included. The fan-in / fan-out roles are built with the smallest pad count
/// the parser would ever give them, since only their existence is asked here.
fn registry_has(registry: &Registry, name: &str) -> bool {
    const PROBE_PADS: usize = 2;
    registry.make_element(name).is_some()
        || registry.make_source(name).is_some()
        || registry.make_muxer(name, PROBE_PADS).is_some()
        || registry.make_demux(name, PROBE_PADS).is_some()
        || registry.make_fanout_src(name, PROBE_PADS).is_some()
}

/// The result of linting a `gst-launch` line for g2g portability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintReport {
    /// True when the line is portable as written (every element resolves and it
    /// parses against `registry`).
    pub ok: bool,
    /// Porting guidance, one per issue. Empty when `ok`. Unportable elements are
    /// reported together (every renamed / unsupported / unknown element in the
    /// line, not just the first), so a port is one pass rather than
    /// fix-one-rerun; a structural / property error is reported on its own once
    /// the element names all resolve.
    pub findings: Vec<String>,
}

/// Every element name a `gst-launch` line references, best-effort: the first
/// token of each `!`-separated segment, skipping inline caps filters
/// (`video/x-raw,...`, which contain `/`), pad references (`t.`, `d.video_0`,
/// `mux.sink_1`), and stray `key=value` tokens. Good enough for a portability
/// scan; the authoritative element set is whatever [`parse_launch`] builds.
fn element_names(line: &str) -> Vec<&str> {
    let mut names = Vec::new();
    for segment in line.split('!') {
        let Some(first) = segment.split_whitespace().next() else {
            continue;
        };
        // Inline caps filter (media/type,fields) or a pad reference (no element
        // name has a dot in it) or a bare property token, none of which is an
        // element to look up.
        if first.contains('/') || first.contains('.') || first.contains('=') {
            continue;
        }
        names.push(first);
    }
    names
}

/// The porting guidance for one element name, `None` when it is portable as
/// written. Shared by the launch linter, the source scanner, and the parse-error
/// explainer, so all three word the same problem identically.
fn finding(name: &str, equivalent: &GstEquivalent) -> Option<String> {
    match equivalent {
        GstEquivalent::Available => None,
        GstEquivalent::Renamed(g) => Some(format!(
            "`{name}` is not a g2g element name; g2g calls it `{g}` (see `g2g-inspect {g}`)"
        )),
        GstEquivalent::Unsupported(hint) => Some(format!("`{name}` has no g2g element: {hint}")),
        GstEquivalent::NotCompiled(feature) => Some(format!(
            "`{name}` is a g2g element but is not compiled into this build; \
             rebuild with `--features {feature}`"
        )),
        GstEquivalent::DidYouMean(near) => Some(format!(
            "`{name}` is not a g2g element; did you mean `{near}`?"
        )),
        GstEquivalent::Unknown => Some(format!(
            "`{name}` is unknown to g2g with no known equivalent; list elements with `g2g-inspect`"
        )),
    }
}

/// The porting findings for a set of element names, in order, portable names
/// skipped. Shared by the linter, the source scanner, and the parse-error
/// explainer.
fn name_findings<'a>(registry: &Registry, names: impl Iterator<Item = &'a str>) -> Vec<String> {
    names
        .filter_map(|name| finding(name, &gst_equivalent(registry, name)))
        .collect()
}

/// Guidance for a [`ParseError`] that `line` already produced, without re-running
/// the parse (a re-parse would repeat its side effects, like a `uridecodebin`
/// file probe logging the same unreadable path twice). Element-name findings when
/// a name is at fault, else the explained error; empty when the explanation would
/// only restate the error's own message.
pub fn explain_parse_error(registry: &Registry, line: &str, error: &ParseError) -> Vec<String> {
    let findings = name_findings(registry, element_names(line).into_iter());
    if !findings.is_empty() {
        return findings;
    }
    let explained = explain(registry, error);
    if explained == error.to_string() {
        return Vec::new();
    }
    Vec::from([explained])
}

/// Lint a `gst-launch` line for g2g portability. First scans every element name
/// and collects guidance for all that are not portable as-is (renamed,
/// unsupported, or unknown); if all elements resolve, runs the authoritative
/// [`parse_launch`] and, on failure, explains that structural / property error.
pub fn lint_launch(registry: &Registry, line: &str) -> LintReport {
    let findings = name_findings(registry, element_names(line).into_iter());
    if !findings.is_empty() {
        return LintReport {
            ok: false,
            findings,
        };
    }
    // Elements all resolve: let the parser catch caps / property / topology
    // issues (one authoritative error).
    match parse_launch(registry, line) {
        Ok(_) => LintReport {
            ok: true,
            findings: Vec::new(),
        },
        Err(e) => LintReport {
            ok: false,
            findings: Vec::from([explain(registry, &e)]),
        },
    }
}

/// The result of scanning GStreamer application source for g2g portability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceScanReport {
    /// Porting guidance for each distinct non-portable element factory the
    /// source instantiates (renamed / unsupported / unknown), deduplicated and
    /// sorted. Empty when every element resolves.
    pub findings: Vec<String>,
    /// Advisories for dynamic-pipeline APIs the source uses (pad-added relink,
    /// pad probes, appsrc/appsink), each pointing at the porting guidance. These
    /// are not errors: they flag idioms that map to a different g2g primitive.
    pub notes: Vec<String>,
}

/// The quoted string argument immediately following each occurrence of `anchor`,
/// best-effort: only when a `"..."` opens before any `)` / `;` / newline, so a
/// call passing a *variable* (e.g. `gst_parse_launch(pipeline, &err)`) is
/// skipped rather than grabbing an unrelated later literal.
fn quoted_args_after(source: &str, anchor: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(pos) = rest.find(anchor) {
        let after = &rest[pos + anchor.len()..];
        if let Some(q) = after.find('"') {
            let pre = &after[..q];
            if !pre.contains(';') && !pre.contains('\n') && !pre.contains(')') {
                let tail = &after[q + 1..];
                if let Some(end) = tail.find('"') {
                    out.push(tail[..end].to_string());
                }
            }
        }
        rest = after; // strictly shorter, so this terminates
    }
    out
}

/// Scan GStreamer *application source* (C or Python) for g2g portability: the
/// element factories it instantiates (`gst_element_factory_make("x", ...)`,
/// `Gst.ElementFactory.make("x")`, and the elements inside any
/// `gst_parse_launch("...")` / `Gst.parse_launch("...")` string) and the
/// dynamic-pipeline APIs it uses. Best-effort and static, it complements
/// [`lint_launch`] (a single launch string) for apps that build pipelines in
/// code; the authoritative check is still running the ported pipeline.
pub fn scan_source(registry: &Registry, source: &str) -> SourceScanReport {
    // Element factories: the first quoted arg of each make-call, plus every
    // element inside each parse_launch string.
    let mut names: BTreeSet<String> = BTreeSet::new();
    for anchor in ["factory_make", "ElementFactory.make"] {
        for name in quoted_args_after(source, anchor) {
            names.insert(name);
        }
    }
    for line in quoted_args_after(source, "parse_launch") {
        for name in element_names(&line) {
            names.insert(name.to_string());
        }
    }

    let findings = name_findings(registry, names.iter().map(String::as_str));

    // Dynamic-pipeline idioms: map each to its g2g primitive (PORTING.md §5.1).
    let mut notes = Vec::new();
    let has = |needle: &str| source.contains(needle);
    if has("pad-added") {
        notes.push(
            "uses `pad-added` dynamic relink: in g2g use `decodebin`/`uridecodebin` auto-plug, \
             or `StreamDemux` / `register_demux` with typed output ports (PORTING.md §5.1)"
                .to_string(),
        );
    }
    if has("add_probe") || has("pad_add_probe") {
        notes.push(
            "uses pad probes: in g2g register a `LinkInterceptor` on the slot (PORTING.md §5.1)"
                .to_string(),
        );
    }
    if has("appsrc") || has("need-data") || has("push-buffer") {
        notes.push(
            "uses appsrc: g2g has `appsrc channel=<name>` + `register_appsrc`, or `g2g-bridge` \
             for a whole embedded sub-graph (PORTING.md §5.1)"
                .to_string(),
        );
    }
    if has("appsink") || has("new-sample") || has("pull-sample") {
        notes.push(
            "uses appsink: g2g has `appsink channel=<name>` + `set_appsink_callback` (callback) \
             or `register_appsink_pull` (pull) (PORTING.md §5.1)"
                .to_string(),
        );
    }

    SourceScanReport { findings, notes }
}

/// Turn a [`ParseError`] into porting-oriented guidance.
fn explain(registry: &Registry, e: &ParseError) -> String {
    match e {
        ParseError::UnknownElement(n) | ParseError::UnknownSource(n) => {
            let equivalent = gst_equivalent(registry, n);
            finding(n, &equivalent).unwrap_or_else(|| {
                format!(
                    "`{n}` is available; re-check spelling or whether its feature is compiled in"
                )
            })
        }
        ParseError::UnknownProperty { element, key } => {
            format!("`{element}` has no property `{key}`; run `g2g-inspect {element}` for its properties")
        }
        ParseError::BadValue {
            element,
            key,
            value,
        } => {
            format!("`{element}` property `{key}` rejects `{value}`; check its type with `g2g-inspect {element}`")
        }
        ParseError::NotAMuxer(n) => {
            format!("`{n}` has several inputs but is not a registered muxer; use a g2g muxer (`funnel`, `audiomixer`, `mpegtsmux`, ...)")
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capsfilter::parse_caps;
    use crate::registry::default_registry;
    use alloc::boxed::Box;
    use g2g_core::{Caps, Dim, Rate, RawVideoFormat};

    #[test]
    fn caps_string_round_trips_through_the_parser() {
        let c = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(parse_caps(&c.to_gst_string()), Some(c));
    }

    #[test]
    fn the_synonym_table_names_the_elements_the_two_engines_disagree_about() {
        let reg = default_registry();
        let pairs = gst_name_synonyms(&reg);
        // The case that makes the table necessary: gst's parser is a type g2g
        // shares between codecs, so a graph dump never pairs the two by name.
        assert!(pairs.contains(&("h264parse", "NalParse")), "got {pairs:?}");
        for (gst_name, g2g_name) in &pairs {
            assert!(
                !same_word(gst_name, g2g_name),
                "{gst_name} and {g2g_name} already pair without the table"
            );
            assert!(
                runtime_name(&reg, gst_name).is_some_and(|n| n == *g2g_name)
                    || matches!(gst_equivalent(&reg, gst_name), GstEquivalent::Renamed(_)),
                "{gst_name} maps to {g2g_name} through the registry or the rename table"
            );
        }
    }

    #[test]
    fn clean_line_lints_ok() {
        let reg = default_registry();
        let r = lint_launch(&reg, "videotestsrc num-buffers=1 ! videoconvert ! fakesink");
        assert!(r.ok, "findings: {:?}", r.findings);
    }

    // Only meaningful when `x264enc` is NOT compiled in: with the `ffmpeg`
    // feature it is a registered element, so the lint reports no finding.
    #[cfg(not(feature = "ffmpeg"))]
    #[test]
    fn unknown_encoder_gets_a_suggestion() {
        let reg = default_registry();
        let r = lint_launch(&reg, "videotestsrc ! x264enc ! fakesink");
        assert!(!r.ok);
        let msg = &r.findings[0];
        assert!(
            msg.contains("x264enc") && (msg.contains("mfencode") || msg.contains("av1enc")),
            "{msg}"
        );
    }

    #[test]
    fn renamed_element_maps_to_g2g_name() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "jpegdec"),
            GstEquivalent::Renamed("mjpegdec")
        );
    }

    #[test]
    fn reports_every_unportable_element_not_just_the_first() {
        let reg = default_registry();
        // Two unsupported encoders (feature-independent) in one line: both must
        // appear, so a port is one pass rather than fix-one-rerun.
        let r = lint_launch(&reg, "videotestsrc ! theoraenc ! x265enc ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 2, "both flagged: {:?}", r.findings);
        assert!(
            r.findings.iter().any(|m| m.contains("theoraenc")),
            "{:?}",
            r.findings
        );
        assert!(
            r.findings.iter().any(|m| m.contains("x265enc")),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn renamed_element_in_a_line_is_flagged_with_its_g2g_name() {
        let reg = default_registry();
        let r = lint_launch(&reg, "filesrc location=x ! jpegdec ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 1, "{:?}", r.findings);
        assert!(
            r.findings[0].contains("jpegdec") && r.findings[0].contains("mjpegdec"),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn caps_filters_and_tee_branches_are_not_mistaken_for_elements() {
        let reg = default_registry();
        // Inline caps filter and a tee branch ref must not be linted as unknown
        // elements; this well-formed line is portable.
        let r = lint_launch(
            &reg,
            "videotestsrc ! video/x-raw,width=320,height=240 ! tee name=t \
             ! queue ! fakesink t. ! queue ! fakesink",
        );
        assert!(r.ok, "findings: {:?}", r.findings);
    }

    #[test]
    fn keyword_and_unknown_classify() {
        let reg = default_registry();
        assert_eq!(gst_equivalent(&reg, "tee"), GstEquivalent::Available);
        assert_eq!(
            gst_equivalent(&reg, "videoconvert"),
            GstEquivalent::Available
        );
        assert_eq!(
            gst_equivalent(&reg, "totally-made-up"),
            GstEquivalent::Unknown
        );
    }

    #[test]
    fn a_misspelled_element_gets_a_suggestion() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "filesrcc"),
            GstEquivalent::DidYouMean("filesrc")
        );
        let r = lint_launch(&reg, "filesrcc location=x ! fakesink");
        assert!(!r.ok);
        assert_eq!(r.findings.len(), 1, "{:?}", r.findings);
        assert!(
            r.findings[0].contains("did you mean `filesrc`"),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn a_garbage_token_gets_no_suggestion() {
        let reg = default_registry();
        for name in ["totally-made-up", "zzzz", "xyzzy", "qqqqqqqqqqqq"] {
            assert_eq!(
                gst_equivalent(&reg, name),
                GstEquivalent::Unknown,
                "`{name}` must not get a suggestion"
            );
        }
    }

    // Only meaningful when `srt` is NOT compiled in: with the feature the element
    // is registered, so it resolves as `Available`.
    #[cfg(not(feature = "srt"))]
    #[test]
    fn a_feature_gated_element_names_its_feature() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "srtsink"),
            GstEquivalent::NotCompiled("srt")
        );
        let r = lint_launch(&reg, "filesrc location=x ! srtsink");
        assert!(!r.ok);
        assert!(r.findings[0].contains("--features srt"), "{:?}", r.findings);
    }

    #[test]
    fn registered_appsrc_appsink_are_available_not_unsupported() {
        let reg = default_registry();
        assert_eq!(gst_equivalent(&reg, "appsrc"), GstEquivalent::Available);
        assert_eq!(gst_equivalent(&reg, "appsink"), GstEquivalent::Available);
    }

    #[test]
    fn auto_capture_aliases_resolve_when_a_capture_element_is_built() {
        const WIDTH: u32 = 320;
        const HEIGHT: u32 = 240;
        const FRAMERATE: u32 = 30;
        const FRAMES: u64 = 1;
        let mut reg = default_registry();
        // A stand-in for whichever platform capture element a build has: the
        // alias must land on it rather than staying unknown.
        reg.register_source(g2g_core::runtime::SourceFactory::new(
            "v4l2src",
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(WIDTH),
                height: Dim::Fixed(HEIGHT),
                framerate: Rate::Fixed(FRAMERATE << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
            },
            || {
                Box::new(crate::videotestsrc::VideoTestSrc::new(
                    WIDTH, HEIGHT, FRAMERATE, FRAMES,
                ))
            },
        ));
        reg.register_alias("autovideosrc", &["v4l2src"]);
        assert_eq!(
            gst_equivalent(&reg, "autovideosrc"),
            GstEquivalent::Available
        );
    }

    /// The fan-out roles are registered through their own factory lists, so the
    /// registry lookup has to ask those too or a registered demuxer reads as
    /// unknown.
    #[test]
    fn fan_out_factories_count_as_registered() {
        let reg = default_registry();
        for name in ["output-selector", "deinterleave"] {
            assert_eq!(
                gst_equivalent(&reg, name),
                GstEquivalent::Available,
                "`{name}` is a registered fan-out factory"
            );
        }
    }

    /// Every gst name g2g covers under another name has to reach an answer: a
    /// typo suggestion or a blank "unknown" is not one.
    #[test]
    fn the_covered_gst_names_all_reach_an_answer() {
        let reg = default_registry();
        for name in [
            "ccextractor",
            "line21encoder",
            "line21decoder",
            "webrtcbin",
            "subtitleoverlay",
            "dvbsuboverlay",
            "dvdspu",
            "ssaparse",
            "subparse_typefind",
            "onnxinference",
            "output-selector",
            "streamiddemux",
            "dashsink",
            "sdpsrc",
            "sdpdemux",
            "dtlsenc",
            "dtlsdec",
            "dtlssrtpenc",
            "dtlssrtpdec",
            "av1dec",
            "multiudpsink",
            "dynudpsink",
            "bin",
            "pipeline",
        ] {
            let answer = gst_equivalent(&reg, name);
            assert!(
                !matches!(
                    answer,
                    GstEquivalent::Unknown | GstEquivalent::DidYouMean(_)
                ),
                "`{name}` answered {answer:?}"
            );
        }
    }

    /// The guidance rows have to name the g2g path, not just say no.
    #[test]
    fn the_covered_name_hints_name_the_g2g_path() {
        let reg = default_registry();
        for (name, needle) in [
            ("ccextractor", "ccextract"),
            ("line21encoder", "cccombiner"),
            ("line21decoder", "ccextract"),
            ("webrtcbin", "webrtcsessionsink"),
            ("subtitleoverlay", "subpictureoverlay"),
            ("dvbsuboverlay", "dvbsubdec"),
            ("dvdspu", "vobsubdec"),
            ("ssaparse", "subparse"),
            ("subparse_typefind", "typefind"),
            ("onnxinference", "ortinfer"),
            ("streamiddemux", "output-selector"),
            ("dashsink", "hlssink"),
            ("sdpdemux", "udpsrc sdp="),
            ("dtlsenc", "dtlssrtpenc"),
            ("dynudpsink", "udpsink"),
            ("pipeline", "flattens bins"),
        ] {
            let GstEquivalent::Unsupported(hint) = gst_equivalent(&reg, name) else {
                panic!("`{name}` must carry a hint");
            };
            assert!(hint.contains(needle), "`{name}`: {hint}");
        }
    }

    /// `av1dec` and `multiudpsink` are registry aliases, so they answer
    /// `Available` in a build with the target and name the way to it otherwise.
    #[test]
    fn the_av1_and_multicast_aliases_answer_either_way() {
        let reg = default_registry();
        let expected = if cfg!(any(feature = "dav1d", feature = "rav1d")) {
            GstEquivalent::Available
        } else {
            GstEquivalent::Renamed("dav1ddec")
        };
        assert_eq!(gst_equivalent(&reg, "av1dec"), expected);
        let expected = if cfg!(feature = "udp-egress") {
            GstEquivalent::Available
        } else {
            GstEquivalent::NotCompiled("udp-egress")
        };
        assert_eq!(gst_equivalent(&reg, "multiudpsink"), expected);
    }

    #[test]
    fn container_and_mixer_aliases_resolve_to_their_g2g_targets() {
        let reg = default_registry();
        for name in ["webmmux", "adder", "liveadder", "videomixer"] {
            assert_eq!(
                gst_equivalent(&reg, name),
                GstEquivalent::Available,
                "`{name}` must resolve through its alias"
            );
        }
    }

    #[test]
    fn the_extra_decoder_names_point_at_the_ffmpeg_decoders() {
        let reg = default_registry();
        let compiled = cfg!(all(target_os = "linux", feature = "ffmpeg"));
        for name in ["vp8dec", "mpeg2dec", "avdec_vp9", "avdec_mpeg4"] {
            let expected = if compiled {
                GstEquivalent::Available
            } else {
                GstEquivalent::Renamed("ffmpegdec")
            };
            assert_eq!(gst_equivalent(&reg, name), expected, "`{name}`");
        }
        for name in ["mpg123audiodec", "a52dec", "faad", "avdec_flac"] {
            let expected = if compiled {
                GstEquivalent::Available
            } else {
                GstEquivalent::Renamed("ffmpegaudiodec")
            };
            assert_eq!(gst_equivalent(&reg, name), expected, "`{name}`");
        }
    }

    #[test]
    fn adaptive_and_srt_rows_name_the_g2g_path() {
        let reg = default_registry();
        for (name, needle) in [
            ("hlsdemux", "hlssrc"),
            ("dashdemux2", "dashsrc"),
            ("mssdemux", "dashsrc"),
            ("srtserversrc", "srtsrc"),
            ("srtclientsink", "srtsink"),
        ] {
            let GstEquivalent::Unsupported(hint) = gst_equivalent(&reg, name) else {
                panic!("`{name}` must carry a hint");
            };
            assert!(hint.contains(needle), "`{name}`: {hint}");
        }
    }

    #[test]
    fn bin_names_point_at_the_launch_keyword_the_parser_accepts() {
        let reg = default_registry();
        assert_eq!(
            gst_equivalent(&reg, "decodebin3"),
            GstEquivalent::Renamed("decodebin")
        );
        assert_eq!(
            gst_equivalent(&reg, "playbin3"),
            GstEquivalent::Renamed("playbin")
        );
        assert_eq!(gst_equivalent(&reg, "queue2"), GstEquivalent::Available);
    }

    #[test]
    fn family_rules_answer_the_names_that_come_by_the_dozen() {
        let reg = default_registry();
        let GstEquivalent::Unsupported(pay) = gst_equivalent(&reg, "rtph264pay") else {
            panic!("rtph264pay must hit the rtp payload family");
        };
        assert!(pay.contains("udpsink"), "{pay}");
        let GstEquivalent::Unsupported(session) = gst_equivalent(&reg, "rtpjitterbuffer") else {
            panic!("rtpjitterbuffer must hit the rtp session family");
        };
        assert!(session.contains("jitter-latency"), "{session}");
        let GstEquivalent::Unsupported(gl) = gst_equivalent(&reg, "gleffects_blur") else {
            panic!("gleffects_blur must hit the gl family");
        };
        assert!(gl.contains("wgpusink"), "{gl}");
        assert_eq!(
            gst_equivalent(&reg, "nvav1dec"),
            GstEquivalent::Renamed("nvdec")
        );
        let GstEquivalent::Unsupported(effect) = gst_equivalent(&reg, "warptv") else {
            panic!("warptv must hit the effectv family");
        };
        assert!(effect.contains("effects"), "{effect}");
        let GstEquivalent::Unsupported(geometry) = gst_equivalent(&reg, "kaleidoscope") else {
            panic!("kaleidoscope must hit the geometrictransform family");
        };
        assert!(geometry.contains("videoflip"), "{geometry}");
    }

    #[test]
    fn an_exact_row_beats_a_family_rule() {
        let reg = default_registry();
        // Both the `rtph264depay` row and the rtp depay family would answer;
        // the row's wording is the one that must come out.
        let GstEquivalent::Unsupported(hint) = gst_equivalent(&reg, "rtph264depay") else {
            panic!("rtph264depay has an exact row");
        };
        assert_eq!(hint, "RTP depayloading is built into `udpsrc` / `rtspsrc`");
    }

    #[test]
    fn a_g2g_element_name_is_never_shadowed_by_a_family_rule() {
        let reg = default_registry();
        // `nvdec` matches the `nv*dec` family and `videoconvert` is registered:
        // neither may come back as a family hint.
        let expected = if cfg!(all(target_os = "linux", feature = "nvdec")) {
            GstEquivalent::Available
        } else {
            GstEquivalent::NotCompiled("nvdec")
        };
        assert_eq!(gst_equivalent(&reg, "nvdec"), expected);
        assert_eq!(
            gst_equivalent(&reg, "videoconvert"),
            GstEquivalent::Available
        );
    }

    #[test]
    fn every_renamed_target_still_exists() {
        let reg = default_registry();
        let targets = GST_MAP
            .iter()
            .map(|(_, equivalent)| equivalent)
            .chain(GST_FAMILY_MAP.iter().map(|(_, _, equivalent)| equivalent))
            .filter_map(|equivalent| match equivalent {
                GstEquivalent::Renamed(target) => Some(*target),
                _ => None,
            });
        for target in targets {
            assert!(
                registry_has(&reg, target)
                    || LAUNCH_KEYWORDS.contains(&target)
                    || crate::registry::required_feature(target).is_some(),
                "`{target}` is named as a g2g equivalent but is neither registered, \
                 a launch keyword, nor a feature-gated element"
            );
        }
    }

    #[test]
    fn scans_c_source_for_factories_and_dynamic_apis() {
        let reg = default_registry();
        // A snippet of a C GStreamer app: factory_make calls (one renamed), a
        // parse_launch string (one unsupported element), a pad-added handler.
        let src = r#"
            GstElement *conv = gst_element_factory_make("videoconvert", "c");
            GstElement *dec  = gst_element_factory_make("jpegdec", "d");
            pipeline = gst_parse_launch("videotestsrc ! theoraenc ! fakesink", &err);
            g_signal_connect(demux, "pad-added", G_CALLBACK(on_pad_added), NULL);
        "#;
        let r = scan_source(&reg, src);
        // videoconvert is available (no finding); jpegdec renamed; theoraenc unsupported.
        assert!(
            r.findings
                .iter()
                .any(|m| m.contains("jpegdec") && m.contains("mjpegdec")),
            "{:?}",
            r.findings
        );
        assert!(
            r.findings.iter().any(|m| m.contains("theoraenc")),
            "{:?}",
            r.findings
        );
        assert!(
            !r.findings.iter().any(|m| m.contains("videoconvert")),
            "available element flagged: {:?}",
            r.findings
        );
        assert!(
            r.notes.iter().any(|n| n.contains("pad-added")),
            "notes: {:?}",
            r.notes
        );
    }

    #[test]
    fn scans_python_source_and_ignores_variable_parse_launch() {
        let reg = default_registry();
        let src = r#"
            conv = Gst.ElementFactory.make("videoconvert", "conv")
            sink = Gst.ElementFactory.make("appsink", "sink")
            pipeline = Gst.parse_launch(user_supplied_string)  # variable, not a literal
        "#;
        let r = scan_source(&reg, src);
        // appsink resolves (registered); videoconvert too; the variable
        // parse_launch yields no phantom element findings.
        assert!(
            r.findings.is_empty(),
            "unexpected findings: {:?}",
            r.findings
        );
        // appsink triggers the dynamic-API note.
        assert!(
            r.notes.iter().any(|n| n.contains("appsink")),
            "notes: {:?}",
            r.notes
        );
    }
}
