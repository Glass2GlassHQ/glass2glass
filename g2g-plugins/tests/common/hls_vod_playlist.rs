// serves the A/V TS fixture as a one-segment VOD HLS playlist, for lines that read an http master URL

#[path = "http_routes.rs"]
mod http_routes;

const TS_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac44100.ts"
);
const MASTER_PATH: &str = "/master.m3u8";
const MEDIA_PATH: &str = "/media.m3u8";
const SEGMENT_PATH: &str = "/segment0.ts";
// the fixture's High 4:4:4 Predictive level 1 video and AAC LC audio
const VARIANT_CODECS: &str = "avc1.f4000a,mp4a.40.2";
const VARIANT_BANDWIDTH: u32 = 500_000;
const SEGMENT_SECONDS: u32 = 1;

// the master playlist's URL
pub(crate) fn serve_vod_playlist() -> String {
    let master = format!(
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH={VARIANT_BANDWIDTH},CODECS=\"{VARIANT_CODECS}\"\n{}\n",
        &MEDIA_PATH[1..]
    );
    let media = format!(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:{SEGMENT_SECONDS}\n\
         #EXTINF:{SEGMENT_SECONDS},\n{}\n#EXT-X-ENDLIST\n",
        &SEGMENT_PATH[1..]
    );
    let segment = std::fs::read(TS_FIXTURE).expect("read the TS fixture");
    let base = http_routes::serve(move |path| match path {
        MASTER_PATH => Some(master.clone().into_bytes()),
        MEDIA_PATH => Some(media.clone().into_bytes()),
        SEGMENT_PATH => Some(segment.clone()),
        _ => None,
    });
    format!("{base}{MASTER_PATH}")
}
