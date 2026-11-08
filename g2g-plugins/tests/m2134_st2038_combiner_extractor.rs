#![cfg(all(feature = "std", feature = "metadata"))]

#[path = "common/st2038_ts.rs"]
mod st2038_ts;

use g2g_plugins::st2038::write_st2038;
use g2g_plugins::st291::{AncPacket, CAPTION_DID, CEA608_SDID};
use st2038_ts::{run_launch, scratch, video_and_st2038_ts, CEA608_S334, FRAMES, GST_CEA708_PACKET};

const CEA608_LINE: u16 = 21;

fn cea608_packet() -> Vec<u8> {
    let mut cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
    cea608.line_number = CEA608_LINE;
    cea608.horizontal_offset = 0;
    write_st2038(&[cea608]).unwrap()
}

#[tokio::test]
async fn st2038_rides_the_video_frames_and_comes_back_out() {
    // line 21 before line 9, so the extractor's sort by line shows in the output
    let payload = [cea608_packet().as_slice(), GST_CEA708_PACKET].concat();
    let sorted = [GST_CEA708_PACKET, cea608_packet().as_slice()].concat();
    let ts = scratch("g2g_m2134.ts");
    let out = scratch("g2g_m2134.bin");
    std::fs::write(&ts, video_and_st2038_ts(&payload)).unwrap();
    run_launch(&format!(
        "filesrc location={ts} bytestream-format=mpegts ! tsdemux stream=st2038 ! c.st2038 \
         filesrc location={ts} bytestream-format=mpegts ! tsdemux ! c.sink \
         st2038combiner name=c ! st2038extractor name=e e.st2038 ! filesink location={out} \
         e.src ! fakesink",
        ts = ts.display(),
        out = out.display()
    ))
    .await;
    assert_eq!(std::fs::read(&out).unwrap(), sorted.repeat(FRAMES));
}
