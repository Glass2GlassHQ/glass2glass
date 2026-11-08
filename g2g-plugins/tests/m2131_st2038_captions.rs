#![cfg(feature = "std")]

#[path = "common/st2038_ts.rs"]
mod st2038_ts;

use g2g_plugins::st2038::{parse_st2038, write_st2038};
use g2g_plugins::st291::{AncPacket, CAPTION_DID, CEA608_SDID};
use st2038_ts::{run_launch, scratch, video_and_st2038_ts, CEA608_S334, FRAMES, GST_CEA708_PACKET};

fn cdp() -> Vec<u8> {
    parse_st2038(GST_CEA708_PACKET).unwrap()[0].user_data_bytes()
}

fn captions_ts() -> Vec<u8> {
    let cea608 = AncPacket::from_bytes(CAPTION_DID, CEA608_SDID, &CEA608_S334);
    let mut payload = GST_CEA708_PACKET.to_vec();
    payload.extend(write_st2038(&[cea608]).unwrap());
    video_and_st2038_ts(&payload)
}

#[tokio::test]
async fn st2038anctocc_and_cctost2038anc_rebuild_the_gst_packet() {
    let ts = scratch("g2g_m2131_rebuild.ts");
    let out = scratch("g2g_m2131_rebuild.bin");
    std::fs::write(&ts, captions_ts()).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038anctocc ! cctost2038anc ! filesink location={}",
        ts.display(),
        out.display()
    ))
    .await;
    assert_eq!(
        std::fs::read(&out).unwrap(),
        GST_CEA708_PACKET.repeat(FRAMES)
    );
}

#[tokio::test]
async fn st2038anctocc_splits_captions_to_named_pads() {
    let ts = scratch("g2g_m2131_split.ts");
    let cea708 = scratch("g2g_m2131_split_708.bin");
    let cea608 = scratch("g2g_m2131_split_608.bin");
    std::fs::write(&ts, captions_ts()).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038anctocc name=d d.src_cea608 ! filesink location={} \
         d.src_cea708 ! filesink location={}",
        ts.display(),
        cea608.display(),
        cea708.display()
    ))
    .await;
    assert_eq!(std::fs::read(&cea708).unwrap(), cdp().repeat(FRAMES));
    assert_eq!(std::fs::read(&cea608).unwrap(), CEA608_S334.repeat(FRAMES));
}

#[tokio::test]
async fn one_named_pad_picks_its_caption_type() {
    let ts = scratch("g2g_m2131_named.ts");
    let cea608 = scratch("g2g_m2131_named_608.bin");
    std::fs::write(&ts, captions_ts()).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038anctocc name=d d.src_cea608 ! filesink location={}",
        ts.display(),
        cea608.display()
    ))
    .await;
    assert_eq!(std::fs::read(&cea608).unwrap(), CEA608_S334.repeat(FRAMES));
}
