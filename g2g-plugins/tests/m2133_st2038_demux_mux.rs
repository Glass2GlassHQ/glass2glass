#![cfg(feature = "std")]

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

fn payload() -> Vec<u8> {
    [GST_CEA708_PACKET, &cea608_packet()].concat()
}

#[tokio::test]
async fn a_named_demux_pad_carries_one_stream() {
    let ts = scratch("g2g_m2133_named.ts");
    let out = scratch("g2g_m2133_named.bin");
    std::fs::write(&ts, video_and_st2038_ts(&payload())).unwrap();
    run_launch(&format!(
        "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
         ! st2038ancdemux name=d d.anc_61_02_at_{CEA608_LINE}_0 ! filesink location={}",
        ts.display(),
        out.display()
    ))
    .await;
    assert_eq!(std::fs::read(&out).unwrap(), cea608_packet().repeat(FRAMES));
}

#[tokio::test]
async fn demuxed_streams_mux_back_into_each_frame() {
    for framerate in ["", ",framerate=25/1"] {
        let ts = scratch("g2g_m2133_remux.ts");
        let out = scratch("g2g_m2133_remux.bin");
        std::fs::write(&ts, video_and_st2038_ts(&payload())).unwrap();
        run_launch(&format!(
            "filesrc location={} bytestream-format=mpegts ! tsdemux stream=st2038 \
             ! st2038ancdemux name=d d.anc_61_02_at_{CEA608_LINE}_0 ! m. \
             d.anc_61_01_at_9_0 ! m. st2038ancmux name=m \
             ! meta/x-st-2038,alignment=frame{framerate} ! filesink location={}",
            ts.display(),
            out.display()
        ))
        .await;
        assert_eq!(
            std::fs::read(&out).unwrap(),
            payload().repeat(FRAMES),
            "framerate field {framerate:?}"
        );
    }
}
