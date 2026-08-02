//! M810: SMPTE ST 2022-1 (Pro-MPEG COP3) FEC for MPEG-TS over RTP.

use g2g_plugins::st2022fec::{
    build_repair_packet, protected_seqs, recover_packet, FecHeader, FecStream, Repair,
    St2022FecDecoder, St2022FecEncoder, FEC_HEADER,
};

const RTP_HEADER: usize = 12;
const TS_PT: u8 = 33;
const SSRC: u32 = 0xABCD_0001;

/// An MPEG-TS-over-RTP media packet: seven 188-byte TS packets, the marker bit
/// set every eighth packet so recovery has an M bit to reconstruct.
fn media(seq: u16, ts: u32) -> Vec<u8> {
    let marker = seq % 8 == 7;
    let mut p = Vec::with_capacity(RTP_HEADER + 7 * 188);
    p.push(0x80);
    p.push(if marker { 0x80 | TS_PT } else { TS_PT });
    p.extend_from_slice(&seq.to_be_bytes());
    p.extend_from_slice(&ts.to_be_bytes());
    p.extend_from_slice(&SSRC.to_be_bytes());
    for i in 0..7 * 188 {
        p.push((seq as usize).wrapping_mul(31).wrapping_add(i) as u8);
    }
    p
}

fn stream(n: u16) -> Vec<Vec<u8>> {
    (0..n).map(|i| media(i, 90_000 + i as u32 * 3600)).collect()
}

fn seq_of(p: &[u8]) -> u16 {
    u16::from_be_bytes([p[2], p[3]])
}

/// Run `media` through the encoder, returning the repairs it emitted.
fn encode(enc: &mut St2022FecEncoder, media: &[Vec<u8>]) -> Vec<Repair> {
    media.iter().flat_map(|p| enc.push(p)).collect()
}

/// Deliver every media packet except `lost`, then all repairs, and take what
/// the decoder rebuilt, sorted by sequence.
fn decode(media: &[Vec<u8>], lost: &[usize], repairs: &[Repair]) -> Vec<Vec<u8>> {
    let mut dec = St2022FecDecoder::new(256);
    for (i, p) in media.iter().enumerate() {
        if !lost.contains(&i) {
            dec.push_media(seq_of(p), p);
        }
    }
    for r in repairs {
        dec.push_fec(&r.packet);
    }
    let mut out = dec.take_recovered();
    out.sort_by_key(|p| seq_of(p));
    out
}

#[test]
fn encoder_emits_one_column_repair_per_column_of_the_block() {
    // L=5, D=4: one set of 5 column repairs closes every 20 media packets.
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0);
    assert_eq!(enc.block(), 20);
    let media = stream(40);
    let repairs = encode(&mut enc, &media);

    assert_eq!(repairs.len(), 10, "5 column repairs per block, 2 blocks");
    assert!(repairs.iter().all(|r| r.stream == FecStream::Column));
    // Column 2 of the first block protects media 2, 7, 12, 17.
    let h = FecHeader::parse(&repairs[2].packet[RTP_HEADER..]).expect("header");
    assert!(!h.d, "D=0 is the column stream");
    assert_eq!(h.offset, 5);
    assert_eq!(h.na, 4);
    assert_eq!(
        protected_seqs(&repairs[2].packet).unwrap(),
        vec![2u16, 7, 12, 17],
    );
    // The two repair streams number independently, from zero.
    assert_eq!(seq_of(&repairs[0].packet), 0);
    assert_eq!(seq_of(&repairs[5].packet), 5);
}

#[test]
fn a_single_loss_in_a_column_is_recovered_byte_exact() {
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);

    // Packet 7 carries the marker bit, so its M and PT recovery both matter.
    assert_eq!(media[7][1] & 0x80, 0x80, "the lost packet has M set");
    let recovered = decode(&media, &[7], &repairs);
    assert_eq!(recovered.len(), 1);
    assert_eq!(
        recovered[0], media[7],
        "the reconstructed packet matches the original byte for byte",
    );
}

#[test]
fn every_column_recovers_its_own_loss_in_the_same_block() {
    // One loss per column, five in the block: each column repair recovers its
    // own, which single-dimension FEC over contiguous groups could not do.
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);

    let lost = [0usize, 6, 12, 18, 19];
    let recovered = decode(&media, &lost, &repairs);
    assert_eq!(recovered.len(), 5);
    for (r, i) in recovered.iter().zip([0usize, 6, 12, 18, 19]) {
        assert_eq!(*r, media[i], "packet {i} reconstructed");
    }
}

#[test]
fn a_burst_of_l_consecutive_losses_is_fully_recovered() {
    // The point of column FEC: a burst of up to L consecutive packets lands one
    // per column, so all L come back.
    let l = 5usize;
    let mut enc = St2022FecEncoder::new(l, 4, 96, 0);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);

    let lost: Vec<usize> = (6..6 + l).collect();
    let recovered = decode(&media, &lost, &repairs);
    assert_eq!(recovered.len(), l, "the whole burst was reconstructed");
    for (r, i) in recovered.iter().zip(lost) {
        assert_eq!(*r, media[i], "burst packet {i} byte-exact");
    }
}

#[test]
fn a_burst_longer_than_l_leaves_the_extra_loss_unrecovered() {
    // L+1 consecutive losses put two packets in one column: that column is
    // beyond a single XOR repair, and the decoder must report it as still lost
    // rather than emitting a wrongly XORed packet.
    let l = 5usize;
    let mut enc = St2022FecEncoder::new(l, 4, 96, 0);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);

    let lost: Vec<usize> = (6..6 + l + 1).collect();
    let recovered = decode(&media, &lost, &repairs);
    assert_eq!(
        recovered.len(),
        4,
        "the four singly-hit columns recover; the doubly-hit one does not",
    );
    for r in &recovered {
        let i = seq_of(r) as usize;
        assert_eq!(*r, media[i], "no corrupt output among the recoveries");
    }
    assert!(
        !recovered.iter().any(|r| seq_of(r) == 6 || seq_of(r) == 11),
        "packets 6 and 11 share column 1 and stay lost",
    );
}

#[test]
fn two_losses_in_one_column_recover_nothing_and_corrupt_nothing() {
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);

    // 3 and 13 are both in column 3.
    let recovered = decode(&media, &[3, 13], &repairs);
    assert!(
        recovered.is_empty(),
        "a doubly-hit column reports no recovery, not a corrupt packet",
    );
}

#[test]
fn row_repairs_rescue_a_column_that_lost_two_packets() {
    // Both dimensions enabled: 3 and 13 defeat their shared column repair, but
    // each sits in a different row, so the row repairs recover them and the
    // decoder chains the result back into the column.
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0).with_row_fec(true);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);
    assert_eq!(
        repairs
            .iter()
            .filter(|r| r.stream == FecStream::Row)
            .count(),
        4,
        "one row repair per row of the block",
    );
    let row = repairs
        .iter()
        .find(|r| r.stream == FecStream::Row)
        .expect("a row repair");
    let h = FecHeader::parse(&row.packet[RTP_HEADER..]).expect("header");
    assert!(h.d, "D=1 is the row stream");
    assert_eq!((h.offset, h.na), (1, 5), "L consecutive packets per row");

    let recovered = decode(&media, &[3, 13], &repairs);
    assert_eq!(recovered.len(), 2);
    assert_eq!(recovered[0], media[3]);
    assert_eq!(recovered[1], media[13]);
}

#[test]
fn repairs_survive_a_reordered_and_partly_lost_repair_stream() {
    // Repairs arrive on their own sessions and can be late, reordered or lost
    // themselves; recovery must not depend on their order.
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0);
    let media = stream(20);
    let mut repairs = encode(&mut enc, &media);
    repairs.reverse();
    repairs.pop(); // one repair never arrives

    let mut dec = St2022FecDecoder::new(256);
    for r in &repairs {
        dec.push_fec(&r.packet); // repairs before any media
    }
    for (i, p) in media.iter().enumerate() {
        if i != 9 {
            dec.push_media(seq_of(p), p);
        }
    }
    let recovered = dec.take_recovered();
    assert!(
        recovered.iter().any(|r| *r == media[9]),
        "the loss is recovered whatever order the repairs arrived in",
    );
    // A repair fires as soon as one member of its group is absent, so packets
    // still in flight at the tail of the block get rebuilt early too. Those are
    // real packets, not corrupt ones, and the jitter buffer drops the duplicate.
    for r in &recovered {
        assert_eq!(
            *r,
            media[seq_of(r) as usize],
            "every recovery is byte-exact"
        );
    }
}

#[test]
fn malformed_repair_packets_are_ignored_rather_than_fatal() {
    let mut enc = St2022FecEncoder::new(5, 4, 96, 0);
    let media = stream(20);
    let repairs = encode(&mut enc, &media);
    let good = repairs[2].packet.clone(); // protects 2, 7, 12, 17

    let mut dec = St2022FecDecoder::new(256);
    // Truncations at every length, a header claiming 255 packets at stride 255,
    // a zero-length descriptor, and pure garbage.
    for n in 0..good.len() {
        dec.push_fec(&good[..n]);
    }
    let mut huge = good.clone();
    huge[RTP_HEADER + 13] = 0xFF; // offset
    huge[RTP_HEADER + 14] = 0xFF; // NA
    dec.push_fec(&huge);
    let mut empty = good.clone();
    empty[RTP_HEADER + 14] = 0; // NA = 0 protects nothing
    dec.push_fec(&empty);
    dec.push_fec(&[]);
    dec.push_fec(&[0xFF; 3]);
    dec.push_fec(&[0x80, 96, 0, 1]);

    for (i, p) in media.iter().enumerate() {
        if i != 12 {
            dec.push_media(seq_of(p), p);
        }
    }
    assert!(
        dec.take_recovered().is_empty(),
        "no truncated or bogus repair produced output",
    );

    // The intact repair still works after all that junk.
    dec.push_fec(&good);
    let recovered = dec.take_recovered();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0], media[12]);
}

#[test]
fn a_repair_built_over_a_non_contiguous_group_is_refused() {
    // A sender with a gap in its media sequence must not emit a repair: the
    // receiver derives the protected set from SNBase / offset / NA alone.
    let media = stream(20);
    let column: Vec<&[u8]> = [2usize, 7, 17]
        .iter()
        .map(|i| media[*i].as_slice())
        .collect();
    assert!(
        build_repair_packet(&column, FecStream::Column, 5, 96, 0, 0, 0).is_none(),
        "the group is not SNBase + i * offset",
    );
    let column: Vec<&[u8]> = [2usize, 7, 12]
        .iter()
        .map(|i| media[*i].as_slice())
        .collect();
    let fec = build_repair_packet(&column, FecStream::Column, 5, 96, 0, 0, 0).expect("built");
    assert_eq!(fec.len(), RTP_HEADER + FEC_HEADER + 7 * 188);
    let present = [(2u16, media[2].as_slice()), (12u16, media[12].as_slice())];
    assert_eq!(recover_packet(&fec, &present).unwrap(), media[7]);
}
