#![allow(dead_code)] // not every test file uses every helper here

// AAC-ELD, 48 kHz, stereo, as ffmpeg's FATE sample er_eld2000np_48_ep0 carries it
pub(crate) const ELD_CONFIG: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
pub(crate) const CHANNELS: u16 = 2;
pub(crate) const SAMPLE_RATE: u32 = 48_000;
const SAMPLES_PER_ACCESS_UNIT: u32 = 480;
const TRACK_ID: u32 = 1;

// the second unit holds a valid ADTS header for 16 kHz mono, which a parser scanning raw bytes would announce
pub(crate) fn access_units() -> Vec<Vec<u8>> {
    vec![
        vec![0x7E, 0x89, 0xB4],
        vec![0xFF, 0xF1, 0x60, 0x40, 0x00, 0xFF, 0xFC],
        vec![0x56, 0x78, 0x9A, 0xBC],
    ]
}

fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn full_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    mp4_box(kind, &[&[0u8; 4][..], payload].concat())
}

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    [&[tag, body.len() as u8][..], body].concat()
}

fn table(entries: &[u32]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

// a progressive file with one AAC-ELD track: ftyp, mdat, then the moov indexing the mdat as one chunk
pub(crate) fn progressive_eld_mp4() -> Vec<u8> {
    let units = access_units();
    let ftyp = mp4_box(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = mp4_box(b"mdat", &units.concat());
    let mdat_payload_offset = (ftyp.len() + 8) as u32;
    let count = units.len() as u32;

    let mut tkhd = vec![0u8; 80];
    tkhd[8..12].copy_from_slice(&TRACK_ID.to_be_bytes());
    let mut mdhd = vec![0u8; 16];
    mdhd[8..12].copy_from_slice(&SAMPLE_RATE.to_be_bytes());
    mdhd[12..16].copy_from_slice(&(count * SAMPLES_PER_ACCESS_UNIT).to_be_bytes());
    let mut hdlr = vec![0u8; 20];
    hdlr[4..8].copy_from_slice(b"soun");

    let decoder_config = [
        &[0x40, 0x15][..],
        &[0u8; 11],
        &descriptor(0x05, &ELD_CONFIG),
    ]
    .concat();
    let es = descriptor(
        0x03,
        &[&[0u8; 3][..], &descriptor(0x04, &decoder_config)].concat(),
    );
    let mut mp4a = vec![0u8; 28];
    mp4a[16..18].copy_from_slice(&CHANNELS.to_be_bytes());
    mp4a.extend_from_slice(&full_box(b"esds", &es));

    let sizes: Vec<u32> = units.iter().map(|unit| unit.len() as u32).collect();
    let stbl = [
        full_box(b"stsd", &[table(&[1]), mp4_box(b"mp4a", &mp4a)].concat()),
        full_box(b"stts", &table(&[1, count, SAMPLES_PER_ACCESS_UNIT])),
        full_box(b"stsc", &table(&[1, 1, count, 1])),
        full_box(b"stsz", &[table(&[0, count]), table(&sizes)].concat()),
        full_box(b"stco", &table(&[1, mdat_payload_offset])),
    ]
    .concat();
    let minf = mp4_box(b"minf", &mp4_box(b"stbl", &stbl));
    let mdia = mp4_box(
        b"mdia",
        &[full_box(b"mdhd", &mdhd), full_box(b"hdlr", &hdlr), minf].concat(),
    );
    let trak = mp4_box(b"trak", &[full_box(b"tkhd", &tkhd), mdia].concat());
    [ftyp, mdat, mp4_box(b"moov", &trak)].concat()
}
