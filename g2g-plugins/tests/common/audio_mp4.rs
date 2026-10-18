#![allow(dead_code)] // not every test file uses every helper here

// AAC-ELD, 48 kHz, stereo, as ffmpeg's FATE sample er_eld2000np_48_ep0 carries it
pub(crate) const ELD_CONFIG: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
pub(crate) const CHANNELS: u16 = 2;
pub(crate) const SAMPLE_RATE: u32 = 48_000;
const SAMPLES_PER_ACCESS_UNIT: u32 = 480;
const TRACK_ID: u32 = 1;
// esds objectTypeIndication of MPEG-4 audio
pub(crate) const MPEG4_AUDIO: u8 = 0x40;
// MPEG-4 descriptor tags (ISO/IEC 14496-1 7.2.2.1)
const ES_DESCRIPTOR_TAG: u8 = 0x03;
const DECODER_CONFIG_DESCRIPTOR_TAG: u8 = 0x04;
const DECODER_SPECIFIC_INFO_TAG: u8 = 0x05;
// the DecoderConfigDescriptor byte holding streamType audio, upStream 0 and the reserved bit
const AUDIO_STREAM_TYPE_BYTE: u8 = 0x15;
// trun flags: data offset present, per-sample duration, per-sample size
const TRUN_DATA_OFFSET: u32 = 0x001;
const TRUN_SAMPLE_DURATION: u32 = 0x100;
const TRUN_SAMPLE_SIZE: u32 = 0x200;

// the second unit holds a valid ADTS header for 16 kHz mono, which a parser scanning raw bytes would announce
pub(crate) fn access_units() -> Vec<Vec<u8>> {
    vec![
        vec![0x7E, 0x89, 0xB4],
        vec![0xFF, 0xF1, 0x60, 0x40, 0x00, 0xFF, 0xFC],
        vec![0x56, 0x78, 0x9A, 0xBC],
    ]
}

pub(crate) fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

pub(crate) fn full_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
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

// an `mp4a` sample entry whose esds names `object_type_indication` and carries `config`
pub(crate) fn mp4a_entry(object_type_indication: u8, config: &[u8]) -> Vec<u8> {
    let decoder_config = [
        &[object_type_indication, AUDIO_STREAM_TYPE_BYTE][..],
        &[0u8; 11],
        &descriptor(DECODER_SPECIFIC_INFO_TAG, config),
    ]
    .concat();
    let es = descriptor(
        ES_DESCRIPTOR_TAG,
        &[
            &[0u8; 3][..],
            &descriptor(DECODER_CONFIG_DESCRIPTOR_TAG, &decoder_config),
        ]
        .concat(),
    );
    let mut mp4a = sound_description(0);
    mp4a.extend_from_slice(&full_box(b"esds", &es));
    mp4_box(b"mp4a", &mp4a)
}

// a version 0 sound description at CHANNELS and SAMPLE_RATE, ahead of any child boxes
fn sound_description(sample_bits: u16) -> Vec<u8> {
    let mut entry = vec![0u8; 28];
    entry[16..18].copy_from_slice(&CHANNELS.to_be_bytes());
    entry[18..20].copy_from_slice(&sample_bits.to_be_bytes());
    let rate = u16::try_from(SAMPLE_RATE).expect("the integer half of a 16.16 rate");
    entry[24..26].copy_from_slice(&rate.to_be_bytes());
    entry
}

// a sample entry with no child boxes, as QuickTime writes for PCM and `.mp3`
pub(crate) fn sound_entry(fourcc: &[u8; 4], sample_bits: u16) -> Vec<u8> {
    mp4_box(fourcc, &sound_description(sample_bits))
}

// QuickTime's default movie timescale
pub(crate) const MOVIE_TIMESCALE: u32 = 600;
// an elst media_rate of 1.0 in 16.16 fixed point
const NORMAL_RATE: u32 = 0x0001_0000;

// one media edit: where it starts in the media, in sample frames, and how long it runs, in movie ticks
#[derive(Debug, Clone, Copy)]
pub(crate) struct AudioEdit {
    pub(crate) media_time: u32,
    pub(crate) duration: u32,
}

// an mvhd at MOVIE_TIMESCALE and the track's one-edit edts
fn edit_boxes(edit: AudioEdit) -> (Vec<u8>, Vec<u8>) {
    let mut mvhd = vec![0u8; 96];
    mvhd[8..12].copy_from_slice(&MOVIE_TIMESCALE.to_be_bytes());
    let elst = full_box(
        b"elst",
        &table(&[1, edit.duration, edit.media_time, NORMAL_RATE]),
    );
    (full_box(b"mvhd", &mvhd), mp4_box(b"edts", &elst))
}

// a moov with one sound track: `sample_entry`, the rest of its sample tables, and an `mvex` when fragmented
fn audio_moov(sample_entry: &[u8], sample_tables: &[u8], mvex: &[u8]) -> Vec<u8> {
    edited_audio_moov(sample_entry, sample_tables, mvex, None)
}

fn edited_audio_moov(
    sample_entry: &[u8],
    sample_tables: &[u8],
    mvex: &[u8],
    edit: Option<AudioEdit>,
) -> Vec<u8> {
    let mut tkhd = vec![0u8; 80];
    tkhd[8..12].copy_from_slice(&TRACK_ID.to_be_bytes());
    let mut mdhd = vec![0u8; 16];
    mdhd[8..12].copy_from_slice(&SAMPLE_RATE.to_be_bytes());
    let mut hdlr = vec![0u8; 20];
    hdlr[4..8].copy_from_slice(b"soun");
    let stsd = full_box(b"stsd", &[table(&[1]), sample_entry.to_vec()].concat());
    let stbl = mp4_box(b"stbl", &[stsd, sample_tables.to_vec()].concat());
    let mdia = mp4_box(
        b"mdia",
        &[
            full_box(b"mdhd", &mdhd),
            full_box(b"hdlr", &hdlr),
            mp4_box(b"minf", &stbl),
        ]
        .concat(),
    );
    let (mvhd, edts) = edit.map(edit_boxes).unwrap_or_default();
    let trak = mp4_box(b"trak", &[full_box(b"tkhd", &tkhd), edts, mdia].concat());
    mp4_box(b"moov", &[mvhd, trak, mvex.to_vec()].concat())
}

// a progressive file with one sound track: ftyp, mdat, then the moov indexing the mdat as one chunk
pub(crate) fn progressive_audio_mp4(sample_entry: &[u8], units: &[Vec<u8>]) -> Vec<u8> {
    let ftyp = mp4_box(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = mp4_box(b"mdat", &units.concat());
    let mdat_payload_offset = (ftyp.len() + 8) as u32;
    let count = units.len() as u32;
    let sizes: Vec<u32> = units.iter().map(|unit| unit.len() as u32).collect();
    let tables = [
        full_box(b"stts", &table(&[1, count, SAMPLES_PER_ACCESS_UNIT])),
        full_box(b"stsc", &table(&[1, 1, count, 1])),
        full_box(b"stsz", &[table(&[0, count]), table(&sizes)].concat()),
        full_box(b"stco", &table(&[1, mdat_payload_offset])),
    ]
    .concat();
    [ftyp, mdat, audio_moov(sample_entry, &tables, &[])].concat()
}

// QuickTime's stsz gives PCM frames this size whatever their real size
const QUICKTIME_PCM_SAMPLE_SIZE: u32 = 1;

// a QuickTime PCM file with each of `chunks` a chunk of `frame_bytes` frames, one stsz sample per frame
pub(crate) fn quicktime_pcm_mov(
    sample_entry: &[u8],
    chunks: &[Vec<u8>],
    frame_bytes: usize,
) -> Vec<u8> {
    edited_quicktime_pcm_mov(sample_entry, chunks, frame_bytes, None)
}

// as `quicktime_pcm_mov`, its track presenting only `edit` when given
pub(crate) fn edited_quicktime_pcm_mov(
    sample_entry: &[u8],
    chunks: &[Vec<u8>],
    frame_bytes: usize,
    edit: Option<AudioEdit>,
) -> Vec<u8> {
    let ftyp = mp4_box(b"ftyp", b"qt  \0\0\0\0qt  ");
    let mdat = mp4_box(b"mdat", &chunks.concat());
    let mut offset = (ftyp.len() + 8) as u32;
    let mut stsc = vec![chunks.len() as u32];
    let mut stco = vec![chunks.len() as u32];
    for (index, chunk) in chunks.iter().enumerate() {
        stsc.extend([index as u32 + 1, (chunk.len() / frame_bytes) as u32, 1]);
        stco.push(offset);
        offset += chunk.len() as u32;
    }
    let frames = (chunks.concat().len() / frame_bytes) as u32;
    let tables = [
        full_box(b"stts", &table(&[1, frames, 1])),
        full_box(b"stsc", &table(&stsc)),
        full_box(b"stsz", &table(&[QUICKTIME_PCM_SAMPLE_SIZE, frames])),
        full_box(b"stco", &table(&stco)),
    ]
    .concat();
    [
        ftyp,
        mdat,
        edited_audio_moov(sample_entry, &tables, &[], edit),
    ]
    .concat()
}

// a fragmented file with one sound track: ftyp, a moov with empty sample tables, then one moof + mdat
pub(crate) fn fragmented_audio_mp4(sample_entry: &[u8], units: &[Vec<u8>]) -> Vec<u8> {
    let ftyp = mp4_box(b"ftyp", b"iso6\0\0\0\0iso6");
    let empty_tables = [
        full_box(b"stts", &table(&[0])),
        full_box(b"stsc", &table(&[0])),
        full_box(b"stsz", &table(&[0, 0])),
        full_box(b"stco", &table(&[0])),
    ]
    .concat();
    let trex = full_box(b"trex", &table(&[TRACK_ID, 1, 0, 0, 0]));
    let moov = audio_moov(sample_entry, &empty_tables, &mp4_box(b"mvex", &trex));

    let mfhd = full_box(b"mfhd", &table(&[1]));
    let tfhd = full_box(b"tfhd", &table(&[TRACK_ID]));
    let tfdt = full_box(b"tfdt", &table(&[0]));
    let moof_with_data_offset = |data_offset: u32| {
        let flags = TRUN_DATA_OFFSET | TRUN_SAMPLE_DURATION | TRUN_SAMPLE_SIZE;
        let mut trun = table(&[flags, units.len() as u32, data_offset]);
        for unit in units {
            trun.extend_from_slice(&table(&[SAMPLES_PER_ACCESS_UNIT, unit.len() as u32]));
        }
        let traf = mp4_box(
            b"traf",
            &[tfhd.clone(), tfdt.clone(), mp4_box(b"trun", &trun)].concat(),
        );
        mp4_box(b"moof", &[mfhd.clone(), traf].concat())
    };
    // the data offset counts from the moof start to the mdat payload, and its value does not change the moof's length
    let data_offset = (moof_with_data_offset(0).len() + 8) as u32;
    let moof = moof_with_data_offset(data_offset);
    [ftyp, moov, moof, mp4_box(b"mdat", &units.concat())].concat()
}

pub(crate) fn progressive_eld_mp4() -> Vec<u8> {
    progressive_audio_mp4(&mp4a_entry(MPEG4_AUDIO, &ELD_CONFIG), &access_units())
}
