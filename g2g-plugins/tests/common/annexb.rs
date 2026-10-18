/// Split an Annex-B stream into NAL units (each retaining its leading start code).
fn nals(mut data: &[u8]) -> Vec<Vec<u8>> {
    // find the first start code
    let mut out = Vec::new();
    let sc = |d: &[u8]| d.windows(3).position(|w| w == [0, 0, 1]);
    let Some(first) = sc(data) else { return out };
    data = &data[first..];
    while !data.is_empty() {
        // skip this NAL's start code, find the next one
        let next = sc(&data[3..]).map(|p| p + 3);
        let end = next.unwrap_or(data.len());
        // trim a 4-byte start code's leading zero back onto the previous NAL split
        let mut nal_end = end;
        if let Some(n) = next {
            if data[n - 1] == 0 {
                nal_end = n - 1;
            }
        }
        out.push(data[..nal_end].to_vec());
        data = &data[nal_end..];
    }
    out
}

/// Group NALs into access units: a VCL NAL (type 1..=5) closes the AU it ends, so
/// SPS/PPS attach to the following IDR and each coded picture is its own AU.
pub(crate) fn access_units(data: &[u8]) -> Vec<Vec<u8>> {
    let mut aus = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut cur_has_vcl = false;
    for nal in nals(data) {
        let sc_len = if nal.starts_with(&[0, 0, 0, 1]) { 4 } else { 3 };
        let nal_type = nal.get(sc_len).map(|b| b & 0x1F).unwrap_or(0);
        let is_vcl = (1..=5).contains(&nal_type);
        if is_vcl && cur_has_vcl {
            aus.push(core::mem::take(&mut cur));
            cur_has_vcl = false;
        }
        cur.extend_from_slice(&nal);
        cur_has_vcl |= is_vcl;
    }
    if !cur.is_empty() {
        aus.push(cur);
    }
    aus
}
