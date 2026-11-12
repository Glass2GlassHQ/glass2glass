// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // no one test file uses every helper here

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
        // h264 7.4.1.2.3: SEI, SPS, PPS, AUD and types 14 to 18 after a slice open the next access unit
        let opens_access_unit = matches!(nal_type, 6..=9 | 14..=18);
        if (is_vcl || opens_access_unit) && cur_has_vcl {
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

pub(crate) fn h265_parameter_sets(data: &[u8]) -> [Vec<u8>; 3] {
    let without_start_codes: Vec<Vec<u8>> = nals(data)
        .into_iter()
        .map(|nal| {
            let start_code_end = nal.iter().position(|&byte| byte == 1).expect("start code") + 1;
            nal[start_code_end..].to_vec()
        })
        .collect();
    [32, 33, 34].map(|wanted_type| {
        without_start_codes
            .iter()
            .find(|nal| nal.first().map(|header| (header >> 1) & 0x3F) == Some(wanted_type))
            .expect("the stream carries VPS, SPS and PPS")
            .clone()
    })
}
