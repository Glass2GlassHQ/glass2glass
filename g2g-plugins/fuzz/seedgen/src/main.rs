use g2g_plugins::rtmphandshake::{build_c1, build_s1};

fn main() {
    let out = std::env::args().nth(1).expect("usage: seedgen <corpus-dir>");
    std::fs::create_dir_all(&out).unwrap();
    for t in [0u32, 1, 42, 1000, 0x0100_0000] {
        std::fs::write(format!("{out}/c1_{t}"), build_c1(t)).unwrap();
        std::fs::write(format!("{out}/s1_{t}"), build_s1(t)).unwrap();
    }
    println!("wrote RTMP C1/S1 seeds to {out}");
}
