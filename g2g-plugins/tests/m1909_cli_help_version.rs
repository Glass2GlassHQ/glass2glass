#![cfg(feature = "std")]

use std::process::Command;

const BINARIES: [(&str, &str); 2] = [
    ("g2g-launch", env!("CARGO_BIN_EXE_g2g-launch")),
    ("g2g-inspect", env!("CARGO_BIN_EXE_g2g-inspect")),
];

#[test]
fn help_prints_the_usage_and_exits_zero() {
    for (name, binary) in BINARIES {
        for flag in ["-h", "--help"] {
            let output = Command::new(binary).arg(flag).output().expect("runs");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(output.status.success(), "{name} {flag}: {output:?}");
            assert!(
                stdout.starts_with(&format!("usage: {name} ")),
                "{name} {flag}: {stdout}"
            );
        }
    }
}

#[test]
fn version_prints_the_crate_version_and_exits_zero() {
    for (name, binary) in BINARIES {
        let output = Command::new(binary)
            .arg("--version")
            .output()
            .expect("runs");
        assert!(output.status.success(), "{name}: {output:?}");
        assert_eq!(
            String::from_utf8_lossy(&output.stdout),
            format!("{name} {}\n", env!("CARGO_PKG_VERSION"))
        );
        assert!(output.stderr.is_empty(), "{name}: {output:?}");
    }
}
