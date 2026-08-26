//! `g2g-plugin-sign`: keygen, sign, and verify for detached plugin signatures
//! (M1061).
//!
//! Usage:
//!   g2g-plugin-sign keygen <out-private> <out-public>
//!   g2g-plugin-sign sign <private-key> <plugin>     # writes <plugin>.sig
//!   g2g-plugin-sign verify <public-key> <plugin>
//!
//! The private key is a PKCS#8 v2 Ed25519 document written 0600, the public key
//! is 64 hex characters, and the `.sig` format is documented in
//! [`g2g_plugins::plugin_loader::signing`]. Point a host at the public key with
//! `$G2G_PLUGIN_TRUSTED_KEYS` (or `g2g-inspect --trusted-key`) and it will load
//! only plugins signed by the matching private key.

use std::process;

use g2g_plugins::plugin_loader::signing::{
    read_public_key_file, verify_plugin, write_public_key_file, SigningKey, TrustedKeys,
};

const USAGE: &str = "usage:
  g2g-plugin-sign keygen <out-private> <out-public>
  g2g-plugin-sign sign <private-key> <plugin>
  g2g-plugin-sign verify <public-key> <plugin>";

fn fail(message: impl core::fmt::Display) -> ! {
    eprintln!("g2g-plugin-sign: {message}");
    process::exit(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("");
    match (command, args.len()) {
        ("keygen", 3) => keygen(&args[1], &args[2]),
        ("sign", 3) => sign(&args[1], &args[2]),
        ("verify", 3) => verify(&args[1], &args[2]),
        _ => {
            eprintln!("{USAGE}");
            process::exit(2);
        }
    }
}

fn keygen(private_path: &str, public_path: &str) {
    let key = SigningKey::generate().unwrap_or_else(|e| fail(e));
    key.write_to(private_path).unwrap_or_else(|e| fail(e));
    write_public_key_file(public_path, &key.public_key()).unwrap_or_else(|e| fail(e));
    println!("private key: {private_path} (mode 0600, keep it off shared machines)");
    println!("public key:  {public_path}");
}

fn sign(private_path: &str, plugin: &str) {
    let key = SigningKey::read_from(private_path).unwrap_or_else(|e| fail(e));
    let written = key.sign_plugin(plugin).unwrap_or_else(|e| fail(e));
    println!("{}", written.display());
}

fn verify(public_path: &str, plugin: &str) {
    let key = read_public_key_file(public_path).unwrap_or_else(|e| fail(e));
    let mut trusted = TrustedKeys::new();
    trusted.trust(key);
    let bytes = std::fs::read(plugin).unwrap_or_else(|e| fail(e));
    match verify_plugin(std::path::Path::new(plugin), &bytes, &trusted) {
        Ok(()) => println!("{plugin}: signature verifies"),
        Err(e) => fail(e),
    }
}
