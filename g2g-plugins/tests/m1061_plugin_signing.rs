//! M1061: detached Ed25519 signatures, checked before the `dlopen`.
//!
//! Every case uses the real v2 example plugin: a `cdylib` that loads and
//! registers `v2counter`. So when a refusal test asserts the registry has no
//! `v2counter`, it is asserting the refusal happened, not that the file was
//! broken anyway.
//!
//! Requires the `plugin-signing` feature:
//!   cargo test -p g2g-plugins --features plugin-signing --test m1061_plugin_signing

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::plugin_loader::signing::{SigningError, SigningKey, TrustedKeys};
use g2g_plugins::plugin_loader::{
    self, default_policy, PluginError, PLUGIN_PATH_ENV, TRUSTED_KEYS_ENV,
};
use g2g_plugins::registry::default_registry;

mod common;
use common::build_fixture;

/// The element the fixture registers, and the only evidence that a load
/// actually happened.
const PLUGIN_ELEMENT: &str = "v2counter";

/// A fresh empty directory for one test to own, under cargo's per-suite temp
/// directory.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("m1061")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

/// Copy the built fixture into `dir`, so a test can sign or corrupt its own
/// copy without touching the shared build output.
fn plugin_copy(dir: &Path) -> PathBuf {
    let built = build_fixture();
    let target = dir.join(built.file_name().expect("library file name"));
    std::fs::copy(&built, &target).expect("copy the fixture next to its signature");
    target
}

fn trusting(key: &SigningKey) -> TrustedKeys {
    let mut trusted = TrustedKeys::new();
    trusted.trust(key.public_key());
    trusted
}

/// Load with verification and report what the registry ended up with.
fn load(plugin: &Path, trusted: &TrustedKeys) -> (Result<(), PluginError>, bool) {
    let mut reg = default_registry();
    assert!(
        !reg.element_names().contains(&PLUGIN_ELEMENT),
        "{PLUGIN_ELEMENT} must come from the plugin, not the default registry"
    );
    let result = plugin_loader::load_plugin_verified(plugin, &mut reg, trusted, &default_policy);
    let registered = reg.element_names().contains(&PLUGIN_ELEMENT);
    (result, registered)
}

#[test]
fn an_empty_trust_set_loads_a_plugin_signed_or_not() {
    // The default: nothing configured, nothing verified, so a host that never
    // heard of signatures behaves exactly as it did before they existed.
    let dir = scratch("no-trust");
    let plugin = plugin_copy(&dir);

    let (unsigned, registered) = load(&plugin, &TrustedKeys::new());
    unsigned.expect("an unsigned plugin loads when no keys are trusted");
    assert!(registered, "the unsigned load registered its element");

    let key = SigningKey::generate().expect("keygen");
    key.sign_plugin(&plugin).expect("sign");
    let (signed, registered) = load(&plugin, &TrustedKeys::new());
    signed.expect("a signed plugin loads when no keys are trusted");
    assert!(registered, "the signed load registered its element");
}

#[test]
fn a_signed_plugin_loads_and_runs_under_its_trusted_key() {
    // The whole happy path, including the sealed-memfd load on Linux: the
    // frames only arrive if the code really got mapped and ran.
    let dir = scratch("signed");
    let plugin = plugin_copy(&dir);
    let key = SigningKey::generate().expect("keygen");
    let signature = key.sign_plugin(&plugin).expect("sign");
    assert!(signature.is_file(), "the .sig sits beside the plugin");
    assert_eq!(
        signature.file_name().unwrap().to_string_lossy(),
        format!("{}.sig", plugin.file_name().unwrap().to_string_lossy())
    );

    let mut reg = default_registry();
    plugin_loader::load_plugin_verified(&plugin, &mut reg, &trusting(&key), &default_policy)
        .expect("a plugin signed by a trusted key loads");
    assert!(reg.element_names().contains(&PLUGIN_ELEMENT));

    let graph = parse_launch(
        &reg,
        &format!("videotestsrc num-buffers=3 ! {PLUGIN_ELEMENT} ! fakesink"),
    )
    .expect("pipeline parses");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let stats = rt
        .block_on(run_graph(graph, &WallClock::new(), 4))
        .expect("pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "the verified image is what got loaded and ran"
    );

    // The TOCTOU closure, checked rather than assumed: on Linux the mapped code
    // comes from the sealed memfd, not from a second read of the path.
    #[cfg(target_os = "linux")]
    {
        let maps = std::fs::read_to_string("/proc/self/maps").expect("read our own mappings");
        assert!(
            maps.contains("/memfd:g2g-plugin"),
            "the verified plugin is mapped from a memfd, not reopened from its path"
        );
    }
}

#[test]
fn an_unsigned_plugin_is_refused_once_a_key_is_trusted() {
    let dir = scratch("unsigned");
    let plugin = plugin_copy(&dir);
    let key = SigningKey::generate().expect("keygen");

    let (result, registered) = load(&plugin, &trusting(&key));
    match result {
        Err(PluginError::SignatureRejected {
            error: SigningError::Missing { signature },
            ..
        }) => assert_eq!(signature, plugin_loader::signing::signature_path(&plugin)),
        other => panic!("an unsigned plugin must be refused, got {other:?}"),
    }
    assert!(!registered, "a refused plugin registers nothing");
}

#[test]
fn a_signature_from_an_untrusted_key_is_refused() {
    // The signature is valid, over these exact bytes. It just is not from a key
    // this host trusts, which is the case a stolen-but-unlisted signer covers.
    let dir = scratch("other-key");
    let plugin = plugin_copy(&dir);
    let signer = SigningKey::generate().expect("keygen");
    let trusted_key = SigningKey::generate().expect("keygen");
    signer.sign_plugin(&plugin).expect("sign");

    let (result, registered) = load(&plugin, &trusting(&trusted_key));
    match result {
        Err(PluginError::SignatureRejected {
            error: SigningError::UntrustedSigner { .. },
            ..
        }) => {}
        other => panic!("a foreign signer must be refused, got {other:?}"),
    }
    assert!(!registered, "a refused plugin registers nothing");
}

#[test]
fn a_plugin_modified_after_signing_is_refused() {
    // Appending to an ELF leaves it perfectly loadable, so this file would load
    // and register its element if the signature were not checked.
    let dir = scratch("modified");
    let plugin = plugin_copy(&dir);
    let key = SigningKey::generate().expect("keygen");
    key.sign_plugin(&plugin).expect("sign");

    let mut bytes = std::fs::read(&plugin).expect("read the signed plugin");
    bytes.push(0);
    std::fs::write(&plugin, &bytes).expect("modify after signing");

    let (result, registered) = load(&plugin, &trusting(&key));
    match result {
        Err(PluginError::SignatureRejected {
            error: SigningError::Invalid,
            ..
        }) => {}
        other => panic!("a modified plugin must be refused, got {other:?}"),
    }
    assert!(!registered, "a refused plugin registers nothing");

    // The same bytes with no trust set do load, which is what makes the refusal
    // above a signature decision rather than a broken file.
    let (unverified, registered) = load(&plugin, &TrustedKeys::new());
    unverified.expect("the modified file is still a loadable library");
    assert!(registered);
}

#[test]
fn a_truncated_signature_file_is_refused() {
    let dir = scratch("truncated");
    let plugin = plugin_copy(&dir);
    let key = SigningKey::generate().expect("keygen");
    let signature = key.sign_plugin(&plugin).expect("sign");
    let full = std::fs::read(&signature).expect("read the .sig");
    std::fs::write(&signature, &full[..full.len() / 2]).expect("truncate the .sig");

    let (result, registered) = load(&plugin, &trusting(&key));
    match result {
        Err(PluginError::SignatureRejected {
            error: SigningError::Malformed { .. },
            ..
        }) => {}
        other => panic!("a truncated signature must be refused, got {other:?}"),
    }
    assert!(!registered, "a refused plugin registers nothing");
}

#[test]
fn the_trusted_keys_environment_variable_gates_a_directory_scan() {
    // $G2G_PLUGIN_TRUSTED_KEYS is what a packaged g2g-launch / g2g-inspect is
    // configured through, so it has to reach load_from_env. Both halves live in
    // one test: the variables are process-global and the rest of this suite runs
    // in parallel threads beside it.
    let dir = scratch("from-env");
    let plugin = plugin_copy(&dir);
    let key = SigningKey::generate().expect("keygen");
    let key_file = dir.join("signer.pub");
    plugin_loader::signing::write_public_key_file(&key_file, &key.public_key()).expect("key file");
    std::env::set_var(PLUGIN_PATH_ENV, &dir);
    std::env::set_var(TRUSTED_KEYS_ENV, &key_file);

    let mut reg = default_registry();
    match plugin_loader::load_from_env(&mut reg) {
        Err(PluginError::SignatureRejected {
            error: SigningError::Missing { .. },
            ..
        }) => {}
        other => panic!("the env trust set must gate the scan, got {other:?}"),
    }
    assert!(!reg.element_names().contains(&PLUGIN_ELEMENT));

    key.sign_plugin(&plugin).expect("sign");
    let mut reg = default_registry();
    let loaded = plugin_loader::load_from_env(&mut reg).expect("the signed plugin passes the gate");
    assert_eq!(loaded, vec![plugin]);
    assert!(reg.element_names().contains(&PLUGIN_ELEMENT));

    std::env::remove_var(PLUGIN_PATH_ENV);
    std::env::remove_var(TRUSTED_KEYS_ENV);
}

#[test]
fn an_unreadable_trusted_key_file_is_an_error_not_an_empty_trust_set() {
    // Fail closed: a typo in the key path must not quietly turn verification
    // off for every plugin the host loads.
    let dir = scratch("bad-key");
    let mut trusted = TrustedKeys::new();
    let missing = dir.join("nonesuch.pub");
    match trusted.trust_key_file(&missing) {
        Err(SigningError::Io { .. }) => {}
        other => panic!("a missing key file must error, got {other:?}"),
    }

    let garbage = dir.join("garbage.pub");
    std::fs::write(&garbage, "not a key").expect("write");
    match trusted.trust_key_file(&garbage) {
        Err(SigningError::BadKeyFile { .. }) => {}
        other => panic!("a non-hex key file must error, got {other:?}"),
    }
    assert!(trusted.is_empty());
}
