//! Detached Ed25519 signatures over a plugin file, checked before `dlopen`
//! (M1061).
//!
//! # The `.sig` file
//!
//! A plugin `libfoo.so` is signed by a sibling `libfoo.so.sig`, 103 bytes with
//! no length fields and nothing to allocate on:
//!
//! ```text
//! offset  len  field
//!      0    6  magic, ASCII "G2GSIG"
//!      6    1  format version, currently 1
//!      7   32  the signer's Ed25519 public key
//!     39   64  Ed25519 signature over the exact bytes of the plugin file
//! ```
//!
//! The public key rides along so one directory can hold plugins from several
//! signers and the loader can name the offending signer in an error. It is a
//! *selector*, not a credential: the key must appear in the caller's trust set
//! before the signature is checked against it, so a forged `.sig` carrying its
//! own key is refused with [`SigningError::UntrustedSigner`].
//!
//! # Trusted key files
//!
//! A public key file is 64 hex characters (32 bytes), surrounding whitespace
//! ignored: text, so it can be pasted into a shell or a config repository
//! without an encoding step. A private key file is the raw PKCS#8 v2 document
//! `ring` generates and parses, written 0600 on Unix.
//!
//! # TOCTOU
//!
//! Verifying bytes read from a path and then handing `dlopen` the same path
//! leaves a window where the file can be swapped. On Linux the verified bytes
//! are copied into a `memfd_create` object, sealed against further writes, and
//! opened through `/proc/self/fd/N`, so the loaded code is the code that was
//! verified. Two consequences of loading from `/proc/self/fd/N`: an `$ORIGIN`
//! rpath in the plugin resolves against `/proc/self`, not the plugin's
//! directory, and `/proc/self/maps` names the memfd rather than the file. Both
//! only apply on the verified path (a caller with an empty trust set still gets
//! a plain path `dlopen`). On other platforms the window stays open and the
//! path is opened directly.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use libloading::Library;
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};

use super::PluginError;

/// Bytes in an Ed25519 public key.
pub const PUBLIC_KEY_LEN: usize = 32;
/// Bytes in an Ed25519 signature.
pub const SIGNATURE_LEN: usize = 64;

const MAGIC: &[u8; 6] = b"G2GSIG";
const FORMAT_VERSION: u8 = 1;

/// Size of a `.sig` file: magic + version + signer key + signature.
pub const SIGNATURE_FILE_LEN: usize = MAGIC.len() + 1 + PUBLIC_KEY_LEN + SIGNATURE_LEN;

/// An Ed25519 public key, the 32-byte encoded point.
pub type PublicKey = [u8; PUBLIC_KEY_LEN];

/// Why a signature check failed.
#[derive(Debug)]
pub enum SigningError {
    /// The trust set is non-empty and the plugin has no `.sig` beside it.
    Missing { signature: PathBuf },
    /// The `.sig` is not this format: wrong size, wrong magic, or a format
    /// version this build does not know.
    Malformed { signature: PathBuf, reason: String },
    /// A file could not be read or written.
    Io { path: PathBuf, message: String },
    /// The `.sig` is well formed but names a signer outside the trust set.
    UntrustedSigner { signer: String },
    /// The signature does not verify: the plugin file changed after signing, or
    /// the `.sig` was made over different bytes.
    Invalid,
    /// A trusted-key file did not hold 64 hex characters.
    BadKeyFile { path: PathBuf, reason: String },
    /// A private key file is not a PKCS#8 v2 Ed25519 document.
    BadPrivateKey { path: PathBuf },
    /// The OS random source would not produce a key.
    KeyGeneration,
}

impl core::fmt::Display for SigningError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SigningError::Missing { signature } => write!(
                f,
                "no signature at {} (this host only loads signed plugins)",
                signature.display()
            ),
            SigningError::Malformed { signature, reason } => {
                write!(
                    f,
                    "signature {} is malformed: {reason}",
                    signature.display()
                )
            }
            SigningError::Io { path, message } => {
                write!(f, "{}: {message}", path.display())
            }
            SigningError::UntrustedSigner { signer } => {
                write!(f, "signed by {signer}, which is not a trusted key")
            }
            SigningError::Invalid => write!(
                f,
                "signature does not verify: the file changed after it was signed"
            ),
            SigningError::BadKeyFile { path, reason } => write!(
                f,
                "trusted key file {} is unusable: {reason}",
                path.display()
            ),
            SigningError::BadPrivateKey { path } => write!(
                f,
                "{} is not a PKCS#8 v2 Ed25519 private key",
                path.display()
            ),
            SigningError::KeyGeneration => {
                write!(f, "the system random source would not produce a key")
            }
        }
    }
}

impl std::error::Error for SigningError {}

/// The `.sig` path for a plugin: the whole file name plus `.sig`, so
/// `libfoo.so` pairs with `libfoo.so.sig`.
pub fn signature_path(plugin: &Path) -> PathBuf {
    let mut name = plugin.as_os_str().to_os_string();
    name.push(".sig");
    PathBuf::from(name)
}

/// A decoded `.sig` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetachedSignature {
    /// The public key the signature was made with.
    pub signer: PublicKey,
    /// The Ed25519 signature over the plugin file's bytes.
    pub signature: [u8; SIGNATURE_LEN],
}

impl DetachedSignature {
    /// Serialize to the on-disk form.
    pub fn encode(&self) -> [u8; SIGNATURE_FILE_LEN] {
        let mut out = [0u8; SIGNATURE_FILE_LEN];
        out[..MAGIC.len()].copy_from_slice(MAGIC);
        out[MAGIC.len()] = FORMAT_VERSION;
        let key_at = MAGIC.len() + 1;
        out[key_at..key_at + PUBLIC_KEY_LEN].copy_from_slice(&self.signer);
        out[key_at + PUBLIC_KEY_LEN..].copy_from_slice(&self.signature);
        out
    }

    /// Parse the on-disk form. Every field is fixed-width, so the only checks
    /// are the exact length, the magic, and the version byte.
    pub fn decode(bytes: &[u8], signature_path: &Path) -> Result<Self, SigningError> {
        let malformed = |reason: String| SigningError::Malformed {
            signature: signature_path.to_path_buf(),
            reason,
        };
        if bytes.len() != SIGNATURE_FILE_LEN {
            return Err(malformed(format!(
                "{} bytes, expected {SIGNATURE_FILE_LEN}",
                bytes.len()
            )));
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(malformed("not a g2g signature (bad magic)".to_string()));
        }
        let version = bytes[MAGIC.len()];
        if version != FORMAT_VERSION {
            return Err(malformed(format!(
                "format version {version}, this build reads {FORMAT_VERSION}"
            )));
        }
        let key_at = MAGIC.len() + 1;
        let mut signer = [0u8; PUBLIC_KEY_LEN];
        signer.copy_from_slice(&bytes[key_at..key_at + PUBLIC_KEY_LEN]);
        let mut signature = [0u8; SIGNATURE_LEN];
        signature.copy_from_slice(&bytes[key_at + PUBLIC_KEY_LEN..]);
        Ok(DetachedSignature { signer, signature })
    }

    /// Read and parse a `.sig`. Reads at most one byte more than a well-formed
    /// file, so a huge file is a `Malformed` error rather than a huge read.
    pub fn read(signature_path: &Path) -> Result<Self, SigningError> {
        let file = match std::fs::File::open(signature_path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(SigningError::Missing {
                    signature: signature_path.to_path_buf(),
                })
            }
            Err(e) => {
                return Err(SigningError::Io {
                    path: signature_path.to_path_buf(),
                    message: e.to_string(),
                })
            }
        };
        let mut bytes = Vec::with_capacity(SIGNATURE_FILE_LEN + 1);
        file.take(SIGNATURE_FILE_LEN as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| SigningError::Io {
                path: signature_path.to_path_buf(),
                message: e.to_string(),
            })?;
        DetachedSignature::decode(&bytes, signature_path)
    }
}

/// The Ed25519 public keys a host will accept a plugin signature from.
///
/// Empty is the default and means **no verification**: plugins load signed or
/// unsigned, exactly as they did before signatures existed. One key or more
/// turns verification on for every plugin this host loads.
#[derive(Debug, Clone, Default)]
pub struct TrustedKeys {
    keys: Vec<PublicKey>,
}

impl TrustedKeys {
    /// An empty set: no verification.
    pub fn new() -> Self {
        TrustedKeys::default()
    }

    /// Whether verification is off.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// How many keys are trusted.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Trust one public key.
    pub fn trust(&mut self, key: PublicKey) -> &mut Self {
        if !self.keys.contains(&key) {
            self.keys.push(key);
        }
        self
    }

    /// Trust the key in a hex key file.
    pub fn trust_key_file(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, SigningError> {
        let key = read_public_key_file(path.as_ref())?;
        Ok(self.trust(key))
    }

    /// Trust every key file named in a `PATH`-style list, the form
    /// `G2G_PLUGIN_TRUSTED_KEYS` takes.
    pub fn trust_key_files(&mut self, list: &std::ffi::OsStr) -> Result<&mut Self, SigningError> {
        for path in std::env::split_paths(list) {
            if path.as_os_str().is_empty() {
                continue;
            }
            self.trust_key_file(&path)?;
        }
        Ok(self)
    }

    fn contains(&self, key: &PublicKey) -> bool {
        self.keys.contains(key)
    }
}

/// Check `bytes` (the exact content of the plugin file at `plugin`) against the
/// plugin's `.sig` and this trust set. A caller with an empty trust set must not
/// call this: verification of an unsigned plugin is a refusal, not a pass.
pub fn verify_plugin(
    plugin: &Path,
    bytes: &[u8],
    trusted: &TrustedKeys,
) -> Result<(), SigningError> {
    let detached = DetachedSignature::read(&signature_path(plugin))?;
    if !trusted.contains(&detached.signer) {
        return Err(SigningError::UntrustedSigner {
            signer: to_hex(&detached.signer),
        });
    }
    UnparsedPublicKey::new(&ED25519, detached.signer)
        .verify(bytes, &detached.signature)
        .map_err(|_| SigningError::Invalid)
}

/// An Ed25519 signing key: the PKCS#8 v2 document plus the parsed pair.
#[derive(Debug)]
pub struct SigningKey {
    pair: Ed25519KeyPair,
    pkcs8: Vec<u8>,
}

impl SigningKey {
    /// Generate a fresh key from the OS random source.
    pub fn generate() -> Result<Self, SigningError> {
        let random = ring::rand::SystemRandom::new();
        let document =
            Ed25519KeyPair::generate_pkcs8(&random).map_err(|_| SigningError::KeyGeneration)?;
        SigningKey::from_pkcs8(document.as_ref().to_vec(), Path::new("<generated>"))
    }

    /// Parse a PKCS#8 v2 Ed25519 document. `path` only names the source in an
    /// error message.
    pub fn from_pkcs8(pkcs8: Vec<u8>, path: &Path) -> Result<Self, SigningError> {
        let pair = Ed25519KeyPair::from_pkcs8(&pkcs8).map_err(|_| SigningError::BadPrivateKey {
            path: path.to_path_buf(),
        })?;
        Ok(SigningKey { pair, pkcs8 })
    }

    /// Read a private key file.
    pub fn read_from(path: impl AsRef<Path>) -> Result<Self, SigningError> {
        let path = path.as_ref();
        let pkcs8 = std::fs::read(path).map_err(|e| SigningError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        SigningKey::from_pkcs8(pkcs8, path)
    }

    /// Write the private key, 0600 on Unix, refusing to overwrite an existing
    /// file.
    pub fn write_to(&self, path: impl AsRef<Path>) -> Result<(), SigningError> {
        let path = path.as_ref();
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path).map_err(|e| SigningError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        file.write_all(&self.pkcs8).map_err(|e| SigningError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    }

    /// The matching public key.
    pub fn public_key(&self) -> PublicKey {
        let mut key = [0u8; PUBLIC_KEY_LEN];
        key.copy_from_slice(self.pair.public_key().as_ref());
        key
    }

    /// Sign a message.
    pub fn sign(&self, message: &[u8]) -> DetachedSignature {
        let mut signature = [0u8; SIGNATURE_LEN];
        signature.copy_from_slice(self.pair.sign(message).as_ref());
        DetachedSignature {
            signer: self.public_key(),
            signature,
        }
    }

    /// Sign a plugin file and write the `.sig` beside it, returning its path.
    pub fn sign_plugin(&self, plugin: impl AsRef<Path>) -> Result<PathBuf, SigningError> {
        let plugin = plugin.as_ref();
        let bytes = std::fs::read(plugin).map_err(|e| SigningError::Io {
            path: plugin.to_path_buf(),
            message: e.to_string(),
        })?;
        let target = signature_path(plugin);
        std::fs::write(&target, self.sign(&bytes).encode()).map_err(|e| SigningError::Io {
            path: target.clone(),
            message: e.to_string(),
        })?;
        Ok(target)
    }
}

/// Write a public key file: 64 hex characters and a newline.
pub fn write_public_key_file(path: impl AsRef<Path>, key: &PublicKey) -> Result<(), SigningError> {
    let path = path.as_ref();
    std::fs::write(path, format!("{}\n", to_hex(key))).map_err(|e| SigningError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })
}

/// Read a public key file: 64 hex characters, surrounding whitespace ignored.
pub fn read_public_key_file(path: impl AsRef<Path>) -> Result<PublicKey, SigningError> {
    let path = path.as_ref();
    let bad = |reason: String| SigningError::BadKeyFile {
        path: path.to_path_buf(),
        reason,
    };
    // A key file is 65 bytes; cap the read so a wrong path cannot pull in a
    // large file.
    const MAX_KEY_FILE_LEN: u64 = 256;
    let file = std::fs::File::open(path).map_err(|e| SigningError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let mut text = String::new();
    file.take(MAX_KEY_FILE_LEN)
        .read_to_string(&mut text)
        .map_err(|e| bad(e.to_string()))?;
    let text = text.trim();
    let bytes = from_hex(text).ok_or_else(|| bad("not hexadecimal".to_string()))?;
    let key: PublicKey = bytes.try_into().map_err(|_| {
        bad(format!(
            "{} hex characters, expected {}",
            text.len(),
            PUBLIC_KEY_LEN * 2
        ))
    })?;
    Ok(key)
}

/// `dlopen` bytes that have already been verified, without going back to the
/// path they came from. See the module "TOCTOU" note.
#[cfg(target_os = "linux")]
pub(super) fn open_verified(plugin: &Path, bytes: &[u8]) -> Result<Library, PluginError> {
    use std::os::fd::AsRawFd;

    let name = c"g2g-plugin";
    // SAFETY: `name` is a NUL-terminated C string that outlives the call, and
    // the flags are the documented memfd_create bits.
    let fd =
        unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if fd < 0 {
        return Err(PluginError::Seal {
            path: plugin.to_path_buf(),
            message: format!("memfd_create: {}", std::io::Error::last_os_error()),
        });
    }
    // SAFETY: `memfd_create` returned a fresh descriptor we own exclusively;
    // `File` becomes its only owner and closes it on drop.
    let mut file = unsafe { <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(fd) };

    let seal = |message: String| PluginError::Seal {
        path: plugin.to_path_buf(),
        message,
    };
    file.write_all(bytes)
        .map_err(|e| seal(format!("writing to the memfd: {e}")))?;

    let seals = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE | libc::F_SEAL_SEAL;
    // SAFETY: `F_ADD_SEALS` takes an int argument, and the descriptor is the
    // memfd we just created with `MFD_ALLOW_SEALING`.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
        return Err(seal(format!(
            "sealing the memfd: {}",
            std::io::Error::last_os_error()
        )));
    }

    let fd_path = format!("/proc/self/fd/{}", file.as_raw_fd());
    // SAFETY: loading a shared object runs its initializers and is inherently
    // unsafe. These bytes verified against a trusted key, and the sealed memfd
    // cannot be swapped between the check and this call.
    let lib = unsafe { Library::new(&fd_path) }.map_err(|e| PluginError::Open {
        path: plugin.to_path_buf(),
        message: e.to_string(),
    });
    // `file` must stay open across `Library::new`: the path names its
    // descriptor.
    drop(file);
    lib
}

/// The other platforms: no memfd, so the verified bytes are dropped and the
/// path is opened. See the module "TOCTOU" note.
#[cfg(not(target_os = "linux"))]
pub(super) fn open_verified(plugin: &Path, _bytes: &[u8]) -> Result<Library, PluginError> {
    // SAFETY: loading a shared object runs its initializers and is inherently
    // unsafe. The bytes at this path verified against a trusted key a moment
    // ago; closing the remaining swap window needs a memfd, which is Linux only.
    unsafe { Library::new(plugin) }.map_err(|e| PluginError::Open {
        path: plugin.to_path_buf(),
        message: e.to_string(),
    })
}

fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

fn from_hex(text: &str) -> Option<Vec<u8>> {
    let (pairs, odd) = text.as_bytes().as_chunks::<2>();
    if !odd.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(pairs.len());
    for [hi, lo] in pairs {
        let hi = (*hi as char).to_digit(16)?;
        let lo = (*lo as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_file_round_trips() {
        let key = SigningKey::generate().expect("keygen");
        let detached = key.sign(b"plugin bytes");
        let encoded = detached.encode();
        assert_eq!(encoded.len(), SIGNATURE_FILE_LEN);
        let decoded = DetachedSignature::decode(&encoded, Path::new("x.so.sig"))
            .expect("a signature we just encoded must decode");
        assert_eq!(decoded, detached);
        assert_eq!(decoded.signer, key.public_key());
        UnparsedPublicKey::new(&ED25519, decoded.signer)
            .verify(b"plugin bytes", &decoded.signature)
            .expect("the round-tripped signature still verifies");
    }

    #[test]
    fn a_truncated_signature_file_is_refused() {
        let key = SigningKey::generate().expect("keygen");
        let encoded = key.sign(b"plugin bytes").encode();
        let err = DetachedSignature::decode(&encoded[..SIGNATURE_FILE_LEN - 1], Path::new("x.sig"))
            .expect_err("a short file must not decode");
        match err {
            SigningError::Malformed { reason, .. } => {
                assert!(reason.contains("102"), "reason names the size: {reason}");
            }
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn a_foreign_magic_is_refused() {
        let mut encoded = SigningKey::generate().expect("keygen").sign(b"x").encode();
        encoded[0] = b'X';
        match DetachedSignature::decode(&encoded, Path::new("x.sig")) {
            Err(SigningError::Malformed { .. }) => {}
            other => panic!("expected Malformed, got {other:?}"),
        }
    }

    #[test]
    fn a_public_key_file_round_trips_through_hex() {
        let key = SigningKey::generate().expect("keygen");
        let dir = std::env::temp_dir().join(format!("g2g-key-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("signer.pub");
        let _ = std::fs::remove_file(&path);
        write_public_key_file(&path, &key.public_key()).expect("write");
        let text = std::fs::read_to_string(&path).expect("read back");
        assert_eq!(text.trim().len(), PUBLIC_KEY_LEN * 2);
        assert_eq!(
            read_public_key_file(&path).expect("parse"),
            key.public_key()
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_empty_trust_set_is_the_no_verification_default() {
        assert!(TrustedKeys::new().is_empty());
        let mut trusted = TrustedKeys::new();
        let key = SigningKey::generate().expect("keygen").public_key();
        trusted.trust(key).trust(key);
        assert_eq!(trusted.len(), 1, "the same key is not trusted twice");
        assert!(trusted.contains(&key));
    }
}
