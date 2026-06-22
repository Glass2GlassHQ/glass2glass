//! Dynamic (`dlopen`) loader for third-party plugins (M201).
//!
//! A plugin is a `cdylib` built with the `g2g-plugin` SDK (its
//! `declare_plugin!` macro emits `g2g_plugin_abi` + `g2g_plugin_register`). This
//! module opens such a shared object, verifies its ABI tag against the host's
//! [`g2g_core::ABI_VERSION`], and, on a match, calls its registration entry to
//! add the plugin's elements to a [`Registry`]. The `gst-launch` parser then
//! resolves those elements by name like any built-in.
//!
//! **ABI safety.** Rust has no stable ABI, so loading a plugin built against a
//! different `g2g-core` version, a different `rustc`, or a different
//! layout-affecting feature set (`metadata`, `multi-thread`) and then passing a
//! `Frame` / `Box<dyn ...>` across the boundary is undefined behavior. The tag
//! folds all three together; [`load_plugin`] refuses a mismatch with
//! [`PluginError::AbiMismatch`] before calling any plugin code beyond the tag
//! query.
//!
//! **Keep-alive.** The registered factories are `fn` pointers into the loaded
//! library's mapped code; dropping the [`libloading::Library`] unmaps that code,
//! so any later element construction or `process` call is a use-after-free. We
//! therefore move every successfully loaded `Library` into a process-lifetime
//! list ([`KEEP_ALIVE`]) and never drop it. Elements hold no back-pointer to
//! their library, so this is the only thing keeping the code resident.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::{Library, Symbol};

use g2g_core::runtime::Registry;
use g2g_core::ABI_VERSION;

/// The C-ABI symbol names a `g2g-plugin` `cdylib` exports. Kept in sync with the
/// `declare_plugin!` expansion in the SDK.
const ABI_SYMBOL: &[u8] = b"g2g_plugin_abi";
const REGISTER_SYMBOL: &[u8] = b"g2g_plugin_register";

/// The environment variable a packaged `g2g-launch` / `g2g-inspect` scans for
/// plugin directories (`:`-separated, like `PATH`).
pub const PLUGIN_PATH_ENV: &str = "G2G_PLUGIN_PATH";

/// Loaded libraries, kept resident for the life of the process. See the
/// module-level "Keep-alive" note: dropping a `Library` would unmap the code the
/// registered elements run from.
static KEEP_ALIVE: OnceLock<Mutex<Vec<Library>>> = OnceLock::new();

fn keep_alive(lib: Library) {
    KEEP_ALIVE
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("plugin keep-alive list poisoned")
        .push(lib);
}

/// Why a plugin could not be loaded.
#[derive(Debug)]
pub enum PluginError {
    /// The shared object could not be opened (`dlopen` failed): missing file,
    /// unresolved symbols, wrong architecture. Carries the OS message.
    Open { path: PathBuf, message: String },
    /// The object opened but is missing one of the required entry points, so it
    /// was not built with the `g2g-plugin` SDK (or with an incompatible one).
    MissingSymbol { path: PathBuf, symbol: String },
    /// The plugin's ABI tag is not valid UTF-8 (a corrupt or non-g2g `.so`).
    BadAbiString { path: PathBuf },
    /// The plugin's ABI tag does not match the host's. Loading would risk UB, so
    /// it is refused. Both tags are reported so the difference is legible
    /// (version, `rustc`, or layout-feature skew).
    AbiMismatch { path: PathBuf, plugin: String, host: String },
    /// A directory scan could not read the directory.
    DirRead { path: PathBuf, message: String },
}

impl core::fmt::Display for PluginError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PluginError::Open { path, message } => {
                write!(f, "cannot open plugin {}: {message}", path.display())
            }
            PluginError::MissingSymbol { path, symbol } => write!(
                f,
                "plugin {} is missing `{symbol}`; not built with the g2g-plugin SDK?",
                path.display()
            ),
            PluginError::BadAbiString { path } => {
                write!(f, "plugin {} returned a non-UTF-8 ABI tag", path.display())
            }
            PluginError::AbiMismatch { path, plugin, host } => write!(
                f,
                "plugin {} ABI mismatch:\n  plugin: {plugin}\n  host:   {host}\n\
                 (rebuild the plugin against this g2g-core version + rustc + features)",
                path.display()
            ),
            PluginError::DirRead { path, message } => {
                write!(f, "cannot scan plugin dir {}: {message}", path.display())
            }
        }
    }
}

impl std::error::Error for PluginError {}

/// Compare a plugin's ABI tag to the host's, the check [`load_plugin`] runs
/// before calling any plugin code. Pulled out as a pure function so the
/// version-lock policy is unit-testable without building a `.so`.
fn check_abi(path: &Path, plugin_abi: &str) -> Result<(), PluginError> {
    if plugin_abi == ABI_VERSION {
        Ok(())
    } else {
        Err(PluginError::AbiMismatch {
            path: path.to_path_buf(),
            plugin: plugin_abi.to_string(),
            host: ABI_VERSION.to_string(),
        })
    }
}

/// Load one plugin shared object and register its elements into `reg`.
///
/// Opens `path`, reads its ABI tag via `g2g_plugin_abi`, and only on an exact
/// match calls `g2g_plugin_register(&mut reg)`. The library is then kept
/// resident for the life of the process (see the module "Keep-alive" note). On
/// any failure `reg` is left untouched and the loaded library, if any, is
/// dropped (no elements were registered from it).
pub fn load_plugin(path: impl AsRef<Path>, reg: &mut Registry) -> Result<(), PluginError> {
    let path = path.as_ref();

    // SAFETY: loading an arbitrary shared object runs its initializers and is
    // inherently unsafe; the caller is trusting `path`. We constrain what we do
    // with it to the documented `g2g-plugin` C-ABI entry points below.
    let lib = unsafe { Library::new(path) }
        .map_err(|e| PluginError::Open { path: path.to_path_buf(), message: e.to_string() })?;

    // Read the ABI tag first, before touching any other plugin code.
    // SAFETY: we assert the symbol's type matches the SDK's `g2g_plugin_abi`
    // signature (`extern "C" fn() -> *const c_char`). A wrong-typed symbol is
    // the plugin author's contract violation; an absent one is handled below.
    let abi_fn: Symbol<unsafe extern "C" fn() -> *const core::ffi::c_char> =
        unsafe { lib.get(ABI_SYMBOL) }.map_err(|_| PluginError::MissingSymbol {
            path: path.to_path_buf(),
            symbol: String::from_utf8_lossy(ABI_SYMBOL).into_owned(),
        })?;

    // SAFETY: `abi_fn` returns a pointer to a NUL-terminated, process-lifetime
    // `CString` inside the plugin (the SDK's `abi_cstr`), valid while `lib` is
    // loaded, which it is here.
    let abi_ptr = unsafe { abi_fn() };
    // SAFETY: `abi_ptr` is the non-null pointer just returned by the plugin's
    // `abi_cstr`; it points to a valid NUL-terminated C string.
    let abi_cstr = unsafe { CStr::from_ptr(abi_ptr) };
    let plugin_abi = abi_cstr
        .to_str()
        .map_err(|_| PluginError::BadAbiString { path: path.to_path_buf() })?;

    check_abi(path, plugin_abi)?;

    // ABI matches: it is now sound to hand the plugin our `Registry`.
    // SAFETY: same-typed symbol contract as `abi_fn`; the matched ABI tag
    // guarantees the plugin's `Registry` layout equals ours, so passing
    // `&mut Registry` across the boundary is well-defined.
    let register_fn: Symbol<unsafe extern "C" fn(&mut Registry)> =
        unsafe { lib.get(REGISTER_SYMBOL) }.map_err(|_| PluginError::MissingSymbol {
            path: path.to_path_buf(),
            symbol: String::from_utf8_lossy(REGISTER_SYMBOL).into_owned(),
        })?;

    // SAFETY: see above; the plugin's entry wraps its body in `catch_unwind`, so
    // no panic unwinds back across this `extern "C"` call.
    unsafe { register_fn(reg) };

    // Keep the code resident: the factories just registered are `fn` pointers
    // into `lib`. Must outlive every element, so it lives for the process.
    keep_alive(lib);
    Ok(())
}

/// Whether a filename has this platform's dynamic-library extension
/// (`.so` / `.dylib` / `.dll`), used to filter a directory scan.
fn is_dylib(path: &Path) -> bool {
    let ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Load every dynamic library in `dir` (non-recursive), registering each into
/// `reg`. Returns the loaded paths on success; the first per-file error aborts
/// the scan. Files without this platform's library extension are skipped.
pub fn load_plugin_dir(
    dir: impl AsRef<Path>,
    reg: &mut Registry,
) -> Result<Vec<PathBuf>, PluginError> {
    let dir = dir.as_ref();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| PluginError::DirRead { path: dir.to_path_buf(), message: e.to_string() })?;
    let mut loaded = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && is_dylib(&path) {
            load_plugin(&path, reg)?;
            loaded.push(path);
        }
    }
    Ok(loaded)
}

/// Scan every directory in the `G2G_PLUGIN_PATH` environment variable
/// (`:`-separated on Unix, `;` on Windows, matching the OS path convention) and
/// load the plugins found, registering them into `reg`. A no-op (returns an
/// empty list) when the variable is unset. The first error aborts.
pub fn load_from_env(reg: &mut Registry) -> Result<Vec<PathBuf>, PluginError> {
    let Some(var) = std::env::var_os(PLUGIN_PATH_ENV) else {
        return Ok(Vec::new());
    };
    let mut loaded = Vec::new();
    for dir in std::env::split_paths(&var) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        loaded.extend(load_plugin_dir(&dir, reg)?);
    }
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_abi_tag_is_accepted() {
        // The host's own tag must pass the check (this is the happy path the
        // loader runs before calling a plugin's register entry).
        assert!(check_abi(Path::new("x.so"), ABI_VERSION).is_ok());
    }

    #[test]
    fn mismatched_abi_tag_is_refused_with_both_tags() {
        let err = check_abi(Path::new("x.so"), "g2g-core 0.0.1 | rustc 1.0.0 | feat:none")
            .expect_err("a foreign tag must be refused");
        match err {
            PluginError::AbiMismatch { plugin, host, .. } => {
                assert_eq!(plugin, "g2g-core 0.0.1 | rustc 1.0.0 | feat:none");
                assert_eq!(host, ABI_VERSION);
            }
            other => panic!("expected AbiMismatch, got {other:?}"),
        }
    }

    #[test]
    fn dylib_extension_filter_matches_platform() {
        // The current platform's extension is accepted; a stray .txt is not.
        let so = if cfg!(target_os = "windows") {
            "p.dll"
        } else if cfg!(target_os = "macos") {
            "p.dylib"
        } else {
            "p.so"
        };
        assert!(is_dylib(Path::new(so)));
        assert!(!is_dylib(Path::new("notes.txt")));
    }
}
