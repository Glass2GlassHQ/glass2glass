//! SDK for building **dynamically loadable** glass2glass plugins.
//!
//! A third party builds a plugin with plain `cargo` against a published
//! `g2g-core` + `g2g-plugin` (the Rust equivalent of a `g2g-devel` package),
//! sets `crate-type = ["cdylib"]`, and drops the resulting `.so` where a
//! system-installed `g2g-launch` will `dlopen` it, no recompile of g2g. This
//! crate provides the one thing the plugin must export: a pair of C-ABI entry
//! points, emitted by the [`declare_plugin!`] macro.
//!
//! ```ignore
//! // In the plugin crate (crate-type = ["cdylib"]):
//! g2g_plugin::declare_plugin! {
//!     elements: [
//!         ("myfilter", MyFilter, || Box::new(MyFilter::default())),
//!     ]
//! }
//! ```
//!
//! The macro emits `g2g_plugin_abi` (returns this build's [`abi_version`] as a C
//! string) and `g2g_plugin_register` (adds each element to the host's
//! [`Registry`]). The host loader ([`g2g_plugins::plugin_loader`]) reads the ABI
//! tag first and refuses to call `register` on a mismatch.
//!
//! **Why the ABI tag matters.** Rust has no stable ABI. The plugin and host must
//! share the same `g2g-core` version, the same `rustc`, and the same
//! layout-affecting features (`metadata`, `multi-thread`), or passing a `Frame`
//! / `Box<dyn ...>` across the boundary is UB. [`g2g_core::ABI_VERSION`] folds
//! all three into the tag; the loader compares it loudly rather than risk that.
//!
//! This version-lock path is the v1 design (DESIGN_TODO "Dynamic plugin loading
//! via cargo"); a future `abi_stable`/`stabby` facade would relax the
//! same-toolchain requirement for cross-compiler binary plugins.

// The SDK itself is std (it emits `std::panic::catch_unwind` and builds a
// `CString`); it is not part of the no_std baseline.

use std::ffi::CString;
use std::sync::OnceLock;

// Re-exported so a plugin author and the `declare_plugin!` expansion name them
// through `g2g_plugin::` without a direct `g2g-core` path dependency mattering
// for the macro hygiene (the author still depends on g2g-core for the traits).
pub use g2g_core::runtime::{LaunchFactory, MuxerFactory, Registry, SourceFactory};
pub use g2g_core::ABI_VERSION;

/// The exported name of the ABI-query entry point a plugin `cdylib` must define.
/// The loader looks this symbol up first.
pub const ABI_SYMBOL: &[u8] = b"g2g_plugin_abi";

/// The exported name of the registration entry point a plugin `cdylib` must
/// define. The loader calls this only after the ABI tag matches.
pub const REGISTER_SYMBOL: &[u8] = b"g2g_plugin_register";

/// This build's ABI tag as a NUL-terminated C string, valid for the life of the
/// process. The `g2g_plugin_abi` entry point returns this pointer; the host
/// reads it back as a `CStr` and compares it to its own [`ABI_VERSION`].
///
/// Building the `CString` once and leaking it via a `OnceLock` guarantees the
/// returned pointer stays valid after the call returns (the host borrows it).
pub fn abi_cstr() -> *const core::ffi::c_char {
    static CSTR: OnceLock<CString> = OnceLock::new();
    // ABI_VERSION is a build-time tag with no interior NUL, so this never fails.
    CSTR.get_or_init(|| CString::new(ABI_VERSION).expect("ABI_VERSION has no interior NUL"))
        .as_ptr()
}

/// Called by the [`declare_plugin!`] expansion when a registration closure
/// panics. Unwinding across the `extern "C"` boundary back into the host is UB,
/// so the macro catches it; this hook just notes it on stderr so the failure is
/// not silent (the host then sees the element simply absent from the registry).
pub fn register_panicked(plugin: &str) {
    eprintln!("g2g-plugin: registration panicked in plugin '{plugin}'; elements not registered");
}

/// Declare a dynamically loadable plugin: emit the C-ABI entry points the host
/// loader expects.
///
/// Each element is `(name, Type, build)` where `name` is the `gst-launch` /
/// `gst-inspect` name, `Type` is the element type (must implement
/// `PadTemplates`, so the macro can pull its pad templates), and `build` is a
/// parameterless constructor closure returning a boxed element
/// (`|| Box::new(MyFilter::default())`).
///
/// The expansion registers every element through `Registry::register_launch`
/// inside a `catch_unwind` (a panic must not unwind across `extern "C"`).
#[macro_export]
macro_rules! declare_plugin {
    ( elements: [ $( ( $name:expr, $ty:ty, $build:expr ) ),* $(,)? ] ) => {
        /// ABI-query entry point. Returns this plugin's compatibility tag as a
        /// C string; the host compares it to its own before loading anything.
        ///
        /// # Safety
        /// Called by the host loader via `dlsym`. Returns a pointer into a
        /// process-lifetime `CString`, valid for the life of the loaded library.
        #[no_mangle]
        pub extern "C" fn g2g_plugin_abi() -> *const ::core::ffi::c_char {
            $crate::abi_cstr()
        }

        /// Registration entry point. Adds every declared element to the host's
        /// `Registry`. The body is wrapped in `catch_unwind` because unwinding
        /// across the `extern "C"` boundary into the host is undefined behavior.
        ///
        /// # Safety
        /// Called by the host loader via `dlsym` with a valid `&mut Registry`
        /// whose layout matches this plugin's (guaranteed by the prior ABI-tag
        /// check). Must not be called concurrently on the same registry.
        #[no_mangle]
        pub extern "C" fn g2g_plugin_register(reg: &mut $crate::Registry) {
            let outcome = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                $(
                    reg.register_launch($crate::LaunchFactory::of::<$ty>($name, $build));
                )*
            }));
            if outcome.is_err() {
                $crate::register_panicked(::core::env!("CARGO_PKG_NAME"));
            }
        }
    };
}
