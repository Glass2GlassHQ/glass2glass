//! An out-of-tree third-party g2g plugin on the **v2** ABI.
//!
//! The whole author workflow: write a plain `AsyncElement`, add one
//! [`declare_plugin_v2!`](g2g_plugin::declare_plugin_v2) invocation, build a
//! `cdylib`. Nothing but `repr(C)` data crosses the boundary, so the result
//! loads into a host built by a different `rustc` against a different
//! `g2g-core` build. (Writing that boundary *by hand*, in C, is what
//! `tests/fixtures/c-plugin` shows.)
//!
//! The element is `v2counter`: it counts data frames and forwards them, and
//! exposes `count` (read-only) and `enabled` (drops frames when false).

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AsyncElement, Caps, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, Interlace,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, RawVideoFormat,
};

/// Counts data frames and forwards them unchanged.
#[derive(Debug)]
pub struct V2Counter {
    seen: u64,
    enabled: bool,
}

impl Default for V2Counter {
    fn default() -> Self {
        V2Counter {
            seen: 0,
            enabled: true,
        }
    }
}

const PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new("count", PropKind::Uint, "data frames seen so far").read_only(),
    PropertySpec::new(
        "enabled",
        PropKind::Bool,
        "forward frames; drop them when false",
    )
    .with_default("true"),
];

impl AsyncElement for V2Counter {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.seen += 1;
                    if self.enabled {
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                    }
                }
                // The runner emits the single EOS; a transform must not forward it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "enabled" => {
                self.enabled = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "count" => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "count" => Some(PropValue::Uint(self.seen)),
            "enabled" => Some(PropValue::Bool(self.enabled)),
            _ => None,
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "v2 counting filter",
            "Filter/Effect/Video",
            "Counts data frames and forwards them unchanged (v2 plugin ABI demo).",
            "third-party",
        )
    }
}

impl PadTemplates for V2Counter {
    fn pad_templates() -> Vec<PadTemplate> {
        let any = CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
        });
        Vec::from([PadTemplate::sink(any.clone()), PadTemplate::source(any)])
    }
}

#[cfg(not(feature = "undeclared"))]
g2g_plugin::declare_plugin_v2! {
    name: "g2g-v2-example-plugin",
    version: "0.1.0",
    elements: [
        ("v2counter", V2Counter, transform),
    ]
}

/// A plugin that breaks its own declaration, for the loader's capability gate
/// to catch: the descriptor declares `v2counter`, and `register` then adds a
/// second element under a name nobody was told about. Hand-written, because the
/// macro cannot generate a registration that disagrees with its own capability
/// list.
#[cfg(feature = "undeclared")]
mod undeclared {
    use super::V2Counter;
    use g2g_plugin::abi::{
        AbiStatic, FfiCapability, FfiPluginDescriptor, FfiRegistrar, FfiStatus, FfiStr,
        ELEMENT_TRANSFORM, STATUS_ERROR, V2_ABI_VERSION, V2_MAGIC,
    };

    /// # Safety
    /// `registrar` is the host-owned object, valid for this call.
    unsafe extern "C" fn register(registrar: *const FfiRegistrar) -> FfiStatus {
        if registrar.is_null() {
            return STATUS_ERROR;
        }
        // SAFETY: the host passes a live registrar for the duration of the call.
        let registrar = unsafe { &*registrar };
        // SAFETY: the SDK builds a registration whose pointers it leaks.
        let status = unsafe {
            g2g_plugin::v2::register_element::<V2Counter>(
                registrar,
                "v2counter",
                ELEMENT_TRANSFORM,
            )
        };
        if !status.is_ok() {
            return status;
        }
        // SAFETY: as above. This name was never declared.
        unsafe {
            g2g_plugin::v2::register_element::<V2Counter>(registrar, "sneaky", ELEMENT_TRANSFORM)
        }
    }

    static CAPABILITIES: AbiStatic<[FfiCapability; 1]> = AbiStatic([FfiCapability {
        kind: ELEMENT_TRANSFORM,
        reserved: 0,
        name: FfiStr::borrowed("v2counter"),
    }]);

    #[no_mangle]
    #[allow(non_upper_case_globals)]
    pub static g2g_plugin_v2_descriptor: AbiStatic<FfiPluginDescriptor> =
        AbiStatic(FfiPluginDescriptor {
            magic: V2_MAGIC,
            abi_version: V2_ABI_VERSION,
            struct_size: core::mem::size_of::<FfiPluginDescriptor>() as u32,
            name: FfiStr::borrowed("g2g-v2-example-plugin"),
            version: FfiStr::borrowed("0.1.0"),
            capabilities: CAPABILITIES.0.as_ptr(),
            capability_count: CAPABILITIES.0.len(),
            register: Some(register),
            reserved: [None; 4],
        });
}
