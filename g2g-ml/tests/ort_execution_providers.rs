//! The `execution-provider` property on `ortinfer`: every provider the build
//! includes parses by name and survives the model load, and a provider the host
//! cannot enable fails that load loud instead of dropping to the CPU unnoticed.
//! Which of the two a GPU provider hits depends on the host, so only the CPU
//! outcome is pinned.
//!
//! ```powershell
//! cargo test -p g2g-ml --features ort,tensorrt,openvino,migraphx --test ort_execution_providers
//! ```

#![cfg(feature = "ort")]

use g2g_core::element::AsyncElement;
use g2g_core::{PropError, PropValue};
use g2g_ml::ortinfer::OrtInference;

// shared hand-encoded ONNX fixture builder (tests/util/onnx_fixture.rs)
mod onnx {
    include!("util/onnx_fixture.rs");
}

const CPU: &str = "cpu";

/// The provider names this build accepts.
fn built_in_providers() -> Vec<&'static str> {
    [
        (true, CPU),
        (cfg!(feature = "cuda"), "cuda"),
        (cfg!(feature = "tensorrt"), "tensorrt"),
        (cfg!(feature = "openvino"), "openvino"),
        (cfg!(feature = "rocm"), "rocm"),
        (cfg!(feature = "migraphx"), "migraphx"),
    ]
    .into_iter()
    .filter(|(built_in, _)| *built_in)
    .map(|(_, name)| name)
    .collect()
}

fn fixture_model() -> String {
    let path = std::env::temp_dir().join("g2g_ort_execution_providers.onnx");
    std::fs::write(&path, onnx::identity_model(&[1, 3, 2, 2])).expect("write fixture model");
    path.to_str().unwrap().to_string()
}

#[test]
fn every_built_in_provider_is_named_and_loads_or_fails_loud() {
    let model = fixture_model();
    for name in built_in_providers() {
        let mut element = OrtInference::new();
        element
            .set_property("execution-provider", PropValue::Str(name.into()))
            .unwrap_or_else(|error| panic!("{name} is built in but rejected: {error:?}"));
        assert_eq!(
            element.get_property("execution-provider"),
            Some(PropValue::Str(name.into()))
        );
        let loaded = element.set_property("model", PropValue::Str(model.clone()));
        if name == CPU {
            assert_eq!(loaded, Ok(()), "the CPU provider always loads");
        }
        match loaded {
            Ok(()) => eprintln!("{name}: enabled on this host"),
            Err(PropError::Value) => {
                eprintln!("{name}: not enabled on this host, load failed loud")
            }
            Err(other) => panic!("{name}: load failed with {other:?} instead of a bad value"),
        }
        assert_eq!(
            element.get_property("execution-provider"),
            Some(PropValue::Str(name.into())),
            "{name} must not be swapped for another provider by the load"
        );
    }
}

#[test]
fn a_provider_the_build_lacks_is_a_bad_value() {
    let mut element = OrtInference::new();
    assert_eq!(
        element.set_property("execution-provider", PropValue::Str("abacus".into())),
        Err(PropError::Value)
    );
    assert_eq!(
        element.get_property("execution-provider"),
        Some(PropValue::Str(CPU.into()))
    );
}
