//! Minimal WebAssembly wrapper around g2g's real caps solver.
//!
//! `validate_pipeline` parses + negotiates a `gst-launch` line with the standard
//! registry and returns `toolingjson::validate_json`'s structured result as a
//! JSON string (the JS side does `JSON.parse`, avoiding serde-wasm-bindgen). The
//! visual builder loads this to authoritatively validate each link's caps,
//! falling back to its coarse family heuristic when the blob is unavailable.

use g2g_core::runtime::Registry;
use g2g_plugins::toolingjson::validate_json;

/// Parse + negotiate `line` against `reg` and return the `validate_json` value
/// serialized to a JSON string. Native-testable (no wasm-bindgen), wrapped by the
/// wasm entry point below.
pub async fn validate(reg: &Registry, line: &str) -> String {
    validate_json(reg, line).await.to_string()
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::validate;
    use g2g_plugins::registry::default_registry;
    use wasm_bindgen::prelude::wasm_bindgen;

    #[wasm_bindgen]
    pub async fn validate_pipeline(launch: String) -> String {
        let reg = default_registry();
        validate(&reg, &launch).await
    }
}

#[cfg(test)]
mod tests {
    use super::validate;
    use g2g_plugins::registry::default_registry;

    #[tokio::test]
    async fn good_pipeline_reports_edges() {
        let reg = default_registry();
        let out = validate(&reg, "videotestsrc num-buffers=1 ! fakesink").await;
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], true);
        let edges = v["edges"].as_array().unwrap();
        assert!(!edges.is_empty());
        assert!(edges[0]["from"].is_number() && edges[0]["to"].is_number());
        assert!(edges[0]["caps"].is_string());
    }

    #[tokio::test]
    async fn incompatible_caps_empties_the_link() {
        let reg = default_registry();
        let out = validate(
            &reg,
            "videotestsrc num-buffers=1 ! video/x-raw,format=NV12 ! fakesink",
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["ok"], false);
        assert_eq!(v["stage"], "negotiate");
        assert_eq!(v["failure"]["kind"], "empty-link");
    }
}
