//! Analytics alert (M1176): fires configurable rules on the detections a frame
//! carries, the gst-python-ml `pyml_alert` analog. A rule names a class, a
//! minimum score and a zone; a detection matching all three raises an alert.
//!
//! Alerts leave the element three ways: as an `alert` blob on the frame (a JSON
//! array, which `metasink` writes and `alertrecorder` records a clip around), as
//! a red border painted on the RGBA8 pixels, and as an HTTP POST to
//! `webhook-url`. A POST that fails is logged and the frame goes on.
//!
//! The cooldown is measured in **stream** time, the frame's pts, not wall time,
//! so replaying a file fires the same alerts at the same frames every run. The
//! Python element uses wall time.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use serde_json::{Map, Value};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_info, g2g_warn, AnalyticsMeta, AsyncElement, BlobMeta, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError, MemoryDomain, OutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, RawVideoFormat,
};

use crate::metasink::{
    caps_dimensions, detection_json, HEIGHT_KEY, LABEL_KEY, NS_PER_SECOND, SCORE_KEY, WIDTH_KEY,
    X_KEY, Y_KEY,
};

/// The blob header the alert array is attached under, the name `metasink` writes
/// it as and `alertrecorder` looks for.
pub const ALERT_BLOB: &str = "alert";

/// The rule keys, all optional: a rule with none matches every detection.
const CLASS_KEY: &str = "class";
const MIN_SCORE_KEY: &str = "min_score";
const ZONE_KEY: &str = "zone";
/// The alert payload keys.
const TIMESTAMP_KEY: &str = "timestamp";
const RULE_KEY: &str = "rule";
const DETECTION_KEY: &str = "detection";

const ZONE_CORNERS: usize = 4;
const DEFAULT_COOLDOWN_SECONDS: u64 = 10;
const DEFAULT_COOLDOWN_TEXT: &str = "10";
const NS_PER_SECOND_U64: u64 = 1_000_000_000;

/// The border a fired frame is outlined with, and how it scales: one pixel per
/// this many of the frame's shorter side, never thinner than [`MIN_BORDER_PX`].
const BORDER_DIVISOR: u32 = 100;
const MIN_BORDER_PX: u32 = 2;
const ALERT_RGBA: [u8; 4] = [255, 0, 0, 255];
const BYTES_PER_PIXEL: usize = 4;

/// How long a webhook POST may take before it is abandoned. The POST is awaited
/// in the streaming path, so this bounds the stall a dead endpoint causes.
const WEBHOOK_TIMEOUT: Duration = Duration::from_secs(5);
const WEBHOOK_CONTENT_TYPE: &str = "application/json";

/// Fires rules on the detections a frame carries and attaches the alerts it
/// raised.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::analyticsalert::AnalyticsAlert;
///
/// // gst-launch equivalent:
/// //   analyticsalert rules='{"class":"person","min_score":0.8}' cooldown=30
/// let alert = AnalyticsAlert::new();
/// ```
#[derive(Debug)]
pub struct AnalyticsAlert {
    rules_json: String,
    rules: Vec<Value>,
    cooldown_seconds: u64,
    draw_alert: bool,
    webhook_url: String,
    webhook: Option<reqwest::Client>,
    /// The stream time each rule last fired at, by rule index. Absent means the
    /// rule has never fired, which no cooldown holds back.
    last_fired_ns: BTreeMap<usize, u64>,
    width: u32,
    height: u32,
    fired: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for AnalyticsAlert {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsAlert {
    pub fn new() -> Self {
        Self {
            rules_json: String::new(),
            rules: Vec::new(),
            cooldown_seconds: DEFAULT_COOLDOWN_SECONDS,
            draw_alert: true,
            webhook_url: String::new(),
            webhook: None,
            last_fired_ns: BTreeMap::new(),
            width: 0,
            height: 0,
            fired: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// Set the rules from their JSON text (one object or an array of them), the
    /// `rules` property.
    pub fn with_rules(mut self, rules: &str) -> Result<Self, PropError> {
        self.set_rules(rules)?;
        Ok(self)
    }

    /// Alerts fired so far.
    pub fn fired(&self) -> u64 {
        self.fired
    }

    fn set_rules(&mut self, text: &str) -> Result<(), PropError> {
        if text.is_empty() {
            self.rules_json = String::new();
            self.rules = Vec::new();
            return Ok(());
        }
        let parsed: Value = serde_json::from_str(text).map_err(|_| PropError::Value)?;
        self.rules = match parsed {
            Value::Object(_) => Vec::from([parsed]),
            Value::Array(rules) => rules,
            _ => return Err(PropError::Value),
        };
        self.rules_json = text.to_string();
        Ok(())
    }

    fn accepts(caps: &Caps) -> bool {
        matches!(
            caps,
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(_),
                height: Dim::Fixed(_),
                ..
            }
        )
    }

    /// Whether a detection (in the record shape `metasink` writes) satisfies a
    /// rule: the class name is a substring of the label, the score clears the
    /// threshold, and the box centre sits in the zone.
    fn rule_matches(rule: &Value, detection: &Value) -> bool {
        if let Some(class) = rule.get(CLASS_KEY).and_then(Value::as_str) {
            let label = detection
                .get(LABEL_KEY)
                .and_then(Value::as_str)
                .unwrap_or("");
            if !class.is_empty() && !label.contains(class) {
                return false;
            }
        }
        if let Some(minimum) = rule.get(MIN_SCORE_KEY).and_then(Value::as_f64) {
            if detection
                .get(SCORE_KEY)
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
                < minimum
            {
                return false;
            }
        }
        if let Some(zone) = rule.get(ZONE_KEY).and_then(Value::as_array) {
            if zone.len() != ZONE_CORNERS {
                return true;
            }
            let corner = |index: usize| zone[index].as_f64().unwrap_or(0.0);
            let field = |key: &str| detection.get(key).and_then(Value::as_f64).unwrap_or(0.0);
            let centre_x = field(X_KEY) + field(WIDTH_KEY) / 2.0;
            let centre_y = field(Y_KEY) + field(HEIGHT_KEY) / 2.0;
            let inside = corner(0) <= centre_x
                && centre_x <= corner(2)
                && corner(1) <= centre_y
                && centre_y <= corner(3);
            if !inside {
                return false;
            }
        }
        true
    }

    /// Whether this rule may fire at `pts_ns`, recording that it did.
    fn cooled_down(&mut self, rule_index: usize, pts_ns: u64) -> bool {
        let cooldown_ns = self.cooldown_seconds.saturating_mul(NS_PER_SECOND_U64);
        match self.last_fired_ns.get(&rule_index) {
            Some(last) if pts_ns < last.saturating_add(cooldown_ns) => false,
            _ => {
                self.last_fired_ns.insert(rule_index, pts_ns);
                true
            }
        }
    }

    /// The alerts this frame's detections raise, at most one per rule.
    fn alerts_for(&mut self, detections: &[Value], pts_ns: u64) -> Vec<Value> {
        let timestamp = g2g_core::log::timestamp_now()
            .map_or(pts_ns as f64 / NS_PER_SECOND, |now| {
                now as f64 / NS_PER_SECOND
            });
        let mut alerts = Vec::new();
        for index in 0..self.rules.len() {
            let Some(detection) = detections
                .iter()
                .find(|detection| Self::rule_matches(&self.rules[index], detection))
            else {
                continue;
            };
            if !self.cooled_down(index, pts_ns) {
                continue;
            }
            let mut alert = Map::new();
            alert.insert(TIMESTAMP_KEY.to_string(), Value::from(timestamp));
            alert.insert(RULE_KEY.to_string(), self.rules[index].clone());
            alert.insert(DETECTION_KEY.to_string(), detection.clone());
            alerts.push(Value::Object(alert));
        }
        alerts
    }

    /// Outline the frame in red, so an alert is visible on the picture itself.
    fn draw_border(&self, pixels: &mut [u8]) {
        let width = self.width as usize;
        let height = self.height as usize;
        let border = (self.width.min(self.height) / BORDER_DIVISOR).max(MIN_BORDER_PX) as usize;
        let border = border.min(height.div_ceil(2)).min(width.div_ceil(2));
        let mut paint = |x: usize, y: usize| {
            let at = (y * width + x) * BYTES_PER_PIXEL;
            pixels[at..at + BYTES_PER_PIXEL].copy_from_slice(&ALERT_RGBA);
        };
        for y in 0..height {
            let full_row = y < border || y >= height - border;
            for x in 0..width {
                if full_row || x < border || x >= width - border {
                    paint(x, y);
                }
            }
        }
    }

    async fn post_webhook(&mut self, alerts: &[Value]) {
        if self.webhook_url.is_empty() {
            return;
        }
        if self.webhook.is_none() {
            match reqwest::Client::builder().timeout(WEBHOOK_TIMEOUT).build() {
                Ok(client) => self.webhook = Some(client),
                Err(error) => {
                    g2g_warn!(self, "cannot build the webhook client: {error}");
                    return;
                }
            }
        }
        let client = self.webhook.as_ref().expect("built above");
        let body = match serde_json::to_string(alerts) {
            Ok(body) => body,
            Err(error) => {
                g2g_warn!(self, "cannot serialize the alert: {error}");
                return;
            }
        };
        let posted = client
            .post(&self.webhook_url)
            .header(reqwest::header::CONTENT_TYPE, WEBHOOK_CONTENT_TYPE)
            .body(body)
            .send()
            .await;
        match posted {
            Ok(response) => g2g_info!(self, "webhook answered {}", response.status()),
            Err(error) => g2g_warn!(self, "webhook POST failed: {error}"),
        }
    }
}

impl AsyncElement for AnalyticsAlert {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Analytics alert",
            "Filter/Analytics",
            "Fires rules on detections, attaching an alert blob and posting a webhook",
            "g2g",
        )
    }

    /// Paints the border into host memory, so it takes system frames only.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (width, height) = caps_dimensions(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.width = width;
        self.height = height;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    // Copy the detections out as records so the meta borrow ends
                    // before the pixels are written, and so a rule sees exactly
                    // what `metasink` would have written.
                    let detections: Vec<Value> = frame
                        .meta
                        .get::<AnalyticsMeta>()
                        .map(|analytics| {
                            analytics
                                .detections()
                                .map(|detection| {
                                    detection_json(
                                        detection,
                                        analytics.class_name(detection.label),
                                        self.width,
                                        self.height,
                                    )
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let alerts = if detections.is_empty() || self.rules.is_empty() {
                        Vec::new()
                    } else {
                        self.alerts_for(&detections, frame.timing.pts_ns)
                    };
                    if !alerts.is_empty() {
                        self.fired += alerts.len() as u64;
                        let payload = serde_json::to_vec(&alerts)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                        if frame.meta.get::<BlobMeta>().is_none() {
                            frame.meta.attach(BlobMeta::new());
                        }
                        frame
                            .meta
                            .get_mut::<BlobMeta>()
                            .expect("just attached")
                            .push(ALERT_BLOB, payload);
                        if self.draw_alert {
                            let MemoryDomain::System(slice) = &mut frame.domain else {
                                return Err(G2gError::UnsupportedDomain);
                            };
                            let need = self.width as usize * self.height as usize * BYTES_PER_PIXEL;
                            let pixels = slice.as_mut_slice();
                            if pixels.len() < need {
                                return Err(G2gError::CapsMismatch);
                            }
                            self.draw_border(&mut pixels[..need]);
                        }
                        self.post_webhook(&alerts).await;
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((width, height)) = caps_dimensions(&caps) {
                        self.width = width;
                        self.height = height;
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // A flush restarts the stream clock, so a rule must be free to
                // fire again at the new first frame.
                PipelinePacket::Flush => {
                    self.last_fired_ns.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner's transform arm forwards EOS; don't double it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        ANALYTICSALERT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "rules" => self.set_rules(value.as_str().ok_or(PropError::Type)?),
            "cooldown" => {
                self.cooldown_seconds = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "draw-alert" => {
                self.draw_alert = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "webhook-url" => {
                self.webhook_url = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "rules" => Some(PropValue::Str(self.rules_json.clone())),
            "cooldown" => Some(PropValue::Uint(self.cooldown_seconds)),
            "draw-alert" => Some(PropValue::Bool(self.draw_alert)),
            "webhook-url" => Some(PropValue::Str(self.webhook_url.clone())),
            _ => None,
        }
    }
}

impl LogSource for AnalyticsAlert {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// `AnalyticsAlert`'s settable properties, named as the Python element's.
static ANALYTICSALERT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "rules",
        PropKind::Str,
        "alert rules as JSON: one object or an array of {class, min_score, zone}",
    )
    .with_default(""),
    PropertySpec::new(
        "cooldown",
        PropKind::Uint,
        "stream seconds between repeated alerts for one rule",
    )
    .with_default(DEFAULT_COOLDOWN_TEXT),
    PropertySpec::new(
        "draw-alert",
        PropKind::Bool,
        "outline an alerting frame in red",
    )
    .with_default("true"),
    PropertySpec::new(
        "webhook-url",
        PropKind::Str,
        "HTTP endpoint each alert is POSTed to, none when empty",
    )
    .with_default(""),
];
