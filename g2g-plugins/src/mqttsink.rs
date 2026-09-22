//! MQTT sink (M1179): publishes the record `metasink` writes, one message per
//! frame, to a topic on an MQTT broker. The payload is the same JSON object,
//! key for key (`pts`, `detections`, `text`, one key per JSON blob), so a
//! subscriber reads what a `metasink` file holds, and a frame whose record
//! says nothing but its time publishes nothing.
//!
//! The rumqttc event loop runs on a task of the caller's tokio runtime,
//! reconnecting when the broker drops. A publish is queued to that task and
//! never awaited, so a slow or dead broker does not stall the streaming path:
//! once the publish queue is full, further ones are dropped
//! and counted. `Eos` drains the queue before the connection closes.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use rumqttc::{AsyncClient, ConnectionError, Event, EventLoop, Packet, QoS};
use tokio::task::JoinHandle;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{g2g_error, g2g_info, g2g_warn};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, HardwareError,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::metasink::{caps_dimensions, frame_record};
use crate::mqtt::{
    broker_property_table, qos_from_level, qos_level, BrokerSettings, DEFAULT_QOS,
    DEFAULT_QOS_TEXT, QOS_MAX_TEXT, RECONNECT_DELAY,
};

/// Records waiting for the event loop before further ones are dropped.
const PUBLISH_QUEUE_LEN: usize = 64;
/// How long `Eos` waits for the queued records to reach the broker.
const EOS_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Publishes the record `metasink` writes to an MQTT topic, one message per
/// frame.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::mqttsink::MqttSink;
///
/// // gst-launch equivalent: mqttsink host=broker.local topic=cameras/north/detections
/// let sink = MqttSink::new().with_host("broker.local").with_topic("cameras/north/detections");
/// assert_eq!(sink.published(), 0);
/// ```
#[derive(Debug)]
pub struct MqttSink {
    broker: BrokerSettings,
    topic: String,
    qos: QoS,
    retain: bool,
    client: Option<AsyncClient>,
    event_loop: Option<JoinHandle<()>>,
    text_input: bool,
    width: u32,
    height: u32,
    published: u64,
    dropped: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for MqttSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MqttSink {
    pub fn new() -> Self {
        Self {
            broker: BrokerSettings::new(),
            topic: String::new(),
            qos: DEFAULT_QOS,
            retain: false,
            client: None,
            event_loop: None,
            text_input: false,
            width: 0,
            height: 0,
            published: 0,
            dropped: 0,
            configured: false,
            log_name: LogName::new(),
        }
    }

    /// The broker's host name (the `host` property).
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.broker.host = host.into();
        self
    }

    /// The broker's port (the `port` property).
    pub fn with_port(mut self, port: u16) -> Self {
        self.broker.port = port;
        self
    }

    /// The topic each record is published to (the `topic` property).
    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = topic.into();
        self
    }

    /// Records handed to the connection so far.
    pub fn published(&self) -> u64 {
        self.published
    }

    /// Records dropped because the publish queue was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    fn fallback_client_id(&self) -> &str {
        self.log_name
            .instance()
            .unwrap_or_else(|| short_type_name::<Self>())
    }

    /// Open the connection: the client handle stays here, the event loop goes
    /// onto a task. Called from `process`, which runs inside the runtime.
    fn connect(&mut self) {
        let options = self.broker.mqtt_options(self.fallback_client_id());
        let (client, event_loop) = AsyncClient::new(options, PUBLISH_QUEUE_LEN);
        let log_source = EventLoopLogSource {
            log_name: self.log_name.clone(),
        };
        self.event_loop = Some(tokio::spawn(run_event_loop(event_loop, log_source)));
        self.client = Some(client);
    }

    fn publish(&mut self, payload: Vec<u8>) {
        if self.client.is_none() {
            self.connect();
        }
        let Some(client) = &self.client else {
            return;
        };
        match client.try_publish(self.topic.clone(), self.qos, self.retain, payload) {
            Ok(()) => self.published += 1,
            Err(error) => {
                self.dropped += 1;
                g2g_warn!(self, "record dropped, publish queue full: {error}");
            }
        }
    }

    /// Queue a disconnect behind the records, close the request channel by
    /// dropping the client, and wait for the event loop to finish sending.
    async fn drain(&mut self) {
        if let Some(client) = self.client.take() {
            if let Err(error) = client.try_disconnect() {
                g2g_warn!(self, "disconnect not queued: {error}");
            }
        }
        let Some(event_loop) = self.event_loop.take() else {
            return;
        };
        if tokio::time::timeout(EOS_DRAIN_TIMEOUT, event_loop)
            .await
            .is_err()
        {
            g2g_warn!(self, "queued records did not reach the broker before eos");
        }
    }
}

impl Drop for MqttSink {
    fn drop(&mut self) {
        if let Some(event_loop) = self.event_loop.take() {
            event_loop.abort();
        }
    }
}

/// What the event-loop task logs as: the sink's own name.
#[derive(Debug)]
struct EventLoopLogSource {
    log_name: LogName,
}

impl LogSource for EventLoopLogSource {
    fn log_category(&self) -> &'static str {
        short_type_name::<MqttSink>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

/// Poll the connection until the request channel closes, reconnecting after a
/// delay when the broker drops or refuses.
async fn run_event_loop(mut event_loop: EventLoop, log_source: EventLoopLogSource) {
    loop {
        match event_loop.poll().await {
            Ok(Event::Incoming(Packet::ConnAck(ack))) => {
                g2g_info!(log_source, "connected: {:?}", ack.code);
            }
            Ok(Event::Outgoing(rumqttc::Outgoing::Disconnect)) => return,
            Ok(_) => {}
            Err(ConnectionError::RequestsDone) => return,
            Err(error) => {
                g2g_warn!(log_source, "connection lost: {error}");
                tokio::time::sleep(RECONNECT_DELAY).await;
            }
        }
    }
}

impl AsyncElement for MqttSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MQTT sink",
            "Sink/Network",
            "Publishes the analytics metadata, blobs and text of each frame as one MQTT message",
            "g2g",
        )
    }

    /// Reads the text payload out of host memory; the metadata itself travels
    /// beside the frame in any domain.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Metadata rides on any media type, so the sink takes whatever arrives.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.topic.is_empty() {
            g2g_error!(self, "topic is required");
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.text_input = matches!(absolute_caps, Caps::Text { .. });
        if let Some((width, height)) = caps_dimensions(absolute_caps) {
            self.width = width;
            self.height = height;
        }
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
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(record) =
                        frame_record(&frame, self.text_input, self.width, self.height)
                    else {
                        return Ok(());
                    };
                    let payload = serde_json::to_vec(&record)
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    self.publish(payload);
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.configure_pipeline(&caps)?;
                }
                PipelinePacket::Eos => self.drain().await,
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MQTTSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.broker.set_property(name, &value) {
            return result;
        }
        match name {
            "topic" => self.topic = value.as_str().ok_or(PropError::Type)?.to_string(),
            "qos" => {
                let level = value.as_uint().ok_or(PropError::Type)?;
                self.qos = qos_from_level(level).ok_or(PropError::Value)?;
            }
            "retain" => self.retain = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.broker.get_property(name) {
            return Some(value);
        }
        match name {
            "topic" => Some(PropValue::Str(self.topic.clone())),
            "qos" => Some(PropValue::Uint(qos_level(self.qos))),
            "retain" => Some(PropValue::Bool(self.retain)),
            _ => None,
        }
    }
}

impl LogSource for MqttSink {
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

impl PadTemplates for MqttSink {
    /// Wildcard sink, matching the `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

/// `MqttSink`'s settable properties: the broker, the topic, and how each
/// record is published.
static MQTTSINK_PROPS: &[PropertySpec] = broker_property_table![
    PropertySpec::new(
        "topic",
        PropKind::Str,
        "topic each record is published to, required",
    )
    .with_default(""),
    PropertySpec::new("qos", PropKind::Uint, "MQTT quality of service level")
        .with_range("0", QOS_MAX_TEXT)
        .with_default(DEFAULT_QOS_TEXT),
    PropertySpec::new(
        "retain",
        PropKind::Bool,
        "ask the broker to keep the last record for new subscribers",
    )
    .with_default("false"),
];
