//! MQTT source (M1179): subscribes to a topic filter on an MQTT broker and
//! emits each message received as one `Caps::Text` frame holding its payload,
//! the control-message side of `mqttsink`. A message is stamped with its
//! arrival time and no presentation time, like a raw datagram out of `udpsrc`.
//!
//! The connection is polled in the source loop itself. When the broker drops,
//! the loop reconnects after a delay and subscribes again. `num-buffers` bounds
//! the run: a broker feed has no in-band end, so Eos comes only from that limit.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use rumqttc::{AsyncClient, Event, EventLoop, Packet, QoS};

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    g2g_error, g2g_info, g2g_warn, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, HardwareError, LatencyReport, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    TextFormat,
};

use crate::mqtt::{
    broker_property_table, qos_from_level, qos_level, BrokerSettings, DEFAULT_QOS,
    DEFAULT_QOS_TEXT, QOS_MAX_TEXT, RECONNECT_DELAY,
};

/// Requests the source can have queued to its event loop: a subscribe per
/// (re)connection, nothing more.
const REQUEST_QUEUE_LEN: usize = 4;

/// Emits each message on an MQTT topic filter as a text frame.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::mqttsrc::MqttSrc;
///
/// // gst-launch equivalent: mqttsrc host=broker.local topic=cameras/+/control
/// let source = MqttSrc::new().with_host("broker.local").with_topic("cameras/+/control");
/// assert_eq!(source.received(), 0);
/// ```
#[derive(Debug)]
pub struct MqttSrc {
    broker: BrokerSettings,
    topic: String,
    qos: QoS,
    message_limit: u64,
    received: u64,
    configured: bool,
    log_name: LogName,
}

impl Default for MqttSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl MqttSrc {
    pub fn new() -> Self {
        Self {
            broker: BrokerSettings::new(),
            topic: String::new(),
            qos: DEFAULT_QOS,
            message_limit: u64::MAX,
            received: 0,
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

    /// The topic filter subscribed to, wildcards included (the `topic`
    /// property).
    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = topic.into();
        self
    }

    /// Messages to emit before Eos (the `num-buffers` property).
    pub fn with_message_limit(mut self, limit: u64) -> Self {
        self.message_limit = limit;
        self
    }

    /// Messages emitted so far.
    pub fn received(&self) -> u64 {
        self.received
    }

    fn output_caps(&self) -> Caps {
        Caps::Text {
            format: TextFormat::Utf8,
        }
    }

    fn fallback_client_id(&self) -> &str {
        self.log_name
            .instance()
            .unwrap_or_else(|| short_type_name::<Self>())
    }

    fn subscribe(&self, client: &AsyncClient) {
        if let Err(error) = client.try_subscribe(self.topic.clone(), self.qos) {
            g2g_warn!(self, "subscribe not queued: {error}");
        }
    }

    /// Poll the connection until `message_limit` messages have gone out,
    /// subscribing on every connection and reconnecting after a delay.
    async fn receive(
        &mut self,
        client: &AsyncClient,
        event_loop: &mut EventLoop,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        while self.received < self.message_limit {
            match event_loop.poll().await {
                Ok(Event::Incoming(Packet::ConnAck(ack))) => {
                    g2g_info!(self, "connected: {:?}", ack.code);
                    self.subscribe(client);
                }
                Ok(Event::Incoming(Packet::Publish(publish))) => {
                    let frame = text_frame(publish.payload.to_vec(), self.received);
                    self.received += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                Ok(_) => {}
                Err(error) => {
                    g2g_warn!(self, "connection lost: {error}");
                    tokio::time::sleep(RECONNECT_DELAY).await;
                }
            }
        }
        Ok(())
    }
}

/// A frame carrying one message's payload, stamped with its arrival time and
/// no presentation time.
fn text_frame(payload: Vec<u8>, sequence: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
        FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
    )
}

impl SourceLoop for MqttSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.output_caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.output_caps(),
        ))))
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.output_caps())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.topic.is_empty() {
            g2g_error!(self, "topic is required");
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "MQTT source",
            "Source/Network",
            "Emits each message on an MQTT topic filter as a text frame",
            "g2g",
        )
    }

    /// Live, with no frame period: a message is emitted the moment it arrives.
    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.message_limit, out).await? {
                return Ok(0);
            }
            let options = self.broker.mqtt_options(self.fallback_client_id());
            let (client, mut event_loop) = AsyncClient::new(options, REQUEST_QUEUE_LEN);
            self.receive(&client, &mut event_loop, out).await?;
            if let Err(error) = client.try_disconnect() {
                g2g_warn!(self, "disconnect not queued: {error}");
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.received)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MQTTSRC_PROPS
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
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.message_limit, &value)?,
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
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.message_limit)),
            _ => None,
        }
    }
}

impl LogSource for MqttSrc {
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

impl PadTemplates for MqttSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::Text {
            format: TextFormat::Utf8,
        }))])
    }
}

/// `MqttSrc`'s settable properties: the broker, the topic filter, and how many
/// messages to emit.
static MQTTSRC_PROPS: &[PropertySpec] = broker_property_table![
    PropertySpec::new(
        "topic",
        PropKind::Str,
        "topic filter subscribed to, wildcards allowed, required",
    )
    .with_default(""),
    PropertySpec::new("qos", PropKind::Uint, "MQTT quality of service level")
        .with_range("0", QOS_MAX_TEXT)
        .with_default(DEFAULT_QOS_TEXT),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "messages to emit then EOS (-1 = until shutdown)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
];
