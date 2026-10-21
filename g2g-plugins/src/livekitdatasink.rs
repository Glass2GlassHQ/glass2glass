use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::time::Instant;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_error, g2g_warn, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata,
    G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::livekit_signal::{
    room_property_table, DataPacketKind, IceSettings, RoomSettings, UserDataPacket, VideoGrant,
    ICE_TRANSPORT_POLICY_PROP, LIVEKIT_TURN_SERVER_PROP, TURN_PASS_PROP, TURN_USER_PROP,
};
use crate::livekitdata::{DataSession, Wake};
use crate::webrtc_util::non_empty;
use crate::webrtcdata::MAX_MESSAGE_SIZE;

const QUEUE_DEPTH: usize = 256;
const EOS_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);
// how often the session task retries a packet the send buffer could not take
const RETRY_PERIOD: Duration = Duration::from_millis(5);

pub struct LiveKitDataSink {
    room_settings: RoomSettings,
    ice: IceSettings,
    topic: String,
    reliable: bool,
    destination_identities: Vec<String>,
    tx: Option<mpsc::Sender<Vec<u8>>>,
    session: Option<JoinHandle<()>>,
    sent: u64,
    log_name: LogName,
}

impl core::fmt::Debug for LiveKitDataSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LiveKitDataSink")
            .field("url", &self.room_settings.url)
            .field("room", &self.room_settings.room)
            .field("identity", &self.room_settings.identity)
            .field("topic", &self.topic)
            .field("reliable", &self.reliable)
            .field("sent", &self.sent)
            .finish()
    }
}

impl LogSource for LiveKitDataSink {
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

impl LiveKitDataSink {
    pub fn new(
        url: impl Into<String>,
        room: impl Into<String>,
        identity: impl Into<String>,
    ) -> Self {
        Self {
            room_settings: RoomSettings::new(url, room, identity),
            ice: IceSettings::default(),
            topic: String::new(),
            reliable: true,
            destination_identities: Vec::new(),
            tx: None,
            session: None,
            sent: 0,
            log_name: LogName::new(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>, secret: impl Into<String>) -> Self {
        self.room_settings.api_key = key.into();
        self.room_settings.api_secret = secret.into();
        self
    }

    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = topic.into();
        self
    }

    pub fn with_reliable(mut self, reliable: bool) -> Self {
        self.reliable = reliable;
        self
    }

    pub fn with_destination_identities<I, S>(mut self, identities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.destination_identities = identities.into_iter().map(Into::into).collect();
        self
    }

    pub fn sent(&self) -> u64 {
        self.sent
    }

    fn kind(&self) -> DataPacketKind {
        if self.reliable {
            DataPacketKind::Reliable
        } else {
            DataPacketKind::Lossy
        }
    }

    fn packet(&self, payload: Vec<u8>) -> UserDataPacket {
        UserDataPacket {
            kind: self.kind(),
            payload,
            topic: non_empty(&self.topic),
            destination_identities: self.destination_identities.clone(),
            participant_identity: String::new(),
        }
    }

    async fn start_session(&mut self) -> Result<mpsc::Sender<Vec<u8>>, G2gError> {
        let grant = VideoGrant::data_sender(self.room_settings.room.clone());
        let session = DataSession::open(&self.room_settings, &self.ice, &grant, self).await?;
        let (tx, rx) = mpsc::channel(QUEUE_DEPTH);
        self.session = Some(tokio::spawn(run_session(session, self.kind(), rx)));
        self.tx = Some(tx.clone());
        Ok(tx)
    }

    async fn drain(&mut self) {
        self.tx = None;
        let Some(session) = self.session.take() else {
            return;
        };
        let abort = session.abort_handle();
        if tokio::time::timeout(EOS_DRAIN_TIMEOUT, session)
            .await
            .is_err()
        {
            g2g_warn!(self, "queued packets did not reach the server before eos");
            abort.abort();
        }
    }
}

impl Drop for LiveKitDataSink {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            session.abort();
        }
    }
}

async fn run_session(
    mut session: DataSession,
    kind: DataPacketKind,
    mut rx: mpsc::Receiver<Vec<u8>>,
) {
    let mut pending: Option<Vec<u8>> = None;
    let mut input_done = false;
    loop {
        let drained = session.drain().await;
        if drained.ended {
            return;
        }
        if let Some(packet) = pending.take() {
            match session.write(kind, &packet) {
                Ok(true) => {}
                Ok(false) => pending = Some(packet),
                Err(_) => return,
            }
        }
        let flushing = pending.is_some() || input_done;
        if input_done && pending.is_none() && session.buffered_amount(kind) == 0 {
            session.leave().await;
            return;
        }
        let deadline = if flushing {
            drained.deadline.min(Instant::now() + RETRY_PERIOD)
        } else {
            drained.deadline
        };
        let accepting = !flushing && session.is_open(kind);
        match session.wait(deadline, accepting.then_some(&mut rx)).await {
            Wake::Outgoing(Some(packet)) => pending = Some(packet),
            Wake::Outgoing(None) => input_done = true,
            Wake::Network => {}
            Wake::Closed => return,
        }
    }
}

impl AsyncElement for LiveKitDataSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "LiveKit data sink",
            "Sink/Network/WebRTC",
            "Sends each frame as a LiveKit user data packet into a room",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let bytes = self.packet(slice.to_vec()).encode();
                    if bytes.len() > MAX_MESSAGE_SIZE {
                        g2g_error!(
                            self,
                            "a {} byte packet exceeds the {MAX_MESSAGE_SIZE} byte data-channel message",
                            bytes.len()
                        );
                        return Err(G2gError::Hardware(HardwareError::Other));
                    }
                    let tx = match &self.tx {
                        Some(tx) => tx.clone(),
                        None => self.start_session().await?,
                    };
                    tx.send(bytes).await.map_err(|_| G2gError::Shutdown)?;
                    self.sent += 1;
                }
                PipelinePacket::Eos => self.drain().await,
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVEKITDATASINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.room_settings.set_property(name, &value) {
            return result;
        }
        if let Some(result) = self.ice.set_property(name, &value) {
            return result;
        }
        match name {
            "topic" => self.topic = value.as_str().ok_or(PropError::Type)?.into(),
            "reliable" => self.reliable = value.as_bool().ok_or(PropError::Type)?,
            "destination-identities" => {
                let list = value.as_str().ok_or(PropError::Type)?;
                self.destination_identities = list
                    .split(',')
                    .map(str::trim)
                    .filter(|identity| !identity.is_empty())
                    .map(String::from)
                    .collect();
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.room_settings.get_property(name) {
            return Some(value);
        }
        if let Some(value) = self.ice.get_property(name) {
            return Some(value);
        }
        match name {
            "topic" => Some(PropValue::Str(self.topic.clone())),
            "reliable" => Some(PropValue::Bool(self.reliable)),
            "destination-identities" => Some(PropValue::Str(self.destination_identities.join(","))),
            _ => None,
        }
    }
}

impl PadTemplates for LiveKitDataSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

static LIVEKITDATASINK_PROPS: &[PropertySpec] = room_property_table![
    PropertySpec::new(
        "topic",
        PropKind::Str,
        "topic every packet carries, none when empty",
    )
    .with_default(""),
    PropertySpec::new(
        "reliable",
        PropKind::Bool,
        "send on the reliable data channel, the lossy one when false",
    )
    .with_default("true"),
    PropertySpec::new(
        "destination-identities",
        PropKind::Str,
        "participants to deliver to, comma-separated, everyone when empty",
    )
    .with_default(""),
    LIVEKIT_TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    ICE_TRANSPORT_POLICY_PROP,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn properties_round_trip() {
        let mut sink = LiveKitDataSink::new("ws://h:7880", "room", "id");
        assert_eq!(sink.get_property("reliable"), Some(PropValue::Bool(true)));
        sink.set_property("reliable", PropValue::Bool(false))
            .unwrap();
        sink.set_property("topic", PropValue::Str("input".into()))
            .unwrap();
        sink.set_property(
            "destination-identities",
            PropValue::Str("alice, bob,,".into()),
        )
        .unwrap();
        sink.set_property("api-key", PropValue::Str("k".into()))
            .unwrap();
        assert_eq!(sink.kind(), DataPacketKind::Lossy);
        assert_eq!(sink.destination_identities, ["alice", "bob"]);
        assert_eq!(
            sink.get_property("destination-identities"),
            Some(PropValue::Str("alice,bob".into()))
        );
        assert_eq!(sink.room_settings.api_key, "k");
        assert_eq!(
            sink.set_property("reliable", PropValue::Str("no".into())),
            Err(PropError::Type)
        );
        assert_eq!(
            sink.set_property("nope", PropValue::Bool(true)),
            Err(PropError::Unknown)
        );
    }

    #[test]
    fn packet_carries_the_configured_topic_and_destinations() {
        let sink = LiveKitDataSink::new("ws://h:7880", "room", "id")
            .with_topic("input")
            .with_destination_identities(["publisher"]);
        let packet = UserDataPacket::decode(&sink.packet(alloc::vec![1, 2, 3]).encode()).unwrap();
        assert_eq!(packet.payload, [1, 2, 3]);
        assert_eq!(packet.topic.as_deref(), Some("input"));
        assert_eq!(packet.destination_identities, ["publisher"]);
        assert_eq!(packet.kind, DataPacketKind::Reliable);
        let untagged = LiveKitDataSink::new("ws://h:7880", "room", "id").packet(Vec::new());
        assert_eq!(untagged.topic, None);
    }
}
