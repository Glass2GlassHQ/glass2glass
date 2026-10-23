use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    g2g_debug, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, LatencyReport, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::livekit_signal::{
    room_property_table, DataPacketKind, IceSettings, RoomSettings, UserDataPacket, VideoGrant,
    ICE_TRANSPORT_POLICY_PROP, LIVEKIT_TURN_SERVER_PROP, TURN_PASS_PROP, TURN_USER_PROP,
};
use crate::livekitdata::{ChannelMessage, DataSession, Wake};

const OUTPUT_CAPS: Caps = Caps::ByteStream {
    encoding: ByteStreamEncoding::Raw,
};

pub struct LiveKitDataSrc {
    room_settings: RoomSettings,
    ice: IceSettings,
    topic: String,
    sender_identity: String,
    message_limit: u64,
    emitted_reliable: u64,
    emitted_lossy: u64,
    log_name: LogName,
}

impl core::fmt::Debug for LiveKitDataSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LiveKitDataSrc")
            .field("url", &self.room_settings.url)
            .field("room", &self.room_settings.room)
            .field("identity", &self.room_settings.identity)
            .field("topic", &self.topic)
            .field("sender_identity", &self.sender_identity)
            .finish()
    }
}

impl LogSource for LiveKitDataSrc {
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

impl LiveKitDataSrc {
    pub fn new(
        url: impl Into<String>,
        room: impl Into<String>,
        identity: impl Into<String>,
    ) -> Self {
        Self {
            room_settings: RoomSettings::new(url, room, identity),
            ice: IceSettings::default(),
            topic: String::new(),
            sender_identity: String::new(),
            message_limit: u64::MAX,
            emitted_reliable: 0,
            emitted_lossy: 0,
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

    pub fn with_sender_identity(mut self, identity: impl Into<String>) -> Self {
        self.sender_identity = identity.into();
        self
    }

    pub fn with_message_limit(mut self, limit: u64) -> Self {
        self.message_limit = limit;
        self
    }

    pub fn emitted_from(&self, kind: DataPacketKind) -> u64 {
        match kind {
            DataPacketKind::Reliable => self.emitted_reliable,
            DataPacketKind::Lossy => self.emitted_lossy,
        }
    }

    fn emitted(&self) -> u64 {
        self.emitted_reliable + self.emitted_lossy
    }

    fn accept(&self, message: &ChannelMessage) -> Option<Vec<u8>> {
        let Some(packet) = UserDataPacket::decode(&message.bytes) else {
            g2g_debug!(
                self,
                "skipped a {} byte non-user packet",
                message.bytes.len()
            );
            return None;
        };
        let topic_matches = self.topic.is_empty() || packet.topic.as_deref() == Some(&self.topic);
        let sender_matches =
            self.sender_identity.is_empty() || packet.participant_identity == self.sender_identity;
        (topic_matches && sender_matches).then_some(packet.payload)
    }

    fn count(&mut self, kind: DataPacketKind) {
        match kind {
            DataPacketKind::Reliable => self.emitted_reliable += 1,
            DataPacketKind::Lossy => self.emitted_lossy += 1,
        }
    }

    async fn receive(
        &mut self,
        session: &mut DataSession,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        loop {
            let drained = session.drain().await;
            for message in &drained.messages {
                let Some(payload) = self.accept(message) else {
                    continue;
                };
                let frame = data_frame(payload, self.emitted());
                self.count(message.kind);
                out.push(PipelinePacket::DataFrame(frame)).await?;
                if self.emitted() >= self.message_limit {
                    return Ok(());
                }
            }
            if drained.ended {
                return Ok(());
            }
            if let Wake::Closed = session.wait(drained.deadline, None).await {
                return Ok(());
            }
        }
    }
}

fn data_frame(payload: Vec<u8>, sequence: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(payload.into_boxed_slice())),
        FrameTiming {
            arrival_ns: g2g_core::metrics::monotonic_ns(),
            ..FrameTiming::default()
        },
        sequence,
    )
}

impl SourceLoop for LiveKitDataSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(OUTPUT_CAPS))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(OUTPUT_CAPS))))
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(OUTPUT_CAPS)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "LiveKit data source",
            "Source/Network/WebRTC",
            "Emits each LiveKit user data packet received in a room as a frame",
            "g2g",
        )
    }

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
            if crate::numbuffers::finished_at_zero_limit(self.message_limit, out).await? {
                return Ok(0);
            }
            let grant = VideoGrant::data_receiver(self.room_settings.room.clone());
            let mut session = DataSession::open(&self.room_settings, &self.ice, &grant, &*self)
                .await
                .map_err(G2gError::during_connect)?;
            out.push(PipelinePacket::CapsChanged(OUTPUT_CAPS)).await?;
            let received = self.receive(&mut session, out).await;
            session.leave().await;
            received?;
            out.push(PipelinePacket::Eos).await?;
            Ok(self.emitted())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVEKITDATASRC_PROPS
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
            "sender-identity" => {
                self.sender_identity = value.as_str().ok_or(PropError::Type)?.into()
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.message_limit, &value)?,
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
            "sender-identity" => Some(PropValue::Str(self.sender_identity.clone())),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.message_limit)),
            _ => None,
        }
    }
}

impl PadTemplates for LiveKitDataSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(OUTPUT_CAPS))])
    }
}

static LIVEKITDATASRC_PROPS: &[PropertySpec] = room_property_table![
    PropertySpec::new(
        "topic",
        PropKind::Str,
        "emit only packets with this topic, every topic when empty",
    )
    .with_default(""),
    PropertySpec::new(
        "sender-identity",
        PropKind::Str,
        "emit only packets from this participant, every sender when empty",
    )
    .with_default(""),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "packets to emit then EOS (-1 = until the room ends)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    LIVEKIT_TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    ICE_TRANSPORT_POLICY_PROP,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn message(kind: DataPacketKind, topic: &str, sender: &str) -> ChannelMessage {
        let packet = UserDataPacket {
            kind,
            payload: alloc::vec![7, 0, 255],
            topic: Some(topic.into()),
            destination_identities: Vec::new(),
            participant_identity: sender.into(),
        };
        ChannelMessage {
            kind,
            bytes: packet.encode(),
        }
    }

    #[test]
    fn topic_and_sender_filters_select_packets() {
        let open = LiveKitDataSrc::new("ws://h:7880", "room", "id");
        let filtered = LiveKitDataSrc::new("ws://h:7880", "room", "id")
            .with_topic("input")
            .with_sender_identity("viewer");
        let wanted = message(DataPacketKind::Lossy, "input", "viewer");
        let other_topic = message(DataPacketKind::Reliable, "chat", "viewer");
        let other_sender = message(DataPacketKind::Reliable, "input", "someone");
        assert_eq!(filtered.accept(&wanted), Some(alloc::vec![7, 0, 255]));
        assert_eq!(filtered.accept(&other_topic), None);
        assert_eq!(filtered.accept(&other_sender), None);
        assert!(open.accept(&other_topic).is_some());
        assert!(open.accept(&other_sender).is_some());
        let garbage = ChannelMessage {
            kind: DataPacketKind::Reliable,
            bytes: alloc::vec![0x12, 0x05, 0x01],
        };
        assert_eq!(open.accept(&garbage), None);
    }

    #[test]
    fn properties_round_trip() {
        let mut source = LiveKitDataSrc::new("ws://h:7880", "room", "id");
        for (name, value) in [
            ("topic", PropValue::Str("input".into())),
            ("sender-identity", PropValue::Str("viewer".into())),
            ("num-buffers", PropValue::Int(3)),
            ("token", PropValue::Str("tok".into())),
            ("ice-transport-policy", PropValue::Str("relay".into())),
        ] {
            source.set_property(name, value.clone()).unwrap();
            assert_eq!(source.get_property(name), Some(value), "{name}");
        }
        assert_eq!(source.message_limit, 3);
        assert_eq!(
            source.set_property("topic", PropValue::Uint(1)),
            Err(PropError::Type)
        );
        assert_eq!(
            source.set_property("nope", PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
    }
}
