//! In-process links between independent graphs (`intersink` / `intersrc`), the
//! gst-plugins-rs `intersink` / `intersrc` analog. An `intersink` publishes its
//! stream under a `producer-name`, and every `intersrc` with that name receives
//! it, through a bounded queue that drops the oldest frame when full.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use spin::Mutex;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{bounded, Receiver, SendError, Sender, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
    LatencyReport, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec,
};

const DEFAULT_PRODUCER_NAME: &str = "default";
const DEFAULT_MAX_BUFFERS: usize = 16;
const DEFAULT_MAX_BUFFERS_TEXT: &str = "16";
// a sink is never told the memory domain its input negotiated
const CARRIED_DOMAIN: MemoryDomainKind = MemoryDomainKind::System;

#[derive(Debug)]
struct Consumer {
    id: usize,
    sender: Sender<PipelinePacket>,
}

#[derive(Debug, Default)]
struct ProducerSlot {
    caps: Option<Caps>,
    streaming: bool,
    consumers: Vec<Consumer>,
}

impl ProducerSlot {
    fn announce_caps(&mut self, caps: &Caps) {
        self.caps = Some(caps.clone());
        self.consumers.retain(|consumer| {
            deliver_dropping_oldest(&consumer.sender, PipelinePacket::CapsChanged(caps.clone()))
        });
    }
}

static PRODUCERS: Mutex<BTreeMap<String, ProducerSlot>> = Mutex::new(BTreeMap::new());
static NEXT_CONSUMER_ID: AtomicUsize = AtomicUsize::new(0);

fn deliver_dropping_oldest(sender: &Sender<PipelinePacket>, packet: PipelinePacket) -> bool {
    let packet = match sender.try_send(packet) {
        Ok(()) => return true,
        Err((_, SendError::Closed)) => return false,
        Err((packet, SendError::Full)) => packet,
    };
    sender.evict_front_matching(|queued| matches!(queued, PipelinePacket::DataFrame(_)));
    !matches!(sender.try_send(packet), Err((_, SendError::Closed)))
}

fn claim(name: &str, caps: &Caps) -> Result<(), G2gError> {
    let mut producers = PRODUCERS.lock();
    let slot = producers.entry(name.to_string()).or_default();
    if slot.caps.is_some() {
        return Err(G2gError::NotConfigured);
    }
    slot.announce_caps(caps);
    Ok(())
}

fn change_caps(name: &str, caps: &Caps) {
    let mut producers = PRODUCERS.lock();
    let Some(slot) = producers.get_mut(name) else {
        return;
    };
    if slot.caps.as_ref() == Some(caps) {
        return;
    }
    slot.announce_caps(caps);
}

fn publish_frame(name: &str, frame: Frame) {
    let mut consumers = match PRODUCERS.lock().get_mut(name) {
        Some(slot) => {
            slot.streaming = true;
            core::mem::take(&mut slot.consumers)
        }
        None => return,
    };
    // frame copies happen outside the lock every graph shares
    let mut frame = Some(frame);
    let mut remaining = consumers.len();
    consumers.retain(|consumer| {
        remaining -= 1;
        let copy = if remaining == 0 {
            frame.take()
        } else {
            frame.as_ref().map(Frame::share)
        };
        copy.is_some_and(|copy| {
            deliver_dropping_oldest(&consumer.sender, PipelinePacket::DataFrame(copy))
        })
    });
    if let Some(slot) = PRODUCERS.lock().get_mut(name) {
        consumers.append(&mut slot.consumers);
        slot.consumers = consumers;
    }
}

fn release(name: &str) {
    PRODUCERS.lock().remove(name);
}

fn subscribe(name: &str, max_buffers: usize) -> Subscription {
    let (sender, receiver) = bounded(max_buffers);
    let id = NEXT_CONSUMER_ID.fetch_add(1, Ordering::Relaxed);
    let mut producers = PRODUCERS.lock();
    let slot = producers.entry(name.to_string()).or_default();
    if let Some(caps) = &slot.caps {
        deliver_dropping_oldest(&sender, PipelinePacket::CapsChanged(caps.clone()));
    }
    slot.consumers.push(Consumer { id, sender });
    Subscription {
        name: name.to_string(),
        id,
        receiver,
        joined_mid_stream: slot.streaming,
    }
}

#[derive(Debug)]
struct Subscription {
    name: String,
    id: usize,
    receiver: Receiver<PipelinePacket>,
    joined_mid_stream: bool,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let mut producers = PRODUCERS.lock();
        let Some(slot) = producers.get_mut(&self.name) else {
            return;
        };
        slot.consumers.retain(|consumer| consumer.id != self.id);
        if slot.caps.is_none() && slot.consumers.is_empty() {
            producers.remove(&self.name);
        }
    }
}

/// Publishes its input to every `intersrc` with the same `producer-name`. One
/// `intersink` per name at a time.
#[derive(Debug)]
pub struct InterSink {
    producer_name: String,
    claimed_name: Option<String>,
}

impl Default for InterSink {
    fn default() -> Self {
        Self {
            producer_name: DEFAULT_PRODUCER_NAME.to_string(),
            claimed_name: None,
        }
    }
}

impl InterSink {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Drop for InterSink {
    fn drop(&mut self) {
        if let Some(name) = self.claimed_name.take() {
            release(&name);
        }
    }
}

impl AsyncElement for InterSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(CARRIED_DOMAIN)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match &self.claimed_name {
            Some(name) => change_caps(name, absolute_caps),
            None => {
                claim(&self.producer_name, absolute_caps)?;
                self.claimed_name = Some(self.producer_name.clone());
            }
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Inter-graph sink",
            "Sink",
            "Publishes its stream to every intersrc with the same producer-name",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let Some(name) = &self.claimed_name else {
            return core::future::ready(Err(G2gError::NotConfigured));
        };
        match packet {
            PipelinePacket::DataFrame(frame) if frame.domain.kind() != CARRIED_DOMAIN => {
                return core::future::ready(Err(G2gError::UnsupportedDomain));
            }
            PipelinePacket::DataFrame(frame) => publish_frame(name, frame),
            PipelinePacket::CapsChanged(caps) => change_caps(name, &caps),
            PipelinePacket::Eos => {
                release(name);
                self.claimed_name = None;
            }
            _ => {}
        }
        core::future::ready(Ok(()))
    }

    fn properties(&self) -> &'static [PropertySpec] {
        INTERSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "producer-name" => {
                self.producer_name = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "producer-name" => Some(PropValue::Str(self.producer_name.clone())),
            _ => None,
        }
    }
}

impl PadTemplates for InterSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

static INTERSINK_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "producer-name",
    PropKind::Str,
    "name the intersrc consumers connect to",
)
.with_default(DEFAULT_PRODUCER_NAME)];

/// Emits the stream of the `intersink` with the same `producer-name`, waiting
/// for one to start if none has.
#[derive(Debug)]
pub struct InterSrc {
    producer_name: String,
    max_buffers: usize,
    subscription: Option<Subscription>,
    caps: Option<Caps>,
}

impl Default for InterSrc {
    fn default() -> Self {
        Self {
            producer_name: DEFAULT_PRODUCER_NAME.to_string(),
            max_buffers: DEFAULT_MAX_BUFFERS,
            subscription: None,
            caps: None,
        }
    }
}

impl InterSrc {
    pub fn new() -> Self {
        Self::default()
    }

    async fn producer_caps(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.caps {
            return Ok(caps.clone());
        }
        let subscription = self
            .subscription
            .get_or_insert_with(|| subscribe(&self.producer_name, self.max_buffers));
        let Some(PipelinePacket::CapsChanged(caps)) = subscription.receiver.recv().await else {
            return Err(G2gError::CapsMismatch);
        };
        self.caps = Some(caps.clone());
        Ok(caps)
    }
}

impl SourceLoop for InterSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(self.producer_caps())
    }

    async fn caps_constraint(&mut self) -> Result<CapsConstraint<'_>, G2gError> {
        Ok(CapsConstraint::Produces(CapsSet::one(
            self.producer_caps().await?,
        )))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.caps.is_none() {
            return Err(G2gError::CapsMismatch);
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let subscription = self.subscription.take().ok_or(G2gError::NotConfigured)?;
            let compressed_video = matches!(self.caps, Some(Caps::CompressedVideo { .. }));
            let mut awaiting_keyframe = compressed_video && subscription.joined_mid_stream;
            let mut pushed = 0u64;
            while let Some(packet) = subscription.receiver.recv().await {
                if let PipelinePacket::DataFrame(frame) = &packet {
                    if awaiting_keyframe && !frame.timing.keyframe {
                        continue;
                    }
                    awaiting_keyframe = false;
                    pushed += 1;
                }
                out.push(packet).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    fn output_memory(&self) -> MemoryDomainKind {
        CARRIED_DOMAIN
    }

    fn properties(&self) -> &'static [PropertySpec] {
        INTERSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "producer-name" => {
                self.producer_name = value.as_str().ok_or(PropError::Type)?.to_string();
            }
            "max-buffers" => {
                let max_buffers = value.as_uint().ok_or(PropError::Type)?;
                self.max_buffers = usize::try_from(max_buffers)
                    .ok()
                    .filter(|max_buffers| *max_buffers > 0)
                    .ok_or(PropError::Value)?;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "producer-name" => Some(PropValue::Str(self.producer_name.clone())),
            "max-buffers" => Some(PropValue::Uint(self.max_buffers as u64)),
            _ => None,
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Inter-graph source",
            "Source",
            "Emits the stream of the intersink with the same producer-name",
            "g2g",
        )
    }
}

static INTERSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "producer-name",
        PropKind::Str,
        "name of the intersink to receive from",
    )
    .with_default(DEFAULT_PRODUCER_NAME),
    PropertySpec::new(
        "max-buffers",
        PropKind::Uint,
        "frames queued from the producer before the oldest is dropped",
    )
    .with_default(DEFAULT_MAX_BUFFERS_TEXT),
];
