use core::future::{poll_fn, Future};
use core::pin::Pin;
use core::task::Poll;

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::{String, ToString};
use alloc::sync::{Arc, Weak};
use alloc::vec;
use alloc::vec::Vec;

use spin::Mutex;

use g2g_core::{
    Caps, ConfigureOutcome, ElementMetadata, Frame, G2gError, MultiInputElement,
    MultiOutputElement, MultiOutputSink, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Segment,
};

const DEFAULT_DISPATCH_LOG: &str = "default";

type PortsInDispatchOrder = Mutex<VecDeque<usize>>;

static DISPATCH_LOGS: Mutex<BTreeMap<String, Weak<PortsInDispatchOrder>>> =
    Mutex::new(BTreeMap::new());

fn shared_dispatch_log(name: &str) -> Arc<PortsInDispatchOrder> {
    let mut logs = DISPATCH_LOGS.lock();
    logs.retain(|_, log| log.strong_count() > 0);
    if let Some(log) = logs.get(name).and_then(Weak::upgrade) {
        return log;
    }
    let log = Arc::new(Mutex::new(VecDeque::new()));
    logs.insert(name.to_string(), Arc::downgrade(&log));
    log
}

static DISPATCH_LOG_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "dispatch-log",
    PropKind::Str,
    "name shared by a framesplit and the framemerge that restores its frame order",
)
.with_default(DEFAULT_DISPATCH_LOG)];

#[derive(Debug)]
pub struct FrameSplit {
    ports: usize,
    dispatch_log_name: String,
    dispatch_log: Arc<PortsInDispatchOrder>,
}

impl FrameSplit {
    pub fn new(ports: usize) -> Self {
        assert!(ports > 0, "FrameSplit needs at least one output");
        Self {
            ports,
            dispatch_log_name: DEFAULT_DISPATCH_LOG.to_string(),
            dispatch_log: shared_dispatch_log(DEFAULT_DISPATCH_LOG),
        }
    }

    pub fn with_dispatch_log(mut self, name: &str) -> Self {
        self.dispatch_log_name = name.to_string();
        self.dispatch_log = shared_dispatch_log(name);
        self
    }
}

impl MultiOutputElement for FrameSplit {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Frame split",
            "Generic",
            "Sends each frame to the first output with room, for framemerge to put back in order",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DISPATCH_LOG_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "dispatch-log" => {
                let log_name = value.as_str().ok_or(PropError::Type)?;
                self.dispatch_log_name = log_name.to_string();
                self.dispatch_log = shared_dispatch_log(log_name);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "dispatch-log" => Some(PropValue::Str(self.dispatch_log_name.clone())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(_) => {
                    let ports = self.ports;
                    let port = poll_fn(|cx| {
                        for port in 0..ports {
                            if let Poll::Ready(ready) = out.poll_port_ready(cx, port) {
                                return Poll::Ready(ready.map(|()| port));
                            }
                        }
                        Poll::Pending
                    })
                    .await?;
                    self.dispatch_log.lock().push_back(port);
                    out.push_to(port, packet).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::CapsChanged(caps.clone()))
                            .await?;
                    }
                }
                PipelinePacket::Segment(segment) => {
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::Segment(segment)).await?;
                    }
                }
                PipelinePacket::Flush => {
                    self.dispatch_log.lock().clear();
                    for port in 0..self.ports {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                // the runner sends the end of stream to every output after this returns
                _ => {}
            }
            Ok(())
        })
    }
}

#[derive(Debug)]
enum Arrival {
    Caps(Caps),
    Frame(Frame),
}

#[derive(Debug)]
pub struct FrameMerge {
    dispatch_log_name: String,
    dispatch_log: Arc<PortsInDispatchOrder>,
    configured: Vec<Option<Caps>>,
    arrivals: Vec<VecDeque<Arrival>>,
    ended: Vec<bool>,
    emitted_caps: Option<Caps>,
    emitted_frames: u64,
    last_segment: Option<Segment>,
}

impl FrameMerge {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "FrameMerge needs at least one input");
        Self {
            dispatch_log_name: DEFAULT_DISPATCH_LOG.to_string(),
            dispatch_log: shared_dispatch_log(DEFAULT_DISPATCH_LOG),
            configured: vec![None; inputs],
            arrivals: (0..inputs).map(|_| VecDeque::new()).collect(),
            ended: vec![false; inputs],
            emitted_caps: None,
            emitted_frames: 0,
            last_segment: None,
        }
    }

    pub fn with_dispatch_log(mut self, name: &str) -> Self {
        self.dispatch_log_name = name.to_string();
        self.dispatch_log = shared_dispatch_log(name);
        self
    }

    async fn emit_in_dispatch_order(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            let Some(port) = self.dispatch_log.lock().front().copied() else {
                return Ok(());
            };
            let frame = loop {
                match self.arrivals[port].pop_front() {
                    Some(Arrival::Caps(caps)) => {
                        if self.emitted_caps.as_ref() != Some(&caps) {
                            self.emitted_caps = Some(caps.clone());
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                        }
                    }
                    Some(Arrival::Frame(frame)) => break Some(frame),
                    None => break None,
                }
            };
            match frame {
                Some(mut frame) => {
                    self.dispatch_log.lock().pop_front();
                    // branch elements number their own output from zero
                    frame.sequence = self.emitted_frames;
                    self.emitted_frames += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                None if self.ended[port] => {
                    self.dispatch_log.lock().pop_front();
                }
                None => return Ok(()),
            }
        }
    }
}

impl MultiInputElement for FrameMerge {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        self.arrivals.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn output_follows_input(&self) -> Option<usize> {
        Some(0)
    }

    fn forwards_caps_changes(&self) -> bool {
        true
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured[input] = Some(absolute_caps.clone());
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self) -> Result<Caps, G2gError> {
        self.configured[0].clone().ok_or(G2gError::NotConfigured)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Frame merge",
            "Generic",
            "Puts the branches of a framesplit back into the order it split the frames in",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DISPATCH_LOG_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "dispatch-log" => {
                let log_name = value.as_str().ok_or(PropError::Type)?;
                self.dispatch_log_name = log_name.to_string();
                self.dispatch_log = shared_dispatch_log(log_name);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "dispatch-log" => Some(PropValue::Str(self.dispatch_log_name.clone())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    self.arrivals[input].push_back(Arrival::Frame(frame));
                    self.emit_in_dispatch_order(out).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.arrivals[input].push_back(Arrival::Caps(caps));
                    self.emit_in_dispatch_order(out).await?;
                }
                PipelinePacket::Eos => {
                    self.ended[input] = true;
                    self.emit_in_dispatch_order(out).await?;
                }
                PipelinePacket::Flush => {
                    self.arrivals[input].clear();
                    if input == 0 {
                        self.last_segment = None;
                        out.push(PipelinePacket::Flush).await?;
                    }
                }
                // framesplit sends every branch the same segment
                PipelinePacket::Segment(segment)
                    if input == 0 && self.last_segment != Some(segment) =>
                {
                    self.last_segment = Some(segment);
                    out.push(PipelinePacket::Segment(segment)).await?;
                }
                _ => {}
            }
            Ok(())
        })
    }
}
