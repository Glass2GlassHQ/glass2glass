//! Cursor-on-Target (CoT) bridge (M811): a drone's decoded MISB ST 0601
//! telemetry becomes a track on a TAK / ATAK situational-awareness network.
//!
//! The event builder ([`cot_event`]) is pure `no_std + alloc`: a
//! [`UasDatalink`] local set in, one CoT XML event out. The `CotSink` element
//! (`cotsink`) is the I/O half behind the `udp-egress` feature: `Caps::Klv`
//! packets in, one datagram (UDP unicast / multicast) or one TCP write per
//! parsed local set out.
//!
//! The identity strings in a local set are attacker-controlled bitstream data,
//! so every interpolated value is XML-escaped and C0 control characters (which
//! XML 1.0 forbids even escaped) are dropped: a hostile platform designation
//! cannot break out of the document.

use alloc::format;
use alloc::string::String;

use crate::klv::UasDatalink;
use crate::xmlutil::{iso8601_utc_us, xml_escape};

/// The CoT `<point>` "value unknown" sentinel: `ce` / `le` always (ST 0601
/// carries no error estimate), and `hae` when the set has no altitude.
/// pytak's `DEFAULT_COT_VAL = "9999999.0"`, applied to hae / ce / le when the
/// caller has no value (snstac/pytak `src/pytak/constants.py`, `functions.py`).
const UNKNOWN: &str = "9999999.0";

/// `<event version>`: the CoT schema types it `xs:decimal` with a minimum of 2;
/// `"2.0"` is what pytak and ATAK emit (ATAK-CIV
/// `takcot/mitre/CoT Base-Event Schema (PUBLIC RELEASE).xsd`).
const COT_VERSION: &str = "2.0";

/// `<event how>`: "m" machine generated, "g" derived from GPS receiver, the
/// provenance of ST 0601 platform position. Table verbatim in the CoT base
/// event schema; pytak defaults `how="m-g"` for the same reason.
const HOW_GPS: &str = "m-g";

/// Default CoT type: atom / friend / Air / Military / Fixed-wing / drone-RPV-UAV.
/// `a-.-A-M-F-Q` is "Air/Mil/Fixed/Drone,RPV,UAV" in the MITRE-derived type
/// table (dB-SPL/cot-types `CoTtypes.xml`); the second letter is the affiliation
/// ("f" friend, "u" unknown, ...), which is a deployment choice, hence the
/// `cot-type` property. A rotary-wing UAS is `a-f-A-M-H-Q`.
pub const DEFAULT_COT_TYPE: &str = "a-f-A-M-F-Q";

/// Default event uid. A TAK client keys the track on it, so one platform must
/// keep one uid across updates.
pub const DEFAULT_UID: &str = "g2g-uas";

/// Default TAK SA mesh multicast group, and the default stale interval: the
/// track disappears this long after its last update.
/// pytak: `DEFAULT_COT_URL = "udp+wo://239.2.3.1:6969"`.
pub const DEFAULT_HOST: &str = "239.2.3.1";
/// Seconds a track survives without an update.
pub const DEFAULT_STALE_SECS: u32 = 10;

/// Per-event knobs the sink passes to the builder.
#[derive(Debug, Clone, Copy)]
pub struct CotOptions<'a> {
    /// The event `uid`: the track identity a TAK client updates in place.
    pub uid: &'a str,
    /// The CoT type string (the MIL-STD-2525 derived atom taxonomy code).
    pub cot_type: &'a str,
    /// Seconds after the event time at which a client drops the track.
    pub stale_secs: u32,
}

impl Default for CotOptions<'_> {
    fn default() -> Self {
        Self {
            uid: DEFAULT_UID,
            cot_type: DEFAULT_COT_TYPE,
            stale_secs: DEFAULT_STALE_SECS,
        }
    }
}

/// Build the CoT event for one ST 0601 local set: the platform position as the
/// event `<point>`, the sensor pointing geometry as the `<detail><sensor>` cone
/// (ATAK draws it from azimuth + fov + range), and the frame center in
/// `<remarks>`, since a single event has no second `<point>`.
///
/// `None` (no event at all, never one at 0/0) when the set carries no platform
/// position or no tag 2 timestamp: CoT requires lat, lon, time, start and stale,
/// and inventing any of them would put a phantom track on the map.
pub fn cot_event(ls: &UasDatalink, opts: CotOptions<'_>) -> Option<String> {
    let time_us = ls.timestamp_us?;
    let (lat, lon) = (ls.sensor_lat_deg?, ls.sensor_lon_deg?);
    let stale_us = time_us.saturating_add((opts.stale_secs as u64).saturating_mul(1_000_000));
    let time = iso8601_utc_us(time_us);

    // ST 0601 tag 15 is altitude above mean sea level, CoT hae is above the
    // WGS-84 ellipsoid: they differ by the local geoid undulation (tens of
    // meters). No geoid model here, so the MSL value is passed through.
    let hae = ls
        .sensor_alt_m
        .map_or_else(|| String::from(UNKNOWN), |v| format!("{v:.1}"));

    let mut detail = String::new();
    // The callsign is the label a TAK client paints under the icon.
    let callsign = ls
        .platform_designation
        .as_deref()
        .or(ls.mission_id.as_deref())
        .unwrap_or(opts.uid);
    detail.push_str(&format!("<contact callsign=\"{}\"/>", xml_escape(callsign)));
    if let Some(course) = ls.heading_deg {
        // ATAK also reads a `speed` here; ST 0601 airspeed is not decoded, and a
        // made-up speed would draw a wrong heading arrow, so course rides alone.
        detail.push_str(&format!("<track course=\"{course:.1}\"/>"));
    }
    detail.push_str(&sensor_detail(ls));
    let remarks = remarks(ls);
    if !remarks.is_empty() {
        detail.push_str(&format!("<remarks>{remarks}</remarks>"));
    }

    Some(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n\
         <event version=\"{COT_VERSION}\" uid=\"{uid}\" type=\"{cot_type}\" time=\"{time}\" \
         start=\"{time}\" stale=\"{stale}\" how=\"{HOW_GPS}\">\
         <point lat=\"{lat:.6}\" lon=\"{lon:.6}\" hae=\"{hae}\" ce=\"{UNKNOWN}\" le=\"{UNKNOWN}\"/>\
         <detail>{detail}</detail></event>",
        uid = xml_escape(opts.uid),
        cot_type = xml_escape(opts.cot_type),
        stale = iso8601_utc_us(stale_us),
    ))
}

/// Build the ST 0805.1 Sensor Point of Interest event for one local set: a
/// second event (`b-m-p-s-p-i`) at the point the sensor looks at, tied to the
/// platform track by a `<link relation="p-p">`. The conventions are jmisb's
/// `KlvToCot` (the only ST 0805 implementation verified against): the point is
/// the target location when the set carries one complete (lat, lon, alt),
/// else the frame center; the SPI uid is the platform uid plus the sensor
/// name; `how` is `m-p`; ce / le are the unknown sentinel (ST 0601 error
/// estimates are not decoded, and jmisb writes the same default without them).
///
/// `None` when the set has no timestamp or neither point is complete.
pub fn cot_spi_event(ls: &UasDatalink, opts: CotOptions<'_>) -> Option<String> {
    let time_us = ls.timestamp_us?;
    let (lat, lon, alt) = match (ls.target_lat_deg, ls.target_lon_deg, ls.target_alt_m) {
        (Some(lat), Some(lon), Some(alt)) => (lat, lon, alt),
        _ => (
            ls.frame_center_lat_deg?,
            ls.frame_center_lon_deg?,
            ls.frame_center_alt_m?,
        ),
    };
    let stale_us = time_us.saturating_add((opts.stale_secs as u64).saturating_mul(1_000_000));
    let time = iso8601_utc_us(time_us);

    let platform_uid = platform_uid(ls, opts.uid);
    let sensor = ls.image_source_sensor.as_deref().unwrap_or("UNKNOWN");

    Some(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n\
         <event version=\"{COT_VERSION}\" uid=\"{uid}_{sensor}\" type=\"b-m-p-s-p-i\" \
         time=\"{time}\" start=\"{time}\" stale=\"{stale}\" how=\"m-p\">\
         <point lat=\"{lat:.6}\" lon=\"{lon:.6}\" hae=\"{alt:.1}\" ce=\"{UNKNOWN}\" le=\"{UNKNOWN}\"/>\
         <detail><link relation=\"p-p\" type=\"{cot_type}\" uid=\"{uid}\"/></detail></event>",
        uid = xml_escape(&platform_uid),
        sensor = xml_escape(sensor),
        cot_type = xml_escape(opts.cot_type),
        stale = iso8601_utc_us(stale_us),
    ))
}

/// The platform uid ST 0805 derives: platform designation and mission id
/// joined by `_` when the set has both, else the sink's configured uid.
fn platform_uid(ls: &UasDatalink, fallback: &str) -> String {
    match (&ls.platform_designation, &ls.mission_id) {
        (Some(platform), Some(mission)) => format!("{platform}_{mission}"),
        _ => String::from(fallback),
    }
}

/// The `<sensor>` cone: where the sensor points and how wide it sees, i.e. the
/// frame center expressed as a bearing and a distance from the platform.
/// Attributes and their units are the MITRE CoT Sensor Schema (ATAK-CIV
/// `takcot/mitre/CoT Sensor Schema (PUBLIC RELEASE).xsd`); ATAK's
/// `SensorDetailHandler` draws the cone once azimuth, fov and range are all
/// present, so a partial set is emitted but simply not drawn.
fn sensor_detail(ls: &UasDatalink) -> String {
    let mut attrs = String::new();
    // Schema azimuth is with respect to true north; ST 0601 tag 18 is relative
    // to the platform nose, so it only becomes a true bearing with tag 5.
    if let (Some(heading), Some(rel_az)) = (ls.heading_deg, ls.rel_azimuth_deg) {
        // `%` keeps the sign of the dividend and `f64::rem_euclid` is std-only,
        // so fold the negative case by hand to stay on the no_std baseline.
        let az = match (heading + rel_az) % 360.0 {
            r if r < 0.0 => r + 360.0,
            r => r,
        };
        attrs.push_str(&format!(" azimuth=\"{az:.1}\""));
    }
    if let Some(fov) = ls.hfov_deg {
        attrs.push_str(&format!(" fov=\"{fov:.1}\""));
    }
    if let Some(vfov) = ls.vfov_deg {
        attrs.push_str(&format!(" vfov=\"{vfov:.1}\""));
    }
    if let Some(range) = ls.slant_range_m {
        attrs.push_str(&format!(" range=\"{range:.1}\""));
    }
    // Tag 19 is elevation relative to the platform plane, the schema's is from
    // level: equal for level flight, off by the pitch angle otherwise.
    if let Some(el) = ls.rel_elevation_deg {
        attrs.push_str(&format!(" elevation=\"{el:.1}\""));
    }
    // Tag 20 spans 0..360, the schema's roll is (-180, 180].
    if let Some(roll) = ls.rel_roll_deg {
        let roll = if roll > 180.0 { roll - 360.0 } else { roll };
        attrs.push_str(&format!(" roll=\"{roll:.1}\""));
    }
    if attrs.is_empty() {
        return String::new();
    }
    format!("<sensor{attrs}/>")
}

/// The text ATAK shows in the marker's remarks pane: the frame center (which a
/// single event cannot carry as a second point), the mission, and any security
/// marking the set declares.
fn remarks(ls: &UasDatalink) -> String {
    let mut parts = String::new();
    let mut push = |s: String| {
        if !parts.is_empty() {
            parts.push(' ');
        }
        parts.push_str(&s);
    };
    if let (Some(lat), Some(lon)) = (ls.frame_center_lat_deg, ls.frame_center_lon_deg) {
        push(format!("frame-center={lat:.6},{lon:.6}"));
        if let Some(alt) = ls.frame_center_alt_m {
            push(format!("frame-center-alt={alt:.1}m"));
        }
    }
    if let Some(mission) = &ls.mission_id {
        push(format!("mission={}", xml_escape(mission)));
    }
    if let Some(sec) = &ls.security {
        if let Some(class) = sec.classification {
            push(format!("classification={}", xml_escape(class.label())));
        }
        if let Some(country) = &sec.classifying_country {
            push(format!("classifying-country={}", xml_escape(country)));
        }
    }
    parts
}

#[cfg(feature = "udp-egress")]
mod element {
    use super::{cot_event, cot_spi_event, CotOptions};

    use core::future::Future;
    use core::pin::Pin;

    use alloc::boxed::Box;
    use alloc::string::{String, ToString};
    use alloc::vec::Vec;

    use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

    use g2g_core::{
        AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError,
        OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
        PropertySpec,
    };

    use crate::filesink::io_err;
    use crate::klv::{split_klv_packets, UasDatalink};

    /// Which transport carries the events.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Protocol {
        Udp,
        Tcp,
    }

    /// Sink that turns each ST 0601 local set into a CoT event on the wire.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use g2g_plugins::cotsink::CotSink;
    ///
    /// let dest = "239.2.3.1:6969".parse().unwrap();
    /// let sink = CotSink::new(dest).with_uid("g2g-uav-1").with_stale_secs(30);
    /// ```
    #[derive(Debug)]
    pub struct CotSink {
        dest: SocketAddr,
        protocol: Protocol,
        uid: String,
        cot_type: String,
        stale_secs: u32,
        multicast_ttl: u32,
        verify_checksum: bool,
        spi: bool,
        // Bound synchronously in `configure_pipeline` (no runtime needed) and
        // wrapped into the tokio socket on first `process`, where a runtime
        // context is guaranteed (`UdpSocket::from_std` requires one). The TCP
        // stream connects lazily there for the same reason.
        std_socket: Option<StdUdpSocket>,
        socket: Option<tokio::net::UdpSocket>,
        stream: Option<tokio::net::TcpStream>,
        events_sent: u64,
        eos_seen: bool,
    }

    impl CotSink {
        pub fn new(dest: SocketAddr) -> Self {
            Self {
                dest,
                protocol: Protocol::Udp,
                uid: String::from(super::DEFAULT_UID),
                cot_type: String::from(super::DEFAULT_COT_TYPE),
                stale_secs: super::DEFAULT_STALE_SECS,
                multicast_ttl: 1,
                verify_checksum: true,
                spi: false,
                std_socket: None,
                socket: None,
                stream: None,
                events_sent: 0,
                eos_seen: false,
            }
        }

        /// Send over TCP to a TAK server instead of UDP.
        pub fn with_tcp(mut self) -> Self {
            self.protocol = Protocol::Tcp;
            self
        }

        /// The event `uid` (the track identity a client updates in place).
        pub fn with_uid(mut self, uid: &str) -> Self {
            self.uid = uid.to_string();
            self
        }

        /// The CoT type string for the track.
        pub fn with_cot_type(mut self, cot_type: &str) -> Self {
            self.cot_type = cot_type.to_string();
            self
        }

        /// Seconds after the event time at which a client drops the track.
        pub fn with_stale_secs(mut self, secs: u32) -> Self {
            self.stale_secs = secs;
            self
        }

        /// Hop limit for a multicast destination (1 = this link only).
        pub fn with_multicast_ttl(mut self, ttl: u32) -> Self {
            self.multicast_ttl = ttl;
            self
        }

        /// Tolerate a missing / wrong ST 0601 checksum (default requires it).
        pub fn with_verify_checksum(mut self, verify: bool) -> Self {
            self.verify_checksum = verify;
            self
        }

        /// Also emit an ST 0805.1 Sensor Point of Interest event per local set.
        pub fn with_spi(mut self, spi: bool) -> Self {
            self.spi = spi;
            self
        }

        /// CoT events written so far.
        pub fn events_sent(&self) -> u64 {
            self.events_sent
        }

        pub fn eos_seen(&self) -> bool {
            self.eos_seen
        }

        /// The events one KLV frame's packets produce, in order.
        fn events(&self, buf: &[u8]) -> Vec<String> {
            let opts = CotOptions {
                uid: &self.uid,
                cot_type: &self.cot_type,
                stale_secs: self.stale_secs,
            };
            split_klv_packets(buf)
                .into_iter()
                .flat_map(|pkt| {
                    let ls = if self.verify_checksum {
                        UasDatalink::parse(pkt)
                    } else {
                        UasDatalink::parse_lenient(pkt)
                    };
                    let mut events = Vec::new();
                    if let Some(ls) = ls {
                        events.extend(cot_event(&ls, opts));
                        if self.spi {
                            events.extend(cot_spi_event(&ls, opts));
                        }
                    }
                    events
                })
                .collect()
        }

        async fn send(&mut self, event: &str) -> Result<(), G2gError> {
            match self.protocol {
                Protocol::Udp => {
                    if self.socket.is_none() {
                        let std = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
                        self.socket = Some(tokio::net::UdpSocket::from_std(std).map_err(io_err)?);
                    }
                    let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                    socket.send(event.as_bytes()).await.map_err(io_err)?;
                }
                Protocol::Tcp => {
                    if self.stream.is_none() {
                        self.stream = Some(
                            tokio::net::TcpStream::connect(self.dest)
                                .await
                                .map_err(io_err)?,
                        );
                    }
                    let stream = self.stream.as_ref().ok_or(G2gError::NotConfigured)?;
                    // write_all without tokio's io-util: wait for writability,
                    // then drain the buffer through try_write.
                    let mut rest = event.as_bytes();
                    while !rest.is_empty() {
                        stream.writable().await.map_err(io_err)?;
                        match stream.try_write(rest) {
                            Ok(n) => rest = &rest[n..],
                            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                            Err(e) => return Err(io_err(e)),
                        }
                    }
                }
            }
            self.events_sent += 1;
            Ok(())
        }
    }

    impl AsyncElement for CotSink {
        type ProcessFuture<'a>
            = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
        where
            Self: 'a;

        /// Reads host memory, so it takes system frames only. The allocation
        /// cascade turns that into a download demand on a GPU producer.
        fn input_domains(&self) -> g2g_core::memory::DomainSet {
            g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
        }

        fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
            upstream_caps.intersect(&Caps::Klv)
        }

        fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
            CapsConstraint::Accepts(CapsSet::one(Caps::Klv))
        }

        fn configure_pipeline(
            &mut self,
            absolute_caps: &Caps,
        ) -> Result<ConfigureOutcome, G2gError> {
            if !matches!(absolute_caps, Caps::Klv) {
                return Err(G2gError::CapsMismatch);
            }
            if self.protocol == Protocol::Udp {
                let socket = StdUdpSocket::bind(("0.0.0.0", 0)).map_err(io_err)?;
                socket.set_nonblocking(true).map_err(io_err)?;
                if self.dest.ip().is_multicast() {
                    socket
                        .set_multicast_ttl_v4(self.multicast_ttl)
                        .map_err(io_err)?;
                }
                socket.connect(self.dest).map_err(io_err)?;
                self.std_socket = Some(socket);
            }
            Ok(ConfigureOutcome::Accepted)
        }

        fn metadata(&self) -> ElementMetadata {
            ElementMetadata::new(
                "Cursor-on-Target sink",
                "Sink/Network",
                "Sends KLV telemetry as CoT events to a TAK network",
                "g2g",
            )
        }

        fn properties(&self) -> &'static [PropertySpec] {
            const PROPS: &[PropertySpec] = &[
                PropertySpec::new("host", PropKind::Str, "destination host (IP to send to)")
                    .with_default(super::DEFAULT_HOST),
                PropertySpec::new("port", PropKind::Uint, "destination port")
                    .with_default("6969")
                    .with_range("0", "65535"),
                PropertySpec::new("protocol", PropKind::Str, "transport carrying the events")
                    .with_default("udp")
                    .with_enum_values("udp | tcp"),
                PropertySpec::new("uid", PropKind::Str, "CoT event uid (the track identity)")
                    .with_default(super::DEFAULT_UID),
                PropertySpec::new("cot-type", PropKind::Str, "CoT type string for the track")
                    .with_default(super::DEFAULT_COT_TYPE),
                PropertySpec::new(
                    "stale-seconds",
                    PropKind::Uint,
                    "seconds after the event time at which a client drops the track",
                )
                .with_default("10"),
                PropertySpec::new(
                    "ttl-mc",
                    PropKind::Uint,
                    "hop limit for a multicast destination (1 = this link only)",
                )
                .with_default("1")
                .with_range("1", "255"),
                PropertySpec::new(
                    "verify-checksum",
                    PropKind::Bool,
                    "drop a local set whose checksum is missing or wrong",
                )
                .with_default("true"),
                PropertySpec::new(
                    "spi",
                    PropKind::Bool,
                    "also emit an ST 0805.1 sensor point of interest event per local set",
                )
                .with_default("false"),
            ];
            PROPS
        }

        fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
            if let Some(r) = crate::netprop::set_addr_prop(&mut self.dest, "host", name, &value) {
                return r;
            }
            match name {
                "protocol" => {
                    self.protocol = match value.as_str().ok_or(PropError::Type)? {
                        "udp" => Protocol::Udp,
                        "tcp" => Protocol::Tcp,
                        _ => return Err(PropError::Value),
                    };
                    Ok(())
                }
                "uid" => {
                    self.uid = value.as_str().ok_or(PropError::Type)?.to_string();
                    Ok(())
                }
                "cot-type" => {
                    self.cot_type = value.as_str().ok_or(PropError::Type)?.to_string();
                    Ok(())
                }
                "stale-seconds" => {
                    let s = value.as_uint().ok_or(PropError::Type)?;
                    if s > u32::MAX as u64 {
                        return Err(PropError::Value);
                    }
                    self.stale_secs = s as u32;
                    Ok(())
                }
                "ttl-mc" => {
                    let t = value.as_uint().ok_or(PropError::Type)?;
                    if !(1..=255).contains(&t) {
                        return Err(PropError::Value);
                    }
                    self.multicast_ttl = t as u32;
                    Ok(())
                }
                "verify-checksum" => {
                    self.verify_checksum = value.as_bool().ok_or(PropError::Type)?;
                    Ok(())
                }
                "spi" => {
                    self.spi = value.as_bool().ok_or(PropError::Type)?;
                    Ok(())
                }
                _ => Err(PropError::Unknown),
            }
        }

        fn get_property(&self, name: &str) -> Option<PropValue> {
            if let Some(v) = crate::netprop::get_addr_prop(&self.dest, "host", name) {
                return Some(v);
            }
            match name {
                "protocol" => Some(PropValue::Str(
                    match self.protocol {
                        Protocol::Udp => "udp",
                        Protocol::Tcp => "tcp",
                    }
                    .into(),
                )),
                "uid" => Some(PropValue::Str(self.uid.clone())),
                "cot-type" => Some(PropValue::Str(self.cot_type.clone())),
                "stale-seconds" => Some(PropValue::Uint(self.stale_secs as u64)),
                "ttl-mc" => Some(PropValue::Uint(self.multicast_ttl as u64)),
                "verify-checksum" => Some(PropValue::Bool(self.verify_checksum)),
                "spi" => Some(PropValue::Bool(self.spi)),
                _ => None,
            }
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
                            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                        for event in self.events(slice) {
                            self.send(&event).await?;
                        }
                    }
                    PipelinePacket::Eos => self.eos_seen = true,
                    // future PipelinePacket variants: no-op (terminal sink).
                    _ => {}
                }
                Ok(())
            })
        }
    }

    impl PadTemplates for CotSink {
        fn pad_templates() -> Vec<PadTemplate> {
            Vec::from([PadTemplate::sink(CapsSet::one(Caps::Klv))])
        }
    }
}

#[cfg(feature = "udp-egress")]
pub use element::CotSink;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::klv::{SecurityClassification, SecurityLocalSet};
    use alloc::string::ToString;

    /// 2023-11-14T22:13:20Z plus 123456 us.
    const T: u64 = 1_700_000_000_123_456;

    fn minimal() -> UasDatalink {
        UasDatalink {
            timestamp_us: Some(T),
            sensor_lat_deg: Some(60.176822),
            sensor_lon_deg: Some(24.828835),
            ..Default::default()
        }
    }

    #[test]
    fn iso8601_is_the_w3c_profile_with_microseconds() {
        assert_eq!(iso8601_utc_us(T), "2023-11-14T22:13:20.123456Z");
        // Epoch and a leap-year date, to exercise the civil-date arithmetic.
        assert_eq!(iso8601_utc_us(0), "1970-01-01T00:00:00.000000Z");
        assert_eq!(
            iso8601_utc_us(1_709_164_800_000_000),
            "2024-02-29T00:00:00.000000Z"
        );
    }

    /// Position and timestamp are both mandatory: a set missing either produces
    /// no event, never one at lat/lon 0.
    #[test]
    fn incomplete_sets_produce_no_event() {
        let opts = CotOptions::default();
        assert!(cot_event(&minimal(), opts).is_some());
        for missing in [
            UasDatalink {
                sensor_lat_deg: None,
                ..minimal()
            },
            UasDatalink {
                sensor_lon_deg: None,
                ..minimal()
            },
            UasDatalink {
                timestamp_us: None,
                ..minimal()
            },
            UasDatalink::default(),
        ] {
            assert_eq!(cot_event(&missing, opts), None);
        }
    }

    /// The stale timestamp is the event time plus the configured interval.
    #[test]
    fn stale_trails_the_event_time() {
        let event = cot_event(
            &minimal(),
            CotOptions {
                stale_secs: 90,
                ..CotOptions::default()
            },
        )
        .expect("event");
        assert!(event.contains("time=\"2023-11-14T22:13:20.123456Z\""));
        assert!(event.contains("start=\"2023-11-14T22:13:20.123456Z\""));
        assert!(event.contains("stale=\"2023-11-14T22:14:50.123456Z\""));
    }

    /// The sensor cone is a true bearing (ST 0601 azimuth is relative to the
    /// nose) and the roll is folded into the schema's (-180, 180].
    #[test]
    fn sensor_cone_is_true_bearing_with_folded_roll() {
        let ls = UasDatalink {
            heading_deg: Some(350.0),
            rel_azimuth_deg: Some(30.0),
            rel_roll_deg: Some(358.2),
            hfov_deg: Some(54.9),
            slant_range_m: Some(68_591.0),
            ..minimal()
        };
        let event = cot_event(&ls, CotOptions::default()).expect("event");
        assert!(
            event.contains(
                "<sensor azimuth=\"20.0\" fov=\"54.9\" range=\"68591.0\" roll=\"-1.8\"/>"
            ),
            "{event}"
        );
        // Without a heading there is no true bearing, so azimuth is omitted.
        let no_heading = UasDatalink {
            heading_deg: None,
            ..ls
        };
        let event = cot_event(&no_heading, CotOptions::default()).expect("event");
        assert!(!event.contains("azimuth="), "{event}");
    }

    /// A set with no pointing data at all carries no `<sensor>` element.
    #[test]
    fn absent_pointing_data_omits_the_sensor_element() {
        let event = cot_event(&minimal(), CotOptions::default()).expect("event");
        assert!(!event.contains("<sensor"), "{event}");
    }

    /// Remarks carry the frame center (the look point a single event cannot
    /// express as a second `<point>`) and the security marking.
    #[test]
    fn remarks_carry_frame_center_and_classification() {
        let ls = UasDatalink {
            frame_center_lat_deg: Some(60.18),
            frame_center_lon_deg: Some(24.84),
            frame_center_alt_m: Some(12.0),
            security: Some(SecurityLocalSet {
                classification: Some(SecurityClassification::Secret),
                ..Default::default()
            }),
            ..minimal()
        };
        let event = cot_event(&ls, CotOptions::default()).expect("event");
        assert!(
            event.contains(
                "<remarks>frame-center=60.180000,24.840000 frame-center-alt=12.0m \
                 classification=SECRET</remarks>"
            ),
            "{event}"
        );
    }

    /// The callsign falls back through platform designation, mission id, uid.
    #[test]
    fn callsign_falls_back_to_mission_then_uid() {
        let mission = UasDatalink {
            mission_id: Some("Mission 12".to_string()),
            ..minimal()
        };
        assert!(cot_event(&mission, CotOptions::default())
            .expect("event")
            .contains("<contact callsign=\"Mission 12\"/>"));
        assert!(cot_event(&minimal(), CotOptions::default())
            .expect("event")
            .contains("<contact callsign=\"g2g-uas\"/>"));
    }

    #[test]
    fn escaping_covers_the_five_xml_metacharacters_and_drops_controls() {
        assert_eq!(
            xml_escape("a<b>c&d\"e'f"),
            "a&lt;b&gt;c&amp;d&quot;e&apos;f"
        );
        // A C0 control is not representable in XML 1.0 even as a reference.
        assert_eq!(xml_escape("a\u{0}b\u{1f}c"), "abc");
    }
}
