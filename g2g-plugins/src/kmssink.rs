// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;
use std::ffi::CStr;
use std::os::fd::BorrowedFd;
use std::path::{Path, PathBuf};

use alloc::boxed::Box;
use alloc::vec::Vec;

use drm::buffer::{Buffer, DrmFourcc, DrmModifier, Handle as BufferHandle, PlanarBuffer};
use drm::control::atomic::AtomicModeReq;
use drm::control::dumbbuffer::DumbBuffer;
use drm::control::{
    connector, crtc, framebuffer, plane, property, AtomicCommitFlags, Device as ControlDevice,
    Event, FbCmd2Flags, Mode, ResourceHandle,
};
use drm::{ClientCapability, Device as _};

use g2g_core::element::{PresentationStats, QosMessage};
use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf, DRM_FORMAT_MOD_LINEAR,
};
use g2g_core::meta::{Orientation, OrientationSet, PlaneLayout};
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ClockSync, ConfigureOutcome, ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate,
    PadTemplates, PipelineClock, PipelinePacket, PixelAspect, PresentationPacer, PropError,
    PropKind, PropValue, PropertySpec, RawVideoFormat, MAX_LATENESS_PROPERTY,
    PROCESSING_DEADLINE_PROPERTY, QOS_INTERVAL_PROPERTY, SYNC_PROPERTY,
};

use crate::clock::wait_to_present;
use crate::display_aspect::{display_size, letterbox, ViewportRectangle};
use crate::dmabufimport::DmaBufImportCache;
use crate::dmabufmap::DmaBufReadMap;
use crate::drm_plane_rotation::{require_unrotated, PlaneRotation};
use crate::drmcard::{Card, DEFAULT_CARD};
use crate::paddedrows::{dmabuf_plane_layout, single_stride_layout};
use crate::pixel::plane_shapes;
use crate::videoconvert::display::{display_sink_caps, DisplayInput};
use crate::videoflip::frame_orientation;
#[cfg(feature = "metadata")]
use crate::{
    drmcard::{draw_composition, same_composition},
    overlaycomposition::{composition_layer, RENDER_OVERLAYS_NAME, RENDER_OVERLAYS_PROPERTY},
    paint::blend_layer_into_nv12,
    yuvmatrix::YuvRgbMatrix,
};

const PICTURE_FORMAT: DrmFourcc = DrmFourcc::Nv12;
const BACKGROUND_FORMAT: DrmFourcc = DrmFourcc::Xrgb8888;
#[cfg(feature = "metadata")]
const COMPOSITION_FORMAT: DrmFourcc = DrmFourcc::Argb8888;

fn hardware<E>(_: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

#[derive(Debug)]
struct Slot {
    db: DumbBuffer,
    fb: framebuffer::Handle,
}

#[derive(Debug)]
struct Nv12Framebuffer {
    width: u32,
    height: u32,
    handles: [Option<BufferHandle>; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: Option<DrmModifier>,
}

impl PlanarBuffer for Nv12Framebuffer {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    fn format(&self) -> DrmFourcc {
        PICTURE_FORMAT
    }
    fn modifier(&self) -> Option<DrmModifier> {
        self.modifier
    }
    fn pitches(&self) -> [u32; 4] {
        self.pitches
    }
    fn handles(&self) -> [Option<BufferHandle>; 4] {
        self.handles
    }
    fn offsets(&self) -> [u32; 4] {
        self.offsets
    }
}

const POOL_SIZE: usize = 2;

fn nv12_slots(card: &Card, (width, height): (u32, u32)) -> Result<Vec<Slot>, G2gError> {
    (0..POOL_SIZE)
        .map(|_| {
            // drm dumb buffers take a bits-per-pixel, not a pixel format
            let db = card
                .create_dumb_buffer((width, height + height / 2), DrmFourcc::C8, 8)
                .map_err(hardware)?;
            let pitch = db.pitch();
            let planar = Nv12Framebuffer {
                width,
                height,
                handles: [Some(db.handle()), Some(db.handle()), None, None],
                pitches: [pitch, pitch, 0, 0],
                offsets: [0, pitch * height, 0, 0],
                modifier: None,
            };
            let fb = card
                .add_planar_framebuffer(&planar, FbCmd2Flags::empty())
                .map_err(hardware)?;
            Ok(Slot { db, fb })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaneKind {
    Primary,
    Overlay,
    Cursor,
}

// drm draws planes with a higher zpos in front
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zpos {
    Fixed(u64),
    Settable { lowest: u64, highest: u64 },
}

impl Zpos {
    fn range(self) -> (u64, u64) {
        match self {
            Zpos::Fixed(zpos) => (zpos, zpos),
            Zpos::Settable { lowest, highest } => (lowest, highest),
        }
    }
}

#[derive(Debug)]
struct PlaneCapabilities {
    kind: PlaneKind,
    formats: Vec<u32>,
    zpos: Option<Zpos>,
}

impl PlaneCapabilities {
    fn lists(&self, format: DrmFourcc) -> bool {
        self.formats.contains(&(format as u32))
    }
}

#[derive(Debug, Clone, Copy)]
struct PlaneProperties {
    fb_id: property::Handle,
    crtc_id: property::Handle,
    src_x: property::Handle,
    src_y: property::Handle,
    src_w: property::Handle,
    src_h: property::Handle,
    crtc_x: property::Handle,
    crtc_y: property::Handle,
    crtc_w: property::Handle,
    crtc_h: property::Handle,
}

#[derive(Debug)]
struct Plane {
    handle: plane::Handle,
    capabilities: PlaneCapabilities,
    properties: PlaneProperties,
    zpos_property: Option<property::Handle>,
    rotation: Option<PlaneRotation>,
}

#[derive(Debug)]
struct ObjectProperties(Vec<(property::Info, u64)>);

impl ObjectProperties {
    fn read<T: ResourceHandle>(card: &Card, object: T) -> Result<Self, G2gError> {
        let set = card.get_properties(object).map_err(hardware)?;
        set.iter()
            .map(|(&handle, &value)| Ok((card.get_property(handle).map_err(hardware)?, value)))
            .collect::<Result<_, G2gError>>()
            .map(Self)
    }

    fn find(&self, name: &CStr) -> Option<&(property::Info, u64)> {
        self.0.iter().find(|(info, _)| info.name() == name)
    }

    fn handle(&self, name: &CStr) -> Result<property::Handle, G2gError> {
        self.find(name)
            .map(|(info, _)| info.handle())
            .ok_or(G2gError::Hardware(HardwareError::Other))
    }

    fn enum_name(&self, name: &CStr) -> Option<&CStr> {
        let (info, value) = self.find(name)?;
        info.value_type()
            .convert_value(*value)
            .as_enum()
            .map(property::EnumValue::name)
    }
}

fn read_plane(card: &Card, handle: plane::Handle, formats: &[u32]) -> Result<Plane, G2gError> {
    let properties = ObjectProperties::read(card, handle)?;
    let kind = match properties.enum_name(c"type").map(CStr::to_bytes) {
        Some(b"Primary") => PlaneKind::Primary,
        Some(b"Overlay") => PlaneKind::Overlay,
        Some(b"Cursor") => PlaneKind::Cursor,
        _ => return Err(G2gError::Hardware(HardwareError::Other)),
    };
    let zpos = properties.find(c"zpos");
    let zpos_range = |(info, value): &(property::Info, u64)| match info.value_type() {
        property::ValueType::UnsignedRange(lowest, highest) if info.mutable() => Zpos::Settable {
            lowest: *lowest,
            highest: *highest,
        },
        _ => Zpos::Fixed(*value),
    };
    let rotation = match properties.find(c"rotation") {
        Some((info, _)) => Some(PlaneRotation::new(card, info.handle())?),
        None => None,
    };
    Ok(Plane {
        handle,
        capabilities: PlaneCapabilities {
            kind,
            formats: formats.to_vec(),
            zpos: zpos.map(zpos_range),
        },
        properties: PlaneProperties {
            fb_id: properties.handle(c"FB_ID")?,
            crtc_id: properties.handle(c"CRTC_ID")?,
            src_x: properties.handle(c"SRC_X")?,
            src_y: properties.handle(c"SRC_Y")?,
            src_w: properties.handle(c"SRC_W")?,
            src_h: properties.handle(c"SRC_H")?,
            crtc_x: properties.handle(c"CRTC_X")?,
            crtc_y: properties.handle(c"CRTC_Y")?,
            crtc_w: properties.handle(c"CRTC_W")?,
            crtc_h: properties.handle(c"CRTC_H")?,
        },
        zpos_property: zpos
            .filter(|(info, _)| info.mutable())
            .map(|(info, _)| info.handle()),
        rotation,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlaneChoice {
    picture: usize,
    background: Option<usize>,
    composition: Option<usize>,
    zpos_to_set: Vec<(usize, u64)>,
}

fn stack(planes: &[&PlaneCapabilities], order: &[usize]) -> Option<Vec<(usize, u64)>> {
    let mut floor = 0;
    let mut settable = Vec::new();
    for &index in order {
        let Some(zpos) = planes[index].zpos else {
            continue;
        };
        let (lowest, highest) = zpos.range();
        let value = lowest.max(floor);
        if value > highest {
            return None;
        }
        floor = value + 1;
        if matches!(zpos, Zpos::Settable { .. }) {
            settable.push((index, value));
        }
    }
    Some(settable)
}

fn choose_planes(
    planes: &[&PlaneCapabilities],
    with_composition: bool,
) -> Result<PlaneChoice, &'static str> {
    let of_kind = |kind| (0..planes.len()).filter(move |&index| planes[index].kind == kind);
    let picture = of_kind(PlaneKind::Overlay)
        .chain(of_kind(PlaneKind::Primary))
        .find(|&index| planes[index].lists(PICTURE_FORMAT))
        .ok_or("no plane on the CRTC takes NV12")?;
    let primary = of_kind(PlaneKind::Primary)
        .next()
        .ok_or("the CRTC has no primary plane")?;
    let background = (primary != picture).then_some(primary);
    if background.is_some() && !planes[primary].lists(BACKGROUND_FORMAT) {
        return Err("the primary plane does not take XRGB8888");
    }
    let under_composition: Vec<usize> = background.into_iter().chain([picture]).collect();
    let without_composition = PlaneChoice {
        picture,
        background,
        composition: None,
        zpos_to_set: stack(planes, &under_composition)
            .ok_or("the picture cannot go above the primary plane")?,
    };
    if !with_composition || planes[picture].zpos.is_none() {
        return Ok(without_composition);
    }
    let above_picture = of_kind(PlaneKind::Overlay)
        .filter(|&index| {
            index != picture
                && planes[index].lists(DrmFourcc::Argb8888)
                && planes[index].zpos.is_some()
        })
        .find_map(|index| {
            let order: Vec<usize> = under_composition.iter().copied().chain([index]).collect();
            Some((index, stack(planes, &order)?))
        });
    let Some((composition, zpos)) = above_picture else {
        return Ok(without_composition);
    };
    Ok(PlaneChoice {
        picture,
        background,
        composition: Some(composition),
        zpos_to_set: zpos,
    })
}

fn centred(
    (mode_width, mode_height): (u32, u32),
    (width, height): (u32, u32),
) -> Option<ViewportRectangle> {
    Some(ViewportRectangle {
        x: mode_width.checked_sub(width)? / 2,
        y: mode_height.checked_sub(height)? / 2,
        width,
        height,
    })
}

fn rotation_value(
    rotation: Option<&PlaneRotation>,
    orientation: Orientation,
) -> Result<Option<u64>, G2gError> {
    match rotation {
        Some(rotation) => rotation.value(orientation).map(Some),
        None => require_unrotated(orientation).map(|()| None),
    }
}

fn show_on_plane(
    request: &mut AtomicModeReq,
    plane: &Plane,
    crtc: crtc::Handle,
    framebuffer: framebuffer::Handle,
    (width, height): (u32, u32),
    on_screen: ViewportRectangle,
    rotation: Option<u64>,
) {
    let properties = &plane.properties;
    // drm takes the source rectangle in 16.16 fixed point
    let fixed = |value: u32| property::Value::UnsignedRange(u64::from(value) << 16);
    for (property, value) in [
        (
            properties.fb_id,
            property::Value::Framebuffer(Some(framebuffer)),
        ),
        (properties.crtc_id, property::Value::CRTC(Some(crtc))),
        (properties.src_x, fixed(0)),
        (properties.src_y, fixed(0)),
        (properties.src_w, fixed(width)),
        (properties.src_h, fixed(height)),
        (
            properties.crtc_x,
            property::Value::SignedRange(i64::from(on_screen.x)),
        ),
        (
            properties.crtc_y,
            property::Value::SignedRange(i64::from(on_screen.y)),
        ),
        (
            properties.crtc_w,
            property::Value::UnsignedRange(u64::from(on_screen.width)),
        ),
        (
            properties.crtc_h,
            property::Value::UnsignedRange(u64::from(on_screen.height)),
        ),
    ] {
        request.add_property(plane.handle, property, value);
    }
    if let (Some(plane_rotation), Some(value)) = (&plane.rotation, rotation) {
        request.add_property(
            plane.handle,
            plane_rotation.property(),
            property::Value::Bitmask(value),
        );
    }
}

#[cfg(feature = "metadata")]
fn hide_plane(request: &mut AtomicModeReq, plane: &Plane) {
    request.add_property(
        plane.handle,
        plane.properties.fb_id,
        property::Value::Framebuffer(None),
    );
    request.add_property(
        plane.handle,
        plane.properties.crtc_id,
        property::Value::CRTC(None),
    );
}

#[derive(Debug)]
struct Output {
    connector: connector::Handle,
    connector_crtc_id: property::Handle,
    crtc: crtc::Handle,
    crtc_mode_id: property::Handle,
    crtc_active: property::Handle,
    mode: Mode,
    planes: Vec<Plane>,
}

impl Output {
    fn mode_size(&self) -> (u32, u32) {
        let (width, height) = self.mode.size();
        (u32::from(width), u32::from(height))
    }
}

fn open_output(card: &Card) -> Result<Output, G2gError> {
    let resources = card.resource_handles().map_err(hardware)?;
    let connector = resources
        .connectors()
        .iter()
        .find_map(|&handle| {
            let info = card.get_connector(handle, true).ok()?;
            (info.state() == connector::State::Connected).then_some(info)
        })
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
    let mode = *connector
        .modes()
        .first()
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
    let crtc = connector
        .encoders()
        .iter()
        .find_map(|&encoder| {
            let encoder = card.get_encoder(encoder).ok()?;
            resources
                .filter_crtcs(encoder.possible_crtcs())
                .first()
                .copied()
        })
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
    let mut planes = Vec::new();
    for handle in card.plane_handles().map_err(hardware)? {
        let info = card.get_plane(handle).map_err(hardware)?;
        if resources
            .filter_crtcs(info.possible_crtcs())
            .contains(&crtc)
        {
            planes.push(read_plane(card, handle, info.formats())?);
        }
    }
    let crtc_properties = ObjectProperties::read(card, crtc)?;
    Ok(Output {
        connector: connector.handle(),
        connector_crtc_id: ObjectProperties::read(card, connector.handle())?.handle(c"CRTC_ID")?,
        crtc,
        crtc_mode_id: crtc_properties.handle(c"MODE_ID")?,
        crtc_active: crtc_properties.handle(c"ACTIVE")?,
        mode,
        planes,
    })
}

fn black_framebuffer(card: &Card, size: (u32, u32)) -> Result<framebuffer::Handle, G2gError> {
    let mut dumb = card
        .create_dumb_buffer(size, BACKGROUND_FORMAT, 32)
        .map_err(hardware)?;
    card.map_dumb_buffer(&mut dumb).map_err(hardware)?.fill(0);
    card.add_framebuffer(&dumb, 24, 32).map_err(hardware)
}

#[cfg(feature = "metadata")]
#[derive(Debug)]
struct CompositionBuffer {
    dumb: DumbBuffer,
    framebuffer: framebuffer::Handle,
}

#[cfg(feature = "metadata")]
#[derive(Debug)]
struct CompositionBuffers {
    buffers: Vec<CompositionBuffer>,
    front: usize,
    shown: Option<g2g_core::OverlayCompositionMeta>,
    layer: Vec<u8>,
    row: Vec<u8>,
}

#[cfg(feature = "metadata")]
impl CompositionBuffers {
    fn new(card: &Card, (width, height): (u32, u32)) -> Result<Self, G2gError> {
        let buffers = (0..POOL_SIZE)
            .map(|_| {
                let dumb = card
                    .create_dumb_buffer((width, height), COMPOSITION_FORMAT, 32)
                    .map_err(hardware)?;
                let framebuffer = card.add_framebuffer(&dumb, 32, 32).map_err(hardware)?;
                Ok(CompositionBuffer { dumb, framebuffer })
            })
            .collect::<Result<_, G2gError>>()?;
        Ok(Self {
            buffers,
            front: 0,
            shown: None,
            layer: Vec::new(),
            row: Vec::new(),
        })
    }

    fn update(
        &mut self,
        card: &Card,
        composition: Option<g2g_core::OverlayCompositionMeta>,
        size: (u32, u32),
    ) -> Result<(), G2gError> {
        let unchanged = match (&composition, &self.shown) {
            (None, None) => true,
            (Some(next), Some(shown)) => same_composition(next, shown),
            _ => false,
        };
        if unchanged {
            return Ok(());
        }
        let Some(composition) = composition else {
            self.shown = None;
            return Ok(());
        };
        let back = (self.front + 1) % self.buffers.len();
        draw_composition(
            card,
            &mut self.buffers[back].dumb,
            &composition,
            size,
            &mut self.layer,
            &mut self.row,
        )?;
        self.front = back;
        self.shown = Some(composition);
        Ok(())
    }

    fn framebuffer(&self) -> Option<framebuffer::Handle> {
        self.shown
            .as_ref()
            .map(|_| self.buffers[self.front].framebuffer)
    }
}

#[derive(Debug)]
enum Picture<'a> {
    System(&'a [u8]),
    DmaBuf(OwnedDmaBuf),
}

#[cfg(feature = "metadata")]
#[derive(Debug)]
enum CompositionScanout {
    Plane(CompositionBuffers),
    BlendedIntoPicture { layer: Vec<u8> },
}

#[derive(Debug)]
struct Scanout {
    // drm turns off the planes showing a closing file's framebuffers
    card: Card,
    output: Output,
    planes: PlaneChoice,
    mode_blob: property::Value<'static>,
    background: Option<framebuffer::Handle>,
    input: DisplayInput,
    scales: bool,
    slots: Vec<Slot>,
    front: usize,
    imports: DmaBufImportCache<framebuffer::Handle>,
    displayed_dmabuf: Option<OwnedDmaBuf>,
    queued_dmabuf: Option<OwnedDmaBuf>,
    #[cfg(feature = "metadata")]
    composition: Option<CompositionScanout>,
    modeset_done: bool,
    flip_pending: bool,
}

fn slot_layout(slot: &Slot, (width, height): (u32, u32)) -> Result<PlaneLayout, G2gError> {
    single_stride_layout(RawVideoFormat::Nv12, width, height, 0, slot.db.pitch())
        .map(|(layout, _)| layout)
        .ok_or(G2gError::CapsMismatch)
}

impl Scanout {
    fn open(path: &Path, input: DisplayInput, with_composition: bool) -> Result<Self, G2gError> {
        let target = Target::category(short_type_name::<KmsSink>());
        let card = Card::open(path)?;
        card.set_client_capability(ClientCapability::UniversalPlanes, true)
            .map_err(hardware)?;
        if card
            .set_client_capability(ClientCapability::Atomic, true)
            .is_err()
        {
            g2g_core::g2g_error!(target, "{} has no atomic modesetting", path.display());
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let output = open_output(&card)?;
        let capabilities: Vec<_> = output
            .planes
            .iter()
            .map(|plane| &plane.capabilities)
            .collect();
        let planes = choose_planes(&capabilities, with_composition).map_err(|reason| {
            g2g_core::g2g_error!(target, "{}: {reason}", path.display());
            G2gError::Hardware(HardwareError::Other)
        })?;
        let mode_blob = card.create_property_blob(&output.mode).map_err(hardware)?;
        let background = match planes.background {
            Some(_) => Some(black_framebuffer(&card, output.mode_size())?),
            None => None,
        };
        #[cfg(feature = "metadata")]
        let composition = match (with_composition, planes.composition) {
            (false, _) => None,
            (true, Some(_)) => Some(CompositionScanout::Plane(CompositionBuffers::new(
                &card,
                (input.width, input.height),
            )?)),
            (true, None) => Some(CompositionScanout::BlendedIntoPicture { layer: Vec::new() }),
        };
        Ok(Self {
            card,
            output,
            planes,
            mode_blob,
            background,
            input,
            scales: false,
            slots: Vec::new(),
            front: 0,
            imports: DmaBufImportCache::new(),
            displayed_dmabuf: None,
            queued_dmabuf: None,
            #[cfg(feature = "metadata")]
            composition,
            modeset_done: false,
            flip_pending: false,
        })
    }

    fn picture_rotation(&self) -> Option<&PlaneRotation> {
        self.output.planes[self.planes.picture].rotation.as_ref()
    }

    fn fill_slot(
        &mut self,
        nv12: &[u8],
        layout: &PlaneLayout,
    ) -> Result<framebuffer::Handle, G2gError> {
        let size = (self.input.width, self.input.height);
        if self.slots.is_empty() {
            self.slots = nv12_slots(&self.card, size)?;
        }
        let back = (self.front + 1) % self.slots.len();
        let slot = &mut self.slots[back];
        let slot_layout = slot_layout(slot, size)?;
        let mut mapping = self.card.map_dumb_buffer(&mut slot.db).map_err(hardware)?;
        copy_nv12(nv12, layout, mapping.as_mut(), &slot_layout, size)?;
        self.front = back;
        Ok(slot.fb)
    }

    fn import(&mut self, dmabuf: &OwnedDmaBuf) -> Result<framebuffer::Handle, G2gError> {
        let (width, height) = (self.input.width, self.input.height);
        let (layout, size) = dmabuf_plane_layout(dmabuf, RawVideoFormat::Nv12, width, height)
            .ok_or(G2gError::CapsMismatch)?;
        let card = &self.card;
        self.imports.get_or_import(
            dmabuf,
            size,
            || {
                // SAFETY: the fd is a live dma-buf owned by `dmabuf` for this call.
                let fd = unsafe { BorrowedFd::borrow_raw(dmabuf.as_raw()) };
                let handle = card.prime_fd_to_buffer(fd).map_err(hardware)?;
                let modifier = (dmabuf.modifier() != DRM_FORMAT_MOD_LINEAR)
                    .then(|| DrmModifier::from(dmabuf.modifier()));
                let mut framebuffer = Nv12Framebuffer {
                    width,
                    height,
                    handles: [None; 4],
                    pitches: [0; 4],
                    offsets: [0; 4],
                    modifier,
                };
                for index in 0..layout.count() {
                    let plane = layout.plane(index).ok_or(G2gError::CapsMismatch)?;
                    let to_u32 =
                        |value: usize| u32::try_from(value).map_err(|_| G2gError::CapsMismatch);
                    *framebuffer
                        .handles
                        .get_mut(index)
                        .ok_or(G2gError::CapsMismatch)? = Some(handle);
                    framebuffer.pitches[index] = to_u32(plane.stride)?;
                    framebuffer.offsets[index] = to_u32(plane.offset)?;
                }
                let flags = match modifier {
                    Some(_) => FbCmd2Flags::MODIFIERS,
                    None => FbCmd2Flags::empty(),
                };
                let added = card.add_planar_framebuffer(&framebuffer, flags);
                // drm keeps the buffer alive while a framebuffer uses it
                card.close_buffer(handle).map_err(hardware)?;
                added.map_err(hardware)
            },
            |framebuffer| card.destroy_framebuffer(framebuffer).map_err(hardware),
        )
    }

    fn copy_into_slot(&mut self, picture: Picture<'_>) -> Result<framebuffer::Handle, G2gError> {
        let (width, height) = (self.input.width, self.input.height);
        match picture {
            Picture::System(nv12) => {
                let tight = PlaneLayout::tight(RawVideoFormat::Nv12, width, height)
                    .ok_or(G2gError::CapsMismatch)?;
                self.fill_slot(nv12, &tight)
            }
            Picture::DmaBuf(dmabuf) => {
                let (layout, _) = dmabuf_plane_layout(&dmabuf, RawVideoFormat::Nv12, width, height)
                    .ok_or(G2gError::CapsMismatch)?;
                let mapped = DmaBufReadMap::read(&dmabuf)?;
                self.fill_slot(mapped.from_fd_start(), &layout)
            }
        }
    }

    #[cfg(feature = "metadata")]
    fn blend_into_front_slot(
        &mut self,
        composition: &g2g_core::OverlayCompositionMeta,
    ) -> Result<(), G2gError> {
        let Some(CompositionScanout::BlendedIntoPicture { layer }) = &mut self.composition else {
            return Ok(());
        };
        let size = (self.input.width, self.input.height);
        let drawn = composition_layer(composition, size, layer)?;
        let slot = &mut self.slots[self.front];
        let slot_layout = slot_layout(slot, size)?;
        let mut mapping = self.card.map_dumb_buffer(&mut slot.db).map_err(hardware)?;
        blend_layer_into_nv12(
            drawn,
            mapping.as_mut(),
            &slot_layout,
            (size.0 as usize, size.1 as usize),
            &YuvRgbMatrix::new(self.input.colorimetry),
        )
    }

    fn present(
        &mut self,
        picture: Picture<'_>,
        orientation: Orientation,
        #[cfg(feature = "metadata")] composition: Option<g2g_core::OverlayCompositionMeta>,
    ) -> Result<(), G2gError> {
        if self.flip_pending {
            self.wait_for_flip()?;
        }
        #[cfg(feature = "metadata")]
        let blended = match (&mut self.composition, composition) {
            (Some(CompositionScanout::Plane(buffers)), composition) => {
                buffers.update(
                    &self.card,
                    composition,
                    (self.input.width, self.input.height),
                )?;
                None
            }
            (Some(CompositionScanout::BlendedIntoPicture { .. }), composition) => composition,
            (None, _) => None,
        };
        #[cfg(feature = "metadata")]
        let blends = blended.is_some();
        #[cfg(not(feature = "metadata"))]
        let blends = false;
        let (framebuffer, dmabuf) = match picture {
            Picture::DmaBuf(dmabuf) if !blends => (self.import(&dmabuf)?, Some(dmabuf)),
            picture => (self.copy_into_slot(picture)?, None),
        };
        #[cfg(feature = "metadata")]
        if let Some(composition) = &blended {
            self.blend_into_front_slot(composition)?;
        }
        if !self.modeset_done {
            let scaled = self.request(framebuffer, orientation, self.scaled(orientation))?;
            let test = AtomicCommitFlags::TEST_ONLY | AtomicCommitFlags::ALLOW_MODESET;
            self.scales = self.card.atomic_commit(test, scaled).is_ok();
        }
        let on_screen = if self.scales {
            self.scaled(orientation)
        } else {
            self.centred(orientation)?
        };
        let request = self.request(framebuffer, orientation, on_screen)?;
        let flags = if self.modeset_done {
            AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK
        } else {
            AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::ALLOW_MODESET
        };
        self.card.atomic_commit(flags, request).map_err(hardware)?;
        self.modeset_done = true;
        self.flip_pending = true;
        self.queued_dmabuf = dmabuf;
        Ok(())
    }

    fn shown_size(&self, orientation: Orientation) -> (u32, u32) {
        let input = &self.input;
        let (width, height) = display_size(input.width, input.height, input.pixel_aspect);
        if orientation.swaps_dims() {
            (height, width)
        } else {
            (width, height)
        }
    }

    fn scaled(&self, orientation: Orientation) -> ViewportRectangle {
        let (width, height) = self.shown_size(orientation);
        let (mode_width, mode_height) = self.output.mode_size();
        letterbox(width, height, PixelAspect::SQUARE, mode_width, mode_height)
    }

    fn centred(&self, orientation: Orientation) -> Result<ViewportRectangle, G2gError> {
        let (width, height) = self.shown_size(orientation);
        let mode_size = self.output.mode_size();
        centred(mode_size, (width, height)).ok_or_else(|| {
            g2g_core::g2g_error!(
                Target::category(short_type_name::<KmsSink>()),
                "the plane cannot scale and the {width}x{height} picture is larger than the {}x{} display mode",
                mode_size.0,
                mode_size.1
            );
            G2gError::CapsMismatch
        })
    }

    fn request(
        &self,
        framebuffer: framebuffer::Handle,
        orientation: Orientation,
        on_screen: ViewportRectangle,
    ) -> Result<AtomicModeReq, G2gError> {
        let output = &self.output;
        let crtc = output.crtc;
        let mut request = AtomicModeReq::new();
        if !self.modeset_done {
            request.add_property(
                output.connector,
                output.connector_crtc_id,
                property::Value::CRTC(Some(crtc)),
            );
            request.add_property(crtc, output.crtc_mode_id, self.mode_blob);
            request.add_property(crtc, output.crtc_active, property::Value::Boolean(true));
            for &(index, zpos) in &self.planes.zpos_to_set {
                let plane = &output.planes[index];
                if let Some(property) = plane.zpos_property {
                    request.add_property(
                        plane.handle,
                        property,
                        property::Value::UnsignedRange(zpos),
                    );
                }
            }
            if let (Some(index), Some(black)) = (self.planes.background, self.background) {
                let mode_size = output.mode_size();
                let whole_screen = centred(mode_size, mode_size).ok_or(G2gError::CapsMismatch)?;
                show_on_plane(
                    &mut request,
                    &output.planes[index],
                    crtc,
                    black,
                    mode_size,
                    whole_screen,
                    None,
                );
            }
        }
        let size = (self.input.width, self.input.height);
        let picture_plane = &output.planes[self.planes.picture];
        show_on_plane(
            &mut request,
            picture_plane,
            crtc,
            framebuffer,
            size,
            on_screen,
            rotation_value(picture_plane.rotation.as_ref(), orientation)?,
        );
        #[cfg(feature = "metadata")]
        if let (Some(index), Some(CompositionScanout::Plane(buffers))) =
            (self.planes.composition, &self.composition)
        {
            let plane = &output.planes[index];
            match buffers.framebuffer() {
                Some(composition) => show_on_plane(
                    &mut request,
                    plane,
                    crtc,
                    composition,
                    size,
                    on_screen,
                    rotation_value(plane.rotation.as_ref(), orientation)?,
                ),
                None => hide_plane(&mut request, plane),
            }
        }
        Ok(request)
    }

    fn wait_for_flip(&mut self) -> Result<(), G2gError> {
        let mut empty_reads = 0u32;
        loop {
            let events = self.card.receive_events().map_err(hardware)?;
            let mut saw_flip = false;
            let mut saw_any = false;
            for ev in events {
                saw_any = true;
                if matches!(ev, Event::PageFlip(_)) {
                    saw_flip = true;
                }
            }
            if saw_flip {
                self.flip_pending = false;
                self.displayed_dmabuf = self.queued_dmabuf.take();
                return Ok(());
            }
            // blocking drm fds return empty reads only after the device is gone
            if saw_any {
                empty_reads = 0;
            } else {
                empty_reads += 1;
                if empty_reads >= 8 {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            }
        }
    }
}

pub struct KmsSink {
    device_path: PathBuf,
    #[cfg(feature = "metadata")]
    render_overlays: bool,
    scanout: Option<Scanout>,
    input: Option<DisplayInput>,
    frames_presented: u64,
    latency: LatencyHistogram,
    pacer: PresentationPacer,
}

impl core::fmt::Debug for KmsSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KmsSink")
            .field("device_path", &self.device_path)
            .field("open", &self.scanout.is_some())
            .field("frames_presented", &self.frames_presented)
            .finish()
    }
}

impl Default for KmsSink {
    fn default() -> Self {
        Self::new()
    }
}

pub(crate) fn default_card_present() -> Result<(), alloc::string::String> {
    if Path::new(DEFAULT_CARD).exists() {
        return Ok(());
    }
    Err(alloc::format!("{DEFAULT_CARD} does not exist"))
}

impl KmsSink {
    pub fn new() -> Self {
        Self {
            device_path: PathBuf::from(DEFAULT_CARD),
            #[cfg(feature = "metadata")]
            render_overlays: false,
            scanout: None,
            input: None,
            frames_presented: 0,
            latency: LatencyHistogram::new(),
            pacer: PresentationPacer::for_video_sink(),
        }
    }

    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }

    pub fn with_device<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.device_path = path.into();
        self
    }

    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented
    }
}

#[derive(Debug)]
struct KmsClock;
impl PipelineClock for KmsClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

const NATIVE_FORMATS: &[RawVideoFormat] = &[RawVideoFormat::Nv12];

fn accepted_caps() -> CapsSet {
    display_sink_caps(NATIVE_FORMATS)
}

fn shown_orientations(rotation: Option<&PlaneRotation>) -> OrientationSet {
    match rotation {
        Some(rotation) if cfg!(feature = "metadata") => rotation.shown_orientations(),
        _ => OrientationSet::EMPTY,
    }
}

impl PadTemplates for KmsSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(accepted_caps())])
    }
}

impl AsyncElement for KmsSink {
    fn latency(&self) -> g2g_core::LatencyReport {
        self.pacer.latency()
    }

    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
    }

    fn absorbed_orientations(&self) -> OrientationSet {
        // the runner asks after configure_pipeline opens the device
        shown_orientations(self.scanout.as_ref().and_then(Scanout::picture_rotation))
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(KmsClock),
        ))
    }

    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        Some(PresentationStats {
            presented: self.frames_presented(),
            dropped: 0,
            late_dropped: self.late_dropped(),
        })
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(accepted_caps())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.pacer.adopt_framerate(absolute_caps);
        let input = DisplayInput::from_caps(absolute_caps, NATIVE_FORMATS)?;
        let (w, h) = (input.width, input.height);
        if w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }

        // RtspSrc's first caps carry placeholder dimensions until the SPS arrives
        self.input = Some(input);
        let same_input = self
            .scanout
            .as_ref()
            .is_some_and(|scanout| scanout.input == input);
        if same_input {
            return Ok(ConfigureOutcome::Accepted);
        }
        self.scanout = None;
        #[cfg(feature = "metadata")]
        let with_composition = self.render_overlays;
        #[cfg(not(feature = "metadata"))]
        let with_composition = false;
        self.scanout = Some(Scanout::open(&self.device_path, input, with_composition)?);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "KMS / DRM video sink",
            "Sink/Video",
            "Presents video via DRM / KMS",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "device",
                PropKind::Str,
                "DRM device node (e.g. /dev/dri/card0)",
            )
            .with_default(DEFAULT_CARD),
            MAX_LATENESS_PROPERTY,
            PROCESSING_DEADLINE_PROPERTY,
            QOS_INTERVAL_PROPERTY,
            SYNC_PROPERTY,
            #[cfg(feature = "metadata")]
            RENDER_OVERLAYS_PROPERTY,
        ];
        PROPS
    }

    #[cfg(feature = "metadata")]
    fn meta_requests(&self) -> g2g_core::meta::MetaRequests {
        crate::overlaycomposition::render_requests(self.render_overlays)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => {
                self.device_path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            #[cfg(feature = "metadata")]
            RENDER_OVERLAYS_NAME => {
                self.render_overlays = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => self
                .pacer
                .set_property(name, &value)
                .unwrap_or(Err(PropError::Unknown)),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "device" => Some(PropValue::Str(
                self.device_path.to_string_lossy().into_owned(),
            )),
            #[cfg(feature = "metadata")]
            RENDER_OVERLAYS_NAME => Some(PropValue::Bool(self.render_overlays)),
            _ => self.pacer.get_property(name),
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::CapsChanged(caps) = &packet {
                self.pacer.adopt_framerate(caps);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let orientation = frame_orientation(&frame);
                    #[cfg(feature = "metadata")]
                    let composition = frame
                        .meta
                        .get::<g2g_core::OverlayCompositionMeta>()
                        .filter(|_| self.render_overlays)
                        .cloned();
                    let Frame { domain, timing, .. } = frame;
                    if !matches!(domain, MemoryDomain::DmaBuf(_)) {
                        domain.require_system_slice(short_type_name::<Self>())?;
                    }
                    let paced = self.pacer.judge(&timing, self.frames_presented);
                    if !wait_to_present(paced).await {
                        return Ok(());
                    }
                    let input = self.input.ok_or(G2gError::NotConfigured)?;
                    if matches!(domain, MemoryDomain::DmaBuf(_))
                        && input.negotiated != RawVideoFormat::Nv12
                    {
                        g2g_core::g2g_error!(
                            Target::category(short_type_name::<Self>()),
                            "{} dma-bufs cannot be scanned out, only NV12",
                            input.negotiated.gst_name()
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                    let scanout = self.scanout.as_mut().ok_or(G2gError::NotConfigured)?;
                    let converted;
                    let picture = match domain {
                        MemoryDomain::DmaBuf(dmabuf) => Picture::DmaBuf(dmabuf),
                        MemoryDomain::System(ref pixels) => {
                            converted = input.convert_frame(pixels.as_slice())?;
                            Picture::System(converted.as_deref().unwrap_or(pixels.as_slice()))
                        }
                        _ => return Err(G2gError::UnsupportedDomain),
                    };
                    scanout.present(
                        picture,
                        orientation,
                        #[cfg(feature = "metadata")]
                        composition,
                    )?;
                    self.frames_presented += 1;
                    if timing.arrival_ns != 0 {
                        let now = monotonic_ns();
                        if now >= timing.arrival_ns {
                            self.latency.record(now - timing.arrival_ns);
                        }
                    }
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) => Ok(()),
                PipelinePacket::Segment(seg) => {
                    self.pacer.set_segment(seg);
                    Ok(())
                }
                PipelinePacket::Flush => {
                    self.pacer.flush();
                    Ok(())
                }
                PipelinePacket::Eos => {
                    if let Some(scanout) =
                        self.scanout.as_mut().filter(|scanout| scanout.flip_pending)
                    {
                        scanout.wait_for_flip()?;
                    }
                    Ok(())
                }
                _ => Ok(()),
            }
        })
    }
}

fn copy_nv12(
    src: &[u8],
    src_layout: &PlaneLayout,
    dst: &mut [u8],
    dst_layout: &PlaneLayout,
    (width, height): (u32, u32),
) -> Result<(), G2gError> {
    let shapes = plane_shapes(RawVideoFormat::Nv12, width as usize, height as usize)
        .ok_or(G2gError::CapsMismatch)?;
    for (plane, &(row_bytes, rows)) in shapes.iter().enumerate() {
        for row in 0..rows {
            let source = src_layout
                .row_range(plane, row, row_bytes)
                .and_then(|range| src.get(range))
                .ok_or(G2gError::CapsMismatch)?;
            dst_layout
                .row_range(plane, row, row_bytes)
                .and_then(|range| dst.get_mut(range))
                .ok_or(G2gError::Hardware(HardwareError::Other))?
                .copy_from_slice(source);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use alloc::vec;
    use g2g_core::{Dim, Rate, VideoCodec};

    #[test]
    fn declares_system_memory_and_dmabuf_only() {
        // a wrong declaration makes `nvdec ! kmssink` fail with UnsupportedDomain
        assert_eq!(
            KmsSink::new().input_domains(),
            DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
        );
    }

    #[test]
    fn intercept_passes_through_any_format() {
        let sink = KmsSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn intercept_passes_through_nv12() {
        let sink = KmsSink::new();
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        assert_eq!(sink.intercept_caps(&nv12), Ok(nv12));
    }

    #[test]
    fn caps_constraint_takes_nv12_first_and_converted_formats() {
        let sink = KmsSink::new();
        let CapsConstraint::Accepts(set) = sink.caps_constraint_as_sink() else {
            panic!("expected Accepts");
        };
        let alternatives = set.alternatives();
        assert_eq!(
            alternatives[0],
            crate::videofx::any_geometry(RawVideoFormat::Nv12)
        );
        assert!(alternatives.contains(&crate::videofx::any_geometry(RawVideoFormat::I420)));
    }

    #[test]
    fn configure_rejects_undecoded_video() {
        let mut sink = KmsSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(
            sink.configure_pipeline(&h264).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(
            sink.scanout.is_none(),
            "no DRM device opened on rejected caps"
        );
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = KmsSink::new();
        let odd = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(641),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        // ConfigureOutcome has no PartialEq
        match sink.configure_pipeline(&odd) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    fn pitched(pitch: u32) -> PlaneLayout {
        single_stride_layout(RawVideoFormat::Nv12, 4, 2, 0, pitch)
            .unwrap()
            .0
    }

    #[test]
    fn copy_nv12_packs_planes_at_pitch() {
        // 4x2 NV12 into rows 8 bytes apart
        let src: Vec<u8> = (0..12).collect();
        let mut dst = vec![0u8; 8 * 3];
        let tight = PlaneLayout::tight(RawVideoFormat::Nv12, 4, 2).unwrap();
        copy_nv12(&src, &tight, &mut dst, &pitched(8), (4, 2)).unwrap();
        assert_eq!(&dst[0..4], &[0, 1, 2, 3]);
        assert_eq!(&dst[8..12], &[4, 5, 6, 7]);
        assert_eq!(&dst[16..20], &[8, 9, 10, 11]);
    }

    #[test]
    fn copy_nv12_reads_a_padded_source_at_its_pitch() {
        let src: Vec<u8> = (0..18).collect();
        let mut dst = vec![0u8; 12];
        let tight = PlaneLayout::tight(RawVideoFormat::Nv12, 4, 2).unwrap();
        copy_nv12(&src, &pitched(6), &mut dst, &tight, (4, 2)).unwrap();
        assert_eq!(dst, [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]);
    }

    #[test]
    fn copy_nv12_rejects_truncated_src() {
        let src = vec![0u8; 10]; // Need 12 for 4x2 NV12.
        let mut dst = vec![0u8; 24];
        let tight = PlaneLayout::tight(RawVideoFormat::Nv12, 4, 2).unwrap();
        assert!(copy_nv12(&src, &tight, &mut dst, &pitched(8), (4, 2)).is_err());
    }

    #[derive(Debug)]
    struct ManualClock(alloc::sync::Arc<core::sync::atomic::AtomicU64>);
    impl PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(core::sync::atomic::Ordering::Relaxed)
        }
    }

    struct NullOut;
    impl OutputSink for NullOut {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    #[tokio::test]
    async fn qos_drops_a_late_frame_before_any_drm_io() {
        // a PTS 0 frame is 1 ms late on this clock
        use g2g_core::clock::PlayAnchor;
        use g2g_core::memory::{MemoryDomain, SystemSlice};
        let (bus, handle) = g2g_core::Bus::new(4);
        let clock = alloc::sync::Arc::new(core::sync::atomic::AtomicU64::new(1_000_000));
        let anchor = PlayAnchor::new();
        anchor.stamp(0);
        let mut sink = KmsSink::new().with_max_lateness_ns(0).with_bus(handle);
        AsyncElement::set_clock_sync(
            &mut sink,
            g2g_core::ClockSync::with_play_anchor(
                alloc::sync::Arc::new(ManualClock(clock)),
                0,
                anchor,
            ),
        );

        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 6]))),
            g2g_core::FrameTiming::default(),
            0,
        );
        sink.process(PipelinePacket::DataFrame(frame), &mut NullOut)
            .await
            .expect("a dropped frame is not an error");

        assert_eq!(sink.late_dropped(), 1, "the late frame was dropped");
        assert_eq!(sink.frames_presented(), 0, "nothing was scanned out");
        assert_eq!(
            AsyncElement::presentation_stats(&sink),
            Some(PresentationStats {
                presented: 0,
                dropped: 0,
                late_dropped: 1,
            })
        );
        let upstream = AsyncElement::take_qos(&mut sink).expect("upstream QoS report");
        assert_eq!(upstream.jitter_ns, 1_000_000);
        match bus.try_recv() {
            Some(g2g_core::BusMessage::Qos { dropped, .. }) => assert_eq!(dropped, 1),
            other => panic!("expected a Qos message, got {other:?}"),
        }
    }

    #[test]
    fn absorbs_the_orientations_the_plane_shows() {
        use crate::drm_plane_rotation::tests::{
            half_turns_and_axis_mirrors, plane_with_every_value, plane_without_quarter_turns,
        };
        let when_metadata_on = |set| {
            if cfg!(feature = "metadata") {
                set
            } else {
                OrientationSet::EMPTY
            }
        };
        assert_eq!(
            KmsSink::new().absorbed_orientations(),
            OrientationSet::EMPTY,
            "no plane before the device opens"
        );
        assert_eq!(
            shown_orientations(Some(&plane_without_quarter_turns())),
            when_metadata_on(half_turns_and_axis_mirrors())
        );
        assert_eq!(
            shown_orientations(Some(&plane_with_every_value())),
            when_metadata_on(OrientationSet::ALL)
        );
    }

    #[test]
    fn plane_without_rotation_shows_only_the_stored_picture() {
        assert_eq!(rotation_value(None, Orientation::Identity), Ok(None));
        assert_eq!(
            rotation_value(None, Orientation::Rotate90Cw),
            Err(G2gError::CapsMismatch)
        );
        let plane = crate::drm_plane_rotation::tests::plane_with_every_value();
        assert_ne!(
            rotation_value(Some(&plane), Orientation::Rotate180),
            rotation_value(Some(&plane), Orientation::Identity)
        );
    }

    fn plane(kind: PlaneKind, formats: &[DrmFourcc], zpos: Option<Zpos>) -> PlaneCapabilities {
        PlaneCapabilities {
            kind,
            formats: formats.iter().map(|&format| format as u32).collect(),
            zpos,
        }
    }

    const LTDC_ZPOS: Option<Zpos> = Some(Zpos::Settable {
        lowest: 0,
        highest: 2,
    });

    fn stm32mp257_planes() -> Vec<PlaneCapabilities> {
        // the STM32MP257's LTDC planes, in the order the kernel lists them
        vec![
            plane(
                PlaneKind::Primary,
                &[DrmFourcc::Xrgb8888, DrmFourcc::Argb8888],
                LTDC_ZPOS,
            ),
            plane(
                PlaneKind::Overlay,
                &[DrmFourcc::Nv12, DrmFourcc::Argb8888],
                LTDC_ZPOS,
            ),
            plane(
                PlaneKind::Overlay,
                &[DrmFourcc::Xrgb8888, DrmFourcc::Argb8888],
                LTDC_ZPOS,
            ),
        ]
    }

    #[test]
    fn picture_takes_the_nv12_overlay_with_the_composition_above_it() {
        let planes = stm32mp257_planes();
        let planes: Vec<_> = planes.iter().collect();
        assert_eq!(
            choose_planes(&planes, true),
            Ok(PlaneChoice {
                picture: 1,
                background: Some(0),
                composition: Some(2),
                zpos_to_set: vec![(0, 0), (1, 1), (2, 2)],
            })
        );
        assert_eq!(
            choose_planes(&planes, false),
            Ok(PlaneChoice {
                picture: 1,
                background: Some(0),
                composition: None,
                zpos_to_set: vec![(0, 0), (1, 1)],
            })
        );
    }

    #[test]
    fn picture_takes_the_primary_plane_when_no_overlay_takes_nv12() {
        let planes = [
            plane(PlaneKind::Cursor, &[DrmFourcc::Nv12], Some(Zpos::Fixed(3))),
            plane(
                PlaneKind::Overlay,
                &[DrmFourcc::Argb8888],
                Some(Zpos::Fixed(1)),
            ),
            plane(
                PlaneKind::Primary,
                &[DrmFourcc::Xrgb8888, DrmFourcc::Nv12],
                Some(Zpos::Fixed(0)),
            ),
        ];
        let planes: Vec<_> = planes.iter().collect();
        assert_eq!(
            choose_planes(&planes, true),
            Ok(PlaneChoice {
                picture: 2,
                background: None,
                composition: Some(1),
                zpos_to_set: Vec::new(),
            })
        );
    }

    #[test]
    fn composition_with_no_plane_above_the_picture_goes_on_no_plane() {
        let mut planes = stm32mp257_planes();
        planes[1].zpos = Some(Zpos::Fixed(1));
        planes[2].zpos = Some(Zpos::Fixed(1));
        let planes: Vec<_> = planes.iter().collect();
        let choice = choose_planes(&planes, true).unwrap();
        assert_eq!(choice.composition, None);
        assert_eq!(choice, choose_planes(&planes, false).unwrap());
    }

    #[test]
    fn picture_sits_in_the_middle_of_the_mode() {
        assert_eq!(
            centred((1024, 600), (640, 480)),
            Some(ViewportRectangle {
                x: 192,
                y: 60,
                width: 640,
                height: 480,
            })
        );
        assert_eq!(centred((1024, 600), (640, 640)), None);
    }

    fn fmt_unused(_: &dyn core::fmt::Debug) {}

    #[test]
    fn debug_impl_does_not_touch_card() {
        let sink = KmsSink::new();
        fmt_unused(&sink);
    }
}
