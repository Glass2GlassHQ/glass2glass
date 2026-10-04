use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::PropError;

const MEMORY_DOMAIN_NAMES: &[(&str, MemoryDomainKind)] = &[
    ("system", MemoryDomainKind::System),
    ("systemview", MemoryDomainKind::SystemView),
    ("dmabuf", MemoryDomainKind::DmaBuf),
    ("vulkantexture", MemoryDomainKind::VulkanTexture),
    ("webgpubuffer", MemoryDomainKind::WebGPUBuffer),
    ("cuda", MemoryDomainKind::Cuda),
    ("d3d11texture", MemoryDomainKind::D3D11Texture),
    ("cvpixelbuffer", MemoryDomainKind::CvPixelBuffer),
    (
        "webgpuexternaltexture",
        MemoryDomainKind::WebGPUExternalTexture,
    ),
    ("wgputexture", MemoryDomainKind::WgpuTexture),
    ("wgpubuffer", MemoryDomainKind::WgpuBuffer),
];

pub(crate) fn parse_domains(names: &str) -> Result<DomainSet, PropError> {
    names.split(',').try_fold(DomainSet::EMPTY, |set, name| {
        let (_, kind) = MEMORY_DOMAIN_NAMES
            .iter()
            .find(|(known, _)| *known == name.trim())
            .ok_or(PropError::Value)?;
        Ok(set.with(*kind))
    })
}

pub(crate) fn domain_names(set: DomainSet) -> String {
    let names: Vec<&str> = set
        .iter()
        .filter_map(|kind| {
            MEMORY_DOMAIN_NAMES
                .iter()
                .find(|(_, known)| *known == kind)
                .map(|(name, _)| *name)
        })
        .collect();
    names.join(",")
}
