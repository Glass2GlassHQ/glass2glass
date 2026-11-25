// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rcgen::{CertificateParams, DnType, KeyPair, PKCS_RSA_SHA256};
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{RsaKeyPair, RSA_PKCS1_SHA256};
use rsa::pkcs8::EncodePrivateKey;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};
use sha2::{Digest, Sha256};

use super::GamestreamError;

const KEY_FILE: &str = "host-key.der";
const CERTIFICATE_FILE: &str = "host-certificate.pem";
const UNIQUE_ID_FILE: &str = "unique-id";
const PAIRED_DIRECTORY: &str = "paired-clients";
const PAIRED_EXTENSION: &str = "der";
// moonlight verifies the host's pairing signature as SHA256withRSA
const KEY_BITS: usize = 2048;
const UNIQUE_ID_BYTES: usize = 16;

#[derive(Debug)]
pub(crate) struct HostIdentity {
    pub(crate) unique_id: String,
    pub(crate) certificate_pem: String,
    pub(crate) certificate_der: CertificateDer<'static>,
    pub(crate) certificate_signature: Vec<u8>,
    key_der: Vec<u8>,
    signing_key: RsaKeyPair,
}

impl HostIdentity {
    pub(crate) fn load_or_create(directory: &Path) -> Result<Self, GamestreamError> {
        fs::create_dir_all(directory)?;
        if !directory.join(KEY_FILE).exists() {
            create(directory)?;
        }
        Self::from_parts(
            fs::read(directory.join(KEY_FILE))?,
            fs::read_to_string(directory.join(CERTIFICATE_FILE))?,
            fs::read_to_string(directory.join(UNIQUE_ID_FILE))?
                .trim()
                .to_string(),
        )
    }

    pub(crate) fn from_parts(
        key_der: Vec<u8>,
        certificate_pem: String,
        unique_id: String,
    ) -> Result<Self, GamestreamError> {
        let certificate_der = CertificateDer::from_pem_slice(certificate_pem.as_bytes())
            .map_err(|_| GamestreamError::BadHostCertificate)?;
        let certificate_signature =
            certificate_signature(&certificate_der).ok_or(GamestreamError::BadHostCertificate)?;
        let signing_key =
            RsaKeyPair::from_pkcs8(&key_der).map_err(|_| GamestreamError::BadHostKey)?;
        Ok(Self {
            unique_id,
            certificate_pem,
            certificate_der,
            certificate_signature,
            key_der,
            signing_key,
        })
    }

    pub(crate) fn private_key(&self) -> PrivatePkcs8KeyDer<'static> {
        PrivatePkcs8KeyDer::from(self.key_der.clone())
    }

    pub(crate) fn sign(&self, data: &[u8]) -> Result<Vec<u8>, GamestreamError> {
        let mut signature = vec![0; self.signing_key.public().modulus_len()];
        self.signing_key
            .sign(
                &RSA_PKCS1_SHA256,
                &SystemRandom::new(),
                data,
                &mut signature,
            )
            .map_err(|_| GamestreamError::Signing)?;
        Ok(signature)
    }
}

fn create(directory: &Path) -> Result<(), GamestreamError> {
    let key = rsa::PrivateKey::new(&mut rand::rng(), KEY_BITS)
        .map_err(|_| GamestreamError::KeyGeneration)?;
    let key_der = key
        .to_pkcs8_der()
        .map_err(|_| GamestreamError::KeyGeneration)?
        .as_bytes()
        .to_vec();
    let key_pair = KeyPair::from_pkcs8_der_and_sign_algo(
        &PrivatePkcs8KeyDer::from(key_der.as_slice()),
        &PKCS_RSA_SHA256,
    )?;
    let mut params = CertificateParams::default();
    params
        .distinguished_name
        .push(DnType::CommonName, "g2g-desktop");
    let certificate = params.self_signed(&key_pair)?;
    let mut unique_id = [0u8; UNIQUE_ID_BYTES];
    SystemRandom::new()
        .fill(&mut unique_id)
        .map_err(|_| GamestreamError::KeyGeneration)?;
    write_private(&directory.join(KEY_FILE), &key_der)?;
    fs::write(directory.join(CERTIFICATE_FILE), certificate.pem())?;
    fs::write(directory.join(UNIQUE_ID_FILE), hex::encode(unique_id))?;
    Ok(())
}

fn write_private(path: &Path, bytes: &[u8]) -> Result<(), GamestreamError> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path)?.write_all(bytes)?;
    Ok(())
}

// moonlight's pairing hashes cover the X.509 signature bytes of each side's certificate
pub(crate) fn certificate_signature(certificate_der: &[u8]) -> Option<Vec<u8>> {
    let (_, certificate) = x509_parser::parse_x509_certificate(certificate_der).ok()?;
    Some(certificate.signature_value.data.to_vec())
}

#[derive(Debug)]
pub(crate) struct PairedClients {
    directory: PathBuf,
    fingerprints: Mutex<BTreeSet<[u8; 32]>>,
}

impl PairedClients {
    pub(crate) fn load(directory: &Path) -> Result<Self, GamestreamError> {
        let directory = directory.join(PAIRED_DIRECTORY);
        fs::create_dir_all(&directory)?;
        let mut fingerprints = BTreeSet::new();
        for entry in fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.extension().is_some_and(|ext| ext == PAIRED_EXTENSION) {
                fingerprints.insert(fingerprint(&fs::read(path)?));
            }
        }
        Ok(Self {
            directory,
            fingerprints: Mutex::new(fingerprints),
        })
    }

    pub(crate) fn contains(&self, certificate_der: &[u8]) -> bool {
        self.fingerprints
            .lock()
            .unwrap()
            .contains(&fingerprint(certificate_der))
    }

    pub(crate) fn add(&self, certificate_der: &[u8]) -> Result<(), GamestreamError> {
        let fingerprint = fingerprint(certificate_der);
        let name = format!("{}.{PAIRED_EXTENSION}", hex::encode(fingerprint));
        fs::write(self.directory.join(name), certificate_der)?;
        self.fingerprints.lock().unwrap().insert(fingerprint);
        Ok(())
    }
}

fn fingerprint(certificate_der: &[u8]) -> [u8; 32] {
    Sha256::digest(certificate_der).into()
}
