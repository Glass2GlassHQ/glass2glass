// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::sync::Arc;

use ring::signature::{
    UnparsedPublicKey, VerificationAlgorithm, RSA_PKCS1_2048_8192_SHA256,
    RSA_PKCS1_2048_8192_SHA384, RSA_PKCS1_2048_8192_SHA512, RSA_PSS_2048_8192_SHA256,
    RSA_PSS_2048_8192_SHA384, RSA_PSS_2048_8192_SHA512,
};
use rustls::client::danger::HandshakeSignatureValid;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{
    CertificateError, DigitallySignedStruct, DistinguishedName, ServerConfig, SignatureScheme,
};

use super::identity::HostIdentity;
use super::GamestreamError;

const SCHEMES: [(SignatureScheme, &dyn VerificationAlgorithm); 6] = [
    (
        SignatureScheme::RSA_PKCS1_SHA256,
        &RSA_PKCS1_2048_8192_SHA256,
    ),
    (
        SignatureScheme::RSA_PKCS1_SHA384,
        &RSA_PKCS1_2048_8192_SHA384,
    ),
    (
        SignatureScheme::RSA_PKCS1_SHA512,
        &RSA_PKCS1_2048_8192_SHA512,
    ),
    (SignatureScheme::RSA_PSS_SHA256, &RSA_PSS_2048_8192_SHA256),
    (SignatureScheme::RSA_PSS_SHA384, &RSA_PSS_2048_8192_SHA384),
    (SignatureScheme::RSA_PSS_SHA512, &RSA_PSS_2048_8192_SHA512),
];

// moonlight's self-signed client certificates are not valid web PKI certificates, so pairing decides trust
#[derive(Debug)]
struct AnyClientCertificate;

fn bad_certificate() -> rustls::Error {
    rustls::Error::InvalidCertificate(CertificateError::BadEncoding)
}

fn verify_handshake(
    message: &[u8],
    certificate: &CertificateDer<'_>,
    signed: &DigitallySignedStruct,
) -> Result<HandshakeSignatureValid, rustls::Error> {
    let (_, algorithm) = SCHEMES
        .iter()
        .find(|(scheme, _)| *scheme == signed.scheme)
        .ok_or(rustls::Error::PeerIncompatible(
            rustls::PeerIncompatible::NoSignatureSchemesInCommon,
        ))?;
    let (_, parsed) =
        x509_parser::parse_x509_certificate(certificate).map_err(|_| bad_certificate())?;
    UnparsedPublicKey::new(*algorithm, &parsed.public_key().subject_public_key.data)
        .verify(message, signed.signature())
        .map_err(|_| rustls::Error::InvalidCertificate(CertificateError::BadSignature))?;
    Ok(HandshakeSignatureValid::assertion())
}

impl ClientCertVerifier for AnyClientCertificate {
    fn offer_client_auth(&self) -> bool {
        true
    }

    fn client_auth_mandatory(&self) -> bool {
        false
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, rustls::Error> {
        x509_parser::parse_x509_certificate(end_entity).map_err(|_| bad_certificate())?;
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signed: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_handshake(message, certificate, signed)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        certificate: &CertificateDer<'_>,
        signed: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_handshake(message, certificate, signed)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        SCHEMES.iter().map(|(scheme, _)| *scheme).collect()
    }
}

pub(crate) fn server_config(host: &HostIdentity) -> Result<Arc<ServerConfig>, GamestreamError> {
    let config =
        ServerConfig::builder_with_provider(Arc::new(rustls::crypto::ring::default_provider()))
            .with_safe_default_protocol_versions()?
            .with_client_cert_verifier(Arc::new(AnyClientCertificate))
            .with_single_cert(
                vec![host.certificate_der.clone()],
                PrivateKeyDer::Pkcs8(host.private_key()),
            )?;
    Ok(Arc::new(config))
}
