use g2g_plugins::rtppay::RtpH264Packetizer;
use g2g_plugins::srtp::{SrtpAuthentication, SrtpCipher, SrtpPolicy, SrtpReceiver, SrtpSender};

const SYNCHRONIZATION_SOURCE: u32 = 0x1020_3040;
const MASTER_KEY: [u8; 16] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
];
const MASTER_SALT: [u8; 12] = *b"Quid pro quo";
const POLICY: SrtpPolicy = SrtpPolicy {
    cipher: SrtpCipher::Aes128Gcm,
    authentication: SrtpAuthentication::Null,
};

fn main() {
    let mut packetizer = RtpH264Packetizer::new(96, SYNCHRONIZATION_SOURCE).with_max_payload(8);
    let mut sender = SrtpSender::new(POLICY, &MASTER_KEY, &MASTER_SALT, SYNCHRONIZATION_SOURCE)
        .expect("valid sender key material");
    let mut receiver = SrtpReceiver::new(POLICY, &MASTER_KEY, &MASTER_SALT, SYNCHRONIZATION_SOURCE)
        .expect("valid receiver key material");

    let access_unit = [
        0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x21, 0xa0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60,
        0x70,
    ];
    let rtp_packets = packetizer.packetize(&access_unit, 90_000);

    for (packet_number, rtp_packet) in rtp_packets.iter().enumerate() {
        let protected_packet = sender.protect_rtp(rtp_packet).expect("protect RTP");
        let recovered_packet = receiver
            .unprotect_rtp(&protected_packet)
            .expect("authenticate and decrypt SRTP");
        assert_eq!(&recovered_packet, rtp_packet);
        println!(
            "packet {packet_number}: RTP {} bytes, SRTP {} bytes, recovered byte-exact",
            rtp_packet.len(),
            protected_packet.len()
        );
    }
}
