# The other room participant in the m1784_livekit_data peer tests.
import argparse
import asyncio
import os
import sys

from livekit import api, rtc

SETTLE_SECONDS = 1.0
LINGER_SECONDS = 2.0
POLL_SECONDS = 0.05
FILTERED_PAYLOAD_LENGTH = 16


def emit(event, **fields):
    pairs = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"peer {event} {pairs}".rstrip(), flush=True)


def token(args):
    grants = api.VideoGrants(room_join=True, room=args.room)
    return (
        api.AccessToken(args.api_key, args.api_secret)
        .with_identity(args.identity)
        .with_grants(grants)
        .to_jwt()
    )


async def wait_for_participant(room, identity, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while identity not in {p.identity for p in room.remote_participants.values()}:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError(f"{identity} never joined")
        await asyncio.sleep(POLL_SECONDS)


async def publish(room, kind, topic, destinations, payload):
    await room.local_participant.publish_data(
        payload,
        reliable=kind == "reliable",
        topic=topic,
        destination_identities=destinations,
    )
    emit(
        "sent",
        kind=kind,
        topic=topic,
        destinations=",".join(destinations),
        payload=payload.hex(),
    )


async def send(room, args):
    await wait_for_participant(room, args.target, args.timeout)
    await asyncio.sleep(SETTLE_SECONDS)
    for kind, lengths in (("reliable", args.reliable_lengths), ("lossy", args.lossy_lengths)):
        for length in lengths:
            await publish(room, kind, args.topic, [args.target], os.urandom(length))
        other_topic = os.urandom(FILTERED_PAYLOAD_LENGTH)
        await publish(room, kind, args.topic + "-other", [], other_topic)
        elsewhere = os.urandom(FILTERED_PAYLOAD_LENGTH)
        await publish(room, kind, args.topic, [args.identity + "-nobody"], elsewhere)
    await asyncio.sleep(LINGER_SECONDS)


async def receive(room, args):
    received = asyncio.Queue()
    room.on("data_received", received.put_nowait)
    # the sdk reports packets from a participant it has not announced yet with no sender
    room.on("participant_connected", lambda p: emit("joined", identity=p.identity))
    emit("ready")
    for _ in range(args.count):
        try:
            packet = await asyncio.wait_for(received.get(), args.timeout)
        except asyncio.TimeoutError:
            break
        emit(
            "received",
            kind="reliable" if packet.kind == rtc.DataPacketKind.KIND_RELIABLE else "lossy",
            topic=packet.topic,
            sender=packet.participant.identity if packet.participant else "",
            payload=packet.data.hex(),
        )


def lengths(text):
    return [int(length) for length in text.split(",")]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key", default="devkey")
    parser.add_argument("--api-secret", default="secret")
    parser.add_argument("--room", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    modes = parser.add_subparsers(dest="mode", required=True)
    send_mode = modes.add_parser("send")
    send_mode.add_argument("--target", required=True)
    send_mode.add_argument("--topic", required=True)
    send_mode.add_argument("--reliable-lengths", type=lengths, required=True)
    send_mode.add_argument("--lossy-lengths", type=lengths, required=True)
    receive_mode = modes.add_parser("receive")
    receive_mode.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    room = rtc.Room()
    await room.connect(args.url, token(args))
    try:
        if args.mode == "send":
            await send(room, args)
        else:
            await receive(room, args)
    finally:
        await room.disconnect()
    emit("done")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
