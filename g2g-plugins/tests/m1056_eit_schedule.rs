//! M1056 - DVB EIT schedule tables, and the start time and duration of an event.
//!
//! An EIT event entry carries a 5-byte `start_time` (a 16-bit Modified Julian
//! Date then three BCD bytes of UTC hh:mm:ss, EN 300 468 Annex C) and a 3-byte
//! BCD `duration`, and the schedule tables (`table_id` 0x50..=0x5F) announce days
//! of events rather than the two present/following holds. The reference for the
//! date arithmetic is the Annex C worked example, whose Unix value is pinned
//! against `date -u`.
//!
//! No external tool: sections are hand-built in `eit_common` with a real MPEG-2
//! CRC-32 (computed there, so a broken CRC in the parser cannot pass by agreeing
//! with itself).
#![cfg(feature = "std")]

use g2g_core::{AsyncElement, Bus, BusMessage, ByteStreamEncoding, Caps, Tag};
use g2g_plugins::mpegts::{
    EitSlot, MAX_EIT_SCHEDULE_EVENTS, TAG_KEY_EVENT_DURATION, TAG_KEY_EVENT_NAME,
    TAG_KEY_EVENT_START, TAG_KEY_NEXT_EVENT_DURATION, TAG_KEY_NEXT_EVENT_START,
    TAG_KEY_SCHEDULE_EVENT_DURATION, TAG_KEY_SCHEDULE_EVENT_ID, TAG_KEY_SCHEDULE_EVENT_NAME,
    TAG_KEY_SCHEDULE_EVENT_START, TAG_KEY_SCHEDULE_EVENT_TEXT,
};
use g2g_plugins::tsdemux::{TsDemux, TsStream};

mod eit_common;
use eit_common::{
    data_frame, eit_event, eit_section, feed_sections, parse_sections, psi_packets,
    short_event_descriptor, CaptureSink, ANNEX_C_START_TIME, ANNEX_C_START_UNIX_SECS,
    EVENT_DURATION, EVENT_DURATION_SECS, PID_EIT, TABLE_ID_EIT_PF, TABLE_ID_EIT_SCHEDULE,
    TABLE_ID_EIT_SCHEDULE_OTHER_TS, UNDEFINED_START_TIME,
};

/// The service every fixture here describes, and the version its first section
/// carries.
const SERVICE_ID: u16 = 9;
const VERSION: u8 = 5;
/// A schedule section number away from 0 and 1, which is what tells a schedule
/// section apart from present/following: the schedule tables are segmented, so
/// any section number is valid there.
const SCHEDULE_SECTION: u8 = 3;
const FIRST_EVENT_ID: u16 = 0x2A01;
const SECOND_EVENT_ID: u16 = 0x2A02;

/// Every `Tag::Number` a run posted, as `(key, value)`.
fn posted_numbers(messages: &[BusMessage]) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    for message in messages {
        if let BusMessage::Tag { tags, .. } = message {
            for tag in tags.tags() {
                if let Tag::Number { key, value } = tag {
                    out.push((key.clone(), *value));
                }
            }
        }
    }
    out
}

/// Every `Tag::Other` a run posted, as `(key, value)`.
fn posted_text(messages: &[BusMessage]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for message in messages {
        if let BusMessage::Tag { tags, .. } = message {
            for tag in tags.tags() {
                if let Tag::Other { key, value } = tag {
                    out.push((key.clone(), value.clone()));
                }
            }
        }
    }
    out
}

/// Feed whole sections to a `TsDemux` element and collect what it posted.
async fn post_sections(sections: &[Vec<u8>]) -> Vec<BusMessage> {
    let (bus, handle) = Bus::new(64);
    let mut demux = TsDemux::new().with_stream(TsStream::Av1).with_bus(handle);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("tsdemux accepts an MPEG-TS byte stream");
    let mut ts = Vec::new();
    for section in sections {
        ts.extend_from_slice(&psi_packets(PID_EIT, section));
    }
    let mut sink = CaptureSink::default();
    demux
        .process(data_frame(&ts), &mut sink)
        .await
        .expect("the demuxer reads the sections");
    let mut posted = Vec::new();
    while let Some(message) = bus.try_recv() {
        posted.push(message);
    }
    posted
}

/// The Annex C start time and a 90-minute duration come off a present/following
/// event, and the element posts both under their own keys.
#[tokio::test]
async fn present_following_carries_the_start_time_and_duration() {
    let present = eit_section(
        TABLE_ID_EIT_PF,
        SERVICE_ID,
        VERSION,
        0,
        &eit_event(
            FIRST_EVENT_ID,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"News at Ten", b"The headlines"),
        ),
    );
    let following = eit_section(
        TABLE_ID_EIT_PF,
        SERVICE_ID,
        VERSION,
        1,
        &eit_event(
            SECOND_EVENT_ID,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Late Film", b"A thriller"),
        ),
    );
    let demux = parse_sections(&[present.clone(), following.clone()]);
    let event = demux
        .eit_events()
        .iter()
        .find(|e| e.slot == EitSlot::Present)
        .expect("present event");
    assert_eq!(event.start_unix_secs, Some(ANNEX_C_START_UNIX_SECS));
    assert_eq!(event.duration_secs, EVENT_DURATION_SECS);

    let numbers = posted_numbers(&post_sections(&[present, following]).await);
    let has = |key: &str, value: u64| numbers.iter().any(|(k, v)| k == key && *v == value);
    assert!(
        has(TAG_KEY_EVENT_START, ANNEX_C_START_UNIX_SECS),
        "posted: {numbers:?}"
    );
    assert!(
        has(TAG_KEY_EVENT_DURATION, EVENT_DURATION_SECS.into()),
        "posted: {numbers:?}"
    );
    assert!(
        has(TAG_KEY_NEXT_EVENT_START, ANNEX_C_START_UNIX_SECS),
        "the following event's start posts under its own key: {numbers:?}"
    );
    assert!(
        has(TAG_KEY_NEXT_EVENT_DURATION, EVENT_DURATION_SECS.into()),
        "posted: {numbers:?}"
    );
}

/// The all-ones start time EN 300 468 defines as undefined reports no time, and
/// posts no tag: a consumer must be able to tell "not announced" from an instant.
#[tokio::test]
async fn an_undefined_start_time_reports_nothing() {
    let section = eit_section(
        TABLE_ID_EIT_PF,
        SERVICE_ID,
        VERSION,
        0,
        &eit_event(
            FIRST_EVENT_ID,
            UNDEFINED_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Continuous", b""),
        ),
    );
    let demux = parse_sections(std::slice::from_ref(&section));
    assert_eq!(demux.eit_events().len(), 1);
    assert_eq!(demux.eit_events()[0].start_unix_secs, None);

    let posted = post_sections(&[section]).await;
    assert!(
        posted_text(&posted)
            .iter()
            .any(|(k, v)| k == TAG_KEY_EVENT_NAME && v == "Continuous"),
        "the event itself still posts"
    );
    assert!(
        !posted_numbers(&posted)
            .iter()
            .any(|(k, _)| k == TAG_KEY_EVENT_START),
        "an undefined start time posts no tag"
    );
}

/// Two events of one schedule section drain off the parser as schedule events.
#[test]
fn a_schedule_section_queues_every_event_it_carries() {
    let mut events = eit_event(
        FIRST_EVENT_ID,
        ANNEX_C_START_TIME,
        EVENT_DURATION,
        &short_event_descriptor(b"Morning Show", b"Live"),
    );
    events.extend_from_slice(&eit_event(
        SECOND_EVENT_ID,
        UNDEFINED_START_TIME,
        EVENT_DURATION,
        &short_event_descriptor(b"Afternoon Film", b"A western"),
    ));
    let section = eit_section(
        TABLE_ID_EIT_SCHEDULE,
        SERVICE_ID,
        VERSION,
        SCHEDULE_SECTION,
        &events,
    );

    let mut demux = parse_sections(std::slice::from_ref(&section));
    let queued = demux.take_eit_schedule();
    assert_eq!(queued.len(), 2, "both events of the section: {queued:?}");
    assert!(
        queued.iter().all(|e| e.slot == EitSlot::Schedule),
        "a schedule table's events are marked as such"
    );
    assert!(
        demux.eit_events().is_empty(),
        "a schedule event is not a present/following one"
    );
    assert_eq!(queued[0].service_id, SERVICE_ID);
    assert_eq!(queued[0].event_id, FIRST_EVENT_ID);
    assert_eq!(queued[0].name, "Morning Show");
    assert_eq!(queued[0].text, "Live");
    assert_eq!(queued[0].start_unix_secs, Some(ANNEX_C_START_UNIX_SECS));
    assert_eq!(queued[0].duration_secs, EVENT_DURATION_SECS);
    assert_eq!(queued[1].event_id, SECOND_EVENT_ID);
    assert_eq!(queued[1].name, "Afternoon Film");
    assert_eq!(queued[1].start_unix_secs, None);

    assert!(
        demux.take_eit_schedule().is_empty(),
        "the queue is drained, not copied"
    );
}

/// A schedule section repeating its version carries the events already handed
/// over, so it is not queued again; a new version is.
#[test]
fn a_repeated_schedule_section_is_not_requeued() {
    let event = eit_event(
        FIRST_EVENT_ID,
        ANNEX_C_START_TIME,
        EVENT_DURATION,
        &short_event_descriptor(b"First", b""),
    );
    let first = eit_section(
        TABLE_ID_EIT_SCHEDULE,
        SERVICE_ID,
        VERSION,
        SCHEDULE_SECTION,
        &event,
    );
    let mut demux = parse_sections(std::slice::from_ref(&first));
    assert_eq!(demux.take_eit_schedule().len(), 1);

    feed_sections(&mut demux, std::slice::from_ref(&first));
    assert!(
        demux.take_eit_schedule().is_empty(),
        "a section repeating its version does not re-queue"
    );

    let updated = eit_section(
        TABLE_ID_EIT_SCHEDULE,
        SERVICE_ID,
        VERSION + 1,
        SCHEDULE_SECTION,
        &eit_event(
            SECOND_EVENT_ID,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Second", b""),
        ),
    );
    feed_sections(&mut demux, &[updated]);
    let requeued = demux.take_eit_schedule();
    assert_eq!(requeued.len(), 1, "a new version re-queues");
    assert_eq!(requeued[0].event_id, SECOND_EVENT_ID);
    assert_eq!(requeued[0].name, "Second");
}

/// The element posts one message per schedule event, so the fields of one event
/// travel together and a consumer can tell whose start time is whose.
#[tokio::test]
async fn tsdemux_posts_one_message_per_schedule_event() {
    let mut events = eit_event(
        FIRST_EVENT_ID,
        ANNEX_C_START_TIME,
        EVENT_DURATION,
        &short_event_descriptor(b"Morning Show", b"Live"),
    );
    events.extend_from_slice(&eit_event(
        SECOND_EVENT_ID,
        ANNEX_C_START_TIME,
        EVENT_DURATION,
        &short_event_descriptor(b"Afternoon Film", b"A western"),
    ));
    let section = eit_section(
        TABLE_ID_EIT_SCHEDULE,
        SERVICE_ID,
        VERSION,
        SCHEDULE_SECTION,
        &events,
    );
    let posted = post_sections(&[section]).await;

    assert_eq!(posted.len(), 2, "one message per event: {posted:?}");
    for message in &posted {
        let BusMessage::Tag { program, .. } = message else {
            panic!("a schedule event posts as a tag message: {message:?}");
        };
        assert_eq!(
            *program,
            Some(SERVICE_ID),
            "scoped to the service it describes"
        );
    }
    let first = posted_numbers(&posted[..1]);
    assert!(
        first.contains(&(TAG_KEY_SCHEDULE_EVENT_ID.into(), FIRST_EVENT_ID.into())),
        "the first message carries the first event's id: {first:?}"
    );
    assert!(
        first.contains(&(TAG_KEY_SCHEDULE_EVENT_START.into(), ANNEX_C_START_UNIX_SECS)),
        "and its start time: {first:?}"
    );
    assert!(
        first.contains(&(
            TAG_KEY_SCHEDULE_EVENT_DURATION.into(),
            EVENT_DURATION_SECS.into()
        )),
        "and its duration: {first:?}"
    );
    let text = posted_text(&posted[..1]);
    assert!(
        text.contains(&(TAG_KEY_SCHEDULE_EVENT_NAME.into(), "Morning Show".into())),
        "and its name: {text:?}"
    );
    assert!(
        text.contains(&(TAG_KEY_SCHEDULE_EVENT_TEXT.into(), "Live".into())),
        "and its short description: {text:?}"
    );
    assert!(
        posted_numbers(&posted[1..])
            .contains(&(TAG_KEY_SCHEDULE_EVENT_ID.into(), SECOND_EVENT_ID.into())),
        "the second message carries the second event"
    );
}

/// A schedule table of another transport stream describes services carried
/// elsewhere, so it is ignored the way the other-TS present/following table is.
#[test]
fn an_other_transport_stream_schedule_is_ignored() {
    let section = eit_section(
        TABLE_ID_EIT_SCHEDULE_OTHER_TS,
        SERVICE_ID,
        VERSION,
        SCHEDULE_SECTION,
        &eit_event(
            FIRST_EVENT_ID,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"Elsewhere", b""),
        ),
    );
    let mut demux = parse_sections(&[section]);
    assert!(demux.take_eit_schedule().is_empty());
    assert!(demux.eit_events().is_empty());
}

/// A stream that never stops announcing events fills the queue and then loses the
/// later ones, keeping what it already holds: the cap bounds the memory an
/// untrusted stream costs, and a section past it still parses cleanly.
#[test]
fn the_schedule_queue_stops_at_its_cap() {
    // The event entries have to fit a 4096-byte section, and each needs a
    // short_event_descriptor to be reported at all.
    const EVENTS_PER_SECTION: usize = 200;
    let sections: Vec<Vec<u8>> = (0..)
        .map(|section_number: u8| {
            let mut events = Vec::new();
            for i in 0..EVENTS_PER_SECTION {
                let event_id = section_number as u16 * EVENTS_PER_SECTION as u16 + i as u16;
                events.extend_from_slice(&eit_event(
                    event_id,
                    ANNEX_C_START_TIME,
                    EVENT_DURATION,
                    &short_event_descriptor(b"", b""),
                ));
            }
            eit_section(
                TABLE_ID_EIT_SCHEDULE,
                SERVICE_ID,
                VERSION,
                section_number,
                &events,
            )
        })
        .take(MAX_EIT_SCHEDULE_EVENTS / EVENTS_PER_SECTION + 1)
        .collect();
    let fed = sections.len() * EVENTS_PER_SECTION;
    assert!(
        fed > MAX_EIT_SCHEDULE_EVENTS,
        "the fixture really does overrun the cap ({fed} events)"
    );

    let mut demux = parse_sections(&sections);
    let queued = demux.take_eit_schedule();
    assert_eq!(queued.len(), MAX_EIT_SCHEDULE_EVENTS);
    assert_eq!(
        queued[0].event_id, 0,
        "the events already held survive, the later ones are dropped"
    );

    // The parser keeps working: a drained queue takes the next section again.
    feed_sections(&mut demux, &[sections[0].clone()]);
    assert!(
        demux.take_eit_schedule().is_empty(),
        "that section's version is already read"
    );
    let after_cap = eit_section(
        TABLE_ID_EIT_SCHEDULE,
        SERVICE_ID,
        VERSION + 1,
        0,
        &eit_event(
            FIRST_EVENT_ID,
            ANNEX_C_START_TIME,
            EVENT_DURATION,
            &short_event_descriptor(b"After", b""),
        ),
    );
    feed_sections(&mut demux, &[after_cap]);
    let fresh = demux.take_eit_schedule();
    assert_eq!(fresh.len(), 1, "a drained queue takes events again");
    assert_eq!(fresh[0].event_id, FIRST_EVENT_ID);
}
