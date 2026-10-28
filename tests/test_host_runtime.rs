//! Integration and Verification Tests for SAIL Native Host Runtime and Event Bus.
//!
//! Verifies acceptance criteria for GitHub Issue #3:
//! 1. Bounded queue and memory growth under synthetic loads; overflow counters track dropped work.
//! 2. Event attribution, causal parent tracking, and sequence gap detection.
//! 3. Cooperative cancellation propagation and worker crash isolation (no deadlocks).
//! 4. Headless deterministic event replay and native host smoke lifecycle.

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use sail::runtime::{
    BoundedEventQueue, BusError, EventBus, EventPayload, EventRecorder, OverflowPolicy,
    RuntimeEvent, SequenceDiscrepancy, SharedAudioBuffer, SharedVideoBuffer, Supervisor,
    WorkerRole, WorkerState,
};

/// Test 1: Bounded queue prevents unbounded growth under burst load; drop counters identify dropped work.
#[test]
fn test_bounded_queue_overflow_and_drop_counters() {
    let capacity = 50;
    let queue = BoundedEventQueue::new(capacity);

    // 1. Fill queue with 200 synthetic events using DropOldest
    for i in 1..=200 {
        let audio = Arc::new(SharedAudioBuffer::new(16000, 1, vec![0.1; 320]));
        let event = RuntimeEvent::new(
            i,
            "session_1",
            "mic",
            i,
            1_000_000 * i,
            EventPayload::AudioInput(audio),
        );
        let res = queue.push(event, OverflowPolicy::DropOldest);
        assert!(res.is_ok());
    }

    // Queue must never exceed capacity
    assert_eq!(queue.len(), capacity);
    assert_eq!(queue.enqueued_count(), 200);
    assert_eq!(queue.dropped_count(), 150); // 200 - 50 = 150 dropped

    // The events remaining in the queue should be the newest ones (51..=200)
    let drained = queue.drain();
    assert_eq!(drained.len(), capacity);
    assert_eq!(drained.first().unwrap().event_id, 151);
    assert_eq!(drained.last().unwrap().event_id, 200);

    // 2. Test DropNewest policy
    let queue_newest = BoundedEventQueue::new(10);
    for i in 1..=25 {
        let event = RuntimeEvent::new(
            i,
            "session_2",
            "screen",
            i,
            1_000_000 * i,
            EventPayload::ControlCommand {
                command: format!("cmd_{i}"),
            },
        );
        let _ = queue_newest.push(event, OverflowPolicy::DropNewest);
    }
    assert_eq!(queue_newest.len(), 10);
    assert_eq!(queue_newest.dropped_count(), 15);
    let drained_newest = queue_newest.drain();
    // In DropNewest, the original 10 remain
    assert_eq!(drained_newest.first().unwrap().event_id, 1);
    assert_eq!(drained_newest.last().unwrap().event_id, 10);

    // 3. Test Reject policy
    let queue_reject = BoundedEventQueue::new(5);
    for i in 1..=5 {
        let event = RuntimeEvent::new(
            i,
            "s",
            "src",
            i,
            i,
            EventPayload::ControlCommand {
                command: "ping".into(),
            },
        );
        assert!(queue_reject.push(event, OverflowPolicy::Reject).is_ok());
    }
    let overflow_event = RuntimeEvent::new(
        6,
        "s",
        "src",
        6,
        6,
        EventPayload::ControlCommand {
            command: "ping".into(),
        },
    );
    let reject_res = queue_reject.push(overflow_event, OverflowPolicy::Reject);
    assert_eq!(reject_res, Err(BusError::QueueFull { capacity: 5 }));
    assert_eq!(queue_reject.dropped_count(), 1);
}

/// Test 2: Audio callback non-blocking try_push never blocks even under high contention.
#[test]
fn test_audio_callback_non_blocking_try_push() {
    let queue = Arc::new(BoundedEventQueue::new(20));

    // Fill queue to capacity
    for i in 1..=20 {
        let event = RuntimeEvent::new(
            i,
            "session",
            "mic",
            i,
            i * 1000,
            EventPayload::AudioInput(Arc::new(SharedAudioBuffer::new(16000, 1, vec![0.0; 160]))),
        );
        queue.push(event, OverflowPolicy::DropOldest).unwrap();
    }

    // High rate non-blocking callback simulation
    let mut pushed_ok = 0;
    for i in 21..=100 {
        let event = RuntimeEvent::new(
            i,
            "session",
            "mic",
            i,
            i * 1000,
            EventPayload::AudioInput(Arc::new(SharedAudioBuffer::new(16000, 1, vec![0.0; 160]))),
        );
        if queue.try_push(event, OverflowPolicy::DropOldest).is_ok() {
            pushed_ok += 1;
        }
    }

    assert_eq!(pushed_ok, 80);
    assert_eq!(queue.len(), 20);
    assert_eq!(queue.dropped_count(), 80);
}

/// Test 3: Causal parent tracking, per-source sequence numbering, and gap detection.
#[test]
fn test_event_ordering_causality_and_gap_detection() {
    let bus = EventBus::new();

    // Allocate sequence numbers for different sources
    let mic_seq1 = bus.next_sequence_for_source("mic");
    let mic_seq2 = bus.next_sequence_for_source("mic");
    let screen_seq1 = bus.next_sequence_for_source("screen");

    assert_eq!(mic_seq1, 1);
    assert_eq!(mic_seq2, 2);
    assert_eq!(screen_seq1, 1);

    // Causal chain: User audio -> Transcription -> Model token -> Tool call
    let audio_event = RuntimeEvent::new(
        bus.next_event_id(),
        "session_main",
        "mic",
        1,
        100_000,
        EventPayload::AudioInput(Arc::new(SharedAudioBuffer::new(16000, 1, vec![0.1; 320]))),
    );

    let transcript_event = RuntimeEvent::new(
        bus.next_event_id(),
        "session_main",
        "perception_asr",
        1,
        120_000,
        EventPayload::UserInputText {
            text: "What is the capital of France?".into(),
            is_final: true,
        },
    )
    .with_parent_event_id(audio_event.event_id);

    let model_chunk_event = RuntimeEvent::new(
        bus.next_event_id(),
        "session_main",
        "inference_engine",
        1,
        150_000,
        EventPayload::ModelOutputChunk {
            generation_id: 1,
            response_id: "resp_001".into(),
            token_id: 42,
            text: "Paris".into(),
            is_final: false,
        },
    )
    .with_parent_event_id(transcript_event.event_id);

    // Verify parent relationships
    assert_eq!(transcript_event.parent_event_id, Some(audio_event.event_id));
    assert_eq!(
        model_chunk_event.parent_event_id,
        Some(transcript_event.event_id)
    );

    // Sequence gap detection
    // Expected next for "mic" is 3. If an event arrives with seq=5, bus detects Gap.
    let gap_event = RuntimeEvent::new(
        100,
        "session_main",
        "mic",
        5,
        200_000,
        EventPayload::ControlCommand {
            command: "test".into(),
        },
    );
    let disc = bus.validate_sequence(&gap_event);
    assert!(disc.is_some());
    match disc.unwrap() {
        SequenceDiscrepancy::Gap {
            source,
            expected,
            actual,
        } => {
            assert_eq!(source, "mic");
            assert_eq!(expected, 3);
            assert_eq!(actual, 5);
        }
        other => panic!("Expected Gap, got {:?}", other),
    }
}

/// Test 4: Cancellation propagates cleanly; worker failure/crash does not deadlock audio or runtime.
#[test]
fn test_cancellation_propagation_and_worker_crash_isolation() {
    let bus = Arc::new(EventBus::new());
    let supervisor = Supervisor::new(bus.clone());

    let audio_running = Arc::new(AtomicBool::new(false));
    let audio_tick_count = Arc::new(AtomicU64::new(0));

    // 1. Spawn perception audio worker (simulating real-time mic streaming)
    let ar_flag = audio_running.clone();
    let at_count = audio_tick_count.clone();
    supervisor
        .spawn_worker(
            "perception_audio",
            WorkerRole::PerceptionAudio,
            move |token, _bus| {
                ar_flag.store(true, Ordering::SeqCst);
                while !token.is_cancelled() {
                    at_count.fetch_add(1, Ordering::Relaxed);
                    thread::sleep(Duration::from_millis(5));
                }
                Ok(())
            },
        )
        .unwrap();

    // 2. Spawn a tool worker that intentionally panics
    supervisor
        .spawn_worker(
            "fragile_tool_worker",
            WorkerRole::ToolExecutor,
            |_token, _bus| {
                thread::sleep(Duration::from_millis(20));
                panic!("Simulated critical hardware / tool failure!");
            },
        )
        .unwrap();

    // Wait briefly for panic to occur
    thread::sleep(Duration::from_millis(80));

    let states = supervisor.get_worker_states();
    // The crashed worker is safely marked Faulted
    match states.get("fragile_tool_worker") {
        Some(WorkerState::Faulted(err)) => {
            assert!(err.contains("Simulated critical hardware / tool failure!"));
        }
        other => panic!("Expected Faulted state for crashed worker, got {:?}", other),
    }

    // Audio capture worker was NOT deadlocked or stopped by the other worker's crash
    assert!(audio_running.load(Ordering::SeqCst));
    assert!(audio_tick_count.load(Ordering::Relaxed) > 5);

    // 3. Test orderly shutdown and cancellation propagation
    supervisor.shutdown(Duration::from_millis(200));
    assert!(supervisor.is_shutting_down());

    let post_shutdown_states = supervisor.get_worker_states();
    assert_eq!(
        post_shutdown_states.get("perception_audio"),
        Some(&WorkerState::Stopped)
    );
}

/// Test 5: Headless deterministic event replay from recorded session JSONL.
#[test]
fn test_deterministic_headless_event_replay() {
    let temp_dir = std::env::temp_dir().join(format!("sail-replay-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let log_file = temp_dir.join("test_session.events.jsonl");

    // 1. Record a deterministic sequence
    let mut recorder = EventRecorder::new();
    let audio = Arc::new(SharedAudioBuffer::new(16000, 1, vec![0.05; 320]));
    let video = Arc::new(SharedVideoBuffer::new(
        64,
        64,
        "RGB24",
        vec![128; 64 * 64 * 3],
    ));

    recorder.record(RuntimeEvent::new(
        1,
        "replay_sess",
        "mic",
        1,
        10_000,
        EventPayload::AudioInput(audio),
    ));
    recorder.record(RuntimeEvent::new(
        2,
        "replay_sess",
        "screen",
        1,
        20_000,
        EventPayload::VideoFrame(video),
    ));
    recorder.record(RuntimeEvent::new(
        3,
        "replay_sess",
        "model",
        1,
        30_000,
        EventPayload::ModelOutputChunk {
            generation_id: 1,
            response_id: "res1".into(),
            token_id: 101,
            text: "Hello".into(),
            is_final: true,
        },
    ));

    // Save to disk
    recorder.save_to_jsonl(&log_file).expect("save jsonl");
    assert!(log_file.exists());

    // 2. Load from disk
    let loaded = EventRecorder::load_from_jsonl(&log_file).expect("load jsonl");
    assert_eq!(loaded.len(), 3);

    // 3. Replay into test observer
    let mut replayed_ids = Vec::new();
    let count = loaded.replay_deterministic(|evt| {
        replayed_ids.push(evt.event_id);
    });

    assert_eq!(count, 3);
    assert_eq!(replayed_ids, vec![1, 2, 3]);

    let _ = fs::remove_dir_all(&temp_dir);
}
