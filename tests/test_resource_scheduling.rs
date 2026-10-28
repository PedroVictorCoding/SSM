//! Integration tests for GPU/CPU Resource Scheduling, Budgets, and Live Concurrency Prioritization.
//!
//! Validates:
//! - Concurrent-load benchmarking reporting p50/p95/p99 voice latency, memory peaks, audio underruns, and frame drops.
//! - Starting conversation triggers background work to yield within measured bounds while playback continues.
//! - Strict tier budget enforcement and total VRAM capacity protection preventing OOM crashes.
//! - Repeated session reloads maintain strictly bounded memory without leaks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use sail::runtime::resource::{DeviceBudgetConfig, PriorityTier, ResourceScheduler};

#[test]
fn test_concurrent_load_benchmark_reports_latencies_and_underruns() {
    let scheduler = ResourceScheduler::default();

    let metrics = scheduler.benchmark_concurrent_load(100);

    // Latency checks: sub-millisecond loop in release/debug
    assert!(
        metrics.p50_voice_latency_ms < 5.0,
        "p50 latency {:.3}ms was too high",
        metrics.p50_voice_latency_ms
    );
    assert!(
        metrics.p95_voice_latency_ms < 10.0,
        "p95 latency {:.3}ms was too high",
        metrics.p95_voice_latency_ms
    );
    assert!(
        metrics.p99_voice_latency_ms < 20.0,
        "p99 latency {:.3}ms was too high",
        metrics.p99_voice_latency_ms
    );

    // Audio underrun check: must be 0
    assert_eq!(
        metrics.audio_underruns, 0,
        "Realtime audio suffered buffer underruns"
    );

    // Memory bounded within hardware configuration
    assert!(
        metrics.peak_vram_mb <= 14336,
        "Peak VRAM exceeded usable hardware limit"
    );

    // Yield latency bound
    assert!(
        metrics.training_yield_latency_ms < 5.0,
        "Yield response was too slow"
    );
}

#[test]
fn test_dialogue_start_triggers_background_training_yield() {
    let scheduler = ResourceScheduler::default();

    // Background training running initially
    assert!(!scheduler.should_training_yield());
    let initial_snapshot = scheduler.get_snapshot();
    assert_eq!(initial_snapshot.perception_fps_cap, 30);
    assert!(!initial_snapshot.is_dialogue_active);

    // User starts speaking -> Dialogue begins
    scheduler.on_dialogue_started();

    // Immediate yield signal
    assert!(
        scheduler.should_training_yield(),
        "Training must immediately be signaled to yield"
    );
    let active_snapshot = scheduler.get_snapshot();
    assert!(active_snapshot.is_dialogue_active);
    assert!(active_snapshot.is_training_paused);
    assert_eq!(
        active_snapshot.perception_fps_cap, 5,
        "Perception FPS must be throttled to conserve bandwidth"
    );

    // Conversation finishes
    scheduler.on_dialogue_ended();
    assert!(!scheduler.should_training_yield());
    let restored_snapshot = scheduler.get_snapshot();
    assert_eq!(restored_snapshot.perception_fps_cap, 30);
}

#[test]
fn test_budget_enforcement_and_oom_prevention() {
    let config = DeviceBudgetConfig {
        total_gpu_vram_mb: 4096,
        os_reserved_vram_mb: 512,
        audio_budget_mb: 128,
        foreground_inference_budget_mb: 2048,
        perception_budget_mb: 512,
        background_training_budget_mb: 1024,
        total_cpu_ram_mb: 8192,
        max_active_sessions: 2,
    };
    let scheduler = ResourceScheduler::new(config);

    // 1. Allocate within budget
    let res_ok = scheduler.allocate_memory(PriorityTier::ForegroundDialogue, 1024);
    assert!(res_ok.is_ok());

    // 2. Request exceeding tier budget (1024 current + 1500 requested > 2048 budget)
    let res_tier_exceeded = scheduler.allocate_memory(PriorityTier::ForegroundDialogue, 1500);
    match res_tier_exceeded {
        Err(err) => assert!(err.contains("MemoryBudgetExceeded")),
        Ok(()) => panic!("Expected MemoryBudgetExceeded error"),
    }

    // 3. Deallocate and re-allocate
    scheduler.deallocate_memory(PriorityTier::ForegroundDialogue, 512);
    let snapshot = scheduler.get_snapshot();
    assert_eq!(
        snapshot.allocations_by_tier.get("ForegroundDialogue"),
        Some(&512)
    );

    // 4. Exceed total hardware usable limit
    let _ = scheduler.allocate_memory(PriorityTier::BackgroundTraining, 1024);
    let _ = scheduler.allocate_memory(PriorityTier::VisualPerception, 512);
    // Usable = 4096 - 512 = 3584 MB
    // Currently allocated: 512 + 1024 + 512 = 2048 MB
    // Requesting 2000 MB on audio (which also exceeds audio budget)
    let res_oom = scheduler.allocate_memory(PriorityTier::RealtimeAudio, 2000);
    assert!(res_oom.is_err());
}

#[test]
fn test_repeated_session_reloads_maintain_bounded_memory() {
    let scheduler = ResourceScheduler::default();

    for _ in 0..10 {
        // Allocate session memory
        scheduler
            .allocate_memory(PriorityTier::ForegroundDialogue, 512)
            .unwrap();
        scheduler
            .allocate_memory(PriorityTier::VisualPerception, 128)
            .unwrap();

        // Release on session close
        scheduler.deallocate_memory(PriorityTier::ForegroundDialogue, 512);
        scheduler.deallocate_memory(PriorityTier::VisualPerception, 128);
    }

    let final_snapshot = scheduler.get_snapshot();
    assert_eq!(
        final_snapshot.total_allocated_vram_mb, 0,
        "Memory must not leak across session reloads"
    );
}

#[test]
fn test_invalid_os_reservation_returns_error_without_allocating() {
    let scheduler = ResourceScheduler::new(DeviceBudgetConfig {
        total_gpu_vram_mb: 1024,
        os_reserved_vram_mb: 2048,
        ..Default::default()
    });
    let error = scheduler
        .allocate_memory(PriorityTier::RealtimeAudio, 1)
        .unwrap_err();
    assert!(error.contains("InvalidDeviceBudget"));
    assert_eq!(scheduler.get_snapshot().total_allocated_vram_mb, 0);
}

#[test]
fn test_tier_allocation_overflow_is_rejected_without_mutating_reservations() {
    let scheduler = ResourceScheduler::new(DeviceBudgetConfig {
        total_gpu_vram_mb: usize::MAX,
        os_reserved_vram_mb: 0,
        foreground_inference_budget_mb: usize::MAX,
        ..Default::default()
    });
    scheduler
        .allocate_memory(PriorityTier::ForegroundDialogue, usize::MAX)
        .unwrap();
    let before = scheduler.get_snapshot();
    let error = scheduler
        .allocate_memory(PriorityTier::ForegroundDialogue, 1)
        .unwrap_err();
    assert!(error.contains("MemoryBudgetExceeded"));
    assert_eq!(scheduler.get_snapshot(), before);
}

#[test]
fn test_cross_tier_total_overflow_is_rejected_without_mutating_reservations() {
    let scheduler = ResourceScheduler::new(DeviceBudgetConfig {
        total_gpu_vram_mb: usize::MAX,
        os_reserved_vram_mb: 0,
        foreground_inference_budget_mb: usize::MAX,
        audio_budget_mb: usize::MAX,
        ..Default::default()
    });
    scheduler
        .allocate_memory(PriorityTier::ForegroundDialogue, usize::MAX - 1)
        .unwrap();
    let before = scheduler.get_snapshot();
    let error = scheduler
        .allocate_memory(PriorityTier::RealtimeAudio, 2)
        .unwrap_err();
    assert!(error.contains("DeviceVramExhausted"));
    assert_eq!(scheduler.get_snapshot(), before);
}

#[test]
fn test_zero_iteration_benchmark_returns_cleanly_without_panic() {
    let scheduler = ResourceScheduler::default();
    let metrics = scheduler.benchmark_concurrent_load(0);
    assert_eq!(metrics.p50_voice_latency_ms, 0.0);
    assert_eq!(metrics.p95_voice_latency_ms, 0.0);
    assert_eq!(metrics.p99_voice_latency_ms, 0.0);
    assert_eq!(metrics.audio_underruns, 0);
    assert_eq!(metrics.dropped_frames, 0);
    assert_eq!(metrics.training_yield_latency_ms, 0.0);
}

#[test]
fn test_raii_memory_reservation_guard_releases_on_drop() {
    let scheduler = Arc::new(ResourceScheduler::default());

    assert_eq!(scheduler.get_snapshot().total_allocated_vram_mb, 0);

    {
        let guard = scheduler
            .allocate_bytes(PriorityTier::ForegroundDialogue, 512 * 1024 * 1024)
            .expect("should allocate 512 MB");
        assert_eq!(guard.bytes(), 512 * 1024 * 1024);
        assert_eq!(guard.allocated_mb(), 512);
        assert_eq!(scheduler.get_snapshot().total_allocated_vram_mb, 512);
    }

    // Automatically deallocated on drop
    assert_eq!(scheduler.get_snapshot().total_allocated_vram_mb, 0);
}

#[test]
fn test_session_guard_and_concurrency_limit() {
    let config = DeviceBudgetConfig {
        max_active_sessions: 2,
        ..Default::default()
    };
    let scheduler = Arc::new(ResourceScheduler::new(config));

    let session1 = scheduler
        .acquire_session("session_1")
        .expect("session 1 ok");
    let session2 = scheduler
        .acquire_session("session_2")
        .expect("session 2 ok");
    assert_eq!(scheduler.get_snapshot().active_sessions, 2);

    // Third session exceeds limit
    let err = scheduler.acquire_session("session_3").unwrap_err();
    assert!(err.contains("MaxActiveSessionsExceeded"));
    assert_eq!(scheduler.get_snapshot().active_sessions, 2);

    // Dropping session 1 releases a slot
    drop(session1);
    assert_eq!(scheduler.get_snapshot().active_sessions, 1);

    let session3 = scheduler
        .acquire_session("session_3")
        .expect("session 3 ok now");
    assert_eq!(scheduler.get_snapshot().active_sessions, 2);
    drop(session2);
    drop(session3);
    assert_eq!(scheduler.get_snapshot().active_sessions, 0);
}

#[test]
fn test_host_cpu_ram_budget_and_guard() {
    let config = DeviceBudgetConfig {
        total_cpu_ram_mb: 256,
        ..Default::default()
    };
    let scheduler = Arc::new(ResourceScheduler::new(config));

    let host_guard = scheduler
        .allocate_host_memory(128 * 1024 * 1024)
        .expect("should allocate 128 MB host RAM");
    assert_eq!(
        scheduler.get_snapshot().total_allocated_host_bytes,
        128 * 1024 * 1024
    );

    // Exceeding 256 MB
    let err = scheduler
        .allocate_host_memory(200 * 1024 * 1024)
        .unwrap_err();
    assert!(err.contains("HostMemoryBudgetExceeded"));

    drop(host_guard);
    assert_eq!(scheduler.get_snapshot().total_allocated_host_bytes, 0);
}

#[test]
fn test_real_memory_buffer_allocation() {
    let scheduler = Arc::new(ResourceScheduler::default());

    {
        let mut allocation = scheduler
            .allocate_real_buffer(PriorityTier::ForegroundDialogue, 1024 * 1024)
            .expect("should allocate 1MB buffer");
        assert_eq!(allocation.len(), 1024 * 1024);
        assert_eq!(allocation.tier(), PriorityTier::ForegroundDialogue);

        // Mutate buffer to verify real memory backing
        allocation.as_mut_slice()[0] = 42;
        allocation.as_mut_slice()[1024 * 1024 - 1] = 99;
        assert_eq!(allocation.as_slice()[0], 42);
        assert_eq!(allocation.as_slice()[1024 * 1024 - 1], 99);

        assert!(scheduler.get_snapshot().total_allocated_vram_mb >= 1);
    }

    assert_eq!(scheduler.get_snapshot().total_allocated_vram_mb, 0);
}

#[test]
fn test_trace_foreground_dialogue_to_observed_background_training_yield() {
    let scheduler = Arc::new(ResourceScheduler::default());
    let stop_signal = Arc::new(AtomicBool::new(false));
    let training_step_counter = Arc::new(AtomicU64::new(0));
    let yield_acknowledged = Arc::new(AtomicBool::new(false));

    // 1. Spawn concurrent simulated background training worker
    let s_train = scheduler.clone();
    let stop_train = stop_signal.clone();
    let step_count = training_step_counter.clone();
    let yield_ack = yield_acknowledged.clone();

    let train_handle = thread::spawn(move || {
        while !stop_train.load(Ordering::SeqCst) {
            // Check yield boundary before bounded work unit
            if s_train.should_training_yield() {
                s_train.acknowledge_training_yield();
                yield_ack.store(true, Ordering::SeqCst);
                // Pause safely without incrementing steps or corrupting state
                while s_train.should_training_yield() && !stop_train.load(Ordering::SeqCst) {
                    thread::sleep(Duration::from_millis(5));
                }
                s_train.record_training_resumed();
            }

            // Simulate one bounded training step (~5ms)
            let step_start = Instant::now();
            thread::sleep(Duration::from_millis(5));
            let step_duration = step_start.elapsed().as_secs_f64() * 1000.0;
            s_train.record_training_step_duration(step_duration);
            step_count.fetch_add(1, Ordering::SeqCst);
        }
    });

    // 2. Spawn concurrent realtime audio playback simulator
    let s_audio = scheduler.clone();
    let stop_audio = stop_signal.clone();
    let audio_packets = Arc::new(AtomicU64::new(0));
    let aud_count = audio_packets.clone();

    let audio_handle = thread::spawn(move || {
        while !stop_audio.load(Ordering::SeqCst) {
            // Audio requires timely servicing every 10ms
            let packet_start = Instant::now();
            thread::sleep(Duration::from_millis(5));
            if packet_start.elapsed().as_millis() > 30 {
                s_audio.record_audio_underrun();
            }
            aud_count.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Let background training run for a few steps
    thread::sleep(Duration::from_millis(30));
    assert!(training_step_counter.load(Ordering::SeqCst) > 0);
    assert!(!scheduler.is_training_paused());

    // 3. User triggers foreground dialogue event
    let event_start = Instant::now();
    scheduler.on_dialogue_started();

    // Verify background worker yields within bounded window (< 50ms)
    let mut yielded = false;
    while event_start.elapsed() < Duration::from_millis(100) {
        if yield_acknowledged.load(Ordering::SeqCst) && scheduler.is_training_paused() {
            yielded = true;
            break;
        }
        thread::sleep(Duration::from_millis(2));
    }
    assert!(
        yielded,
        "Background training worker failed to yield within 100ms"
    );

    let yield_latency = scheduler.last_yield_latency_ms();
    assert!(
        yield_latency < 50.0,
        "Observed yield latency {:.2}ms exceeded 50ms bound",
        yield_latency
    );

    // While dialogue is active, training step counter must remain frozen
    let step_at_yield = training_step_counter.load(Ordering::SeqCst);
    thread::sleep(Duration::from_millis(40));
    let step_after_pause = training_step_counter.load(Ordering::SeqCst);
    assert_eq!(
        step_at_yield, step_after_pause,
        "Background worker continued executing steps while paused for dialogue"
    );

    // Audio playback continued without buffer underruns
    assert_eq!(
        scheduler.get_snapshot().audio_underruns,
        0,
        "Realtime audio suffered underruns during conversation"
    );

    // 4. Foreground dialogue concludes
    scheduler.on_dialogue_ended();
    assert!(!scheduler.should_training_yield());

    // Verify background worker resumes training
    thread::sleep(Duration::from_millis(30));
    let step_after_resume = training_step_counter.load(Ordering::SeqCst);
    assert!(
        step_after_resume > step_at_yield,
        "Background worker did not resume training after dialogue concluded"
    );

    // Clean shutdown
    stop_signal.store(true, Ordering::SeqCst);
    train_handle.join().expect("train thread join");
    audio_handle.join().expect("audio thread join");
}
