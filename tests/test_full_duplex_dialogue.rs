//! Comprehensive integration tests for full-duplex dialogue, barge-in interruption,
//! causal response validation, tool state tracking, and interrupt latency objectives.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use sail::capability::{Capability, CapabilityPolicy};
use sail::runtime::adapter::MockModelAdapter;
use sail::runtime::dialogue::{
    DialogueEvent, DialogueState, FullDuplexCoordinator, ToolExecutionStatus,
};
use sail::tool::{Tool, ToolRegistry};
use sail::voice::PlaybackPipeline;
use sail::voice::tts::FormantNeuralTtsEngine;

struct DummyWeatherTool;

impl Tool for DummyWeatherTool {
    fn name(&self) -> &'static str {
        "get_weather"
    }

    fn description(&self) -> &'static str {
        "Returns weather conditions"
    }

    fn execute(&self, query: &str) -> Result<String, String> {
        Ok(format!("Weather for {query}: Sunny 21C"))
    }
}

fn create_test_coordinator(
    mock_responses: Vec<(&str, &str)>,
) -> (FullDuplexCoordinator, ToolRegistry, Arc<PlaybackPipeline>) {
    let mut adapter = MockModelAdapter::new("test-full-duplex-model");

    for (prompt, reply) in mock_responses {
        adapter = adapter.with_mock_response(prompt, reply);
    }
    let adapter_arc: Arc<dyn sail::runtime::adapter::ModelBackendAdapter> = Arc::new(adapter);

    let tts_engine = Arc::new(Mutex::new(FormantNeuralTtsEngine::new(16000)));
    let playback_pipeline = Arc::new(PlaybackPipeline::new(16000, 64));

    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("get_weather".into()));
    let policy = CapabilityPolicy::new(
        granted,
        vec![PathBuf::from(".")],
        vec![PathBuf::from("/tmp")],
    );

    let mut registry = ToolRegistry::new();
    registry.register(Arc::new(DummyWeatherTool));

    let coordinator = FullDuplexCoordinator::new(
        "test_session_duplex",
        adapter_arc,
        tts_engine,
        playback_pipeline.clone(),
        policy,
    );

    (coordinator, registry, playback_pipeline)
}

#[test]
fn test_normal_dialogue_turn_execution_and_completion() {
    let (mut coordinator, registry, _) =
        create_test_coordinator(vec![("hello", "Hello there! How can I assist you today?")]);

    assert_eq!(coordinator.state(), DialogueState::Idle);

    let mut events = coordinator
        .submit_user_turn("hello", &registry, 1000)
        .expect("submit user turn");

    // During playback, state is Speaking
    assert_eq!(coordinator.state(), DialogueState::Speaking);

    // Drain playback to simulate user hearing full response
    let drain_events = coordinator.drain_playback(1200);
    events.extend(drain_events);

    // Must reach Idle at the end of normal turn
    assert_eq!(coordinator.state(), DialogueState::Idle);

    // Verify event progression: Interpreting -> Speaking -> Idle
    let state_changes: Vec<DialogueState> = events
        .iter()
        .filter_map(|e| match e {
            DialogueEvent::StateChanged(ev) => Some(ev.to_state),
            _ => None,
        })
        .collect();

    assert!(state_changes.contains(&DialogueState::Interpreting));
    assert!(state_changes.contains(&DialogueState::Speaking));
    assert_eq!(*state_changes.last().unwrap(), DialogueState::Idle);

    // Conversation history commits the turn
    let history = coordinator.conversation_history();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].user_text, "hello");
    assert!(!history[0].was_interrupted);
    assert_eq!(
        history[0].assistant_played_text,
        "Hello there! How can I assist you today?"
    );
}

#[test]
fn test_mid_speech_barge_in_and_playback_interruption() {
    let (mut coordinator, registry, playback) = create_test_coordinator(vec![(
        "tell me a story",
        "Once upon a time in a peaceful valley there lived a wise traveler who explored the world.",
    )]);

    // Start a long assistant response
    let _ = coordinator
        .submit_user_turn("tell me a story", &registry, 2000)
        .expect("submit story");

    // Queue has chunks buffered for continuous speaker playback
    assert!(playback.queued_chunks() > 0);

    // Simulate speaker playing the first 3 words (~960 samples)
    let mut speaker_buf = vec![0.0f32; 960];
    playback.pull_samples(&mut speaker_buf);
    assert_eq!(playback.playback_cursor(), 960);

    // User barges in with new microphone input during speech!
    // Endpoint detector requires min_speech_frames (4 frames = 80ms) to confirm speech start
    let voice_frame = vec![0.35f32; 320]; // 20ms voice frame
    let mut barge_events = Vec::new();
    for f in 0..4 {
        let ev = coordinator
            .process_audio_frame(&voice_frame, 2200 + f * 20)
            .expect("process audio frame");
        barge_events.extend(ev);
    }

    assert!(
        barge_events
            .iter()
            .any(|e| matches!(e, DialogueEvent::InterruptionTriggered { .. })),
        "Should trigger interruption on speech during playback"
    );

    // Playback queue must be instantly flushed on barge-in
    assert_eq!(
        playback.queued_chunks(),
        0,
        "Speaker buffer must be flushed on barge-in"
    );

    // State moves to Interrupted then Listening for user correction
    assert_eq!(coordinator.state(), DialogueState::Listening);

    // Verify history records that the turn was interrupted and unspoken text was preserved separately
    let history = coordinator.conversation_history();
    assert!(!history.is_empty());
    let interrupted_turn = history.last().unwrap();
    assert!(interrupted_turn.was_interrupted);
    assert!(interrupted_turn.assistant_unspoken_text.is_some());

    // Only heard words are committed in assistant_played_text
    println!(
        "Played prefix on barge-in: \"{}\"",
        interrupted_turn.assistant_played_text
    );
    assert!(
        !interrupted_turn
            .assistant_played_text
            .contains("explored the world")
    );
}

#[test]
fn test_barge_in_latency_objective_under_150ms() {
    let (mut coordinator, registry, playback) = create_test_coordinator(vec![(
        "stream response",
        "Streaming test sentence for latency measurement with multiple words.",
    )]);

    // Run 10 rapid barge-in trials to compute p95 latency
    for i in 0..10 {
        let _ = coordinator.submit_user_turn("stream response", &registry, 3000 + i * 500);

        // Pull some samples
        let mut buf = vec![0.0f32; 480];
        playback.pull_samples(&mut buf);

        // Interrupt
        let _ = coordinator.trigger_barge_in(3050 + i * 500);
    }

    let (p50, p95, count) = coordinator.interrupt_latency_stats();
    println!(
        "Interrupt-to-playback-stop latency (N={count}): p50 = {p50:.3} ms | p95 = {p95:.3} ms (Objective: <= 150 ms)"
    );

    assert_eq!(count, 10);
    assert!(
        p95 <= 150.0,
        "p95 latency must meet <= 150ms objective, measured: {p95:.3}ms"
    );
    // On native host, in-memory stop is typically < 2ms!
    assert!(p95 < 20.0);
}

#[test]
fn test_stale_response_id_rejection() {
    let (mut coordinator, registry, _) =
        create_test_coordinator(vec![("question one", "Answer for question one.")]);

    let _ = coordinator.submit_user_turn("question one", &registry, 4000);
    coordinator.drain_playback(4100);

    // Turn completed, active_response_id is reset
    assert!(coordinator.active_response_id().is_none());

    // Stale generation chunk bearing old or fabricated response ID
    let accepted = coordinator.accept_chunk_for_response("resp_stale_old_id");
    assert!(
        !accepted,
        "Coordinator must reject stale response ID chunks"
    );
    assert_eq!(coordinator.stale_rejection_count(), 1);
}

#[test]
fn test_tool_execution_lifecycle_and_cancellation_semantics() {
    let (mut coordinator, registry, _) = create_test_coordinator(vec![(
        "what is the weather in Tokyo",
        "Checking now <tool_call>get_weather(Tokyo)</tool_call>",
    )]);

    let events = coordinator
        .submit_user_turn("what is the weather in Tokyo", &registry, 5000)
        .expect("submit tool turn");
    coordinator.drain_playback(5100);

    // Verify tool execution events
    let tool_completed = events.iter().any(|e| {
        matches!(
            e,
            DialogueEvent::ToolExecutionCompleted {
                status: ToolExecutionStatus::Completed,
                ..
            }
        )
    });
    assert!(
        tool_completed,
        "Tool should complete with side effects committed"
    );

    let history = coordinator.conversation_history();
    assert_eq!(history.len(), 1);
    let tools = &history[0].executed_tools;
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].tool_name, "get_weather");
    assert_eq!(tools[0].status, ToolExecutionStatus::Completed);
    assert!(tools[0].side_effects_committed);
    assert!(tools[0].output.as_ref().unwrap().contains("Sunny 21C"));
}

#[test]
fn test_overlapping_conversation_with_user_correction() {
    let (mut coordinator, registry, playback) = create_test_coordinator(vec![
        (
            "play classical music",
            "Starting classical playlist symphony number nine by Beethoven.",
        ),
        (
            "wait stop play jazz instead",
            "Switching playlist to smooth jazz radio.",
        ),
    ]);

    // 1. User asks for classical music
    let _ = coordinator
        .submit_user_turn("play classical music", &registry, 6000)
        .expect("turn 1");

    // Assistant starts speaking
    let mut buf = vec![0.0f32; 640];
    playback.pull_samples(&mut buf);

    // 2. User interrupts with a correction
    let _correction_events = coordinator
        .submit_user_turn("wait stop play jazz instead", &registry, 6500)
        .expect("turn 2 correction");

    // Drain turn 2 audio playback
    coordinator.drain_playback(7000);

    // Verify first turn was marked interrupted, and second turn completed
    let history = coordinator.conversation_history();
    assert_eq!(history.len(), 2);
    assert!(history[0].was_interrupted);
    assert_eq!(history[0].user_text, "play classical music");

    assert!(!history[1].was_interrupted);
    assert_eq!(history[1].user_text, "wait stop play jazz instead");
    assert_eq!(
        history[1].assistant_played_text,
        "Switching playlist to smooth jazz radio."
    );

    assert_eq!(coordinator.state(), DialogueState::Idle);
}

#[test]
fn test_tool_cancellation_preserves_completed_and_cancels_unstarted() {
    let (mut coordinator, registry, _) = create_test_coordinator(vec![(
        "search and execute",
        "Beginning execution <tool_call>get_weather(Paris)</tool_call>",
    )]);

    // Start turn with tool
    let _ = coordinator
        .submit_user_turn("search and execute", &registry, 8000)
        .expect("submit turn");

    // Before or during speech, barge-in occurs
    let _cancel_events = coordinator.trigger_barge_in(8050).expect("barge in");

    // History should retain the completed tool with side_effects_committed == true
    let history = coordinator.conversation_history();
    assert_eq!(history.len(), 1);
    let record = &history[0];
    assert!(record.was_interrupted);

    let tools = &record.executed_tools;
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].status, ToolExecutionStatus::Completed);
    assert!(tools[0].side_effects_committed);
    assert!(tools[0].output.as_ref().unwrap().contains("Paris"));
}
