//! Integration tests for Host-Owned Explicit Session & Task State Machine, Bounded Model Memory, and Crash Recovery.
//!
//! Validates:
//! - Multi-session state isolation between concurrent users/sessions.
//! - Revision of user objective during barge-in without replaying completed actions.
//! - Crash recovery reconciliation: marking running tools as UncertainOutcome rather than blind retry.
//! - Bounded session recurrent/KV cache limits and explicit session reset/termination.
//! - Inspectable session reports.

use sail::runtime::dialogue::{ToolExecutionStatus, TrackedToolExecution};
use sail::runtime::session::{BoundedSessionCache, SessionManager, TaskLifecycleStatus};

#[test]
fn test_multi_session_isolation() {
    let manager = SessionManager::new(10);

    // Initialize two distinct sessions
    let s1 = manager.get_or_create_session("session_alpha", 1000);
    let s2 = manager.get_or_create_session("session_beta", 1000);

    assert_eq!(s1.session_id, "session_alpha");
    assert_eq!(s2.session_id, "session_beta");

    // Start task in Alpha
    manager
        .start_task(
            "session_alpha",
            "task_a1",
            "Find all rust files in src/",
            1010,
        )
        .expect("start task alpha");

    // Start task in Beta
    manager
        .start_task("session_beta", "task_b1", "Synthesize audio waveform", 1015)
        .expect("start task beta");

    // Record tool execution only in Alpha
    let tool_call = TrackedToolExecution {
        call_id: "call_find_1".into(),
        tool_name: "file_search".into(),
        input: "path: src/".into(),
        status: ToolExecutionStatus::Completed,
        response_id: "resp_1".into(),
        output: Some("[\"main.rs\", \"lib.rs\"]".into()),
        side_effects_committed: true,
        start_ms: 1020,
        end_ms: Some(1030),
    };
    manager
        .record_tool_result("session_alpha", tool_call)
        .expect("record tool in alpha");

    // Verify alpha state
    let rep_alpha = manager
        .inspect_session("session_alpha")
        .expect("inspect alpha");
    assert_eq!(
        rep_alpha.active_objective,
        Some("Find all rust files in src/".into())
    );
    assert_eq!(rep_alpha.completed_tools_count, 1);

    // Verify beta state has zero tools and different objective
    let rep_beta = manager
        .inspect_session("session_beta")
        .expect("inspect beta");
    assert_eq!(
        rep_beta.active_objective,
        Some("Synthesize audio waveform".into())
    );
    assert_eq!(rep_beta.completed_tools_count, 0);
}

#[test]
fn test_task_revision_on_user_barge_in_prevents_replaying_completed_tools() {
    let manager = SessionManager::new(5);
    manager.get_or_create_session("sess_barge_in", 2000);

    manager
        .start_task(
            "sess_barge_in",
            "task_1",
            "Download dataset and format JSON",
            2005,
        )
        .expect("start task");

    // 1. Tool 1: Download dataset (Completed and committed)
    let tool_download = TrackedToolExecution {
        call_id: "dl_01".into(),
        tool_name: "download_dataset".into(),
        input: "url: https://example.com/data.parquet".into(),
        status: ToolExecutionStatus::Completed,
        response_id: "resp_2".into(),
        output: Some("Downloaded 50MB to /tmp/data.parquet".into()),
        side_effects_committed: true,
        start_ms: 2010,
        end_ms: Some(2050),
    };
    manager
        .record_tool_result("sess_barge_in", tool_download)
        .expect("record download");

    // 2. User interrupts: "Stop formatting as JSON, export to SQLite instead!"
    manager
        .revise_active_objective(
            "sess_barge_in",
            "Download dataset and export to SQLite",
            "User corrected destination format via barge-in speech",
            2060,
        )
        .expect("revise objective");

    let rep = manager.inspect_session("sess_barge_in").expect("inspect");
    assert_eq!(
        rep.active_objective,
        Some("Download dataset and export to SQLite".into())
    );
    assert_eq!(rep.revisions_count, 1);
    assert_eq!(rep.task_status, Some(TaskLifecycleStatus::InProgress));

    // Verify that the completed download tool is preserved and can be checked
    let sess = manager.get_or_create_session("sess_barge_in", 2070);
    let task = sess.active_task.as_ref().expect("has active task");
    assert!(task.has_completed_tool("download_dataset", "url: https://example.com/data.parquet"));

    // Ensure downstream planner knows download is already completed, preventing redundant re-execution
    assert!(!task.has_completed_tool("export_sqlite", "target: db.sqlite"));
}

#[test]
fn test_worker_crash_recovery_reconciles_running_tools_as_uncertain() {
    let manager = SessionManager::new(5);
    manager.get_or_create_session("sess_crash", 3000);

    manager
        .start_task(
            "sess_crash",
            "task_db_write",
            "Perform database schema migration",
            3005,
        )
        .expect("start task");

    // Tool 1: Completed backup
    manager
        .record_tool_result(
            "sess_crash",
            TrackedToolExecution {
                call_id: "t_backup".into(),
                tool_name: "backup_db".into(),
                input: "tables: all".into(),
                status: ToolExecutionStatus::Completed,
                response_id: "resp_crash_1".into(),
                output: Some("Backup saved".into()),
                side_effects_committed: true,
                start_ms: 3010,
                end_ms: Some(3020),
            },
        )
        .expect("record backup");

    // Tool 2: Was actively running when worker crashed
    manager
        .record_tool_result(
            "sess_crash",
            TrackedToolExecution {
                call_id: "t_migrate".into(),
                tool_name: "execute_sql_migration".into(),
                input: "ALTER TABLE users ADD COLUMN phone TEXT".into(),
                status: ToolExecutionStatus::Running,
                response_id: "resp_crash_2".into(),
                output: None,
                side_effects_committed: false,
                start_ms: 3025,
                end_ms: None,
            },
        )
        .expect("record migrate running");

    // Update volatile cache metrics prior to crash
    {
        let mut map = manager.get_or_create_session("sess_crash", 3030);
        map.cache_metrics.update_usage(1024, 500);
    }

    // Worker crashes and supervisor initiates session crash recovery
    let recovered_report = manager
        .recover_from_worker_crash("sess_crash")
        .expect("recover");

    // Verification:
    // 1. Task enters AwaitingClarification
    assert_eq!(
        recovered_report.task_status,
        Some(TaskLifecycleStatus::AwaitingClarification)
    );
    // 2. Running tool transitioned to UncertainOutcome
    assert_eq!(recovered_report.pending_or_uncertain_tools_count, 1);
    assert_eq!(recovered_report.completed_tools_count, 1);
    // 3. Volatile model cache reset
    assert_eq!(recovered_report.memory_tokens, 0);
    assert_eq!(recovered_report.ssm_steps, 0);

    // Inspect the specific tool status
    let sess = manager.get_or_create_session("sess_crash", 3050);
    let active_task = sess.active_task.as_ref().unwrap();
    let migrate_tool = active_task
        .tool_executions
        .iter()
        .find(|t| t.call_id == "t_migrate")
        .unwrap();
    assert_eq!(migrate_tool.status, ToolExecutionStatus::UncertainOutcome);
    assert!(
        migrate_tool
            .output
            .as_ref()
            .unwrap()
            .contains("Worker restart occurred")
    );
}

#[test]
fn test_bounded_model_memory_limits_and_explicit_termination() {
    let mut cache = BoundedSessionCache {
        max_tokens_retained: 100,
        current_token_count: 0,
        ssm_step_count: 0,
        last_compacted_ms: 0,
    };

    cache.update_usage(50, 100);
    assert_eq!(cache.current_token_count, 50);
    assert_eq!(cache.ssm_step_count, 100);

    // Update exceeding max capacity is clamped
    cache.update_usage(500, 200);
    assert_eq!(cache.current_token_count, 100);
    assert_eq!(cache.ssm_step_count, 300);

    let manager = SessionManager::new(2);
    manager.get_or_create_session("sess_to_end", 4000);

    manager
        .start_task("sess_to_end", "task_term", "Generate report", 4005)
        .expect("start");

    // Terminate session
    manager.terminate_session("sess_to_end").expect("terminate");

    let rep = manager.inspect_session("sess_to_end").expect("inspect");
    assert!(rep.is_terminated);
    assert_eq!(rep.active_objective, None);

    // Cannot start new task on terminated session
    let start_res = manager.start_task("sess_to_end", "task_fail", "Another task", 4010);
    assert!(start_res.is_err());
}
