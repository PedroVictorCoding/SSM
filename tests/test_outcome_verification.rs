//! Integration tests for Outcome Verification and Multi-Step Task Coordination.
//!
//! Validates:
//! - A tool reporting success without the requested postcondition CANNOT mark the task complete.
//! - User corrections dynamically cancel obsolete queued steps before execution.
//! - Failure and timeout fixtures never produce false spoken success and prevent duplicate non-idempotent actions.
//! - The editor-error/test scenario produces end-to-end traceable evidence connecting request, tools, observations, and speech.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use sail::capability::{Capability, CapabilityPolicy};
use sail::runtime::verification::{
    MultiStepTaskCoordinator, StepExecutionStatus, StepPostcondition, TaskPlan, TaskPlanStep,
};
use sail::tool::broker::{
    BrokerExecutionStatus, ToolExecutionBroker, ToolExecutionContext, ToolExecutionOutput,
    ToolProposal, TypedTool,
};
use sail::tool::schema::{ParameterType, ToolParameter, ToolSafetyLevel, ToolSchema};
use sail::tool::workspace::{
    WorkspaceFileReadTool, WorkspaceFileWriteTool, WorkspaceTestRunnerTool,
};

#[test]
fn test_test_passed_requires_matching_complete_nonempty_evidence() {
    let postcondition = StepPostcondition::TestPassed {
        test_name: "requested_test".into(),
    };
    let valid = json!({
        "tool_name": "workspace_test_runner",
        "success": true,
        "speech_summary": "One test passed",
        "detailed_payload": {
            "test_name": "requested_test", "passed": 1, "failed": 0, "exit_code": 0
        },
        "warnings": []
    });
    let verify = |artifact: &serde_json::Value| {
        postcondition.verify(std::path::Path::new("."), None, Some(&artifact.to_string()))
    };
    assert!(verify(&valid).is_ok());
    for (field, replacement) in [
        ("test_name", json!("different_test")),
        ("passed", json!(0)),
        ("passed", json!(-1)),
        ("passed", json!("1")),
        ("failed", json!(1)),
        ("failed", json!("0")),
        ("exit_code", json!(1)),
        ("exit_code", json!(null)),
    ] {
        let mut artifact = valid.clone();
        artifact["detailed_payload"][field] = replacement;
        assert!(verify(&artifact).is_err(), "accepted invalid {field}");
    }
    for field in ["passed", "failed", "exit_code", "test_name"] {
        let mut artifact = valid.clone();
        artifact["detailed_payload"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(verify(&artifact).is_err(), "accepted missing {field}");
    }
    let mut unrelated = valid.clone();
    unrelated["tool_name"] = json!("unrelated_tool");
    assert!(verify(&unrelated).is_err());
    let mut unsuccessful = valid.clone();
    unsuccessful["success"] = json!(false);
    assert!(verify(&unsuccessful).is_err());
}

/// Mock tool simulating a faulty tool that claims success (exit_code 0)
/// without actually performing the requested disk mutation.
struct FaultyNoOpWriteTool {
    schema: ToolSchema,
}

impl FaultyNoOpWriteTool {
    fn new() -> Self {
        let schema = ToolSchema::new(
            "faulty_writer",
            "1.0.0",
            "Simulates a faulty tool reporting success without writing",
            ToolSafetyLevel::DestructiveOrExternal,
            false,
        )
        .with_parameter(ToolParameter {
            name: "path".into(),
            param_type: ParameterType::String,
            description: "Target path".into(),
            required: true,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "content".into(),
            param_type: ParameterType::String,
            description: "Content that tool falsely claims to write".into(),
            required: true,
            enum_values: None,
        });

        Self { schema }
    }
}

impl TypedTool for FaultyNoOpWriteTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn execute(&self, _args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        // Falsely reports success and committed side effects, but does NOTHING to disk!
        Ok(ToolExecutionOutput {
            result_text: "{\"status\": \"ok\", \"written\": true}".into(),
            exit_code: 0,
            side_effects_committed: true,
        })
    }
}

#[test]
fn test_tool_success_without_postcondition_rejects_task_completion() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("faulty_writer".into()));
    let policy = CapabilityPolicy::new(granted, vec![], vec![]);

    let broker = Arc::new(ToolExecutionBroker::new(policy));
    broker.register_tool(Arc::new(FaultyNoOpWriteTool::new()));

    let temp_dir = std::env::temp_dir();
    let target_file = temp_dir.join(format!("sail_faulty_test_{}.rs", rand::random::<u64>()));
    // Write initial state without the fix
    fs::write(&target_file, b"fn buggy_func() { panic!(); }").unwrap();

    let coordinator = MultiStepTaskCoordinator::new(broker, temp_dir.clone());

    let proposal = ToolProposal {
        call_id: "step_fix".into(),
        task_id: "task_verify_01".into(),
        tool_name: "faulty_writer".into(),
        arguments: json!({
            "path": target_file.to_str().unwrap(),
            "content": "fn buggy_func() { /* fixed */ }"
        }),
        idempotency_key: None,
        proposed_at_ms: 1000,
    };

    let postcondition = StepPostcondition::FileContainsText {
        path: target_file.clone(),
        expected_snippet: "/* fixed */".into(),
    };

    let step = TaskPlanStep {
        step_id: "s1".into(),
        description: "Apply bug fix to target file".into(),
        proposal,
        postconditions: vec![postcondition],
        status: StepExecutionStatus::Pending,
        verified_observation: None,
    };

    let mut plan = TaskPlan::new("task_verify_01", "Fix panic in buggy_func").with_step(step);

    let ctx = ToolExecutionContext {
        interactive_confirmation_granted: true,
        ..Default::default()
    };

    let trace = coordinator.execute_plan(&mut plan, None, &ctx);

    // CRITICAL ACCEPTANCE CRITERIA:
    // 1. Tool execution reported exit_code 0, BUT task cannot be marked complete because postcondition failed!
    assert!(
        !trace.all_postconditions_verified,
        "Task must NOT be verified complete when postcondition fails"
    );
    assert_eq!(trace.completed_steps_count, 0);
    assert_eq!(trace.failed_steps_count, 1);

    // 2. Step status explicitly reports VerificationFailed
    let failed_step = &trace.step_traces[0];
    match &failed_step.status {
        StepExecutionStatus::VerificationFailed(err) => {
            assert!(err.contains("does not contain expected snippet '/* fixed */'"));
        }
        other => panic!("Expected VerificationFailed status, got {:?}", other),
    }

    // 3. Spoken status tells user the truth instead of hallucinating success
    assert!(
        trace
            .spoken_status
            .contains("failed postcondition verification")
    );
    assert!(
        trace
            .spoken_status
            .contains("Desired outcome was not observed on disk")
    );

    // Clean up
    let _ = fs::remove_file(target_file);
}

#[test]
fn test_user_corrections_cancel_obsolete_queued_steps() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_file_read".into()));
    granted.insert(Capability::ToolExecution("workspace_file_write".into()));
    let policy = CapabilityPolicy::new(granted, vec![PathBuf::from(".")], vec![PathBuf::from(".")]);

    let broker = Arc::new(ToolExecutionBroker::new(policy));
    broker.register_tool(Arc::new(WorkspaceFileReadTool::new(vec![PathBuf::from(
        ".",
    )])));
    broker.register_tool(Arc::new(WorkspaceFileWriteTool::new(vec![PathBuf::from(
        ".",
    )])));

    let coordinator = MultiStepTaskCoordinator::new(broker, PathBuf::from("."));

    // Plan with 2 steps: Step 1 (Read Cargo.toml), Step 2 (Write obsolete file)
    let step1 = TaskPlanStep {
        step_id: "step_read".into(),
        description: "Read Cargo.toml".into(),
        proposal: ToolProposal {
            call_id: "c1".into(),
            task_id: "task_corr".into(),
            tool_name: "workspace_file_read".into(),
            arguments: json!({ "path": "Cargo.toml", "start_line": 1, "end_line": 3 }),
            idempotency_key: None,
            proposed_at_ms: 2000,
        },
        postconditions: vec![],
        status: StepExecutionStatus::Pending,
        verified_observation: None,
    };

    let step2 = TaskPlanStep {
        step_id: "step_write_obsolete".into(),
        description: "Write obsolete config".into(),
        proposal: ToolProposal {
            call_id: "c2".into(),
            task_id: "task_corr".into(),
            tool_name: "workspace_file_write".into(),
            arguments: json!({ "path": "/tmp/obsolete.txt", "content": "obsolete" }),
            idempotency_key: None,
            proposed_at_ms: 2010,
        },
        postconditions: vec![],
        status: StepExecutionStatus::Pending,
        verified_observation: None,
    };

    let mut plan = TaskPlan::new("task_corr", "Read Cargo and write config")
        .with_step(step1)
        .with_step(step2);

    let ctx = ToolExecutionContext {
        interactive_confirmation_granted: true,
        ..Default::default()
    };

    // Step 1 executes successfully
    let step1_ok = coordinator.execute_step(&mut plan.steps[0], None, &ctx);
    assert!(step1_ok);
    assert_eq!(plan.steps[0].status, StepExecutionStatus::VerifiedComplete);

    // User barges in BEFORE step 2 runs:
    // "Wait, don't write that config file!"
    plan.revise_objective(
        "Only inspect Cargo.toml without writing",
        "User barged in and cancelled write action",
    );

    let trace = coordinator.execute_plan(&mut plan, None, &ctx);

    // Step 2 was marked obsolete and NEVER executed
    assert_eq!(trace.completed_steps_count, 1);
    assert_eq!(trace.obsolete_steps_count, 1);
    match &trace.step_traces[1].status {
        StepExecutionStatus::SkippedObsolete(reason) => {
            assert!(reason.contains("User barged in and cancelled write action"));
        }
        other => panic!("Expected SkippedObsolete for step 2, got {:?}", other),
    }
}

#[test]
fn test_failure_or_timeout_avoids_false_spoken_success() {
    let temp_dir = std::env::temp_dir();
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_file_write".into()));
    let policy = CapabilityPolicy::new(granted, vec![temp_dir.clone()], vec![temp_dir.clone()]);

    let broker = Arc::new(ToolExecutionBroker::new(policy));
    broker.register_tool(Arc::new(WorkspaceFileWriteTool::new(vec![
        temp_dir.clone(),
    ])));

    let coordinator = MultiStepTaskCoordinator::new(broker, std::env::temp_dir());

    let proposal = ToolProposal {
        call_id: "call_timeout".into(),
        task_id: "task_timeout".into(),
        tool_name: "workspace_file_write".into(),
        arguments: json!({
            "path": "/tmp/test_timeout.txt",
            "content": "payload"
        }),
        idempotency_key: Some("idem_timeout_guard".into()),
        proposed_at_ms: 3000,
    };

    let step = TaskPlanStep {
        step_id: "s_time".into(),
        description: "Timed out write".into(),
        proposal,
        postconditions: vec![],
        status: StepExecutionStatus::Pending,
        verified_observation: None,
    };

    let mut plan = TaskPlan::new("task_timeout", "Perform write").with_step(step);

    // Context with expired deadline
    let ctx = ToolExecutionContext {
        deadline_ms: Some(2500),
        current_timestamp_ms: 3000,
        interactive_confirmation_granted: true,
        ..Default::default()
    };

    let trace = coordinator.execute_plan(&mut plan, None, &ctx);

    assert!(!trace.all_postconditions_verified);
    assert_eq!(trace.failed_steps_count, 1);
    assert!(trace.spoken_status.contains("Task stopped with errors"));

    // Verify step was classified as UncertainOutcome preventing blind retries
    match &trace.step_traces[0].status {
        StepExecutionStatus::UncertainOutcome(msg) => {
            assert!(msg.contains("blocked from blind retry"));
        }
        other => panic!("Expected UncertainOutcome, got {:?}", other),
    }
}

#[test]
fn test_editor_error_and_test_scenario_traceability() {
    let temp_dir = std::env::temp_dir();
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_file_read".into()));
    granted.insert(Capability::ToolExecution("workspace_file_write".into()));
    let policy = CapabilityPolicy::new(
        granted,
        vec![PathBuf::from("."), temp_dir.clone()],
        vec![temp_dir.clone()],
    );

    let broker = Arc::new(ToolExecutionBroker::new(policy));
    broker.register_tool(Arc::new(WorkspaceFileReadTool::new(vec![
        PathBuf::from("."),
        temp_dir.clone(),
    ])));
    broker.register_tool(Arc::new(WorkspaceFileWriteTool::new(vec![
        temp_dir.clone(),
    ])));

    let code_file = temp_dir.join(format!("sail_verified_edit_{}.rs", rand::random::<u64>()));
    fs::write(
        &code_file,
        b"pub fn calculate_sum(a: i32, b: i32) -> i32 { a - b }",
    )
    .unwrap();

    let coordinator = MultiStepTaskCoordinator::new(broker, temp_dir.clone());

    // Step 1: Read the buggy file
    let step1 = TaskPlanStep {
        step_id: "step_read_error".into(),
        description: "Read code around subtraction bug".into(),
        proposal: ToolProposal {
            call_id: "c_read".into(),
            task_id: "task_editor_fix".into(),
            tool_name: "workspace_file_read".into(),
            arguments: json!({
                "path": code_file.to_str().unwrap(),
                "start_line": 1,
                "end_line": 1
            }),
            idempotency_key: None,
            proposed_at_ms: 4000,
        },
        postconditions: vec![StepPostcondition::FileExists {
            path: code_file.clone(),
        }],
        status: StepExecutionStatus::Pending,
        verified_observation: None,
    };

    // Step 2: Apply the correction (a + b)
    let step2 = TaskPlanStep {
        step_id: "step_write_fix".into(),
        description: "Apply addition correction to calculate_sum".into(),
        proposal: ToolProposal {
            call_id: "c_write".into(),
            task_id: "task_editor_fix".into(),
            tool_name: "workspace_file_write".into(),
            arguments: json!({
                "path": code_file.to_str().unwrap(),
                "content": "pub fn calculate_sum(a: i32, b: i32) -> i32 { a + b }"
            }),
            idempotency_key: None,
            proposed_at_ms: 4010,
        },
        postconditions: vec![StepPostcondition::FileContainsText {
            path: code_file.clone(),
            expected_snippet: "a + b".into(),
        }],
        status: StepExecutionStatus::Pending,
        verified_observation: None,
    };

    let mut plan = TaskPlan::new("task_editor_fix", "Fix calculate_sum operator in editor")
        .with_step(step1)
        .with_step(step2);

    let ctx = ToolExecutionContext {
        interactive_confirmation_granted: true,
        ..Default::default()
    };

    let trace = coordinator.execute_plan(&mut plan, None, &ctx);

    // End-to-end verification
    assert!(trace.all_postconditions_verified);
    assert_eq!(trace.completed_steps_count, 2);
    assert_eq!(trace.failed_steps_count, 0);

    // Check step 1 & 2 verified
    assert_eq!(
        trace.step_traces[0].status,
        StepExecutionStatus::VerifiedComplete
    );
    assert_eq!(
        trace.step_traces[1].status,
        StepExecutionStatus::VerifiedComplete
    );

    // Verify disk content actually changed
    let final_content = fs::read_to_string(&code_file).unwrap();
    assert!(final_content.contains("a + b"));

    // Verify spoken status confirms verified completion
    assert!(trace.spoken_status.contains("Task completed and verified"));
    assert!(trace.spoken_status.contains("all 2 steps"));

    // Clean up
    let _ = fs::remove_file(code_file);
}

#[test]
fn test_qa10_real_cargo_test_broker_to_postcondition_end_to_end() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_test_runner".into()));
    let policy = CapabilityPolicy::new(granted, vec![PathBuf::from(".")], vec![]);

    let broker = ToolExecutionBroker::new(policy);
    broker.register_tool(Arc::new(WorkspaceTestRunnerTool::new(PathBuf::from("."))));
    // Standing authorization required since test execution is classified as DestructiveOrExternal
    broker.grant_standing_authorization("workspace_test_runner");

    let ctx = ToolExecutionContext::default();

    // 1. Authentic matching test invocation: test_alignment_document_preserved_verbatim
    let prop_match = ToolProposal {
        call_id: "call_real_test_match".into(),
        task_id: "task_qa10".into(),
        tool_name: "workspace_test_runner".into(),
        arguments: json!({
            "test_name": "test_alignment_document_preserved_verbatim",
            "package": "sail"
        }),
        idempotency_key: None,
        proposed_at_ms: 100,
    };

    let res_match = broker.execute_proposal(&prop_match, &ctx);
    assert_eq!(res_match.status, BrokerExecutionStatus::Success);
    assert_eq!(res_match.exit_code, 0);
    assert!(
        res_match.side_effects_committed,
        "Test execution must commit side effects"
    );

    let postcondition_match = StepPostcondition::TestPassed {
        test_name: "test_alignment_document_preserved_verbatim".into(),
    };
    let verify_res =
        postcondition_match.verify(std::path::Path::new("."), None, res_match.output.as_deref());
    assert!(
        verify_res.is_ok(),
        "Authentic passing test artifact must satisfy TestPassed postcondition: {:?}",
        verify_res
    );

    // 2. Unmatched test filter: returns 0 passing tests
    let prop_unmatched = ToolProposal {
        call_id: "call_real_test_unmatched".into(),
        task_id: "task_qa10".into(),
        tool_name: "workspace_test_runner".into(),
        arguments: json!({
            "test_name": "nonexistent_test_filter_xyz_99999",
            "package": "sail"
        }),
        idempotency_key: None,
        proposed_at_ms: 200,
    };

    let res_unmatched = broker.execute_proposal(&prop_unmatched, &ctx);
    assert_eq!(res_unmatched.status, BrokerExecutionStatus::Success);
    // Exit code is 0 from Cargo, but passed count is 0 and artifact success is false
    let postcondition_unmatched = StepPostcondition::TestPassed {
        test_name: "nonexistent_test_filter_xyz_99999".into(),
    };
    let verify_unmatched = postcondition_unmatched.verify(
        std::path::Path::new("."),
        None,
        res_unmatched.output.as_deref(),
    );
    assert!(
        verify_unmatched.is_err(),
        "Zero matching tests must be rejected by TestPassed postcondition"
    );
    let err_msg = verify_unmatched.unwrap_err();
    assert!(
        err_msg.contains("reported failure in test artifact")
            || err_msg.contains("No passing tests"),
        "Error message should explain rejection: {}",
        err_msg
    );

    // 3. Multi-target aggregation check: filter 'test_bpe' matches test_bpe and test_bpe_train
    let prop_multi = ToolProposal {
        call_id: "call_real_test_multi".into(),
        task_id: "task_qa10".into(),
        tool_name: "workspace_test_runner".into(),
        arguments: json!({
            "test_name": "test_bpe",
            "package": "sail"
        }),
        idempotency_key: None,
        proposed_at_ms: 300,
    };

    let res_multi = broker.execute_proposal(&prop_multi, &ctx);
    assert_eq!(res_multi.status, BrokerExecutionStatus::Success);
    assert_eq!(res_multi.exit_code, 0);
    let output_multi_str = res_multi.output.expect("multi target output");
    assert!(output_multi_str.contains("target_runs"));
    let postcondition_multi = StepPostcondition::TestPassed {
        test_name: "test_bpe".into(),
    };
    assert!(
        postcondition_multi
            .verify(std::path::Path::new("."), None, Some(&output_multi_str))
            .is_ok()
    );

    // 4. Cancellation check: cancelled test produces no output and rejects postcondition
    let cancel_token = sail::runtime::CancellationToken::new();
    cancel_token.cancel();
    let ctx_cancel = ToolExecutionContext {
        cancellation_token: Some(cancel_token),
        ..Default::default()
    };
    let res_cancel = broker.execute_proposal(&prop_match, &ctx_cancel);
    assert_eq!(res_cancel.status, BrokerExecutionStatus::Cancelled);
    let verify_cancelled = postcondition_match.verify(
        std::path::Path::new("."),
        None,
        res_cancel.output.as_deref(),
    );
    assert!(
        verify_cancelled.is_err(),
        "Cancelled execution cannot satisfy postcondition"
    );
}
