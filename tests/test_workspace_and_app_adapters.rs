//! Integration tests for Workspace, Application, and Browser Tool Adapters.
//!
//! Validates:
//! - Inspecting workspace files and running targeted tests from visible error scenarios.
//! - Path traversal rejection and stale file version concurrency protections.
//! - Reporting of unsupported applications and invalid URL schemes without silent simulation.
//! - Detection of moved UI controls and DPI scale mismatches before executing actions.
//! - Strict capability boundaries and confirmation gates for write actions.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use sail::capability::{Capability, CapabilityPolicy};
use sail::runtime::CancellationToken;
use sail::screen::automation::{Rect, UiElement};
use sail::tool::application::{ApplicationControlTool, BrowserInteractionTool};
use sail::tool::broker::{
    BrokerExecutionStatus, ToolExecutionBroker, ToolExecutionContext, ToolProposal,
};
use sail::tool::workspace::{
    WorkspaceFileReadTool, WorkspaceFileSearchTool, WorkspaceFileWriteTool, WorkspaceTestRunnerTool,
};

#[test]
fn test_visible_error_code_inspection_and_test_runner() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_file_read".into()));
    granted.insert(Capability::ToolExecution("workspace_test_runner".into()));
    granted.insert(Capability::ToolExecution("workspace_file_search".into()));
    let policy = CapabilityPolicy::new(granted, vec![PathBuf::from(".")], vec![]);

    let broker = ToolExecutionBroker::new(policy);
    broker.register_tool(Arc::new(WorkspaceFileReadTool::new(vec![
        PathBuf::from(".gen_roots"),
        PathBuf::from("."),
    ])));
    broker.register_tool(Arc::new(WorkspaceTestRunnerTool::new(PathBuf::from("."))));
    broker.grant_standing_authorization("workspace_test_runner");

    // 1. Read lines from Cargo.toml
    let prop_read = ToolProposal {
        call_id: "read_cargo".into(),
        task_id: "task_err_fix".into(),
        tool_name: "workspace_file_read".into(),
        arguments: json!({
            "path": "Cargo.toml",
            "start_line": 1,
            "end_line": 5
        }),
        idempotency_key: None,
        proposed_at_ms: 1000,
    };

    let res_read = broker.execute_proposal(&prop_read, &ToolExecutionContext::default());
    assert_eq!(res_read.status, BrokerExecutionStatus::Success);
    let output = res_read.output.unwrap();
    assert!(output.contains("[package]"));
    assert!(output.contains("Read lines 1-5"));

    // 2. Run targeted test: test_bpe_save_and_load
    let prop_test = ToolProposal {
        call_id: "run_bpe_test".into(),
        task_id: "task_err_fix".into(),
        tool_name: "workspace_test_runner".into(),
        arguments: json!({
            "test_name": "test_bpe_save_and_load",
            "release": false
        }),
        idempotency_key: None,
        proposed_at_ms: 1010,
    };

    let res_test = broker.execute_proposal(&prop_test, &ToolExecutionContext::default());
    assert_eq!(res_test.status, BrokerExecutionStatus::Success);
    assert_eq!(res_test.exit_code, 0);
    let test_out = res_test.output.unwrap();
    assert!(test_out.contains("passed"));
    assert!(test_out.contains("test_bpe_save_and_load"));

    // 3. Search files using WorkspaceFileSearchTool
    let searcher = Arc::new(WorkspaceFileSearchTool::new(vec![PathBuf::from(".")]));
    broker.register_tool(searcher);

    let prop_search = ToolProposal {
        call_id: "search_src".into(),
        task_id: "task_err_fix".into(),
        tool_name: "workspace_file_search".into(),
        arguments: json!({
            "query": "lib.rs",
            "root_dir": "src"
        }),
        idempotency_key: None,
        proposed_at_ms: 1020,
    };

    let res_search = broker.execute_proposal(&prop_search, &ToolExecutionContext::default());
    assert_eq!(res_search.status, BrokerExecutionStatus::Success);
    assert!(res_search.output.unwrap().contains("lib.rs"));
}

#[test]
fn test_path_traversal_and_stale_file_version_prevention() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_file_read".into()));
    granted.insert(Capability::ToolExecution("workspace_file_write".into()));
    granted.insert(Capability::ToolExecution("workspace_file_search".into()));
    let temp_dir = std::env::temp_dir();
    let policy = CapabilityPolicy::new(
        granted,
        vec![PathBuf::from("."), temp_dir.clone()],
        vec![PathBuf::from("."), temp_dir.clone()],
    );

    let broker = ToolExecutionBroker::new(policy);
    broker.register_tool(Arc::new(WorkspaceFileReadTool::new(vec![
        PathBuf::from("."),
        temp_dir.clone(),
    ])));
    broker.register_tool(Arc::new(WorkspaceFileWriteTool::new(vec![
        PathBuf::from("."),
        temp_dir.clone(),
    ])));

    // 1. Path traversal attempt outside workspace root
    let prop_traversal = ToolProposal {
        call_id: "read_passwd".into(),
        task_id: "task_sec".into(),
        tool_name: "workspace_file_read".into(),
        arguments: json!({
            "path": "/etc/passwd"
        }),
        idempotency_key: None,
        proposed_at_ms: 2000,
    };
    let res_trav = broker.execute_proposal(&prop_traversal, &ToolExecutionContext::default());
    match res_trav.status {
        BrokerExecutionStatus::ExecutionFailed(msg) | BrokerExecutionStatus::Unauthorized(msg) => {
            assert!(
                msg.contains("Access denied")
                    || msg.contains("Capability Not Granted")
                    || msg.contains("not within configured read roots")
            );
        }
        other => panic!(
            "Expected ExecutionFailed or Unauthorized for traversal, got {:?}",
            other
        ),
    }

    // 2. Concurrency StaleFileVersion protection on write
    let temp_file =
        std::env::temp_dir().join(format!("sail_stale_test_{}.txt", rand::random::<u64>()));
    fs::write(&temp_file, b"initial content").unwrap();

    let writer = Arc::new(WorkspaceFileWriteTool::new(vec![std::env::temp_dir()]));
    broker.register_tool(writer);

    let prop_stale_write = ToolProposal {
        call_id: "write_stale".into(),
        task_id: "task_sec".into(),
        tool_name: "workspace_file_write".into(),
        arguments: json!({
            "path": temp_file.to_str().unwrap(),
            "content": "new conflicting content",
            "expected_mtime_ms": 100 // Old mtime long in the past
        }),
        idempotency_key: None,
        proposed_at_ms: 2010,
    };

    let ctx = ToolExecutionContext {
        interactive_confirmation_granted: true,
        ..Default::default()
    };
    let res_stale = broker.execute_proposal(&prop_stale_write, &ctx);
    match res_stale.status {
        BrokerExecutionStatus::ExecutionFailed(msg) => {
            assert!(msg.contains("StaleFileVersion"));
        }
        other => panic!("Expected StaleFileVersion rejection, got {:?}", other),
    }

    // Clean up
    let _ = fs::remove_file(temp_file);
}

#[test]
fn test_unsupported_application_and_invalid_url_reporting() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("application_control".into()));
    granted.insert(Capability::ToolExecution("browser_interaction".into()));
    let policy = CapabilityPolicy::new(granted, vec![], vec![]);

    let broker = ToolExecutionBroker::new(policy);
    broker.register_tool(Arc::new(ApplicationControlTool::new()));
    broker.register_tool(Arc::new(BrowserInteractionTool::new()));

    let ctx = ToolExecutionContext {
        interactive_confirmation_granted: true,
        ..Default::default()
    };

    // 1. Unsupported application
    let prop_unsupported = ToolProposal {
        call_id: "app_gimp".into(),
        task_id: "task_app".into(),
        tool_name: "application_control".into(),
        arguments: json!({
            "app_name": "gimp_image_editor",
            "action": "FocusWindow"
        }),
        idempotency_key: None,
        proposed_at_ms: 3000,
    };
    let res_app = broker.execute_proposal(&prop_unsupported, &ctx);
    match res_app.status {
        BrokerExecutionStatus::ExecutionFailed(msg) => {
            assert!(msg.contains("UnsupportedApplication"));
        }
        other => panic!("Expected UnsupportedApplication error, got {:?}", other),
    }

    // 2. Dangerous browser URL scheme (file://)
    let prop_bad_url = ToolProposal {
        call_id: "nav_shadow".into(),
        task_id: "task_app".into(),
        tool_name: "browser_interaction".into(),
        arguments: json!({
            "action": "Navigate",
            "url": "file:///etc/shadow"
        }),
        idempotency_key: None,
        proposed_at_ms: 3010,
    };
    let res_url = broker.execute_proposal(&prop_bad_url, &ctx);
    match res_url.status {
        BrokerExecutionStatus::ExecutionFailed(msg) => {
            assert!(msg.contains("InvalidUrlScheme"));
        }
        other => panic!("Expected InvalidUrlScheme, got {:?}", other),
    }
}

#[test]
fn test_moved_control_and_dpi_change_detection() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("application_control".into()));
    let policy = CapabilityPolicy::new(granted, vec![], vec![]);

    let broker = ToolExecutionBroker::new(policy);
    let app_tool = Arc::new(ApplicationControlTool::new());

    // Setup active UI tree with a button at [100, 200, 80, 30]
    let ui_root = UiElement {
        element_id: "window_root".into(),
        name: "Main Editor Window".into(),
        role: "Window".into(),
        bounds: Rect::new(0, 0, 1920, 1080),
        text_value: None,
        supported_actions: vec!["Focus".into()],
        is_focused: true,
        is_enabled: true,
        children: vec![UiElement {
            element_id: "btn_save".into(),
            name: "Save".into(),
            role: "Button".into(),
            bounds: Rect::new(100, 200, 80, 30),
            text_value: Some("Save".into()),
            supported_actions: vec!["Click".into()],
            is_focused: false,
            is_enabled: true,
            children: Vec::new(),
        }],
    };
    app_tool.set_active_ui_root(Some(ui_root));
    broker.register_tool(app_tool);

    let ctx = ToolExecutionContext {
        interactive_confirmation_granted: true,
        ..Default::default()
    };

    // 1. Target control moved from expected position
    let prop_moved = ToolProposal {
        call_id: "click_moved".into(),
        task_id: "task_ui".into(),
        tool_name: "application_control".into(),
        arguments: json!({
            "app_name": "editor",
            "action": "ClickElement",
            "target_element_id": "btn_save",
            "expected_bounds": {
                "x": 300, // Mismatched expectation!
                "y": 400,
                "width": 80,
                "height": 30
            }
        }),
        idempotency_key: None,
        proposed_at_ms: 4000,
    };
    let res_moved = broker.execute_proposal(&prop_moved, &ctx);
    match res_moved.status {
        BrokerExecutionStatus::ExecutionFailed(msg) => {
            assert!(msg.contains("ControlMovedError"));
        }
        other => panic!("Expected ControlMovedError, got {:?}", other),
    }

    // 2. Display scale factor changed (e.g. window moved to 4K 1.5x monitor)
    let prop_dpi = ToolProposal {
        call_id: "click_dpi".into(),
        task_id: "task_ui".into(),
        tool_name: "application_control".into(),
        arguments: json!({
            "app_name": "editor",
            "action": "ClickElement",
            "target_element_id": "btn_save",
            "expected_scale_factor": 1.5 // Active display is 1.0
        }),
        idempotency_key: None,
        proposed_at_ms: 4010,
    };
    let res_dpi = broker.execute_proposal(&prop_dpi, &ctx);
    match res_dpi.status {
        BrokerExecutionStatus::ExecutionFailed(msg) => {
            assert!(msg.contains("DpiMismatchError"));
        }
        other => panic!("Expected DpiMismatchError, got {:?}", other),
    }

    // 3. Perfect match: coordinates and scale match -> Success
    let prop_ok = ToolProposal {
        call_id: "click_ok".into(),
        task_id: "task_ui".into(),
        tool_name: "application_control".into(),
        arguments: json!({
            "app_name": "editor",
            "action": "ClickElement",
            "target_element_id": "btn_save",
            "expected_bounds": {
                "x": 100,
                "y": 200,
                "width": 80,
                "height": 30
            },
            "expected_scale_factor": 1.0
        }),
        idempotency_key: None,
        proposed_at_ms: 4020,
    };
    let res_ok = broker.execute_proposal(&prop_ok, &ctx);
    assert_eq!(res_ok.status, BrokerExecutionStatus::Success);
    assert!(
        res_ok
            .output
            .unwrap()
            .contains("Clicked UI element 'btn_save'")
    );
}

#[test]
fn test_workspace_test_runner_supervision_and_cancellation() {
    let mut granted = HashSet::new();
    granted.insert(Capability::ToolExecution("workspace_test_runner".into()));
    let policy = CapabilityPolicy::new(granted, vec![PathBuf::from(".")], vec![]);

    let broker = ToolExecutionBroker::new(policy);
    broker.register_tool(Arc::new(WorkspaceTestRunnerTool::new(PathBuf::from("."))));
    broker.grant_standing_authorization("workspace_test_runner");

    // 1. Pre-cancelled execution: must stop before execution begins
    let cancel_token = CancellationToken::new();
    cancel_token.cancel();
    let ctx_cancelled = ToolExecutionContext {
        cancellation_token: Some(cancel_token),
        ..Default::default()
    };

    let prop = ToolProposal {
        call_id: "test_cancel".into(),
        task_id: "task_test".into(),
        tool_name: "workspace_test_runner".into(),
        arguments: json!({
            "test_name": "test_alignment_boundaries",
            "package": "sail"
        }),
        idempotency_key: None,
        proposed_at_ms: 100,
    };

    let res_cancel = broker.execute_proposal(&prop, &ctx_cancelled);
    assert_eq!(res_cancel.status, BrokerExecutionStatus::Cancelled);

    // 2. Ultra-short deadline on test command: process supervisor kills child process
    let ctx_deadline = ToolExecutionContext {
        deadline_ms: Some(1),
        current_timestamp_ms: 0,
        ..Default::default()
    };
    let res_deadline = broker.execute_proposal(&prop, &ctx_deadline);
    assert_eq!(
        res_deadline.status,
        BrokerExecutionStatus::TimedOut,
        "Supervised test runner must return exact TimedOut on deadline expiry"
    );

    // 3. In-flight cancellation: worker spawns test runner, token cancels mid-run, supervisor terminates process tree
    let in_flight_token = CancellationToken::new();
    let ctx_in_flight = ToolExecutionContext {
        cancellation_token: Some(in_flight_token.clone()),
        ..Default::default()
    };
    let broker_arc = Arc::new(broker);
    let prop_clone = prop.clone();
    let runner_thread =
        std::thread::spawn(move || broker_arc.execute_proposal(&prop_clone, &ctx_in_flight));

    // Cancel after slight delay so child starts spawning
    std::thread::sleep(std::time::Duration::from_millis(15));
    in_flight_token.cancel();

    let res_in_flight = runner_thread.join().unwrap();
    assert_eq!(
        res_in_flight.status,
        BrokerExecutionStatus::Cancelled,
        "Supervised test runner must return exact Cancelled status on in-flight cancellation"
    );
}
