//! Deterministic QA probe for in-flight deadline and cancellation behavior.
//!
//! This uses only in-process synchronization and counters. It does not execute
//! external commands, access the network, or mutate the filesystem.

use std::collections::HashSet;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use sail::capability::{Capability, CapabilityPolicy};
use sail::runtime::bus::CancellationToken;
use sail::tool::broker::{
    BrokerExecutionStatus, ToolExecutionBroker, ToolExecutionContext,
    ToolExecutionOutput, ToolProposal, TypedTool,
};
use sail::tool::schema::{ToolSafetyLevel, ToolSchema};
use serde_json::json;

struct DelayTool;

impl TypedTool for DelayTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "delay_fixture",
            "1.0.0",
            "Side-effect-free deadline fixture",
            ToolSafetyLevel::ReadOnly,
            true,
        )
    }

    fn execute(&self, _args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        thread::sleep(Duration::from_millis(50));
        Ok(ToolExecutionOutput {
            result_text: "finished".into(),
            exit_code: 0,
            side_effects_committed: false,
        })
    }
}

struct GatedTool {
    started: Mutex<Option<mpsc::Sender<()>>>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl TypedTool for GatedTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "gated_fixture",
            "1.0.0",
            "Side-effect-free cancellation fixture",
            ToolSafetyLevel::ReadOnly,
            true,
        )
    }

    fn execute(&self, _args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        if let Some(tx) = self.started.lock().unwrap().take() {
            tx.send(()).unwrap();
        }
        self.release.lock().unwrap().recv().unwrap();
        Ok(ToolExecutionOutput {
            result_text: "released".into(),
            exit_code: 0,
            side_effects_committed: false,
        })
    }
}

fn policy_for(tool: &str) -> CapabilityPolicy {
    let mut grants = HashSet::new();
    grants.insert(Capability::ToolExecution(tool.into()));
    CapabilityPolicy::new(grants, vec![], vec![])
}

fn proposal(tool: &str, call: &str) -> ToolProposal {
    ToolProposal {
        call_id: call.into(),
        task_id: "qa08-probe".into(),
        tool_name: tool.into(),
        arguments: json!({}),
        idempotency_key: None,
        proposed_at_ms: 0,
    }
}

fn main() {
    let deadline_broker = ToolExecutionBroker::new(policy_for("delay_fixture"));
    deadline_broker.register_tool(Arc::new(DelayTool));
    let deadline_context = ToolExecutionContext {
        deadline_ms: Some(5),
        current_timestamp_ms: 0,
        ..Default::default()
    };
    let wall_start = Instant::now();
    let deadline_result = deadline_broker.execute_proposal(
        &proposal("delay_fixture", "deadline"),
        &deadline_context,
    );
    println!(
        "deadline result: {:?}; returned after {}ms",
        deadline_result.status,
        wall_start.elapsed().as_millis()
    );

    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let cancellation_broker = Arc::new(ToolExecutionBroker::new(policy_for("gated_fixture")));
    cancellation_broker.register_tool(Arc::new(GatedTool {
        started: Mutex::new(Some(started_tx)),
        release: Mutex::new(release_rx),
    }));
    let token = CancellationToken::new();
    let worker_token = token.clone();
    let worker_broker = Arc::clone(&cancellation_broker);
    let worker = thread::spawn(move || {
        let context = ToolExecutionContext {
            cancellation_token: Some(worker_token),
            ..Default::default()
        };
        let result = worker_broker.execute_proposal(
            &proposal("gated_fixture", "cancel"),
            &context,
        );
        result_tx.send(result).unwrap();
    });

    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    token.cancel();
    let prompt_result = result_rx.recv_timeout(Duration::from_millis(30));
    println!(
        "result available within 30ms of in-flight cancellation: {}",
        prompt_result.is_ok()
    );
    release_tx.send(()).unwrap();
    let final_result = match prompt_result {
        Ok(result) => result,
        Err(_) => result_rx.recv_timeout(Duration::from_secs(2)).unwrap(),
    };
    println!("result after tool release: {:?}", final_result.status);
    assert!(matches!(deadline_result.status, BrokerExecutionStatus::TimedOut));
    assert!(matches!(final_result.status, BrokerExecutionStatus::Cancelled));
    worker.join().unwrap();
}
