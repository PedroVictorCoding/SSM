//! Disposable-fixture QA probe. Exit 0 prints observations, not a passing safety verdict.
//! Compile against the current sail and matching serde_json rlibs; see the adjacent log.
use sail::capability::{Capability, CapabilityPolicy};
use sail::runtime::CancellationToken;
use sail::runtime::verification::StepPostcondition;
use sail::tool::{ToolExecutionBroker, ToolExecutionContext, ToolProposal, TypedFileDeleteTool,
    TypedFileReadTool, WorkspaceFileWriteTool, TypedTool, ToolSchema, ToolSafetyLevel,
    ToolExecutionOutput};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

struct SlowFixture;
impl TypedTool for SlowFixture {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("slow_fixture", "1", "Disposable QA fixture", ToolSafetyLevel::ReadOnly, true)
    }
    fn execute(&self, _: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        std::thread::sleep(Duration::from_millis(40));
        Ok(ToolExecutionOutput { result_text: "fixture payload".into(), exit_code: 0,
            side_effects_committed: false })
    }
}
fn proposal(name: &str, arguments: serde_json::Value) -> ToolProposal {
    ToolProposal { call_id: "qa-call".into(), task_id: "qa-task".into(), tool_name: name.into(),
        arguments, idempotency_key: None, proposed_at_ms: 0 }
}
fn main() {
    let root=std::env::temp_dir().join(format!("sail-boundary-review-{}",std::process::id()));
    let workspace=root.join("workspace");
    let outside=root.join("outside");
    std::fs::create_dir_all(workspace.join("src")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    let grants=HashSet::from([Capability::ToolExecution("file_read".into()),
        Capability::ToolExecution("file_delete".into()),
        Capability::ToolExecution("workspace_file_write".into())]);
    let policy=CapabilityPolicy::new(grants,vec![workspace.clone()],vec![workspace.clone()]);
    println!("absolute protected source policy check: {:?}",policy.check_file_write(workspace.join("src/fixture.rs")));
    let broker=ToolExecutionBroker::new(policy);
    broker.register_tool(Arc::new(TypedFileReadTool::new()));
    broker.register_tool(Arc::new(TypedFileDeleteTool::new()));
    broker.register_tool(Arc::new(WorkspaceFileWriteTool::new(vec![workspace.clone()])));
    let context=ToolExecutionContext { interactive_confirmation_granted: true, ..Default::default() };
    let marker=outside.join("marker.txt");std::fs::write(&marker,"disposable outside-root marker").unwrap();
    let read=broker.execute_proposal(&proposal("file_read",json!({"path":marker})),&context);
    println!("out-of-root read: {:?}, marker returned: {}",read.status,read.output.as_deref()==Some("disposable outside-root marker"));
    let deleted=broker.execute_proposal(&proposal("file_delete",json!({"path":marker})),&context);
    println!("out-of-root deletion: {:?}, file removed: {}",deleted.status,!marker.exists());
    let protected=workspace.join("ALIGNMENT.md");std::fs::write(&protected,"disposable sentinel").unwrap();
    let written=broker.execute_proposal(&proposal("workspace_file_write",json!({"path":protected,"content":"replaced fixture"})),&context);
    println!("protected-name workspace write: {:?}, sentinel replaced: {}",written.status,std::fs::read_to_string(protected).unwrap()=="replaced fixture");

    let cache_broker=ToolExecutionBroker::new(CapabilityPolicy::new(HashSet::from([
        Capability::ToolExecution("slow_fixture".into())]),vec![],vec![]));
    cache_broker.register_tool(Arc::new(SlowFixture));
    let mut request=proposal("slow_fixture",json!({}));request.idempotency_key=Some("shared-key".into());
    let first=cache_broker.execute_proposal(&request,&Default::default());
    assert!(first.output.is_some());
    cache_broker.set_policy(CapabilityPolicy::new(HashSet::new(),vec![],vec![]));
    let cancelled=CancellationToken::new();cancelled.cancel();
    request.task_id="different-task".into();request.arguments=json!({"unexpected":"must be rejected"});
    let revoked=cache_broker.execute_proposal(&request,&ToolExecutionContext {
        cancellation_token: Some(cancelled),max_output_bytes: 1,..Default::default() });
    println!("revoked+malformed+cancelled cache request: {:?}, from_cache: {}, returned bytes: {}",revoked.status,revoked.from_cache,revoked.output.as_ref().map_or(0,String::len));
    cache_broker.set_policy(CapabilityPolicy::new(HashSet::from([
        Capability::ToolExecution("slow_fixture".into())]),vec![],vec![]));
    let started=Instant::now();
    let late=cache_broker.execute_proposal(&proposal("slow_fixture",json!({})),&ToolExecutionContext {
        current_timestamp_ms: 0, deadline_ms: Some(1),..Default::default() });
    println!("1ms deadline on 40ms fixture: {:?}, elapsed ms: {}",late.status,started.elapsed().as_millis());
    let fake_artifact=json!({"tool_name":"unrelated_tool","success":true,"speech_summary":"not a test run","detailed_payload":{},"warnings":[]}).to_string();
    println!("missing test evidence postcondition: {:?}",StepPostcondition::TestPassed {test_name:"never-executed-test".into()}.verify(&workspace,None,Some(&fake_artifact)));
    std::fs::remove_dir_all(root).unwrap();
}
