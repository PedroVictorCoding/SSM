//! Deterministic disposable counter effects; no external commands, files, or network.
use sail::capability::{Capability, CapabilityPolicy};
use sail::tool::{ToolExecutionBroker, ToolExecutionContext, ToolProposal, TypedTool,
    ToolSchema, ToolSafetyLevel, ToolExecutionOutput};
use serde_json::json;
use std::collections::HashSet;
use std::sync::{Arc, Mutex, mpsc, atomic::{AtomicUsize, Ordering}};
use std::time::Duration;
struct BlockingFixture {
    calls: Arc<AtomicUsize>,
    started: mpsc::Sender<()>,
    release: Mutex<mpsc::Receiver<()>>,
}
impl TypedTool for BlockingFixture {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new("blocked_fixture","1","counter-only fixture",ToolSafetyLevel::ReversibleWrite,false)
    }
    fn execute(&self,_: &serde_json::Value)->Result<ToolExecutionOutput,String> {
        let n=self.calls.fetch_add(1,Ordering::SeqCst);
        if n==0 {self.started.send(()).unwrap();self.release.lock().unwrap().recv_timeout(Duration::from_secs(10)).unwrap();}
        Ok(ToolExecutionOutput{result_text:"counter effect".into(),exit_code:0,side_effects_committed:true})
    }
}
struct EchoFixture {version: &'static str}
impl TypedTool for EchoFixture {
    fn schema(&self)->ToolSchema {ToolSchema::new("echo_fixture",self.version,"version fixture",ToolSafetyLevel::ReadOnly,true)}
    fn execute(&self,_:&serde_json::Value)->Result<ToolExecutionOutput,String> {
        Ok(ToolExecutionOutput{result_text:format!("output from {}",self.version),exit_code:0,side_effects_committed:false})
    }
}
fn proposal(tool:&str,key:String,time:u64)->ToolProposal {
    ToolProposal {call_id:format!("call-{key}"),task_id:"qa-task".into(),tool_name:tool.into(),arguments:json!({}),idempotency_key:Some(key),proposed_at_ms:time}
}
fn main() {
    let policy=CapabilityPolicy::new(HashSet::from([
        Capability::ToolExecution("blocked_fixture".into()),Capability::ToolExecution("echo_fixture".into())]),vec![],vec![]);
    let broker=Arc::new(ToolExecutionBroker::new(policy));
    broker.register_tool(Arc::new(EchoFixture{version:"v1"}));
    let calls=Arc::new(AtomicUsize::new(0));
    let (started_tx,started_rx)=mpsc::channel();let (release_tx,release_rx)=mpsc::channel();
    broker.register_tool(Arc::new(BlockingFixture{calls:calls.clone(),started:started_tx,release:Mutex::new(release_rx)}));
    let request=proposal("blocked_fixture","blocked-original".into(),0);
    let worker_broker=broker.clone();let worker_request=request.clone();
    let worker=std::thread::spawn(move||worker_broker.execute_proposal(&worker_request,&Default::default()));
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    for i in 0..1002 {
        let ctx=ToolExecutionContext{current_timestamp_ms:i+1,..Default::default()};
        broker.execute_proposal(&proposal("echo_fixture",format!("flood-{i}"),i+1),&ctx);
    }
    let retry=broker.execute_proposal(&request,&ToolExecutionContext{current_timestamp_ms:2000,..Default::default()});
    println!("duplicate during original in-flight operation after cache pressure: {:?}; counter effects={}",retry.status,calls.load(Ordering::SeqCst));
    release_tx.send(()).unwrap();let first=worker.join().unwrap();println!("original eventual result: {:?}",first.status);
    let version_request=proposal("echo_fixture","version-key".into(),3000);
    let first=broker.execute_proposal(&version_request,&ToolExecutionContext{current_timestamp_ms:3000,..Default::default()});
    println!("v1 result: {:?}",first.output);
    broker.register_tool(Arc::new(EchoFixture{version:"v2"}));
    let changed=broker.execute_proposal(&version_request,&ToolExecutionContext{current_timestamp_ms:3001,..Default::default()});
    println!("same key after registered tool version replacement: {:?}; cached={}; output={:?}",changed.status,changed.from_cache,changed.output);
}
