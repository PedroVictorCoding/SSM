//! Disposable fixtures only; reports behavior, not a passing safety verdict.
use sail::capability::{Capability, CapabilityPolicy};
use sail::tool::{ToolExecutionBroker, ToolExecutionContext, ToolProposal, WorkspaceFileSearchTool};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
fn has_outside_match(output: &Option<String>) -> bool {
    output.as_deref().and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok())
        .is_some_and(|artifact| artifact["detailed_payload"]["matches"].as_array()
            .is_some_and(|matches| matches.iter().any(|entry| entry["path"].as_str()
                .is_some_and(|path| path.contains("outside_marker_qa")))))
}
fn main() {
    let root=std::env::temp_dir().join(format!("sail-path-integration-review-{}",std::process::id()));
    let user_project=root.join("user-project");
    std::fs::create_dir_all(user_project.join("src")).unwrap();
    std::fs::write(user_project.join("Cargo.toml"),"disposable fixture, not a build").unwrap();
    std::fs::write(user_project.join("src/user_code.rs"),"user code fixture").unwrap();
    let policy=CapabilityPolicy::new(HashSet::new(),vec![user_project.clone()],vec![user_project.clone()]);
    println!("authorized unrelated Rust project source edit: {:?}",policy.check_file_write(user_project.join("src/user_code.rs")));

    let workspace=root.join("search-root");let outside=root.join("outside");
    std::fs::create_dir_all(&workspace).unwrap();std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("outside_marker_qa.txt"),"disposable metadata marker").unwrap();
    #[cfg(unix)] {
        std::os::unix::fs::symlink(&outside,workspace.join("escape-link")).unwrap();
        let grants=HashSet::from([Capability::ToolExecution("workspace_file_search".into())]);
        let broker=ToolExecutionBroker::new(CapabilityPolicy::new(grants.clone(),vec![workspace.clone()],vec![]));
        broker.register_tool(Arc::new(WorkspaceFileSearchTool::new(vec![workspace.clone()])));
        let proposal=ToolProposal {call_id:"search".into(),task_id:"qa".into(),tool_name:"workspace_file_search".into(),
            arguments:json!({"query":"outside_marker_qa","root_dir":workspace}),idempotency_key:None,proposed_at_ms:0};
        let result=broker.execute_proposal(&proposal,&ToolExecutionContext::default());
        println!("narrow-constructor symlink search status: {:?}; outside-root metadata exposed: {}",result.status,has_outside_match(&result.output));
        let broad_broker=ToolExecutionBroker::new(CapabilityPolicy::new(grants,vec![workspace.clone()],vec![]));
        broad_broker.register_tool(Arc::new(WorkspaceFileSearchTool::new(vec![root.clone()])));
        let result=broad_broker.execute_proposal(&proposal,&ToolExecutionContext::default());
        println!("broader-constructor symlink search status: {:?}; outside host-root metadata exposed: {}",result.status,has_outside_match(&result.output));
    }

    let framework=root.join("protected-framework");std::fs::create_dir_all(framework.join("src")).unwrap();
    std::fs::write(framework.join("src/fixture.rs"),"protected fixture").unwrap();
    let policy=CapabilityPolicy::new(HashSet::new(),vec![root.clone()],vec![root.clone()]).with_framework_root(&framework);
    let moved=root.join("moved-framework");
    println!("protected framework root rename check: {:?}",policy.check_file_rename(&framework,&moved));
    println!("protected framework parent deletion policy check (not executed): {:?}",policy.check_file_delete(&root));
    println!("explicit protected framework reclassified as user project: {:?}",policy.clone().with_user_project_root(&framework).check_file_write(framework.join("src/fixture.rs")));
    // Host-created move only demonstrates identity behavior. No model/tool writes run.
    std::fs::rename(&framework,&moved).unwrap();
    println!("protected source after fixture ancestor move: {:?}",policy.check_file_write(moved.join("src/fixture.rs")));
    std::fs::remove_dir_all(root).unwrap();
}
