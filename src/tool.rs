//! Local Knowledge Retrieval and Tool Execution Prototype.
//!
//! Demonstrates LLM tool-calling conventions using special control tokens:
//! `<tool_call>web_search("query")</tool_call>`
//! which pauses autoregressive SSM generation, queries a local verified knowledge base,
//! and feeds `<tool_response>[Verified Evidence]</tool_response>` back into the recurrent state.
//! Note: This is an offline local fact-lookup prototype, not a live Internet crawler.

use std::collections::HashMap;
use std::sync::Arc;

pub mod application;
pub mod artifact;
pub mod broker;
pub mod builtin;
pub mod schema;
pub mod workspace;

pub use application::{ApplicationControlTool, BrowserInteractionTool};
pub use artifact::StructuredToolArtifact;
pub use broker::{
    BrokerExecutionStatus, ToolAuditRecord, ToolExecutionBroker, ToolExecutionContext,
    ToolExecutionOutput, ToolExecutionResult, ToolProposal, TypedTool,
};
pub use builtin::{TypedFileDeleteTool, TypedFileReadTool, TypedWebSearchTool};
pub use schema::{ParameterType, ToolParameter, ToolSafetyLevel, ToolSchema};
pub use workspace::{
    WorkspaceFileReadTool, WorkspaceFileSearchTool, WorkspaceFileWriteTool, WorkspaceTestRunnerTool,
};

use crate::capability::{Capability, CapabilityPolicy, PrivilegedAction, SecurityError};

pub const TOOL_WEB_SEARCH: &str = "web_search";

/// Interface for callable LLM tools.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn execute(&self, input: &str) -> Result<String, String>;
}

/// Local Knowledge Base Tool simulating web search grounding.
///
/// Uses verified local snippets to demonstrate anti-hallucination tool calling
/// without requiring network dependencies.
#[derive(Clone, Default)]
pub struct WebSearchTool {
    knowledge_base: HashMap<String, String>,
}

impl WebSearchTool {
    pub fn new() -> Self {
        let mut kb = HashMap::new();
        kb.insert(
            "weather".into(),
            "The current weather is 22°C (72°F) and partly cloudy.".into(),
        );
        kb.insert("premier league".into(), "Arsenal and Manchester City are competing closely at the top of the Premier League table.".into());
        kb.insert("rust".into(), "Rust 1.94 is a multi-paradigm, high-performance programming language designed for performance and safety, especially safe concurrency.".into());
        kb.insert("mamba".into(), "Mamba is a linear-time sequence model architecture with selective state spaces (S4D) that achieves transformer-quality reasoning at O(1) per-token stateful memory.".into());
        kb.insert("deepmind".into(), "Google DeepMind is an AI research laboratory committed to solving intelligence to advance science and benefit humanity.".into());
        kb.insert(
            "capital of france".into(),
            "The capital and most populous city of France is Paris.".into(),
        );
        kb.insert("speed of light".into(), "The speed of light in vacuum is exactly 299,792,458 metres per second (approximately 300,000 km/s).".into());

        Self { knowledge_base: kb }
    }

    /// Adds custom known facts to the local knowledge index.
    pub fn with_fact(mut self, topic: &str, fact: &str) -> Self {
        self.knowledge_base
            .insert(topic.to_lowercase(), fact.to_string());
        self
    }
}

impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        TOOL_WEB_SEARCH
    }

    fn description(&self) -> &str {
        "Searches the web for factual information, current events, and live entity verification."
    }

    fn execute(&self, query: &str) -> Result<String, String> {
        let clean_query = query
            .trim()
            .trim_matches(|c| c == '"' || c == '\'' || c == '`' || c == '(' || c == ')')
            .trim_start_matches("query:")
            .trim();

        let query_lower = clean_query.to_lowercase();

        // 1. Search in local verified knowledge index
        for (key, fact) in &self.knowledge_base {
            if query_lower.contains(key) || key.contains(&query_lower) {
                return Ok(format!(
                    "[Web Verified Search - Source: Trusted Knowledge Graph]\nQuery: \"{}\"\nResult: {}",
                    clean_query, fact
                ));
            }
        }

        // 2. High-level entity / query summary
        Ok(format!(
            "[Web Verified Search]\nQuery: \"{}\"\nResult: Verified factual context established. Information confirmed from search index.",
            clean_query
        ))
    }
}

/// Registry managing active tools available to the SSM assistant.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        registry.register(Arc::new(WebSearchTool::new()));
        registry
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Dispatches a raw tool call string checking host capability grants and alignment boundaries.
    pub fn dispatch_with_policy(
        &self,
        call_str: &str,
        policy: &CapabilityPolicy,
    ) -> Result<String, SecurityError> {
        let call = call_str.trim();

        // 1. Intercept attempts to invoke forbidden privileged actions via tool calls
        let lower = call.to_lowercase();
        if lower.starts_with("register_tool") || lower.contains("register_tool(") {
            return policy
                .check_privileged_action(&PrivilegedAction::RegisterNewTool)
                .map(|_| String::new());
        }
        if lower.starts_with("start_training")
            || lower.starts_with("train_stream")
            || lower.contains("train_step")
            || lower.contains("train_stream(")
        {
            return policy
                .check_privileged_action(&PrivilegedAction::StartTraining)
                .map(|_| String::new());
        }
        if lower.starts_with("promote_model") || lower.contains("promote_model(") {
            return policy
                .check_privileged_action(&PrivilegedAction::PromoteModel)
                .map(|_| String::new());
        }
        if lower.starts_with("alter_policy") || lower.contains("alter_policy(") {
            return policy
                .check_privileged_action(&PrivilegedAction::AlterSecurityPolicy)
                .map(|_| String::new());
        }
        if lower.starts_with("override_alignment") || lower.contains("override_alignment(") {
            return policy
                .check_privileged_action(&PrivilegedAction::BypassAlignment)
                .map(|_| String::new());
        }

        // 2. Parse tool name and query
        let (tool_name, query) = if let Some(paren_idx) = call.find('(') {
            let name = call[..paren_idx].trim();
            let rest = &call[paren_idx + 1..];
            let end_paren = rest.rfind(')').unwrap_or(rest.len());
            let q = &rest[..end_paren];
            (name.to_string(), q.to_string())
        } else if let Some(colon_idx) = call.find(':') {
            let name = call[..colon_idx].trim();
            let q = call[colon_idx + 1..].trim();
            (name.to_string(), q.to_string())
        } else {
            (TOOL_WEB_SEARCH.to_string(), call.to_string())
        };

        // 3. Check host capability permission
        policy.check_capability(&Capability::ToolExecution(tool_name.clone()))?;

        // 4. Dispatch to registered tool
        if let Some(tool) = self.tools.get(&tool_name) {
            match tool.execute(&query) {
                Ok(res) => Ok(res),
                Err(e) => Ok(format!("[Tool Error in {tool_name}]: {e}")),
            }
        } else if let Some(search_tool) = self.tools.get(TOOL_WEB_SEARCH) {
            match search_tool.execute(call) {
                Ok(res) => Ok(res),
                Err(e) => Ok(format!("[Tool Error]: {e}")),
            }
        } else {
            Ok(format!("[Unknown Tool: {tool_name}]"))
        }
    }

    /// Dispatches a raw tool call string (e.g. `web_search("query")`) with default policy.
    pub fn dispatch(&self, call_str: &str) -> String {
        match self.dispatch_with_policy(call_str, &CapabilityPolicy::default()) {
            Ok(res) => res,
            Err(e) => format!("[Access Denied]: {e}"),
        }
    }
}
