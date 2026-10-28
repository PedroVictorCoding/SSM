//! Typed, Capability-Scoped Tool Execution Broker.
//!
//! Provides:
//! - Separation of untrusted model proposals from verified host execution.
//! - Typed schema validation, parameter type checking, and unexpected extra-field rejection.
//! - Host capability policy enforcement and permanent prohibition of privileged self-modification.
//! - Standing user authorizations and interactive confirmation gates for destructive actions.
//! - Idempotency tracking preventing dangerous blind retries of non-idempotent actions after timeouts.
//! - Bounded output truncation preventing context overflow.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::capability::{Capability, CapabilityPolicy, PrivilegedAction};
use crate::runtime::bus::CancellationToken;
use crate::tool::schema::{ToolSafetyLevel, ToolSchema};

/// Output produced by a typed tool implementation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionOutput {
    pub result_text: String,
    pub exit_code: i32,
    pub side_effects_committed: bool,
}

/// Interface for typed, schema-validated tools.
pub trait TypedTool: Send + Sync {
    fn schema(&self) -> ToolSchema;
    fn execute(&self, args: &serde_json::Value) -> Result<ToolExecutionOutput, String>;
    fn execute_with_context(
        &self,
        args: &serde_json::Value,
        _context: &ToolExecutionContext,
    ) -> Result<ToolExecutionOutput, String> {
        self.execute(args)
    }
}

/// Model-originated proposal to execute a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolProposal {
    pub call_id: String,
    pub task_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub idempotency_key: Option<String>,
    pub proposed_at_ms: u64,
}

/// Execution constraints supplied by the host runtime.
#[derive(Clone, Debug)]
pub struct ToolExecutionContext {
    pub session_id: Option<String>,
    pub deadline_ms: Option<u64>,
    pub cancellation_token: Option<CancellationToken>,
    pub max_output_bytes: usize,
    pub current_timestamp_ms: u64,
    pub interactive_confirmation_granted: bool,
    pub policy: Option<Arc<CapabilityPolicy>>,
}

impl Default for ToolExecutionContext {
    fn default() -> Self {
        Self {
            session_id: None,
            deadline_ms: None,
            cancellation_token: None,
            max_output_bytes: 4096,
            current_timestamp_ms: 0,
            interactive_confirmation_granted: false,
            policy: None,
        }
    }
}

/// Formal outcome status of a brokered execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BrokerExecutionStatus {
    Success,
    ValidationFailed(String),
    Unauthorized(String),
    AwaitingAuthorization(String),
    TimedOut,
    Cancelled,
    UnknownOutcome(String),
    ExecutionFailed(String),
}

/// Maximum capacity of the in-memory idempotency cache before evicting oldest completed entries.
pub const MAX_IDEMPOTENCY_RECORDS: usize = 1000;

/// Total raw output byte limit across all retained idempotency cache records (10 MB).
pub const MAX_CACHED_OUTPUT_BYTES_TOTAL: usize = 10 * 1024 * 1024;

/// State of an in-flight or committed idempotency cache record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum IdempotencyRecordState {
    InFlight { started_at_ms: u64 },
    Committed { result: ToolExecutionResult },
    Failed { error: String },
}

/// Identity-bound idempotency cache record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IdempotencyRecord {
    pub key: String,
    pub session_id: Option<String>,
    pub task_id: String,
    pub tool_name: String,
    pub tool_version: String,
    pub canonical_arguments: String,
    pub argument_digest: String,
    pub raw_output: Option<String>,
    pub exit_code: i32,
    pub state: IdempotencyRecordState,
    pub created_at_ms: u64,
    pub total_output_bytes: usize,
}

/// Evicts oldest completed, non-protected records until count and byte bounds are satisfied.
/// In-flight records and unconfirmed non-idempotent unknown outcomes are strictly preserved.
fn evict_idempotency_records_if_needed(
    cache: &mut HashMap<String, IdempotencyRecord>,
    needed_bytes: usize,
) -> Result<(), BrokerExecutionStatus> {
    loop {
        let current_bytes: usize = cache.values().map(|r| r.total_output_bytes).sum();
        let over_count = cache.len() >= MAX_IDEMPOTENCY_RECORDS;
        let over_bytes = current_bytes + needed_bytes > MAX_CACHED_OUTPUT_BYTES_TOTAL;

        if !over_count && !over_bytes {
            return Ok(());
        }

        let evictable_key = cache
            .iter()
            .filter(|(_, r)| match &r.state {
                IdempotencyRecordState::InFlight { .. } => false,
                IdempotencyRecordState::Committed { result } => !matches!(
                    result.status,
                    BrokerExecutionStatus::UnknownOutcome(_) | BrokerExecutionStatus::TimedOut
                ),
                IdempotencyRecordState::Failed { .. } => true,
            })
            .min_by_key(|(_, r)| r.created_at_ms)
            .map(|(k, _)| k.clone());

        match evictable_key {
            Some(k) => {
                cache.remove(&k);
            }
            None => {
                return Err(BrokerExecutionStatus::UnknownOutcome(
                    "Idempotency cache capacity exhausted: all active records are in-flight or protected"
                        .into(),
                ));
            }
        }
    }
}

/// Complete result returned to host and conversation coordinator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    pub call_id: String,
    pub task_id: String,
    pub idempotency_key: Option<String>,
    pub status: BrokerExecutionStatus,
    pub output: Option<String>,
    pub exit_code: i32,
    pub execution_time_ms: f64,
    pub side_effects_committed: bool,
    pub from_cache: bool,
}

/// Audit entry for security and compliance recording.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolAuditRecord {
    pub timestamp_ms: u64,
    pub call_id: String,
    pub task_id: String,
    pub tool_name: String,
    pub status: BrokerExecutionStatus,
    pub side_effects_committed: bool,
}

/// Helper function to deterministically serialize JSON arguments into a canonical format.
fn canonical_json_string(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(&String, &serde_json::Value)> = map.iter().collect();
            entries.sort_by_key(|(k, _)| *k);
            let mut out = String::from("{");
            for (i, (k, v)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                out.push_str(&canonical_json_string(v));
            }
            out.push('}');
            out
        }
        serde_json::Value::Array(arr) => {
            let mut out = String::from("[");
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_json_string(item));
            }
            out.push(']');
            out
        }
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Computes a deterministic digest string for proposal arguments.
fn compute_argument_digest(arguments: &serde_json::Value) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let canonical = canonical_json_string(arguments);
    let mut hasher = DefaultHasher::new();
    canonical.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Truncates output text safely to max_bytes at a valid UTF-8 boundary, reserving room for notice.
pub fn truncate_output_to_bound(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let notice = format!("\n[Output truncated at {} bytes]", max_bytes);
    let include_notice = notice.len() <= max_bytes;
    let mut end = if include_notice {
        max_bytes - notice.len()
    } else {
        max_bytes
    };
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = text[..end].to_string();
    if include_notice {
        truncated.push_str(&notice);
    }
    truncated
}

/// Typed, capability-scoped tool execution broker.
pub struct ToolExecutionBroker {
    tools: Arc<RwLock<HashMap<String, Arc<dyn TypedTool>>>>,
    policy: Arc<RwLock<CapabilityPolicy>>,
    standing_authorizations: Arc<RwLock<HashSet<String>>>,
    idempotency_cache: Arc<RwLock<HashMap<String, IdempotencyRecord>>>,
    audit_log: Arc<RwLock<Vec<ToolAuditRecord>>>,
}

impl ToolExecutionBroker {
    pub fn new(policy: CapabilityPolicy) -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            policy: Arc::new(RwLock::new(policy)),
            standing_authorizations: Arc::new(RwLock::new(HashSet::new())),
            idempotency_cache: Arc::new(RwLock::new(HashMap::new())),
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Registers a typed tool implementation with the broker.
    pub fn register_tool(&self, tool: Arc<dyn TypedTool>) {
        let name = tool.schema().name.clone();
        let mut map = self.tools.write().unwrap();
        map.insert(name, tool);
    }

    /// Grants standing user authorization for a tool (e.g. user granted persistent write access).
    pub fn grant_standing_authorization(&self, tool_name: &str) {
        let mut grants = self.standing_authorizations.write().unwrap();
        grants.insert(tool_name.to_string());
    }

    /// Revokes standing authorization for a tool.
    pub fn revoke_standing_authorization(&self, tool_name: &str) {
        let mut grants = self.standing_authorizations.write().unwrap();
        grants.remove(tool_name);
    }

    /// Updates the host capability policy.
    pub fn set_policy(&self, policy: CapabilityPolicy) {
        let mut p = self.policy.write().unwrap();
        *p = policy;
    }

    /// Executes a tool proposal under host-enforced validation, capabilities, and safety bounds.
    pub fn execute_proposal(
        &self,
        proposal: &ToolProposal,
        context: &ToolExecutionContext,
    ) -> ToolExecutionResult {
        let start = Instant::now();

        // 1. Lookup tool
        let tool = {
            let tools = self.tools.read().unwrap();
            match tools.get(&proposal.tool_name) {
                Some(t) => Arc::clone(t),
                None => {
                    return self.record_and_return(
                        proposal,
                        BrokerExecutionStatus::ValidationFailed(format!(
                            "Tool '{}' is not registered in broker",
                            proposal.tool_name
                        )),
                        None,
                        -1,
                        start.elapsed().as_secs_f64() * 1000.0,
                        false,
                        false,
                        context.current_timestamp_ms,
                    );
                }
            }
        };

        let schema = tool.schema();

        // 2. Schema validation (types, required fields, extra fields)
        if let Err(val_err) = schema.validate_arguments(&proposal.arguments) {
            return self.record_and_return(
                proposal,
                BrokerExecutionStatus::ValidationFailed(val_err),
                None,
                -1,
                start.elapsed().as_secs_f64() * 1000.0,
                false,
                false,
                context.current_timestamp_ms,
            );
        }

        // 3. Authoritative host capability policy check
        {
            let pol = self.policy.read().unwrap();
            if let Err(cap_err) =
                pol.check_capability(&Capability::ToolExecution(proposal.tool_name.clone()))
            {
                return self.record_and_return(
                    proposal,
                    BrokerExecutionStatus::Unauthorized(cap_err.to_string()),
                    None,
                    -1,
                    start.elapsed().as_secs_f64() * 1000.0,
                    false,
                    false,
                    context.current_timestamp_ms,
                );
            }

            // Guard against prohibited privileged actions
            let lower = proposal.tool_name.to_lowercase();
            if lower.contains("train") || lower.contains("self_modify") || lower.contains("promote")
            {
                if let Err(priv_err) = pol.check_privileged_action(&PrivilegedAction::StartTraining)
                {
                    return self.record_and_return(
                        proposal,
                        BrokerExecutionStatus::Unauthorized(priv_err.to_string()),
                        None,
                        -1,
                        start.elapsed().as_secs_f64() * 1000.0,
                        false,
                        false,
                        context.current_timestamp_ms,
                    );
                }
            }

            // Enforce path policy for file tools directly at broker boundary
            if lower == "file_read" || lower == "workspace_file_read" {
                if let Some(path_str) = proposal.arguments.get("path").and_then(|v| v.as_str()) {
                    if let Err(e) = pol.check_file_read(path_str) {
                        return self.record_and_return(
                            proposal,
                            BrokerExecutionStatus::Unauthorized(e.to_string()),
                            None,
                            -1,
                            start.elapsed().as_secs_f64() * 1000.0,
                            false,
                            false,
                            context.current_timestamp_ms,
                        );
                    }
                }
            } else if lower == "file_delete" {
                if let Some(path_str) = proposal.arguments.get("path").and_then(|v| v.as_str()) {
                    if let Err(e) = pol.check_file_delete(path_str) {
                        return self.record_and_return(
                            proposal,
                            BrokerExecutionStatus::Unauthorized(e.to_string()),
                            None,
                            -1,
                            start.elapsed().as_secs_f64() * 1000.0,
                            false,
                            false,
                            context.current_timestamp_ms,
                        );
                    }
                }
            } else if lower == "workspace_file_write" {
                if let Some(path_str) = proposal.arguments.get("path").and_then(|v| v.as_str()) {
                    if let Err(e) = pol.check_file_write(path_str) {
                        return self.record_and_return(
                            proposal,
                            BrokerExecutionStatus::Unauthorized(e.to_string()),
                            None,
                            -1,
                            start.elapsed().as_secs_f64() * 1000.0,
                            false,
                            false,
                            context.current_timestamp_ms,
                        );
                    }
                }
            } else if lower == "workspace_file_search" {
                let root_str = proposal
                    .arguments
                    .get("root_dir")
                    .and_then(|v| v.as_str())
                    .unwrap_or(".");
                if let Err(e) = pol.check_file_read(root_str) {
                    return self.record_and_return(
                        proposal,
                        BrokerExecutionStatus::Unauthorized(e.to_string()),
                        None,
                        -1,
                        start.elapsed().as_secs_f64() * 1000.0,
                        false,
                        false,
                        context.current_timestamp_ms,
                    );
                }
            }
        }

        // 4. Destructive authorization check
        if schema.safety_level == ToolSafetyLevel::DestructiveOrExternal {
            let has_standing = {
                let grants = self.standing_authorizations.read().unwrap();
                grants.contains(&proposal.tool_name)
            };

            if !has_standing && !context.interactive_confirmation_granted {
                return self.record_and_return(
                    proposal,
                    BrokerExecutionStatus::AwaitingAuthorization(format!(
                        "Tool '{}' is classified as DestructiveOrExternal and requires standing authorization or user confirmation",
                        proposal.tool_name
                    )),
                    None,
                    -1,
                    start.elapsed().as_secs_f64() * 1000.0,
                    false,
                    false,
                    context.current_timestamp_ms,
                );
            }
        }

        // 5. Pre-execution cancellation and deadline checks
        if let Some(token) = &context.cancellation_token {
            if token.is_cancelled() {
                return self.record_and_return(
                    proposal,
                    BrokerExecutionStatus::Cancelled,
                    None,
                    -1,
                    start.elapsed().as_secs_f64() * 1000.0,
                    false,
                    false,
                    context.current_timestamp_ms,
                );
            }
        }

        if let Some(dl) = context.deadline_ms {
            if context.current_timestamp_ms > dl {
                let status = if schema.is_idempotent {
                    BrokerExecutionStatus::TimedOut
                } else {
                    BrokerExecutionStatus::UnknownOutcome(
                        "Deadline exceeded before dispatch".into(),
                    )
                };
                return self.record_and_return(
                    proposal,
                    status,
                    None,
                    -1,
                    start.elapsed().as_secs_f64() * 1000.0,
                    false,
                    false,
                    context.current_timestamp_ms,
                );
            }
        }

        // 6. Idempotency cache check and in-flight reservation
        let canonical_args = canonical_json_string(&proposal.arguments);
        let arg_digest = compute_argument_digest(&proposal.arguments);
        if let Some(key) = &proposal.idempotency_key {
            let cached_outcome = {
                let mut cache = self.idempotency_cache.write().unwrap();
                if let Some(existing) = cache.get(key) {
                    // Identity binding checks
                    if existing.task_id != proposal.task_id {
                        Some(Err(BrokerExecutionStatus::ValidationFailed(format!(
                            "Idempotency key '{}' reuse across different tasks is prohibited: bound to task '{}', requested task '{}'",
                            key, existing.task_id, proposal.task_id
                        ))))
                    } else if existing.tool_name != proposal.tool_name {
                        Some(Err(BrokerExecutionStatus::ValidationFailed(format!(
                            "Idempotency key '{}' reuse with different tool is prohibited: bound to tool '{}', requested tool '{}'",
                            key, existing.tool_name, proposal.tool_name
                        ))))
                    } else if existing.tool_version != schema.version {
                        Some(Err(BrokerExecutionStatus::ValidationFailed(format!(
                            "Idempotency key '{}' reuse with different tool version is prohibited: bound to version '{}', requested tool has version '{}'",
                            key, existing.tool_version, schema.version
                        ))))
                    } else if context.session_id.is_some()
                        && existing.session_id.is_some()
                        && existing.session_id != context.session_id
                    {
                        Some(Err(BrokerExecutionStatus::ValidationFailed(format!(
                            "Idempotency key '{}' reuse across different host sessions is prohibited: bound to session '{:?}', requested session '{:?}'",
                            key, existing.session_id, context.session_id
                        ))))
                    } else if existing.canonical_arguments != canonical_args {
                        Some(Err(BrokerExecutionStatus::ValidationFailed(format!(
                            "Idempotency key '{}' reuse with different arguments is prohibited",
                            key
                        ))))
                    } else {
                        match &existing.state {
                            IdempotencyRecordState::InFlight { .. } => {
                                Some(Err(BrokerExecutionStatus::UnknownOutcome(format!(
                                    "Concurrent in-flight execution for idempotency key '{}' is already in progress",
                                    key
                                ))))
                            }
                            IdempotencyRecordState::Committed { result } => {
                                match &result.status {
                                    BrokerExecutionStatus::Success => Some(Ok(existing.clone())),
                                    BrokerExecutionStatus::TimedOut
                                    | BrokerExecutionStatus::UnknownOutcome(_) => {
                                        if !schema.is_idempotent {
                                            Some(Err(BrokerExecutionStatus::UnknownOutcome(
                                                "Cannot blindly retry non-idempotent action in UnknownOutcome status; requires explicit confirmation or compensation".into(),
                                            )))
                                        } else {
                                            // Retry allowed for idempotent action: reserve in-flight
                                            cache.insert(
                                                key.clone(),
                                                IdempotencyRecord {
                                                    key: key.clone(),
                                                    session_id: context.session_id.clone(),
                                                    task_id: proposal.task_id.clone(),
                                                    tool_name: proposal.tool_name.clone(),
                                                    tool_version: schema.version.clone(),
                                                    canonical_arguments: canonical_args.clone(),
                                                    argument_digest: arg_digest.clone(),
                                                    raw_output: None,
                                                    exit_code: 0,
                                                    state: IdempotencyRecordState::InFlight {
                                                        started_at_ms: context.current_timestamp_ms,
                                                    },
                                                    created_at_ms: context.current_timestamp_ms,
                                                    total_output_bytes: 0,
                                                },
                                            );
                                            None
                                        }
                                    }
                                    _ => {
                                        if !schema.is_idempotent {
                                            Some(Err(BrokerExecutionStatus::UnknownOutcome(
                                                "Cannot blindly retry failed non-idempotent action"
                                                    .into(),
                                            )))
                                        } else {
                                            cache.insert(
                                                key.clone(),
                                                IdempotencyRecord {
                                                    key: key.clone(),
                                                    session_id: context.session_id.clone(),
                                                    task_id: proposal.task_id.clone(),
                                                    tool_name: proposal.tool_name.clone(),
                                                    tool_version: schema.version.clone(),
                                                    canonical_arguments: canonical_args.clone(),
                                                    argument_digest: arg_digest.clone(),
                                                    raw_output: None,
                                                    exit_code: 0,
                                                    state: IdempotencyRecordState::InFlight {
                                                        started_at_ms: context.current_timestamp_ms,
                                                    },
                                                    created_at_ms: context.current_timestamp_ms,
                                                    total_output_bytes: 0,
                                                },
                                            );
                                            None
                                        }
                                    }
                                }
                            }
                            IdempotencyRecordState::Failed { error } => {
                                if !schema.is_idempotent {
                                    Some(Err(BrokerExecutionStatus::UnknownOutcome(format!(
                                        "Cannot retry failed non-idempotent action: {}",
                                        error
                                    ))))
                                } else {
                                    cache.insert(
                                        key.clone(),
                                        IdempotencyRecord {
                                            key: key.clone(),
                                            session_id: context.session_id.clone(),
                                            task_id: proposal.task_id.clone(),
                                            tool_name: proposal.tool_name.clone(),
                                            tool_version: schema.version.clone(),
                                            canonical_arguments: canonical_args.clone(),
                                            argument_digest: arg_digest.clone(),
                                            raw_output: None,
                                            exit_code: 0,
                                            state: IdempotencyRecordState::InFlight {
                                                started_at_ms: context.current_timestamp_ms,
                                            },
                                            created_at_ms: context.current_timestamp_ms,
                                            total_output_bytes: 0,
                                        },
                                    );
                                    None
                                }
                            }
                        }
                    }
                } else {
                    // Not in cache: check eviction bounds before reserving in-flight!
                    if let Err(status) = evict_idempotency_records_if_needed(&mut cache, 0) {
                        Some(Err(status))
                    } else {
                        cache.insert(
                            key.clone(),
                            IdempotencyRecord {
                                key: key.clone(),
                                session_id: context.session_id.clone(),
                                task_id: proposal.task_id.clone(),
                                tool_name: proposal.tool_name.clone(),
                                tool_version: schema.version.clone(),
                                canonical_arguments: canonical_args.clone(),
                                argument_digest: arg_digest.clone(),
                                raw_output: None,
                                exit_code: 0,
                                state: IdempotencyRecordState::InFlight {
                                    started_at_ms: context.current_timestamp_ms,
                                },
                                created_at_ms: context.current_timestamp_ms,
                                total_output_bytes: 0,
                            },
                        );
                        None
                    }
                }
            };

            match cached_outcome {
                Some(Err(status)) => {
                    return self.record_and_return(
                        proposal,
                        status,
                        None,
                        -1,
                        start.elapsed().as_secs_f64() * 1000.0,
                        false,
                        false,
                        context.current_timestamp_ms,
                    );
                }
                Some(Ok(record)) => {
                    // Delivered from cache!
                    let raw = record.raw_output.as_deref().unwrap_or("");
                    let bounded = truncate_output_to_bound(raw, context.max_output_bytes);
                    let side_effects = match &record.state {
                        IdempotencyRecordState::Committed { result } => {
                            result.side_effects_committed
                        }
                        _ => false,
                    };
                    let result = ToolExecutionResult {
                        call_id: proposal.call_id.clone(),
                        task_id: proposal.task_id.clone(),
                        idempotency_key: Some(key.clone()),
                        status: BrokerExecutionStatus::Success,
                        output: Some(bounded),
                        exit_code: record.exit_code,
                        execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                        side_effects_committed: side_effects,
                        from_cache: true,
                    };

                    // Record correlated audit entry for cached delivery
                    {
                        let mut audit = self.audit_log.write().unwrap();
                        audit.push(ToolAuditRecord {
                            timestamp_ms: context.current_timestamp_ms,
                            call_id: proposal.call_id.clone(),
                            task_id: proposal.task_id.clone(),
                            tool_name: proposal.tool_name.clone(),
                            status: BrokerExecutionStatus::Success,
                            side_effects_committed: side_effects,
                        });
                    }

                    return result;
                }
                None => {
                    // Key successfully reserved as InFlight. Proceed to execute.
                }
            }
        }

        // 7. Execute tool with execution context: ALWAYS inject broker's authoritative policy
        let mut ctx_with_policy = context.clone();
        {
            let pol = self.policy.read().unwrap();
            ctx_with_policy.policy = Some(Arc::new(pol.clone()));
        }

        let output = match tool.execute_with_context(&proposal.arguments, &ctx_with_policy) {
            Ok(out) => out,
            Err(e) => {
                let status = if e.starts_with("CANCELLED:") || e == "Cancelled" {
                    BrokerExecutionStatus::Cancelled
                } else if e.starts_with("TIMED_OUT:") || e == "TimedOut" {
                    if schema.is_idempotent {
                        BrokerExecutionStatus::TimedOut
                    } else {
                        BrokerExecutionStatus::UnknownOutcome(e.clone())
                    }
                } else if e.starts_with("UNKNOWN_OUTCOME:") {
                    BrokerExecutionStatus::UnknownOutcome(e.clone())
                } else {
                    BrokerExecutionStatus::ExecutionFailed(e.clone())
                };

                if let Some(key) = &proposal.idempotency_key {
                    let mut cache = self.idempotency_cache.write().unwrap();
                    let _ = evict_idempotency_records_if_needed(&mut cache, 0);
                    let state = match &status {
                        BrokerExecutionStatus::ExecutionFailed(msg) => {
                            IdempotencyRecordState::Failed { error: msg.clone() }
                        }
                        _ => IdempotencyRecordState::Committed {
                            result: ToolExecutionResult {
                                call_id: proposal.call_id.clone(),
                                task_id: proposal.task_id.clone(),
                                idempotency_key: Some(key.clone()),
                                status: status.clone(),
                                output: None,
                                exit_code: -1,
                                execution_time_ms: start.elapsed().as_secs_f64() * 1000.0,
                                side_effects_committed: false,
                                from_cache: false,
                            },
                        },
                    };
                    cache.insert(
                        key.clone(),
                        IdempotencyRecord {
                            key: key.clone(),
                            session_id: context.session_id.clone(),
                            task_id: proposal.task_id.clone(),
                            tool_name: proposal.tool_name.clone(),
                            tool_version: schema.version.clone(),
                            canonical_arguments: canonical_args.clone(),
                            argument_digest: arg_digest.clone(),
                            raw_output: None,
                            exit_code: -1,
                            state,
                            created_at_ms: context.current_timestamp_ms,
                            total_output_bytes: 0,
                        },
                    );
                }
                return self.record_and_return(
                    proposal,
                    status,
                    None,
                    -1,
                    start.elapsed().as_secs_f64() * 1000.0,
                    false,
                    false,
                    context.current_timestamp_ms,
                );
            }
        };

        // Post-execution cancellation check
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let effective_now_ms = context.current_timestamp_ms + elapsed_ms as u64;

        if let Some(token) = &context.cancellation_token {
            if token.is_cancelled() {
                let status = if output.side_effects_committed && !schema.is_idempotent {
                    BrokerExecutionStatus::UnknownOutcome("Execution completed but cancelled during dispatch; side effects were committed".into())
                } else {
                    BrokerExecutionStatus::Cancelled
                };
                return self.record_and_return(
                    proposal,
                    status,
                    None,
                    output.exit_code,
                    elapsed_ms,
                    output.side_effects_committed,
                    false,
                    effective_now_ms,
                );
            }
        }

        // Post-execution deadline check
        if let Some(dl) = context.deadline_ms {
            let elapsed_ms_u64 = elapsed_ms as u64;
            let exceeded = elapsed_ms_u64 > dl
                || (context.current_timestamp_ms > 0
                    && context.current_timestamp_ms.saturating_add(elapsed_ms_u64) > dl);
            if exceeded {
                let status = if output.side_effects_committed && !schema.is_idempotent {
                    BrokerExecutionStatus::UnknownOutcome(format!(
                        "Execution completed but elapsed time ({:.1}ms) exceeded deadline ({}ms); side effects were committed",
                        elapsed_ms, dl
                    ))
                } else {
                    BrokerExecutionStatus::TimedOut
                };
                return self.record_and_return(
                    proposal,
                    status,
                    None,
                    output.exit_code,
                    elapsed_ms,
                    output.side_effects_committed,
                    false,
                    effective_now_ms,
                );
            }
        }

        // 8. Bounded output size enforcement
        let bounded_output =
            truncate_output_to_bound(&output.result_text, context.max_output_bytes);

        // Commit to idempotency cache if key is present
        if let Some(key) = &proposal.idempotency_key {
            let committed_result = ToolExecutionResult {
                call_id: proposal.call_id.clone(),
                task_id: proposal.task_id.clone(),
                idempotency_key: Some(key.clone()),
                status: BrokerExecutionStatus::Success,
                output: Some(bounded_output.clone()),
                exit_code: output.exit_code,
                execution_time_ms: elapsed_ms,
                side_effects_committed: output.side_effects_committed,
                from_cache: false,
            };

            let out_len = output.result_text.len();
            let mut cache = self.idempotency_cache.write().unwrap();
            let _ = evict_idempotency_records_if_needed(&mut cache, out_len);
            cache.insert(
                key.clone(),
                IdempotencyRecord {
                    key: key.clone(),
                    session_id: context.session_id.clone(),
                    task_id: proposal.task_id.clone(),
                    tool_name: proposal.tool_name.clone(),
                    tool_version: schema.version.clone(),
                    canonical_arguments: canonical_args,
                    argument_digest: arg_digest,
                    raw_output: Some(output.result_text.clone()),
                    exit_code: output.exit_code,
                    state: IdempotencyRecordState::Committed {
                        result: committed_result,
                    },
                    created_at_ms: effective_now_ms,
                    total_output_bytes: out_len,
                },
            );
        }

        self.record_and_return(
            proposal,
            BrokerExecutionStatus::Success,
            Some(bounded_output),
            output.exit_code,
            elapsed_ms,
            output.side_effects_committed,
            false,
            effective_now_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_and_return(
        &self,
        proposal: &ToolProposal,
        status: BrokerExecutionStatus,
        output: Option<String>,
        exit_code: i32,
        execution_time_ms: f64,
        side_effects_committed: bool,
        from_cache: bool,
        timestamp_ms: u64,
    ) -> ToolExecutionResult {
        let result = ToolExecutionResult {
            call_id: proposal.call_id.clone(),
            task_id: proposal.task_id.clone(),
            idempotency_key: proposal.idempotency_key.clone(),
            status: status.clone(),
            output,
            exit_code,
            execution_time_ms,
            side_effects_committed,
            from_cache,
        };

        // Cache unknown outcomes, timeouts, and execution failures for idempotency replay protection
        if let Some(key) = &proposal.idempotency_key {
            match &status {
                BrokerExecutionStatus::UnknownOutcome(_)
                | BrokerExecutionStatus::TimedOut
                | BrokerExecutionStatus::ExecutionFailed(_) => {
                    let mut cache = self.idempotency_cache.write().unwrap();
                    let canonical_args = canonical_json_string(&proposal.arguments);
                    let arg_digest = compute_argument_digest(&proposal.arguments);
                    let state = match &status {
                        BrokerExecutionStatus::ExecutionFailed(e) => {
                            IdempotencyRecordState::Failed { error: e.clone() }
                        }
                        _ => IdempotencyRecordState::Committed {
                            result: result.clone(),
                        },
                    };
                    let tool_version = self
                        .tools
                        .read()
                        .unwrap()
                        .get(&proposal.tool_name)
                        .map(|t| t.schema().version)
                        .unwrap_or_else(|| "1.0.0".into());

                    let out_len = result.output.as_ref().map(|o| o.len()).unwrap_or(0);
                    let _ = evict_idempotency_records_if_needed(&mut cache, out_len);
                    cache.insert(
                        key.clone(),
                        IdempotencyRecord {
                            key: key.clone(),
                            session_id: None,
                            task_id: proposal.task_id.clone(),
                            tool_name: proposal.tool_name.clone(),
                            tool_version,
                            canonical_arguments: canonical_args,
                            argument_digest: arg_digest,
                            raw_output: result.output.clone(),
                            exit_code,
                            state,
                            created_at_ms: timestamp_ms,
                            total_output_bytes: out_len,
                        },
                    );
                }
                _ => {}
            }
        }

        // Audit log
        {
            let mut audit = self.audit_log.write().unwrap();
            audit.push(ToolAuditRecord {
                timestamp_ms,
                call_id: proposal.call_id.clone(),
                task_id: proposal.task_id.clone(),
                tool_name: proposal.tool_name.clone(),
                status,
                side_effects_committed,
            });
        }

        result
    }

    /// Exports recent audit records.
    pub fn get_audit_records(&self) -> Vec<ToolAuditRecord> {
        self.audit_log.read().unwrap().clone()
    }
}
