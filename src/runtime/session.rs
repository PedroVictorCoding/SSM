//! Host-Owned Explicit Session & Task State Machine, Bounded Model Memory, and Crash Recovery.
//!
//! Provides:
//! - Exact host-owned record of user intent, request revisions, and verified task outcomes
//!   independently from model recurrent/KV caches.
//! - Fine-grained task lifecycle (`Pending`, `InProgress`, `AwaitingClarification`, `Completed`, `Interrupted`, `Failed`).
//! - Reconciliation across interruptions and worker restarts: prevents replaying obsolete actions
//!   and marks in-flight calls as `UncertainOutcome` rather than hazardous blind retries.
//! - Strict multi-session state isolation and bounded per-session recurrent/KV memory limits.
//! - Inspectable session reports and explicit session reset/termination without weight changes.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::runtime::dialogue::{ToolExecutionStatus, TrackedToolExecution};

/// Formal lifecycle status of a user task within a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskLifecycleStatus {
    Pending,
    InProgress,
    AwaitingClarification,
    Completed,
    Interrupted,
    Failed,
}

/// Recorded revision to a user request or objective (e.g. user barge-in correction).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskRevision {
    pub revision_id: u32,
    pub timestamp_ms: u64,
    pub revised_objective: String,
    pub reason: String,
}

/// Host-owned state of a conversational task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionTask {
    pub task_id: String,
    pub initial_prompt: String,
    pub current_objective: String,
    pub revisions: Vec<TaskRevision>,
    pub status: TaskLifecycleStatus,
    pub tool_executions: Vec<TrackedToolExecution>,
    pub delivered_speech: String,
    pub unspoken_speech: Option<String>,
    pub observation_references: Vec<String>,
    pub start_timestamp_ms: u64,
    pub completion_timestamp_ms: Option<u64>,
}

impl SessionTask {
    pub fn new(
        task_id: impl Into<String>,
        initial_prompt: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        let p = initial_prompt.into();
        Self {
            task_id: task_id.into(),
            initial_prompt: p.clone(),
            current_objective: p,
            revisions: Vec::new(),
            status: TaskLifecycleStatus::Pending,
            tool_executions: Vec::new(),
            delivered_speech: String::new(),
            unspoken_speech: None,
            observation_references: Vec::new(),
            start_timestamp_ms: timestamp_ms,
            completion_timestamp_ms: None,
        }
    }

    /// Records a user correction or redirection, updating current objective without losing history.
    pub fn revise_objective(&mut self, new_objective: &str, reason: &str, timestamp_ms: u64) {
        let rev_id = self.revisions.len() as u32 + 1;
        self.revisions.push(TaskRevision {
            revision_id: rev_id,
            timestamp_ms,
            revised_objective: new_objective.to_string(),
            reason: reason.to_string(),
        });
        self.current_objective = new_objective.to_string();
        self.status = TaskLifecycleStatus::InProgress;
    }

    /// Records completed speech delivered through the speaker.
    pub fn record_delivered_speech(&mut self, speech: &str, unspoken: Option<String>) {
        self.delivered_speech = speech.to_string();
        self.unspoken_speech = unspoken;
    }

    /// Checks if a tool call was already completed with committed side effects.
    pub fn has_completed_tool(&self, tool_name: &str, input: &str) -> bool {
        self.tool_executions.iter().any(|t| {
            t.tool_name == tool_name
                && t.input == input
                && t.status == ToolExecutionStatus::Completed
                && t.side_effects_committed
        })
    }
}

/// Bounded per-session model memory (recurrent SSM state / KV cache tracker).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoundedSessionCache {
    pub max_tokens_retained: usize,
    pub current_token_count: usize,
    pub ssm_step_count: u64,
    pub last_compacted_ms: u64,
}

impl Default for BoundedSessionCache {
    fn default() -> Self {
        Self {
            max_tokens_retained: 2048,
            current_token_count: 0,
            ssm_step_count: 0,
            last_compacted_ms: 0,
        }
    }
}

impl BoundedSessionCache {
    pub fn update_usage(&mut self, tokens: usize, ssm_steps: u64) {
        self.current_token_count = tokens.min(self.max_tokens_retained);
        self.ssm_step_count += ssm_steps;
    }

    pub fn reset(&mut self) {
        self.current_token_count = 0;
        self.ssm_step_count = 0;
    }
}

/// Complete record of an active or archived user session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub session_id: String,
    pub created_at_ms: u64,
    pub last_active_ms: u64,
    pub active_task: Option<SessionTask>,
    pub archived_tasks: Vec<SessionTask>,
    pub cache_metrics: BoundedSessionCache,
    pub is_terminated: bool,
}

/// Inspectable snapshot of session status.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionInspectionReport {
    pub session_id: String,
    pub is_terminated: bool,
    pub active_objective: Option<String>,
    pub task_status: Option<TaskLifecycleStatus>,
    pub revisions_count: usize,
    pub completed_tools_count: usize,
    pub pending_or_uncertain_tools_count: usize,
    pub delivered_speech: String,
    pub memory_tokens: usize,
    pub ssm_steps: u64,
}

/// Thread-safe host-owned session manager.
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<String, SessionRecord>>>,
    max_active_sessions: usize,
}

impl SessionManager {
    pub fn new(max_active_sessions: usize) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            max_active_sessions: max_active_sessions.max(1),
        }
    }

    /// Obtains or initializes an isolated session.
    pub fn get_or_create_session(&self, session_id: &str, timestamp_ms: u64) -> SessionRecord {
        let mut map = self.sessions.write().unwrap();

        // Enforce bounded sessions capacity: evict oldest terminated or inactive session if full
        if map.len() >= self.max_active_sessions && !map.contains_key(session_id) {
            let oldest_key = map
                .iter()
                .min_by_key(|(_, s)| (s.is_terminated as u8, s.last_active_ms))
                .map(|(k, _)| k.clone());
            if let Some(k) = oldest_key {
                map.remove(&k);
            }
        }

        map.entry(session_id.to_string())
            .or_insert_with(|| SessionRecord {
                session_id: session_id.to_string(),
                created_at_ms: timestamp_ms,
                last_active_ms: timestamp_ms,
                active_task: None,
                archived_tasks: Vec::new(),
                cache_metrics: BoundedSessionCache::default(),
                is_terminated: false,
            })
            .clone()
    }

    /// Starts a new task within a session.
    pub fn start_task(
        &self,
        session_id: &str,
        task_id: &str,
        prompt: &str,
        timestamp_ms: u64,
    ) -> Result<(), String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        if session.is_terminated {
            return Err(format!("Session '{session_id}' has been terminated"));
        }

        // Archive prior task if incomplete
        if let Some(mut prev) = session.active_task.take() {
            if prev.status == TaskLifecycleStatus::InProgress
                || prev.status == TaskLifecycleStatus::Pending
            {
                prev.status = TaskLifecycleStatus::Interrupted;
            }
            session.archived_tasks.push(prev);
        }

        let task = SessionTask::new(task_id, prompt, timestamp_ms);
        session.active_task = Some(task);
        session.last_active_ms = timestamp_ms;

        Ok(())
    }

    /// Updates active task objective following user interruption or clarification.
    pub fn revise_active_objective(
        &self,
        session_id: &str,
        new_objective: &str,
        reason: &str,
        timestamp_ms: u64,
    ) -> Result<(), String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        let task = session
            .active_task
            .as_mut()
            .ok_or_else(|| "No active task in session".to_string())?;

        task.revise_objective(new_objective, reason, timestamp_ms);
        session.last_active_ms = timestamp_ms;
        Ok(())
    }

    /// Records tool execution result and side-effect commitment into host state.
    pub fn record_tool_result(
        &self,
        session_id: &str,
        tool: TrackedToolExecution,
    ) -> Result<(), String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        let task = session
            .active_task
            .as_mut()
            .ok_or_else(|| "No active task in session".to_string())?;

        // Update existing call if present, or push
        if let Some(pos) = task
            .tool_executions
            .iter()
            .position(|t| t.call_id == tool.call_id)
        {
            task.tool_executions[pos] = tool;
        } else {
            task.tool_executions.push(tool);
        }

        Ok(())
    }

    /// Marks the active task as completed with confirmed delivered speech.
    pub fn complete_active_task(
        &self,
        session_id: &str,
        delivered_speech: &str,
        unspoken: Option<String>,
        timestamp_ms: u64,
    ) -> Result<(), String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        let mut task = session
            .active_task
            .take()
            .ok_or_else(|| "No active task to complete".to_string())?;

        task.status = TaskLifecycleStatus::Completed;
        task.record_delivered_speech(delivered_speech, unspoken);
        task.completion_timestamp_ms = Some(timestamp_ms);

        session.archived_tasks.push(task);
        session.last_active_ms = timestamp_ms;

        Ok(())
    }

    /// Simulates worker crash recovery: restores state from committed records
    /// and reconciles in-flight tasks/tools as `UncertainOutcome`.
    pub fn recover_from_worker_crash(
        &self,
        session_id: &str,
    ) -> Result<SessionInspectionReport, String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        // Reconcile active task tools: any tool left in Running becomes UncertainOutcome
        if let Some(task) = session.active_task.as_mut() {
            for tool in &mut task.tool_executions {
                if tool.status == ToolExecutionStatus::Running {
                    tool.status = ToolExecutionStatus::UncertainOutcome;
                    tool.output = Some(
                        "Worker restart occurred while tool was running; side effects unconfirmed"
                            .into(),
                    );
                } else if tool.status == ToolExecutionStatus::NotStarted {
                    tool.status = ToolExecutionStatus::Cancelled;
                }
            }
            task.status = TaskLifecycleStatus::AwaitingClarification;
        }

        // Reset volatile model cache (KV / SSM) while strictly preserving host database
        session.cache_metrics.reset();

        let report = Self::build_inspection_report(session);
        Ok(report)
    }

    /// Resets model caches and active task for a session without changing model weights.
    pub fn reset_session(&self, session_id: &str) -> Result<(), String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        session.active_task = None;
        session.cache_metrics.reset();
        Ok(())
    }

    /// Explicitly terminates a session.
    pub fn terminate_session(&self, session_id: &str) -> Result<(), String> {
        let mut map = self.sessions.write().unwrap();
        let session = map
            .get_mut(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        session.is_terminated = true;
        session.active_task = None;
        session.cache_metrics.reset();
        Ok(())
    }

    /// Generates an inspectable report for telemetry and user auditing.
    pub fn inspect_session(&self, session_id: &str) -> Result<SessionInspectionReport, String> {
        let map = self.sessions.read().unwrap();
        let session = map
            .get(session_id)
            .ok_or_else(|| format!("Session '{session_id}' not found"))?;

        Ok(Self::build_inspection_report(session))
    }

    fn build_inspection_report(session: &SessionRecord) -> SessionInspectionReport {
        let (active_obj, status, rev_count, comp_count, uncert_count, speech) =
            if let Some(t) = &session.active_task {
                let comp = t
                    .tool_executions
                    .iter()
                    .filter(|x| x.status == ToolExecutionStatus::Completed)
                    .count();
                let uncert = t
                    .tool_executions
                    .iter()
                    .filter(|x| {
                        x.status == ToolExecutionStatus::Running
                            || x.status == ToolExecutionStatus::UncertainOutcome
                    })
                    .count();
                (
                    Some(t.current_objective.clone()),
                    Some(t.status),
                    t.revisions.len(),
                    comp,
                    uncert,
                    t.delivered_speech.clone(),
                )
            } else {
                (None, None, 0, 0, 0, String::new())
            };

        SessionInspectionReport {
            session_id: session.session_id.clone(),
            is_terminated: session.is_terminated,
            active_objective: active_obj,
            task_status: status,
            revisions_count: rev_count,
            completed_tools_count: comp_count,
            pending_or_uncertain_tools_count: uncert_count,
            delivered_speech: speech,
            memory_tokens: session.cache_metrics.current_token_count,
            ssm_steps: session.cache_metrics.ssm_step_count,
        }
    }
}
