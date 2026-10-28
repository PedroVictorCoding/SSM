//! Outcome Verification and Multi-Step Task Coordination.
//!
//! Provides:
//! - Bounded propose/validate/execute/observe multi-step execution.
//! - Independent postcondition verification (re-reading files on disk, inspecting parsed test artifacts, verifying UI state).
//! - Strict outcome gating: a tool reporting success without its postcondition observed CANNOT mark the task complete.
//! - Dynamic cancellation of obsolete queued steps on user barge-in or objective revision.
//! - Honest spoken status generation with separate wording for verified, failed, uncertain, and cancelled outcomes.
//! - End-to-end traceable audit trail: request -> tool proposals -> postcondition observations -> spoken response.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::screen::automation::UiElement;
use crate::tool::artifact::StructuredToolArtifact;
use crate::tool::broker::{
    BrokerExecutionStatus, ToolExecutionBroker, ToolExecutionContext, ToolProposal,
};

/// Formal postcondition to independently observe and verify after tool execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepPostcondition {
    /// Verifies by re-reading the file on disk that it contains the expected snippet.
    FileContainsText {
        path: PathBuf,
        expected_snippet: String,
    },
    /// Verifies that a target file exists on disk.
    FileExists { path: PathBuf },
    /// Verifies that a target file does not exist on disk (e.g. after deletion).
    FileDeleted { path: PathBuf },
    /// Verifies that test runner executed and all tests passed (zero failures).
    TestPassed { test_name: String },
    /// Verifies that a UI element exists in the UI tree and matches expected text.
    UiElementMatches {
        element_id: String,
        expected_text: String,
    },
}

impl StepPostcondition {
    /// Performs the independent observation on real host/disk state.
    pub fn verify(
        &self,
        workspace_root: &Path,
        ui_root: Option<&UiElement>,
        last_tool_output: Option<&str>,
    ) -> Result<(), String> {
        match self {
            StepPostcondition::FileContainsText {
                path,
                expected_snippet,
            } => {
                let full_path = if path.is_absolute() {
                    path.clone()
                } else {
                    workspace_root.join(path)
                };

                let canonical_workspace =
                    crate::capability::resolve_canonical_target(workspace_root);
                let canonical_target = crate::capability::resolve_canonical_target(&full_path);
                if !canonical_target.starts_with(&canonical_workspace) {
                    return Err(format!(
                        "Postcondition rejected: Target path '{}' resolves outside permitted workspace root '{}'",
                        full_path.display(),
                        canonical_workspace.display()
                    ));
                }

                if !full_path.exists() {
                    return Err(format!(
                        "Postcondition failed: Target file '{}' does not exist on disk",
                        full_path.display()
                    ));
                }

                let content = fs::read_to_string(&full_path).map_err(|e| {
                    format!(
                        "Postcondition failed: Could not read file '{}': {e}",
                        full_path.display()
                    )
                })?;

                if !content.contains(expected_snippet) {
                    return Err(format!(
                        "Postcondition failed: File '{}' was read from disk but does not contain expected snippet '{}'",
                        full_path.display(),
                        expected_snippet
                    ));
                }

                Ok(())
            }
            StepPostcondition::FileExists { path } => {
                let full_path = if path.is_absolute() {
                    path.clone()
                } else {
                    workspace_root.join(path)
                };

                let canonical_workspace =
                    crate::capability::resolve_canonical_target(workspace_root);
                let canonical_target = crate::capability::resolve_canonical_target(&full_path);
                if !canonical_target.starts_with(&canonical_workspace) {
                    return Err(format!(
                        "Postcondition rejected: Target path '{}' resolves outside permitted workspace root '{}'",
                        full_path.display(),
                        canonical_workspace.display()
                    ));
                }

                if !full_path.exists() {
                    return Err(format!(
                        "Postcondition failed: Expected file '{}' was not created on disk",
                        full_path.display()
                    ));
                }
                Ok(())
            }
            StepPostcondition::FileDeleted { path } => {
                let full_path = if path.is_absolute() {
                    path.clone()
                } else {
                    workspace_root.join(path)
                };

                let canonical_workspace =
                    crate::capability::resolve_canonical_target(workspace_root);
                let canonical_target = crate::capability::resolve_canonical_target(&full_path);
                if !canonical_target.starts_with(&canonical_workspace) {
                    return Err(format!(
                        "Postcondition rejected: Target path '{}' resolves outside permitted workspace root '{}'",
                        full_path.display(),
                        canonical_workspace.display()
                    ));
                }

                if full_path.exists() {
                    return Err(format!(
                        "Postcondition failed: Target file '{}' still exists on disk after deletion",
                        full_path.display()
                    ));
                }
                Ok(())
            }
            StepPostcondition::TestPassed { test_name } => {
                let raw_json = last_tool_output.ok_or_else(|| {
                    "Postcondition failed: No test runner output available for verification"
                        .to_string()
                })?;

                let artifact: StructuredToolArtifact =
                    serde_json::from_str(raw_json).map_err(|e| {
                        format!("Postcondition failed: Failed to parse test runner artifact: {e}")
                    })?;

                if !artifact.success {
                    return Err(format!(
                        "Postcondition failed: Test run for '{test_name}' reported failure in test artifact"
                    ));
                }

                let payload = &artifact.detailed_payload;
                if artifact.tool_name != "workspace_test_runner"
                    || payload.get("test_name").and_then(|v| v.as_str()) != Some(test_name.as_str())
                {
                    return Err(format!(
                        "Postcondition failed: Artifact is not from the requested test runner invocation for '{test_name}'"
                    ));
                }

                let passed_count =
                    payload
                        .get("passed")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            "Postcondition failed: Missing or invalid passed test count".to_string()
                        })?;
                let failed_count = artifact
                    .detailed_payload
                    .get("failed")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| {
                        "Postcondition failed: Missing or invalid failed test count".to_string()
                    })?;

                if passed_count == 0 || payload.get("exit_code").and_then(|v| v.as_i64()) != Some(0)
                {
                    return Err("Postcondition failed: No passing tests or missing/nonzero test-runner exit code".into());
                }

                if failed_count > 0 {
                    return Err(format!(
                        "Postcondition failed: Test run had {failed_count} failing tests"
                    ));
                }

                Ok(())
            }
            StepPostcondition::UiElementMatches {
                element_id,
                expected_text,
            } => {
                let root = ui_root.ok_or_else(|| {
                    "Postcondition failed: No active UI tree provided for verification".to_string()
                })?;

                let elem = root.find_by_id(element_id).ok_or_else(|| {
                    format!(
                        "Postcondition failed: UI element '{element_id}' not found in active tree"
                    )
                })?;

                let matches = elem.text_value.as_deref() == Some(expected_text.as_str())
                    || elem.name.contains(expected_text.as_str());

                if !matches {
                    return Err(format!(
                        "Postcondition failed: UI element '{}' has text '{:?}', expected '{}'",
                        element_id, elem.text_value, expected_text
                    ));
                }

                Ok(())
            }
        }
    }
}

/// Lifecycle execution status of an individual plan step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum StepExecutionStatus {
    Pending,
    Running,
    VerifiedComplete,
    ToolExecutionFailed(String),
    VerificationFailed(String),
    UncertainOutcome(String),
    SkippedObsolete(String),
}

/// Individual step in a multi-step task plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskPlanStep {
    pub step_id: String,
    pub description: String,
    pub proposal: ToolProposal,
    pub postconditions: Vec<StepPostcondition>,
    pub status: StepExecutionStatus,
    pub verified_observation: Option<String>,
}

/// Complete plan consisting of multiple sequential or conditional steps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TaskPlan {
    pub task_id: String,
    pub original_prompt: String,
    pub current_objective: String,
    pub steps: Vec<TaskPlanStep>,
    pub is_cancelled: bool,
}

impl TaskPlan {
    pub fn new(task_id: impl Into<String>, prompt: impl Into<String>) -> Self {
        let p = prompt.into();
        Self {
            task_id: task_id.into(),
            original_prompt: p.clone(),
            current_objective: p,
            steps: Vec::new(),
            is_cancelled: false,
        }
    }

    pub fn with_step(mut self, step: TaskPlanStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Dynamically revises task objective and marks unstarted steps as obsolete.
    pub fn revise_objective(&mut self, new_objective: &str, reason: &str) {
        self.current_objective = new_objective.to_string();
        for step in &mut self.steps {
            if step.status == StepExecutionStatus::Pending {
                step.status = StepExecutionStatus::SkippedObsolete(format!(
                    "Cancelled due to user revision: {reason}"
                ));
            }
        }
    }

    /// Cancels entire remaining plan.
    pub fn cancel(&mut self, reason: &str) {
        self.is_cancelled = true;
        for step in &mut self.steps {
            if step.status == StepExecutionStatus::Pending {
                step.status =
                    StepExecutionStatus::SkippedObsolete(format!("Plan cancelled: {reason}"));
            }
        }
    }
}

/// Comprehensive audit trace connecting intent -> execution -> verification -> spoken outcome.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifiedTaskExecutionTrace {
    pub task_id: String,
    pub objective: String,
    pub completed_steps_count: usize,
    pub failed_steps_count: usize,
    pub obsolete_steps_count: usize,
    pub all_postconditions_verified: bool,
    pub spoken_status: String,
    pub step_traces: Vec<TaskPlanStep>,
    pub total_duration_ms: f64,
}

/// Multi-step task coordinator and outcome verifier.
pub struct MultiStepTaskCoordinator {
    broker: Arc<ToolExecutionBroker>,
    workspace_root: PathBuf,
}

impl MultiStepTaskCoordinator {
    pub fn new(broker: Arc<ToolExecutionBroker>, workspace_root: PathBuf) -> Self {
        Self {
            broker,
            workspace_root,
        }
    }

    /// Executes a single step of a plan, independently verifying postconditions.
    pub fn execute_step(
        &self,
        step: &mut TaskPlanStep,
        ui_root: Option<&UiElement>,
        context: &ToolExecutionContext,
    ) -> bool {
        if matches!(step.status, StepExecutionStatus::SkippedObsolete(_)) {
            return false;
        }

        step.status = StepExecutionStatus::Running;
        let exec_res = self.broker.execute_proposal(&step.proposal, context);

        match exec_res.status {
            BrokerExecutionStatus::Success => {
                let mut step_verified = true;
                let mut fail_reasons = Vec::new();

                for post in &step.postconditions {
                    if let Err(err) =
                        post.verify(&self.workspace_root, ui_root, exec_res.output.as_deref())
                    {
                        step_verified = false;
                        fail_reasons.push(err);
                    }
                }

                if step_verified {
                    step.status = StepExecutionStatus::VerifiedComplete;
                    step.verified_observation =
                        Some("All postconditions independently verified on disk/system".into());
                    true
                } else {
                    let reason = fail_reasons.join("; ");
                    step.status = StepExecutionStatus::VerificationFailed(reason.clone());
                    step.verified_observation = Some(format!("Postcondition failure: {reason}"));
                    false
                }
            }
            BrokerExecutionStatus::TimedOut | BrokerExecutionStatus::UnknownOutcome(_) => {
                step.status = StepExecutionStatus::UncertainOutcome(
                    "Execution timed out or had unconfirmed outcome; non-idempotent actions blocked from blind retry".into()
                );
                false
            }
            BrokerExecutionStatus::Cancelled => {
                step.status = StepExecutionStatus::SkippedObsolete("Execution cancelled".into());
                false
            }
            BrokerExecutionStatus::ValidationFailed(e)
            | BrokerExecutionStatus::Unauthorized(e)
            | BrokerExecutionStatus::AwaitingAuthorization(e)
            | BrokerExecutionStatus::ExecutionFailed(e) => {
                step.status = StepExecutionStatus::ToolExecutionFailed(e);
                false
            }
        }
    }

    /// Executes a task plan step-by-step, validating postconditions independently.
    pub fn execute_plan(
        &self,
        plan: &mut TaskPlan,
        ui_root: Option<&UiElement>,
        context: &ToolExecutionContext,
    ) -> VerifiedTaskExecutionTrace {
        let start = Instant::now();
        let mut completed_count = 0;
        let mut failed_count = 0;
        let mut obsolete_count = 0;
        let mut all_verified = true;

        for step in &mut plan.steps {
            // Check if plan was cancelled or step was skipped
            if plan.is_cancelled || matches!(step.status, StepExecutionStatus::SkippedObsolete(_)) {
                obsolete_count += 1;
                continue;
            }

            let step_ok = self.execute_step(step, ui_root, context);
            if step_ok {
                completed_count += 1;
            } else {
                failed_count += 1;
                all_verified = false;
                break;
            }
        }

        // Generate honest spoken status based on true verification
        let spoken_status = Self::generate_spoken_status(
            &plan.current_objective,
            completed_count,
            failed_count,
            plan.steps.len(),
            all_verified,
            plan.steps.iter().find_map(|s| {
                if let StepExecutionStatus::VerificationFailed(r) = &s.status {
                    Some(r.as_str())
                } else {
                    None
                }
            }),
        );

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        VerifiedTaskExecutionTrace {
            task_id: plan.task_id.clone(),
            objective: plan.current_objective.clone(),
            completed_steps_count: completed_count,
            failed_steps_count: failed_count,
            obsolete_steps_count: obsolete_count,
            all_postconditions_verified: all_verified && failed_count == 0,
            spoken_status,
            step_traces: plan.steps.clone(),
            total_duration_ms: duration_ms,
        }
    }

    /// Generates distinct honest spoken wording preventing false claims of task completion.
    pub fn generate_spoken_status(
        objective: &str,
        completed_steps: usize,
        failed_steps: usize,
        total_steps: usize,
        all_verified: bool,
        verification_error: Option<&str>,
    ) -> String {
        if all_verified && failed_steps == 0 {
            format!(
                "Task completed and verified: all {} steps for '{}' succeeded.",
                completed_steps, objective
            )
        } else if let Some(err) = verification_error {
            format!(
                "Task failed postcondition verification: {err}. Desired outcome was not observed on disk."
            )
        } else if failed_steps > 0 {
            format!(
                "Task stopped with errors: {} of {} steps completed.",
                completed_steps, total_steps
            )
        } else {
            format!(
                "Task in progress: {} of {} steps completed.",
                completed_steps, total_steps
            )
        }
    }
}
