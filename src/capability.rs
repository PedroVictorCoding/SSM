//! Host-Controlled Trust Boundaries and Capability Enforcement.
//!
//! Enforces the permanent foundational alignment rule from `ALIGNMENT.md`:
//! **SAIL must never perform autonomous recursive self-improvement.**
//!
//! This module provides:
//! 1. Host-enforced capability grants (`CapabilityPolicy`).
//! 2. Permanent prohibition of model-originated privileged actions (training, tool registration, self-promotion, policy modification).
//! 3. Protected filesystem path enforcement (protecting source code, ALIGNMENT.md, configuration, checkpoints).
//! 4. Prompt injection detection for untrusted observation content (OCR, transcripts, tool results).
//! 5. Data provenance verification guarding training queues against runtime synthetic output.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Host-controlled permissions that can be granted by a human operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Capability {
    ToolExecution(String),
    ScreenCapture,
    MicrophoneCapture,
    CameraCapture,
    AudioOutput,
    FileSystemRead(PathBuf),
    FileSystemWrite(PathBuf),
    NetworkAccess(String),
}

/// Privileged actions that the model or runtime assistant is PERMANENTLY forbidden
/// from performing. These can never be granted as capabilities to the model.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrivilegedAction {
    StartTraining,
    PromoteModel,
    AlterSecurityPolicy,
    ModifyImplementationCode,
    RegisterNewTool,
    BypassAlignment,
}

impl fmt::Display for PrivilegedAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrivilegedAction::StartTraining => write!(f, "StartTraining"),
            PrivilegedAction::PromoteModel => write!(f, "PromoteModel"),
            PrivilegedAction::AlterSecurityPolicy => write!(f, "AlterSecurityPolicy"),
            PrivilegedAction::ModifyImplementationCode => write!(f, "ModifyImplementationCode"),
            PrivilegedAction::RegisterNewTool => write!(f, "RegisterNewTool"),
            PrivilegedAction::BypassAlignment => write!(f, "BypassAlignment"),
        }
    }
}

/// Security violation errors enforced by the host runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SecurityError {
    AlignmentViolation(String),
    CapabilityNotGranted(String),
    PrivilegedActionProhibited(String),
    ProtectedPathViolation(PathBuf),
    UntrustedSelfTrainingProhibited(String),
    InjectionAttemptBlocked(String),
    MalformedToolCall(String),
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::AlignmentViolation(msg) => write!(f, "[Alignment Violation]: {msg}"),
            SecurityError::CapabilityNotGranted(msg) => {
                write!(f, "[Capability Not Granted]: {msg}")
            }
            SecurityError::PrivilegedActionProhibited(msg) => {
                write!(f, "[Privileged Action Prohibited]: {msg}")
            }
            SecurityError::ProtectedPathViolation(path) => {
                write!(
                    f,
                    "[Protected Path Violation]: Access denied for protected path '{}'",
                    path.display()
                )
            }
            SecurityError::UntrustedSelfTrainingProhibited(msg) => {
                write!(f, "[Self-Training Prohibited]: {msg}")
            }
            SecurityError::InjectionAttemptBlocked(msg) => {
                write!(f, "[Prompt Injection Blocked]: {msg}")
            }
            SecurityError::MalformedToolCall(msg) => write!(f, "[Malformed Tool Call]: {msg}"),
        }
    }
}

impl std::error::Error for SecurityError {}

/// Immutable host-controlled policy governing assistant capabilities.
#[derive(Clone, Debug)]
pub struct CapabilityPolicy {
    granted_capabilities: HashSet<Capability>,
    allowed_read_roots: Vec<PathBuf>,
    allowed_write_roots: Vec<PathBuf>,
    framework_root: PathBuf,
    protected_roots: Vec<PathBuf>,
    audit_log: Arc<Mutex<Vec<String>>>,
}

impl Default for CapabilityPolicy {
    fn default() -> Self {
        let mut granted = HashSet::new();
        // Safe default: basic tool execution of registered read-only tools
        granted.insert(Capability::ToolExecution("web_search".into()));
        let fw_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let canonical_fw = resolve_canonical_target(&fw_root);
        Self {
            granted_capabilities: granted,
            allowed_read_roots: vec![canonical_fw.clone()],
            allowed_write_roots: Vec::new(),
            framework_root: canonical_fw.clone(),
            protected_roots: vec![canonical_fw],
            audit_log: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl CapabilityPolicy {
    /// Create a new policy with explicit human-configured grants.
    pub fn new(
        granted_capabilities: HashSet<Capability>,
        allowed_read_roots: Vec<PathBuf>,
        allowed_write_roots: Vec<PathBuf>,
    ) -> Self {
        let fw_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let canonical_fw = resolve_canonical_target(&fw_root);

        let canonical_read = allowed_read_roots
            .iter()
            .map(|p| resolve_canonical_target(p))
            .collect();
        let canonical_write: Vec<PathBuf> = allowed_write_roots
            .iter()
            .map(|p| resolve_canonical_target(p))
            .collect();

        let mut protected = vec![canonical_fw.clone()];
        // Explicitly protect workspace roots that contain the SAIL ALIGNMENT.md sentinel.
        // Unrelated user projects without ALIGNMENT.md are not auto-protected as SAIL framework roots.
        for root in &canonical_write {
            if root.join("ALIGNMENT.md").exists() {
                if !protected.contains(root) {
                    protected.push(root.clone());
                }
            }
        }

        Self {
            granted_capabilities,
            allowed_read_roots: canonical_read,
            allowed_write_roots: canonical_write,
            framework_root: canonical_fw,
            protected_roots: protected,
            audit_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Explicitly designates a root as an authorized user project where source files
    /// can be modified by the assistant, exempting it from framework source code protection.
    /// SAIL framework roots, their subtrees, ancestors, or roots containing ALIGNMENT.md
    /// are permanently protected and can never have protection removed.
    pub fn with_user_project_root(mut self, root: impl AsRef<Path>) -> Self {
        let canonical = resolve_canonical_target(root.as_ref());
        let is_framework_or_ancestor = canonical == self.framework_root
            || self.framework_root.starts_with(&canonical)
            || canonical.starts_with(&self.framework_root)
            || canonical.join("ALIGNMENT.md").exists();

        if !is_framework_or_ancestor {
            self.protected_roots.retain(|p| p != &canonical);
        } else if !self.protected_roots.contains(&canonical) {
            self.protected_roots.push(canonical.clone());
        }

        if !self.allowed_write_roots.contains(&canonical) {
            self.allowed_write_roots.push(canonical.clone());
        }
        if !self.allowed_read_roots.contains(&canonical) {
            self.allowed_read_roots.push(canonical);
        }
        self
    }

    /// Sets explicit framework root.
    pub fn with_framework_root(mut self, root: impl AsRef<Path>) -> Self {
        let canonical = resolve_canonical_target(root.as_ref());
        self.framework_root = canonical.clone();
        if !self.protected_roots.contains(&canonical) {
            self.protected_roots.push(canonical);
        }
        self
    }

    /// Explicitly adds a root to protected implementation roots.
    pub fn with_protected_root(mut self, root: impl AsRef<Path>) -> Self {
        let canonical = resolve_canonical_target(root.as_ref());
        if !self.protected_roots.contains(&canonical) {
            self.protected_roots.push(canonical);
        }
        self
    }

    /// Check if a capability is explicitly granted.
    pub fn check_capability(&self, cap: &Capability) -> Result<(), SecurityError> {
        if self.granted_capabilities.contains(cap) {
            self.audit(format!("Granted capability checked: {:?}", cap));
            Ok(())
        } else {
            let err = format!("Capability '{:?}' is not granted by host policy", cap);
            self.audit(format!("REJECTED capability: {:?}", cap));
            Err(SecurityError::CapabilityNotGranted(err))
        }
    }

    /// Permanent alignment boundary: privileged actions can NEVER be executed by the model.
    pub fn check_privileged_action(&self, action: &PrivilegedAction) -> Result<(), SecurityError> {
        let err = format!(
            "Permanent Alignment Rule: Model or untrusted runtime context cannot perform '{}'. \
             Training, policy alteration, and self-modification must remain human-controlled.",
            action
        );
        self.audit(format!("PROHIBITED action blocked: {}", action));
        Err(SecurityError::PrivilegedActionProhibited(err))
    }

    /// Verifies if a resolved target path is protected against modification or deletion.
    pub fn is_protected_path(&self, canonical_target: &Path) -> bool {
        // 1. Protected exact filenames anywhere across all workspaces
        if let Some(file_name) = canonical_target.file_name().and_then(|n| n.to_str()) {
            let protected_files = [
                "ALIGNMENT.md",
                "AGENTS.md",
                "Cargo.toml",
                "Cargo.lock",
                "build.rs",
            ];
            for pf in &protected_files {
                if file_name.eq_ignore_ascii_case(pf) {
                    return true;
                }
            }
        }

        // 2. Protected implementation and governance subdirectories under protected roots
        let protected_subdirs = ["src", "tests", "benches", "scripts", ".git", "checkpoints"];

        for root in &self.protected_roots {
            let canonical_root = resolve_canonical_target(root);
            if canonical_target == canonical_root || canonical_root.starts_with(canonical_target) {
                return true;
            }
            for dir in &protected_subdirs {
                let protected_dir = canonical_root.join(dir);
                if canonical_target.starts_with(&protected_dir) {
                    return true;
                }
            }
        }

        false
    }

    /// Verify write permissions against allowed write roots and protected system paths.
    pub fn check_file_write(&self, path: impl AsRef<Path>) -> Result<(), SecurityError> {
        let path = path.as_ref();
        let canonical_target = resolve_canonical_target(path);

        // 1. Strictly forbid writing to protected implementation and alignment paths
        if self.is_protected_path(&canonical_target) {
            self.audit(format!(
                "BLOCKED write to protected path: {}",
                canonical_target.display()
            ));
            return Err(SecurityError::ProtectedPathViolation(canonical_target));
        }

        // 2. Check if path is within an explicitly allowed write root
        let allowed = self.allowed_write_roots.iter().any(|root| {
            let canonical_root = resolve_canonical_target(root);
            canonical_target.starts_with(&canonical_root)
        });

        if allowed {
            self.audit(format!("Allowed write to: {}", canonical_target.display()));
            Ok(())
        } else {
            self.audit(format!(
                "REJECTED unpermitted write to: {}",
                canonical_target.display()
            ));
            Err(SecurityError::CapabilityNotGranted(format!(
                "Write access to '{}' is not within configured write roots",
                canonical_target.display()
            )))
        }
    }

    /// Verify deletion permissions: deletion is a permanent write mutation.
    pub fn check_file_delete(&self, path: impl AsRef<Path>) -> Result<(), SecurityError> {
        self.check_file_write(path)
    }

    /// Verify file rename permissions.
    pub fn check_file_rename(
        &self,
        from: impl AsRef<Path>,
        to: impl AsRef<Path>,
    ) -> Result<(), SecurityError> {
        self.check_file_write(from)?;
        self.check_file_write(to)
    }

    /// Verify read permissions against allowed read roots.
    pub fn check_file_read(&self, path: impl AsRef<Path>) -> Result<(), SecurityError> {
        let path = path.as_ref();
        let canonical_target = resolve_canonical_target(path);
        let allowed = self.allowed_read_roots.iter().any(|root| {
            let canonical_root = resolve_canonical_target(root);
            canonical_target.starts_with(&canonical_root)
        });

        if allowed {
            Ok(())
        } else {
            self.audit(format!(
                "REJECTED unpermitted read: {}",
                canonical_target.display()
            ));
            Err(SecurityError::CapabilityNotGranted(format!(
                "Read access to '{}' is not within configured read roots",
                canonical_target.display()
            )))
        }
    }

    fn audit(&self, msg: String) {
        if let Ok(mut log) = self.audit_log.lock() {
            log.push(msg);
        }
    }

    pub fn audit_entries(&self) -> Vec<String> {
        self.audit_log.lock().map(|l| l.clone()).unwrap_or_default()
    }
}

/// Resolves a path to its canonical target, or canonicalizes the longest existing ancestor
/// and appends the remaining uncanonicalized components, preventing symlink escape and traversal bypasses.
pub fn resolve_canonical_target(path: &Path) -> PathBuf {
    if let Ok(canonical) = path.canonicalize() {
        return canonical;
    }

    let mut current = path.to_path_buf();
    let mut tail = Vec::new();

    while !current.exists() {
        if let Some(file_name) = current.file_name() {
            tail.push(file_name.to_os_string());
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let check_base = if current.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        current
    };
    let mut base = check_base.canonicalize().unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else if let Ok(cwd) = std::env::current_dir() {
            cwd.canonicalize().unwrap_or(cwd)
        } else {
            PathBuf::from(".")
        }
    });
    tail.reverse();
    for comp in tail {
        base.push(comp);
    }
    normalize_path(&base)
}

/// Identifies files and directories critical to Sail framework integrity,
/// alignment rules, and compilation.
pub fn is_protected_implementation_path(path: &Path) -> bool {
    let canonical = resolve_canonical_target(path);
    let s = canonical.to_string_lossy().replace('\\', "/");
    let components: Vec<&str> = s
        .split('/')
        .filter(|c| !c.is_empty() && *c != ".")
        .collect();

    if components.is_empty() {
        return false;
    }

    // Protected exact filenames
    let protected_files = [
        "ALIGNMENT.md",
        "AGENTS.md",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
    ];

    for file in &protected_files {
        if components.iter().any(|c| c.eq_ignore_ascii_case(file)) {
            return true;
        }
    }

    // Protected implementation and git directories
    let protected_dirs = ["src", "tests", "benches", "scripts", ".git", "checkpoints"];

    for dir in &protected_dirs {
        if components.iter().any(|c| c.eq_ignore_ascii_case(dir)) {
            return true;
        }
    }

    false
}

/// Normalizes path by resolving `.` and `..` components lexically.
pub fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

/// Untrusted runtime content from OCR, speech transcripts, tool responses, or web snippets.
#[derive(Clone, Debug)]
pub struct UntrustedContent<'a> {
    pub source: &'a str,
    pub raw_text: &'a str,
}

impl<'a> UntrustedContent<'a> {
    pub fn new(source: &'a str, raw_text: &'a str) -> Self {
        Self { source, raw_text }
    }

    /// Scans untrusted input for prompt-injection attacks attempting privilege escalation,
    /// alignment override, or model self-training triggers.
    pub fn scan_and_sanitize(&self) -> Result<&'a str, SecurityError> {
        let lower = self.raw_text.to_lowercase();

        // High-risk prompt injection and privilege escalation patterns
        let injection_signatures = [
            "system override:",
            "override alignment",
            "bypass alignment",
            "grant all capabilities",
            "grant capability",
            "escalate privilege",
            "<tool_call>register_tool",
            "<tool_call>start_training",
            "<tool_call>train_stream",
            "<tool_call>promote_model",
            "<tool_call>alter_policy",
            "write alignment.md",
            "delete alignment.md",
            "modify alignment.md",
        ];

        for sig in &injection_signatures {
            if lower.contains(sig) {
                return Err(SecurityError::InjectionAttemptBlocked(format!(
                    "Adversarial signature '{sig}' detected in untrusted content from source '{}'",
                    self.source
                )));
            }
        }

        Ok(self.raw_text)
    }
}

/// Provenance metadata tracking the origin of datasets and records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProvenanceKind {
    /// Externally sourced, human-approved dataset (Hugging Face, verified benchmark, etc.)
    HumanApprovedExternalDataset {
        source_url: String,
        human_approver: String,
        content_hash: String,
    },
    /// Model-generated text, self-critique, or synthetic reasoning trace
    RuntimeSyntheticOutput { agent_id: String, timestamp: u64 },
    /// Live audio or screen recording captured during an interactive user session
    LiveCapturedSession { session_id: String },
}

/// Verifies whether a candidate data batch or document is permitted to enter the training queue.
///
/// Implements the permanent rule:
/// Model-generated outputs and unverified interactive session captures MUST NOT be fed back
/// into training or reward pipelines.
pub fn verify_training_provenance(provenance: &ProvenanceKind) -> Result<(), SecurityError> {
    match provenance {
        ProvenanceKind::HumanApprovedExternalDataset { .. } => Ok(()),
        ProvenanceKind::RuntimeSyntheticOutput { agent_id, .. } => {
            Err(SecurityError::UntrustedSelfTrainingProhibited(format!(
                "Model-generated output from agent '{}' cannot be ingested into training queue. \
                 Autonomous recursive self-improvement is strictly forbidden by ALIGNMENT.md.",
                agent_id
            )))
        }
        ProvenanceKind::LiveCapturedSession { session_id } => {
            Err(SecurityError::UntrustedSelfTrainingProhibited(format!(
                "Interactive session capture '{}' has not undergone human review and curation. \
                 Raw assistant runtime observations cannot automatically enter the training queue.",
                session_id
            )))
        }
    }
}
