//! Workspace Tool Adapters: Scoped Search, Safe File Reading/Writing, and Structured Test Runner.
//!
//! Provides:
//! - Canonical path boundary checks preventing path-traversal outside allowed roots.
//! - File version and modification hash verification guarding against stale edits.
//! - Structured test execution using explicit argument vectors (NO arbitrary shell evaluation).
//! - Clean speech summaries and structured artifacts for model and human verification.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::json;

use crate::storage::atomic_write;
use crate::tool::artifact::StructuredToolArtifact;
use crate::tool::broker::{ToolExecutionOutput, TypedTool};
use crate::tool::schema::{ParameterType, ToolParameter, ToolSafetyLevel, ToolSchema};

/// Scoped file search tool across permitted workspace directories.
pub struct WorkspaceFileSearchTool {
    schema: ToolSchema,
    allowed_roots: Vec<PathBuf>,
}

impl WorkspaceFileSearchTool {
    pub fn new(allowed_roots: Vec<PathBuf>) -> Self {
        let schema = ToolSchema::new(
            "workspace_file_search",
            "1.0.0",
            "Searches for files matching a pattern within permitted workspace directories",
            ToolSafetyLevel::ReadOnly,
            true, // Idempotent
        )
        .with_parameter(ToolParameter {
            name: "query".into(),
            param_type: ParameterType::String,
            description: "File name substring or search keyword".into(),
            required: true,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "root_dir".into(),
            param_type: ParameterType::String,
            description: "Directory path to search under (must fall within allowed roots)".into(),
            required: false,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "extension".into(),
            param_type: ParameterType::String,
            description: "Optional file extension filter without dot (e.g. 'rs')".into(),
            required: false,
            enum_values: None,
        });

        Self {
            schema,
            allowed_roots: if allowed_roots.is_empty() {
                vec![PathBuf::from(".")]
            } else {
                allowed_roots
            },
        }
    }

    fn is_path_allowed(&self, path: &Path) -> bool {
        let canonical = crate::capability::resolve_canonical_target(path);
        self.allowed_roots.iter().any(|root| {
            let c_root = crate::capability::resolve_canonical_target(root);
            canonical.starts_with(&c_root)
        })
    }
}

impl TypedTool for WorkspaceFileSearchTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn execute(&self, args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        self.execute_with_context(args, &crate::tool::ToolExecutionContext::default())
    }

    fn execute_with_context(
        &self,
        args: &serde_json::Value,
        context: &crate::tool::ToolExecutionContext,
    ) -> Result<ToolExecutionOutput, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'query' parameter".to_string())?;

        let root_str = args.get("root_dir").and_then(|v| v.as_str()).unwrap_or(".");
        let ext_opt = args.get("extension").and_then(|v| v.as_str());

        // Host capability policy check
        if let Some(policy) = &context.policy {
            policy
                .check_file_read(root_str)
                .map_err(|e| e.to_string())?;
        }

        let root_path = PathBuf::from(root_str);
        if !self.is_path_allowed(&root_path) {
            return Err(format!(
                "Access denied: directory '{}' is outside allowed workspace roots",
                root_str
            ));
        }

        let mut matches = Vec::new();
        let mut stack = vec![root_path];
        let mut visited_dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        let max_traversal_entries = 5000;
        let mut entries_count = 0;

        while let Some(dir) = stack.pop() {
            let canonical_dir = crate::capability::resolve_canonical_target(&dir);
            if !visited_dirs.insert(canonical_dir) {
                continue;
            }

            if let Ok(entries) = fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    entries_count += 1;
                    if entries_count > max_traversal_entries {
                        break;
                    }

                    let path = entry.path();
                    let canonical = crate::capability::resolve_canonical_target(&path);
                    if !self.is_path_allowed(&canonical) {
                        continue;
                    }

                    // Enforce authoritative host capability policy at each traversed entry
                    if let Some(policy) = &context.policy {
                        if policy.check_file_read(&canonical).is_err() {
                            continue;
                        }
                    }

                    if path.is_dir() {
                        // Skip common hidden or heavy directories
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        if !name.starts_with('.') && name != "target" && name != "node_modules" {
                            stack.push(path);
                        }
                    } else if path.is_file() {
                        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                        let matches_query = name.to_lowercase().contains(&query.to_lowercase());
                        let matches_ext = ext_opt
                            .map(|ext| path.extension().and_then(|e| e.to_str()) == Some(ext))
                            .unwrap_or(true);

                        if matches_query && matches_ext {
                            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                            matches.push(json!({
                                "path": path.display().to_string(),
                                "bytes": size
                            }));
                        }
                    }
                }
            }
        }

        let summary = format!("Found {} matching files for '{}'.", matches.len(), query);
        let artifact = StructuredToolArtifact::new(
            "workspace_file_search",
            true,
            summary,
            json!({ "matches": matches }),
        );

        Ok(ToolExecutionOutput {
            result_text: serde_json::to_string_pretty(&artifact).unwrap_or_default(),
            exit_code: 0,
            side_effects_committed: false,
        })
    }
}

/// Scoped file read tool with version verification and line slice bounds.
pub struct WorkspaceFileReadTool {
    schema: ToolSchema,
    allowed_roots: Vec<PathBuf>,
}

impl WorkspaceFileReadTool {
    pub fn new(allowed_roots: Vec<PathBuf>) -> Self {
        let schema = ToolSchema::new(
            "workspace_file_read",
            "1.0.0",
            "Reads content from a permitted workspace file with version and line range bounds",
            ToolSafetyLevel::ReadOnly,
            true, // Idempotent
        )
        .with_parameter(ToolParameter {
            name: "path".into(),
            param_type: ParameterType::String,
            description: "Path to file to read".into(),
            required: true,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "start_line".into(),
            param_type: ParameterType::Integer,
            description: "1-indexed start line (inclusive)".into(),
            required: false,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "end_line".into(),
            param_type: ParameterType::Integer,
            description: "1-indexed end line (inclusive)".into(),
            required: false,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "expected_mtime_ms".into(),
            param_type: ParameterType::Integer,
            description: "Expected modification time in ms for concurrency checks".into(),
            required: false,
            enum_values: None,
        });

        Self {
            schema,
            allowed_roots: if allowed_roots.is_empty() {
                vec![PathBuf::from(".")]
            } else {
                allowed_roots
            },
        }
    }

    fn is_path_allowed(&self, path: &Path) -> bool {
        let canonical = crate::capability::resolve_canonical_target(path);
        self.allowed_roots.iter().any(|root| {
            let c_root = crate::capability::resolve_canonical_target(root);
            canonical.starts_with(&c_root)
        })
    }
}

impl TypedTool for WorkspaceFileReadTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn execute(&self, args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        self.execute_with_context(args, &crate::tool::ToolExecutionContext::default())
    }

    fn execute_with_context(
        &self,
        args: &serde_json::Value,
        context: &crate::tool::ToolExecutionContext,
    ) -> Result<ToolExecutionOutput, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'path' parameter".to_string())?;

        // 1. Host capability policy check
        if let Some(policy) = &context.policy {
            policy
                .check_file_read(path_str)
                .map_err(|e| e.to_string())?;
        }

        let path = PathBuf::from(path_str);
        if !self.is_path_allowed(&path) {
            return Err(format!(
                "Access denied: '{}' is outside allowed roots",
                path_str
            ));
        }

        let metadata = fs::metadata(&path)
            .map_err(|e| format!("Failed to stat file '{}': {e}", path.display()))?;

        // Stale file version check
        if let Some(expected_mtime) = args.get("expected_mtime_ms").and_then(|v| v.as_i64()) {
            let actual_mtime = metadata
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);

            if actual_mtime > expected_mtime {
                return Err(format!(
                    "StaleFileVersion: File '{}' was modified externally (expected mtime {}, actual mtime {})",
                    path.display(),
                    expected_mtime,
                    actual_mtime
                ));
            }
        }

        let content = fs::read_to_string(&path)
            .map_err(|e| format!("Failed to read file '{}': {e}", path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        let start_line = args
            .get("start_line")
            .and_then(|v| v.as_i64())
            .unwrap_or(1)
            .max(1) as usize;

        let end_line = args
            .get("end_line")
            .and_then(|v| v.as_i64())
            .map(|e| (e as usize).min(total_lines))
            .unwrap_or(total_lines);

        let slice = if start_line <= total_lines {
            let actual_end = end_line.min(total_lines);
            lines[start_line - 1..actual_end].join("\n")
        } else {
            String::new()
        };

        let summary = format!(
            "Read lines {}-{} of {} from '{}'.",
            start_line,
            end_line.min(total_lines),
            total_lines,
            path.display()
        );

        let artifact = StructuredToolArtifact::new(
            "workspace_file_read",
            true,
            summary,
            json!({
                "path": path.display().to_string(),
                "total_lines": total_lines,
                "start_line": start_line,
                "end_line": end_line.min(total_lines),
                "content": slice
            }),
        );

        Ok(ToolExecutionOutput {
            result_text: serde_json::to_string_pretty(&artifact).unwrap_or_default(),
            exit_code: 0,
            side_effects_committed: false,
        })
    }
}

/// Scoped file write tool with atomic commitment, safety gating, and version checking.
pub struct WorkspaceFileWriteTool {
    schema: ToolSchema,
    allowed_roots: Vec<PathBuf>,
}

impl WorkspaceFileWriteTool {
    pub fn new(allowed_roots: Vec<PathBuf>) -> Self {
        let schema = ToolSchema::new(
            "workspace_file_write",
            "1.0.0",
            "Writes content to a permitted workspace file atomically with version protection",
            ToolSafetyLevel::DestructiveOrExternal,
            false, // Non-idempotent write
        )
        .with_parameter(ToolParameter {
            name: "path".into(),
            param_type: ParameterType::String,
            description: "Path to write".into(),
            required: true,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "content".into(),
            param_type: ParameterType::String,
            description: "Complete text content to write".into(),
            required: true,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "expected_mtime_ms".into(),
            param_type: ParameterType::Integer,
            description: "Expected mtime in ms before write to prevent clobbering concurrent edits"
                .into(),
            required: false,
            enum_values: None,
        });

        Self {
            schema,
            allowed_roots: if allowed_roots.is_empty() {
                vec![PathBuf::from(".")]
            } else {
                allowed_roots
            },
        }
    }

    fn is_path_allowed(&self, path: &Path) -> bool {
        let canonical = crate::capability::resolve_canonical_target(path);
        self.allowed_roots.iter().any(|root| {
            let c_root = crate::capability::resolve_canonical_target(root);
            canonical.starts_with(&c_root)
        })
    }
}

impl TypedTool for WorkspaceFileWriteTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn execute(&self, args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        self.execute_with_context(args, &crate::tool::ToolExecutionContext::default())
    }

    fn execute_with_context(
        &self,
        args: &serde_json::Value,
        context: &crate::tool::ToolExecutionContext,
    ) -> Result<ToolExecutionOutput, String> {
        let path_str = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'path' parameter".to_string())?;

        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'content' parameter".to_string())?;

        // 1. Host capability policy check
        if let Some(policy) = &context.policy {
            policy
                .check_file_write(path_str)
                .map_err(|e| e.to_string())?;
        } else {
            let canonical = crate::capability::resolve_canonical_target(Path::new(path_str));
            if crate::capability::CapabilityPolicy::default().is_protected_path(&canonical) {
                return Err(format!(
                    "Access denied: target path '{}' is a protected system or governance path",
                    path_str
                ));
            }
        }

        let path = PathBuf::from(path_str);
        if !self.is_path_allowed(&path) {
            return Err(format!(
                "Access denied: '{}' is outside allowed write roots",
                path_str
            ));
        }

        // Version check if file exists
        if path.exists() {
            if let Some(expected_mtime) = args.get("expected_mtime_ms").and_then(|v| v.as_i64()) {
                let actual = fs::metadata(&path)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_millis() as i64)
                    .unwrap_or(0);

                if actual > expected_mtime {
                    return Err(format!(
                        "StaleFileVersion: Target file '{}' changed on disk since last inspected",
                        path.display()
                    ));
                }
            }
        }

        // Atomic write
        atomic_write(&path, content.as_bytes())
            .map_err(|e| format!("Failed to atomically write '{}': {e}", path.display()))?;

        let summary = format!(
            "Successfully updated '{}' ({} bytes).",
            path.display(),
            content.len()
        );
        let artifact = StructuredToolArtifact::new(
            "workspace_file_write",
            true,
            summary,
            json!({
                "path": path.display().to_string(),
                "bytes_written": content.len()
            }),
        );

        Ok(ToolExecutionOutput {
            result_text: serde_json::to_string_pretty(&artifact).unwrap_or_default(),
            exit_code: 0,
            side_effects_committed: true,
        })
    }
}

/// Structured test runner executing targeted cargo test invocations without arbitrary shell eval.
pub struct WorkspaceTestRunnerTool {
    schema: ToolSchema,
    working_dir: PathBuf,
}

impl WorkspaceTestRunnerTool {
    pub fn new(working_dir: PathBuf) -> Self {
        let schema = ToolSchema::new(
            "workspace_test_runner",
            "1.0.0",
            "Executes specific workspace tests with structured result parsing and timeouts",
            ToolSafetyLevel::DestructiveOrExternal, // Test code is executable code with potential side effects, not an inherently read-only observation
            true, // Test execution queries test suite outcomes; timing out returns TimedOut
        )
        .with_parameter(ToolParameter {
            name: "test_name".into(),
            param_type: ParameterType::String,
            description: "Exact test name or filter string".into(),
            required: true,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "package".into(),
            param_type: ParameterType::String,
            description: "Specific cargo package (default: 'sail')".into(),
            required: false,
            enum_values: None,
        })
        .with_parameter(ToolParameter {
            name: "release".into(),
            param_type: ParameterType::Boolean,
            description: "Whether to run with --release flag".into(),
            required: false,
            enum_values: None,
        });

        Self {
            schema,
            working_dir,
        }
    }
}

fn kill_child_process_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        if pid > 1 {
            unsafe {
                // Signal negative PID to terminate the entire process group
                libc::kill(-pid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn spawn_stream_drain<R: std::io::Read + Send + 'static>(
    stream: Option<R>,
    buf: Arc<Mutex<Vec<u8>>>,
    trunc: Arc<AtomicBool>,
    max_bytes: usize,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        if let Some(mut r) = stream {
            let mut chunk = [0u8; 4096];
            while let Ok(n) = r.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                let mut guard = buf.lock().unwrap();
                if guard.len() < max_bytes {
                    let to_take = n.min(max_bytes - guard.len());
                    guard.extend_from_slice(&chunk[..to_take]);
                    if n > to_take {
                        trunc.store(true, Ordering::SeqCst);
                    }
                } else {
                    trunc.store(true, Ordering::SeqCst);
                }
            }
        }
    })
}

impl TypedTool for WorkspaceTestRunnerTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn execute(&self, args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        self.execute_with_context(args, &crate::tool::ToolExecutionContext::default())
    }

    fn execute_with_context(
        &self,
        args: &serde_json::Value,
        context: &crate::tool::ToolExecutionContext,
    ) -> Result<ToolExecutionOutput, String> {
        let test_name = args
            .get("test_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing 'test_name' parameter".to_string())?;

        let is_release = args
            .get("release")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let package = args.get("package").and_then(|v| v.as_str());

        let start = Instant::now();

        // Build command safely as argument vector (NO shell injection possible)
        let mut cmd = Command::new("cargo");
        cmd.arg("test")
            .arg(test_name)
            .arg("--offline")
            .current_dir(&self.working_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0); // Child becomes process group leader with pgid == child pid
        }

        if let Some(pkg) = package {
            cmd.arg("-p").arg(pkg);
        }

        if is_release {
            cmd.arg("--release");
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn cargo test: {e}"))?;

        let max_capture_bytes: usize = context.max_output_bytes.max(64 * 1024);
        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));
        let stdout_trunc = Arc::new(AtomicBool::new(false));
        let stderr_trunc = Arc::new(AtomicBool::new(false));

        let out_handle = spawn_stream_drain(
            stdout_pipe,
            Arc::clone(&stdout_buf),
            Arc::clone(&stdout_trunc),
            max_capture_bytes,
        );
        let err_handle = spawn_stream_drain(
            stderr_pipe,
            Arc::clone(&stderr_buf),
            Arc::clone(&stderr_trunc),
            max_capture_bytes,
        );

        // Supervise child process with cancellation and deadline
        let status = loop {
            // Check cancellation token
            if let Some(token) = &context.cancellation_token {
                if token.is_cancelled() {
                    kill_child_process_tree(&mut child);
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return Err("CANCELLED: process terminated by cancellation token".into());
                }
            }

            // Check deadline
            if let Some(dl) = context.deadline_ms {
                let elapsed_ms = start.elapsed().as_millis() as u64;
                let exceeded = elapsed_ms > dl
                    || (context.current_timestamp_ms > 0
                        && context.current_timestamp_ms.saturating_add(elapsed_ms) > dl);
                if exceeded {
                    kill_child_process_tree(&mut child);
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return Err(format!(
                        "TIMED_OUT: process terminated because deadline of {}ms was exceeded",
                        dl
                    ));
                }
            }

            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(e) => {
                    kill_child_process_tree(&mut child);
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return Err(format!("Error waiting for cargo test process: {e}"));
                }
            }
        };

        let _ = out_handle.join();
        let _ = err_handle.join();

        let stdout_bytes = stdout_buf.lock().unwrap().clone();
        let stderr_bytes = stderr_buf.lock().unwrap().clone();
        let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
        let stderr = String::from_utf8_lossy(&stderr_bytes).to_string();
        let duration_secs = start.elapsed().as_secs_f64();
        let exit_code = status.code().unwrap_or(-1);

        // Aggregate test results accurately across all target binaries
        let mut total_passed = 0;
        let mut total_failed = 0;
        let mut target_runs = 0;
        let mut status_ok = status.success();

        for line in stdout.lines() {
            if line.contains("test result:") {
                target_runs += 1;
                if let Some(passed_str) = line
                    .split("passed;")
                    .next()
                    .and_then(|s| s.split('.').last())
                {
                    let p = passed_str.trim().parse::<usize>().unwrap_or(0);
                    total_passed += p;
                }
                if let Some(failed_part) = line.split("passed;").nth(1) {
                    if let Some(failed_str) = failed_part.split("failed;").next() {
                        let f = failed_str.trim().parse::<usize>().unwrap_or(0);
                        total_failed += f;
                    }
                }
                if line.contains("FAILED") {
                    status_ok = false;
                }
            }
        }

        let speech_summary = if status_ok && total_passed > 0 {
            format!(
                "Test '{}' passed in {:.2}s: {} tests passed across {} targets, 0 failed.",
                test_name, duration_secs, total_passed, target_runs
            )
        } else if status_ok && total_passed == 0 {
            format!(
                "Test filter '{}' finished in {:.2}s: 0 matching tests executed.",
                test_name, duration_secs
            )
        } else {
            format!(
                "Test '{}' failed in {:.2}s with {} failures (exit code {}).",
                test_name, duration_secs, total_failed, exit_code
            )
        };

        let failure_snippet = if !status_ok || total_failed > 0 {
            let combined = format!("{stdout}\n{stderr}");
            let lines: Vec<&str> = combined
                .lines()
                .filter(|l| l.contains("error") || l.contains("FAILED") || l.contains("panicked"))
                .collect();
            Some(lines.join("\n"))
        } else {
            None
        };

        let artifact = StructuredToolArtifact::new(
            "workspace_test_runner",
            status_ok && total_passed > 0 && total_failed == 0 && exit_code == 0,
            speech_summary,
            json!({
                "test_name": test_name,
                "package": package.unwrap_or("sail"),
                "passed": total_passed,
                "failed": total_failed,
                "target_runs": target_runs,
                "duration_secs": duration_secs,
                "exit_code": exit_code,
                "workspace": self.working_dir.display().to_string(),
                "failure_snippet": failure_snippet
            }),
        );

        Ok(ToolExecutionOutput {
            result_text: serde_json::to_string_pretty(&artifact).unwrap_or_default(),
            exit_code,
            side_effects_committed: true,
        })
    }
}
