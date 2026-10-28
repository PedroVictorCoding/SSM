//! Built-in Typed Tools for the Execution Broker.
//!
//! Provides:
//! - `TypedWebSearchTool`: Read-only, idempotent local knowledge lookup.
//! - `TypedFileReadTool`: Read-only, idempotent file reading bounded by paths.
//! - `TypedFileDeleteTool`: Destructive, non-idempotent file deletion requiring user confirmation.
//! - `TypedCommandTool`: External process execution requiring explicit authorization.

use std::fs;
use std::path::PathBuf;

use crate::tool::broker::{ToolExecutionOutput, TypedTool};
use crate::tool::schema::{ParameterType, ToolParameter, ToolSafetyLevel, ToolSchema};

/// Typed Web Search and factual lookup tool.
pub struct TypedWebSearchTool {
    schema: ToolSchema,
    web_search: crate::tool::WebSearchTool,
}

impl Default for TypedWebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TypedWebSearchTool {
    pub fn new() -> Self {
        let schema = ToolSchema::new(
            "web_search",
            "1.0.0",
            "Searches local verified knowledge index for facts and entities",
            ToolSafetyLevel::ReadOnly,
            true, // Idempotent
        )
        .with_parameter(ToolParameter {
            name: "query".into(),
            param_type: ParameterType::String,
            description: "Search query string".into(),
            required: true,
            enum_values: None,
        });

        Self {
            schema,
            web_search: crate::tool::WebSearchTool::new(),
        }
    }
}

impl TypedTool for TypedWebSearchTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    fn execute(&self, args: &serde_json::Value) -> Result<ToolExecutionOutput, String> {
        let query = args
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing query parameter".to_string())?;

        let res = crate::tool::Tool::execute(&self.web_search, query)?;
        Ok(ToolExecutionOutput {
            result_text: res,
            exit_code: 0,
            side_effects_committed: false,
        })
    }
}

/// Typed File Reader Tool.
pub struct TypedFileReadTool {
    schema: ToolSchema,
}

impl Default for TypedFileReadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TypedFileReadTool {
    pub fn new() -> Self {
        let schema = ToolSchema::new(
            "file_read",
            "1.0.0",
            "Reads the contents of an approved local file",
            ToolSafetyLevel::ReadOnly,
            true, // Idempotent
        )
        .with_parameter(ToolParameter {
            name: "path".into(),
            param_type: ParameterType::String,
            description: "Relative or absolute path to read".into(),
            required: true,
            enum_values: None,
        });

        Self { schema }
    }
}

impl TypedTool for TypedFileReadTool {
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
            .ok_or_else(|| "Missing path parameter".to_string())?;

        // Enforce file read policy
        if let Some(policy) = &context.policy {
            policy
                .check_file_read(path_str)
                .map_err(|e| e.to_string())?;
        } else {
            crate::capability::CapabilityPolicy::default()
                .check_file_read(path_str)
                .map_err(|e| e.to_string())?;
        }

        let path = PathBuf::from(path_str);
        if !path.exists() {
            return Ok(ToolExecutionOutput {
                result_text: format!("File not found: {}", path.display()),
                exit_code: 1,
                side_effects_committed: false,
            });
        }

        match fs::read_to_string(&path) {
            Ok(content) => Ok(ToolExecutionOutput {
                result_text: content,
                exit_code: 0,
                side_effects_committed: false,
            }),
            Err(e) => Err(format!("Failed to read file '{}': {e}", path.display())),
        }
    }
}

/// Typed Destructive File Deletion Tool (Non-Idempotent, DestructiveOrExternal).
pub struct TypedFileDeleteTool {
    schema: ToolSchema,
}

impl Default for TypedFileDeleteTool {
    fn default() -> Self {
        Self::new()
    }
}

impl TypedFileDeleteTool {
    pub fn new() -> Self {
        let schema = ToolSchema::new(
            "file_delete",
            "1.0.0",
            "Deletes an unneeded temporary or user-specified file",
            ToolSafetyLevel::DestructiveOrExternal,
            false, // Non-idempotent: second attempt on non-existent file differs
        )
        .with_parameter(ToolParameter {
            name: "path".into(),
            param_type: ParameterType::String,
            description: "Path of the file to delete".into(),
            required: true,
            enum_values: None,
        });

        Self { schema }
    }
}

impl TypedTool for TypedFileDeleteTool {
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
            .ok_or_else(|| "Missing path parameter".to_string())?;

        // Enforce file deletion / write policy
        if let Some(policy) = &context.policy {
            policy
                .check_file_delete(path_str)
                .map_err(|e| e.to_string())?;
        } else {
            crate::capability::CapabilityPolicy::default()
                .check_file_delete(path_str)
                .map_err(|e| e.to_string())?;
        }

        let path = PathBuf::from(path_str);
        if !path.exists() {
            return Ok(ToolExecutionOutput {
                result_text: format!("Cannot delete non-existent file: {}", path.display()),
                exit_code: 1,
                side_effects_committed: false,
            });
        }

        match fs::remove_file(&path) {
            Ok(()) => Ok(ToolExecutionOutput {
                result_text: format!("Successfully deleted file: {}", path.display()),
                exit_code: 0,
                side_effects_committed: true, // Permanent side effect committed
            }),
            Err(e) => Err(format!("Failed to delete file '{}': {e}", path.display())),
        }
    }
}
