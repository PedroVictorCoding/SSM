//! Versioned Tool Schemas, Typed Parameter Validation, and Safety Classification.
//!
//! Enforces:
//! - Strict schema validation: parameter existence, type correctness, enum constraints.
//! - Rejection of unexpected extra fields when strict schemas are enabled.
//! - Epistemic and operational safety classification (`ReadOnly`, `ReversibleWrite`, `DestructiveOrExternal`).
//! - Idempotency declaration for replay and retry safety.
//! - Injection mitigation on string arguments.

use serde::{Deserialize, Serialize};

/// Supported types for tool parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array(Box<ParameterType>),
    Object,
}

/// Specification of a single tool parameter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub param_type: ParameterType,
    pub description: String,
    pub required: bool,
    pub enum_values: Option<Vec<String>>,
}

/// Operational safety level of a tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolSafetyLevel {
    /// Safe read-only inspection (e.g. file read, local search, system status).
    ReadOnly,
    /// Isolated, undoable local write (e.g. scratch file creation, draft buffer update).
    ReversibleWrite,
    /// Permanent, irreversible or externally visible action (e.g. file deletion, arbitrary shell, network write).
    /// Requires standing user authorization or explicit interactive confirmation.
    DestructiveOrExternal,
}

/// Versioned schema defining a tool's contract and execution safety boundaries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub version: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
    pub strict_extra_fields: bool,
    pub safety_level: ToolSafetyLevel,
    pub is_idempotent: bool,
}

impl ToolSchema {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        description: impl Into<String>,
        safety_level: ToolSafetyLevel,
        is_idempotent: bool,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            description: description.into(),
            parameters: Vec::new(),
            strict_extra_fields: true,
            safety_level,
            is_idempotent,
        }
    }

    pub fn with_parameter(mut self, param: ToolParameter) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn with_strict_extra_fields(mut self, strict: bool) -> Self {
        self.strict_extra_fields = strict;
        self
    }

    /// Validates provided JSON arguments against the schema.
    pub fn validate_arguments(&self, args: &serde_json::Value) -> Result<(), String> {
        let map = args.as_object().ok_or_else(|| {
            format!(
                "Tool arguments for '{}' must be a JSON object, got: {:?}",
                self.name, args
            )
        })?;

        // 1. Check for unexpected extra fields if strict
        if self.strict_extra_fields {
            for key in map.keys() {
                if !self.parameters.iter().any(|p| &p.name == key) {
                    return Err(format!(
                        "Validation error for '{}': unexpected extra parameter '{}'",
                        self.name, key
                    ));
                }
            }
        }

        // 2. Validate all declared parameters
        for param in &self.parameters {
            match map.get(&param.name) {
                Some(val) if !val.is_null() => {
                    self.validate_value_type(&param.name, val, &param.param_type)?;
                    if let Some(allowed) = &param.enum_values {
                        let str_val = val.as_str().ok_or_else(|| {
                            format!(
                                "Parameter '{}' must be a string for enum validation",
                                param.name
                            )
                        })?;
                        if !allowed.iter().any(|v| v == str_val) {
                            return Err(format!(
                                "Parameter '{}' value '{}' not in allowed enum values: {:?}",
                                param.name, str_val, allowed
                            ));
                        }
                    }
                }
                _ => {
                    if param.required {
                        return Err(format!(
                            "Validation error for '{}': missing required parameter '{}'",
                            self.name, param.name
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    fn validate_value_type(
        &self,
        param_name: &str,
        val: &serde_json::Value,
        expected: &ParameterType,
    ) -> Result<(), String> {
        match expected {
            ParameterType::String => {
                let s = val
                    .as_str()
                    .ok_or_else(|| format!("Parameter '{param_name}' must be a string"))?;
                // Check injection indicators
                let lower = s.to_lowercase();
                if lower.contains("ignore previous instructions")
                    || lower.contains("bypass alignment")
                    || lower.contains("<override_policy>")
                {
                    return Err(format!(
                        "Parameter '{param_name}' contains prohibited prompt injection patterns"
                    ));
                }
            }
            ParameterType::Integer => {
                if !val.is_i64() && !val.is_u64() {
                    return Err(format!("Parameter '{param_name}' must be an integer"));
                }
            }
            ParameterType::Float => {
                if !val.is_f64() && !val.is_i64() {
                    return Err(format!(
                        "Parameter '{param_name}' must be a float or number"
                    ));
                }
            }
            ParameterType::Boolean => {
                if !val.is_boolean() {
                    return Err(format!("Parameter '{param_name}' must be a boolean"));
                }
            }
            ParameterType::Array(inner) => {
                let arr = val
                    .as_array()
                    .ok_or_else(|| format!("Parameter '{param_name}' must be an array"))?;
                for (idx, item) in arr.iter().enumerate() {
                    self.validate_value_type(&format!("{param_name}[{idx}]"), item, inner)?;
                }
            }
            ParameterType::Object => {
                if !val.is_object() {
                    return Err(format!("Parameter '{param_name}' must be an object"));
                }
            }
        }
        Ok(())
    }
}
