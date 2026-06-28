use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub const ASK_USER_TOOL_NAME: &str = "ask_user";

pub struct AskUserTool;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AskUserRequest {
    pub questions: Vec<AskUserQuestion>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AskUserQuestion {
    pub id: String,
    #[serde(default)]
    pub header: Option<String>,
    pub question: String,
    #[serde(default)]
    pub options: Vec<AskUserOption>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AskUserOption {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AskUserResponse {
    pub answers: Vec<AskUserAnswer>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AskUserAnswer {
    Option {
        id: String,
        answer: String,
        option_index: usize,
        option_label: String,
    },
    Freeform {
        id: String,
        answer: String,
    },
}

impl AskUserRequest {
    pub fn parse(input: Value) -> Result<Self, ToolError> {
        let request: AskUserRequest =
            serde_json::from_value(input).map_err(|error| ToolError::InvalidInput {
                tool: ASK_USER_TOOL_NAME.to_string(),
                message: error.to_string(),
            })?;
        request.validate()?;
        Ok(request)
    }

    fn validate(&self) -> Result<(), ToolError> {
        if self.questions.is_empty() {
            return invalid("questions must not be empty");
        }

        let mut ids = BTreeSet::new();
        for question in &self.questions {
            let id = question.id.trim();
            if id.is_empty() {
                return invalid("question id must not be empty");
            }
            if !ids.insert(id.to_string()) {
                return invalid(format!("duplicate question id '{id}'"));
            }
            if question.question.trim().is_empty() {
                return invalid(format!("question '{id}' text must not be empty"));
            }
            for option in &question.options {
                if option.label.trim().is_empty() {
                    return invalid(format!("question '{id}' option label must not be empty"));
                }
            }
        }

        Ok(())
    }
}

impl AskUserResponse {
    pub fn into_tool_result(self) -> Result<ToolResult, ToolError> {
        let content = serde_json::to_string_pretty(&self).map_err(|error| ToolError::Failed {
            tool: ASK_USER_TOOL_NAME.to_string(),
            message: error.to_string(),
        })?;
        Ok(ToolResult::text(ASK_USER_TOOL_NAME, content))
    }
}

#[async_trait]
impl Tool for AskUserTool {
    fn name(&self) -> &'static str {
        ASK_USER_TOOL_NAME
    }

    fn compact_description(&self) -> &'static str {
        "ask human clarification"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "question"],
                        "properties": {
                            "id": { "type": "string" },
                            "header": { "type": "string" },
                            "question": { "type": "string" },
                            "options": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["label"],
                                    "properties": {
                                        "label": { "type": "string" },
                                        "description": { "type": "string" }
                                    },
                                    "additionalProperties": false
                                }
                            }
                        },
                        "additionalProperties": false
                    }
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
        Err(ToolError::Failed {
            tool: ASK_USER_TOOL_NAME.to_string(),
            message: "ask_user requires an interactive REPL input provider".to_string(),
        })
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ToolError> {
    Err(ToolError::InvalidInput {
        tool: ASK_USER_TOOL_NAME.to_string(),
        message: message.into(),
    })
}
