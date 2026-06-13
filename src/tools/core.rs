use crate::llm::ImageAttachment;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool '{0}' was not found")]
    UnknownTool(String),
    #[error("invalid input for tool '{tool}': {message}")]
    InvalidInput { tool: String, message: String },
    #[error("tool '{tool}' failed: {message}")]
    Failed { tool: String, message: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    pub name: String,
    pub content: String,
    #[serde(default)]
    pub refresh_manifest: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ImageAttachment>,
}

impl ToolResult {
    pub fn text(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            content: content.into(),
            refresh_manifest: false,
            images: Vec::new(),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn compact_description(&self) -> &'static str;
    fn json_schema(&self) -> Value;
    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError>;

    fn openai_schema(&self) -> Value {
        json!({
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.compact_description(),
                "parameters": self.json_schema()
            }
        })
    }
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<&'static str, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn core() -> Self {
        let mut registry = Self::default();
        registry.insert(crate::tools::file::ReadFileTool);
        registry.insert(crate::tools::image::ReadImageTool);
        registry.insert(crate::tools::file::WriteFileTool);
        registry.insert(crate::tools::file::EditFileTool);
        registry.insert(crate::tools::batch::BatchTool);
        registry.insert(RefreshManifestTool);
        registry
    }

    pub fn insert<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name(), Box::new(tool));
    }

    pub fn schemas(&self) -> Vec<Value> {
        self.tools
            .values()
            .map(|tool| tool.openai_schema())
            .collect()
    }

    pub fn compact_descriptions(&self) -> String {
        self.tools
            .values()
            .map(|tool| format!("{}:{}", tool.name(), tool.compact_description()))
            .collect::<Vec<_>>()
            .join("; ")
    }

    pub async fn execute(&self, name: &str, input: Value) -> Result<ToolResult, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::UnknownTool(name.to_string()))?;
        tool.execute(input).await
    }
}

struct RefreshManifestTool;

#[async_trait]
impl Tool for RefreshManifestTool {
    fn name(&self) -> &'static str {
        "refresh_manifest"
    }

    fn compact_description(&self) -> &'static str {
        "rescan host manifest"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "additionalProperties": false
        })
    }

    async fn execute(&self, _input: Value) -> Result<ToolResult, ToolError> {
        let mut result = ToolResult::text(self.name(), "manifest refresh requested");
        result.refresh_manifest = true;
        Ok(result)
    }
}
