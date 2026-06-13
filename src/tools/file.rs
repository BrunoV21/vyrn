use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

const READ_LIMIT_BYTES: u64 = 200_000;

pub struct ReadFileTool;
pub struct WriteFileTool;
pub struct EditFileTool;

#[derive(Debug, Deserialize)]
struct ReadFileInput {
    path: String,
}

#[derive(Debug, Deserialize)]
struct WriteFileInput {
    path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct EditFileInput {
    path: String,
    old: String,
    new: String,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn compact_description(&self) -> &'static str {
        "read file path"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": { "type": "string" }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let input: ReadFileInput = parse_input(self.name(), input)?;
        let path = resolve_path(&input.path)?;
        let metadata = tokio::fs::metadata(&path).await?;
        if metadata.len() > READ_LIMIT_BYTES {
            return Err(ToolError::Failed {
                tool: self.name().to_string(),
                message: format!(
                    "{} is {} bytes, above read limit {}",
                    path.display(),
                    metadata.len(),
                    READ_LIMIT_BYTES
                ),
            });
        }
        let content = tokio::fs::read_to_string(&path).await?;
        Ok(ToolResult {
            name: self.name().to_string(),
            content,
            refresh_manifest: false,
        })
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &'static str {
        "write_file"
    }

    fn compact_description(&self) -> &'static str {
        "write file path"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let input: WriteFileInput = parse_input(self.name(), input)?;
        let path = resolve_path(&input.path)?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, input.content).await?;
        Ok(ToolResult {
            name: self.name().to_string(),
            content: format!("wrote {}", path.display()),
            refresh_manifest: false,
        })
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &'static str {
        "edit_file"
    }

    fn compact_description(&self) -> &'static str {
        "exact string replace"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["path", "old", "new"],
            "properties": {
                "path": { "type": "string" },
                "old": { "type": "string" },
                "new": { "type": "string" }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let input: EditFileInput = parse_input(self.name(), input)?;
        if input.old.is_empty() {
            return Err(ToolError::InvalidInput {
                tool: self.name().to_string(),
                message: "old string must not be empty".to_string(),
            });
        }

        let path = resolve_path(&input.path)?;
        let content = tokio::fs::read_to_string(&path).await?;
        let count = content.matches(&input.old).count();
        if count == 0 {
            return Err(ToolError::Failed {
                tool: self.name().to_string(),
                message: "old string was not found".to_string(),
            });
        }
        if count > 1 {
            return Err(ToolError::Failed {
                tool: self.name().to_string(),
                message: format!("old string matched {count} times; expected exactly one"),
            });
        }

        let edited = content.replacen(&input.old, &input.new, 1);
        tokio::fs::write(&path, edited).await?;
        Ok(ToolResult {
            name: self.name().to_string(),
            content: format!("edited {}", path.display()),
            refresh_manifest: false,
        })
    }
}

fn parse_input<T: for<'de> Deserialize<'de>>(tool: &str, input: Value) -> Result<T, ToolError> {
    serde_json::from_value(input).map_err(|error| ToolError::InvalidInput {
        tool: tool.to_string(),
        message: error.to_string(),
    })
}

fn resolve_path(path: &str) -> Result<PathBuf, ToolError> {
    let path = Path::new(path);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}
