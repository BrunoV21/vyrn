use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

const COMMAND_TIMEOUT_SECONDS: u64 = 120;

pub struct BatchTool;

#[derive(Debug, Deserialize)]
struct BatchInput {
    commands: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BatchCommandResult {
    command: String,
    status: Option<i32>,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

#[async_trait]
impl Tool for BatchTool {
    fn name(&self) -> &'static str {
        "batch"
    }

    fn compact_description(&self) -> &'static str {
        "run shell commands"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["commands"],
            "properties": {
                "commands": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let input: BatchInput =
            serde_json::from_value(input).map_err(|error| ToolError::InvalidInput {
                tool: self.name().to_string(),
                message: error.to_string(),
            })?;

        let mut results = Vec::with_capacity(input.commands.len());
        for command in input.commands {
            results.push(run_command(command).await?);
        }

        let content =
            serde_json::to_string_pretty(&results).map_err(|error| ToolError::Failed {
                tool: self.name().to_string(),
                message: error.to_string(),
            })?;

        Ok(ToolResult {
            name: self.name().to_string(),
            content,
            refresh_manifest: false,
        })
    }
}

async fn run_command(command: String) -> Result<BatchCommandResult, ToolError> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut child = Command::new(shell);
    child.arg("-lc").arg(&command);

    let output = match timeout(Duration::from_secs(COMMAND_TIMEOUT_SECONDS), child.output()).await {
        Ok(output) => output?,
        Err(_) => {
            return Ok(BatchCommandResult {
                command,
                status: None,
                stdout: String::new(),
                stderr: format!("timed out after {COMMAND_TIMEOUT_SECONDS}s"),
                timed_out: true,
            });
        }
    };

    Ok(BatchCommandResult {
        command,
        status: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        timed_out: false,
    })
}
