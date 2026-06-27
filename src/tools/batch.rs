use crate::tools::{Tool, ToolError, ToolResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::time::{Duration, timeout};

const COMMAND_TIMEOUT_SECONDS: u64 = 120;
const COMMAND_STREAM_LIMIT_CHARS: usize = 8000;
const BATCH_STREAM_TOTAL_LIMIT_CHARS: usize = 16_000;
const MIN_COMMAND_STREAM_LIMIT_CHARS: usize = 120;

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
        "run shell commands in cwd by default"
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

        let stream_limit = batch_stream_limit(input.commands.len());
        let mut results = Vec::with_capacity(input.commands.len());
        for command in input.commands {
            results.push(run_command(command, stream_limit).await?);
        }

        let content =
            serde_json::to_string_pretty(&results).map_err(|error| ToolError::Failed {
                tool: self.name().to_string(),
                message: error.to_string(),
            })?;

        Ok(ToolResult::text(self.name(), content))
    }
}

fn batch_stream_limit(command_count: usize) -> usize {
    let stream_count = command_count.saturating_mul(2).max(1);
    let limit = BATCH_STREAM_TOTAL_LIMIT_CHARS / stream_count;
    limit.clamp(MIN_COMMAND_STREAM_LIMIT_CHARS, COMMAND_STREAM_LIMIT_CHARS)
}

async fn run_command(
    command: String,
    stream_limit: usize,
) -> Result<BatchCommandResult, ToolError> {
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut child = Command::new(shell);
    child.arg("-lc").arg(&command);
    child.kill_on_drop(true);

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
        stdout: trim_stream(&String::from_utf8_lossy(&output.stdout), stream_limit),
        stderr: trim_stream(&String::from_utf8_lossy(&output.stderr), stream_limit),
        timed_out: false,
    })
}

fn trim_stream(value: &str, stream_limit: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= stream_limit {
        return value.to_string();
    }
    let head_chars = stream_limit / 2;
    let tail_chars = stream_limit - head_chars;
    let head = value.chars().take(head_chars).collect::<String>();
    let tail = value
        .chars()
        .skip(char_count.saturating_sub(tail_chars))
        .collect::<String>();
    format!(
        "{head}\n[trimmed {} chars from batch output]\n{tail}",
        char_count - stream_limit
    )
}
