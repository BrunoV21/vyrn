use crate::tools::{Tool, ToolError, ToolResult};
use crate::vision;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

pub struct ReadImageTool;

#[derive(Debug, Deserialize)]
struct ReadImageInput {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    paths: Vec<String>,
}

#[async_trait]
impl Tool for ReadImageTool {
    fn name(&self) -> &'static str {
        "read_image"
    }

    fn compact_description(&self) -> &'static str {
        "attach image paths"
    }

    fn json_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "PNG/JPG/JPEG/WEBP/GIF image paths to attach"
                },
                "path": {
                    "type": "string",
                    "description": "Single image path; prefer paths for multiple images"
                }
            },
            "additionalProperties": false
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolResult, ToolError> {
        let input: ReadImageInput =
            serde_json::from_value(input).map_err(|error| ToolError::InvalidInput {
                tool: self.name().to_string(),
                message: error.to_string(),
            })?;
        let mut paths = input.paths;
        if let Some(path) = input.path {
            paths.push(path);
        }
        if paths.is_empty() {
            return Err(ToolError::InvalidInput {
                tool: self.name().to_string(),
                message: "expected path or paths".to_string(),
            });
        }

        let images = vision::attachments_from_paths(&paths)
            .await
            .map_err(|error| ToolError::Failed {
                tool: self.name().to_string(),
                message: error.to_string(),
            })?;
        if images.is_empty() {
            return Err(ToolError::Failed {
                tool: self.name().to_string(),
                message: "no supported image files found".to_string(),
            });
        }

        let sources = images
            .iter()
            .map(|image| image.source.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Ok(ToolResult {
            name: self.name().to_string(),
            content: format!("attached {} image(s): {sources}", images.len()),
            refresh_manifest: false,
            images,
        })
    }
}
