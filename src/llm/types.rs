use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user_with_images(content: impl Into<String>, images: &[ImageAttachment]) -> Self {
        let mut parts = vec![ContentPart::Text {
            text: content.into(),
        }];
        parts.extend(images.iter().map(|image| ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: image.data_url(),
            },
        }));
        Self {
            role: "user".to_string(),
            content: Some(MessageContent::Parts(parts)),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant_tool_calls(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: if content.is_empty() {
                None
            } else {
                Some(MessageContent::Text(content))
            },
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(MessageContent::Text(content.into())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    pub fn content_text(&self) -> Option<&str> {
        match self.content.as_ref()? {
            MessageContent::Text(text) => Some(text),
            MessageContent::Parts(parts) => parts.iter().find_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                ContentPart::ImageUrl { .. } => None,
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageAttachment {
    pub source: String,
    pub mime_type: String,
    pub base64_data: String,
}

impl ImageAttachment {
    pub fn new(
        source: impl Into<String>,
        mime_type: impl Into<String>,
        base64_data: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            mime_type: mime_type.into(),
            base64_data: base64_data.into(),
        }
    }

    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.base64_data)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: usize,
    #[serde(default)]
    pub completion_tokens: usize,
    #[serde(default)]
    pub total_tokens: usize,
}
