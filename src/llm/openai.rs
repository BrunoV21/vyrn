use crate::config::ModelProfile;
use crate::llm::stream::StreamEvent;
use crate::llm::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ToolCall, ToolCallFunction, Usage,
};
use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP request failed while calling {url}: {source}")]
    Request { url: String, source: reqwest::Error },
    #[error("OpenAI-compatible endpoint returned {status} from {url}: {body}")]
    HttpStatus {
        url: String,
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("failed to parse stream event: {0}")]
    ParseStream(String),
    #[error("model response did not include a choice")]
    MissingChoice,
}

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    profile: ModelProfile,
    client: reqwest::Client,
}

impl OpenAiClient {
    pub fn new(profile: ModelProfile) -> Self {
        Self {
            profile,
            client: reqwest::Client::new(),
        }
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.profile
    }

    pub fn switch_profile(&mut self, profile: ModelProfile) {
        self.profile = profile;
    }

    pub async fn complete_chat(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, LlmError> {
        request.model = self.profile.model.clone();
        request.stream = false;
        let url = self.chat_completions_url();
        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&request)
            .send()
            .await
            .map_err(|source| LlmError::Request {
                url: url.clone(),
                source,
            })?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpStatus { url, status, body });
        }
        response
            .json::<ChatCompletionResponse>()
            .await
            .map_err(|source| LlmError::Request { url, source })
    }

    pub async fn stream_chat<F>(
        &self,
        mut request: ChatCompletionRequest,
        mut on_event: F,
    ) -> Result<ChatCompletionResponse, LlmError>
    where
        F: FnMut(StreamEvent),
    {
        request.model = self.profile.model.clone();
        request.stream = true;
        let url = self.chat_completions_url();
        let response = self
            .client
            .post(&url)
            .headers(self.headers())
            .json(&request)
            .send()
            .await
            .map_err(|source| LlmError::Request {
                url: url.clone(),
                source,
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(LlmError::HttpStatus { url, status, body });
        }

        let mut content = String::new();
        let mut tool_calls: BTreeMap<usize, ToolCallAccumulator> = BTreeMap::new();
        let mut usage = None;
        let mut buffer = String::new();
        let mut bytes = response.bytes_stream();

        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|source| LlmError::Request {
                url: url.clone(),
                source,
            })?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    on_event(StreamEvent::Finished);
                    continue;
                }
                let event: StreamChunk = serde_json::from_str(data)
                    .map_err(|error| LlmError::ParseStream(error.to_string()))?;
                if let Some(event_usage) = event.usage {
                    usage = Some(event_usage);
                }
                for choice in event.choices {
                    if let Some(delta_content) = choice.delta.content {
                        content.push_str(&delta_content);
                        on_event(StreamEvent::TextDelta(delta_content));
                    }
                    for delta_call in choice.delta.tool_calls {
                        let accumulator = tool_calls
                            .entry(delta_call.index)
                            .or_insert_with(ToolCallAccumulator::default);
                        if let Some(id) = delta_call.id {
                            accumulator.id = id;
                        }
                        if let Some(kind) = delta_call.kind {
                            accumulator.kind = kind;
                        }
                        if let Some(function) = delta_call.function {
                            if let Some(name) = function.name {
                                accumulator.name = name;
                            }
                            if let Some(arguments) = function.arguments {
                                accumulator.arguments.push_str(&arguments);
                            }
                        }
                    }
                }
            }
        }

        let calls = tool_calls
            .into_values()
            .filter_map(ToolCallAccumulator::finish)
            .collect::<Vec<_>>();
        for call in &calls {
            on_event(StreamEvent::ToolCallDone(call.clone()));
        }

        Ok(ChatCompletionResponse {
            choices: vec![crate::llm::types::ChatChoice {
                message: if calls.is_empty() {
                    ChatMessage::assistant(content)
                } else {
                    ChatMessage::assistant_tool_calls(content, calls)
                },
            }],
            usage,
        })
    }

    fn chat_completions_url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.profile.base_url.trim_end_matches('/')
        )
    }

    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        if !self.profile.api_key.is_empty() {
            let value = format!("Bearer {}", self.profile.api_key);
            if let Ok(value) = HeaderValue::from_str(&value) {
                headers.insert(AUTHORIZATION, value);
            }
        }
        headers
    }
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    kind: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    fn finish(self) -> Option<ToolCall> {
        if self.name.is_empty() {
            return None;
        }
        Some(ToolCall {
            id: self.id,
            kind: if self.kind.is_empty() {
                "function".to_string()
            } else {
                self.kind
            },
            function: ToolCallFunction {
                name: self.name,
                arguments: self.arguments,
            },
        })
    }
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Debug, Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<StreamToolCallDelta>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCallDelta {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    function: Option<StreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct StreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
