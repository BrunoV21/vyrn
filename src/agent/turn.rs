use crate::agent::tokens::{estimate_chat_request_breakdown, estimate_messages_breakdown};
use crate::agent::transcript::truncate;
use crate::llm::{ChatMessage, LlmError, MessageContent, ToolCall};

const TOOL_CONTEXT_COMPACTION_PERCENT: usize = 70;
const COMPACTED_TOOL_RESULT_CONTENT: &str = "[tool output compacted into turn scratchpad]";
const COMPACTED_TOOL_IMAGE_CONTENT: &str =
    "Attached image(s) from read_image compacted into turn scratchpad.";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnScratchpad {
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolChainPreparation {
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub threshold: usize,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NextTurnContext {
    pub scratchpad: TurnScratchpad,
    pub tool_batch: Vec<ChatMessage>,
    pub preparation: ToolChainPreparation,
}

pub fn build_turn_messages(
    base_messages: &[ChatMessage],
    scratchpad: &TurnScratchpad,
    current_tool_batch: &[ChatMessage],
) -> Vec<ChatMessage> {
    let mut messages = base_messages.to_vec();
    if !scratchpad.summary.trim().is_empty() {
        messages.push(ChatMessage::system(format!(
            "[turn scratchpad]\n{}",
            scratchpad.summary.trim()
        )));
    }
    messages.extend(current_tool_batch.iter().cloned());
    messages
}

pub fn live_steering_message(text: &str) -> ChatMessage {
    ChatMessage::user(format!(
        "[live steering from the human]\n{text}\nApply this immediately. Reconsider the next action before continuing."
    ))
}

pub fn apply_live_steering_to_tool_batch(
    tool_batch: &mut Vec<ChatMessage>,
    interrupted_calls: &[ToolCall],
    text: &str,
) {
    for call in interrupted_calls {
        tool_batch.push(ChatMessage::tool(
            call.id.clone(),
            "tool execution interrupted by live user steering",
        ));
    }
    tool_batch.push(live_steering_message(text));
}

pub fn prepare_next_turn_context(
    base_messages: &[ChatMessage],
    scratchpad: &TurnScratchpad,
    tool_batch: &[ChatMessage],
    tools: &[serde_json::Value],
    max_tokens: usize,
) -> Result<NextTurnContext, LlmError> {
    let threshold = tool_context_compaction_threshold(max_tokens);
    let original_messages = build_turn_messages(base_messages, scratchpad, tool_batch);
    let before_tokens = estimate_chat_request_breakdown(&original_messages, tools).total();
    if before_tokens <= max_tokens {
        return Ok(NextTurnContext {
            scratchpad: scratchpad.clone(),
            tool_batch: tool_batch.to_vec(),
            preparation: ToolChainPreparation {
                before_tokens,
                after_tokens: before_tokens,
                threshold,
                max_tokens,
            },
        });
    }

    for candidate_scratchpad in scratchpad_candidates(&scratchpad.summary) {
        for candidate_batch in tool_batch_candidates(tool_batch) {
            let messages =
                build_turn_messages(base_messages, &candidate_scratchpad, &candidate_batch);
            let after_tokens = estimate_chat_request_breakdown(&messages, tools).total();
            if after_tokens <= max_tokens {
                return Ok(NextTurnContext {
                    scratchpad: candidate_scratchpad,
                    tool_batch: candidate_batch,
                    preparation: ToolChainPreparation {
                        before_tokens,
                        after_tokens,
                        threshold,
                        max_tokens,
                    },
                });
            }
        }
    }

    Err(LlmError::Input(format!(
        "context budget exceeded: next tool request estimates {before_tokens} tokens before compaction and could not fit configured {max_tokens}"
    )))
}

pub fn turn_scratchpad_update_source(
    consumed_tool_batch: &[ChatMessage],
    assistant_response: &ChatMessage,
) -> String {
    let mut lines = Vec::new();
    if !consumed_tool_batch.is_empty() {
        lines.push(tool_round_compaction_source(consumed_tool_batch, 1));
    }
    lines.push(assistant_response_source(assistant_response));
    lines.join("\n\n")
}

pub fn build_turn_scratchpad_update_messages(
    current_scratchpad: &str,
    source: &str,
) -> Vec<ChatMessage> {
    vec![
        ChatMessage::system(
            "You update a compact scratchpad for one active terminal-agent turn. Preserve task-relevant facts, decisions, file paths, commands and outcomes, errors that changed behavior, artifacts created, and next steps. Drop raw tool JSON, full stdout/stderr, repeated diffs, and dead ends. Return only concise bullets for the full updated scratchpad.",
        ),
        ChatMessage::user(format!(
            "Current scratchpad:\n{}\n\nNew consumed tool batch and assistant response:\n{}",
            if current_scratchpad.trim().is_empty() {
                "none"
            } else {
                current_scratchpad.trim()
            },
            source
        )),
    ]
}

pub fn build_fitted_turn_scratchpad_update_messages(
    current_scratchpad: &str,
    source: &str,
    max_tokens: usize,
) -> Vec<ChatMessage> {
    let mut fitted_source = source.to_string();
    let mut fitted_scratchpad = current_scratchpad.to_string();
    loop {
        let messages = build_turn_scratchpad_update_messages(&fitted_scratchpad, &fitted_source);
        if estimate_messages_breakdown(&messages).total() <= max_tokens {
            return messages;
        }

        let source_chars = fitted_source.chars().count();
        if source_chars > 256 {
            fitted_source = truncate(&fitted_source, source_chars.saturating_mul(3) / 4);
            continue;
        }

        let scratchpad_chars = fitted_scratchpad.chars().count();
        if scratchpad_chars > 256 {
            fitted_scratchpad =
                truncate(&fitted_scratchpad, scratchpad_chars.saturating_mul(3) / 4);
            continue;
        }

        return build_turn_scratchpad_update_messages(
            &truncate(
                &fitted_scratchpad,
                scratchpad_chars.saturating_sub(1).max(1),
            ),
            &truncate(&fitted_source, source_chars.saturating_sub(1).max(1)),
        );
    }
}

fn tool_context_compaction_threshold(max_tokens: usize) -> usize {
    max_tokens
        .saturating_mul(TOOL_CONTEXT_COMPACTION_PERCENT)
        .div_ceil(100)
        .max(1)
}

fn tool_batch_candidates(tool_batch: &[ChatMessage]) -> Vec<Vec<ChatMessage>> {
    let mut candidates = Vec::new();
    candidates.push(tool_batch.to_vec());
    for limit in [2000, 1000, 400, 120, 0] {
        let candidate = tool_batch
            .iter()
            .map(|message| compact_tool_batch_message(message, limit))
            .collect::<Vec<_>>();
        if candidates.last() != Some(&candidate) {
            candidates.push(candidate);
        }
    }
    candidates
}

fn compact_tool_batch_message(message: &ChatMessage, tool_content_limit: usize) -> ChatMessage {
    if message.role == "tool" {
        let content = message
            .content_text()
            .filter(|content| tool_content_limit > 0 && !content.trim().is_empty())
            .map(|content| truncate(content, tool_content_limit))
            .unwrap_or_else(|| COMPACTED_TOOL_RESULT_CONTENT.to_string());
        return ChatMessage {
            role: message.role.clone(),
            content: Some(MessageContent::Text(content)),
            tool_calls: None,
            tool_call_id: message.tool_call_id.clone(),
        };
    }

    if is_tool_image_attachment_message(message) {
        return ChatMessage::user(COMPACTED_TOOL_IMAGE_CONTENT);
    }

    if message.role == "assistant" {
        let content = message.content_text().and_then(|content| {
            let content = truncate(content, 400);
            (!content.trim().is_empty()).then_some(MessageContent::Text(content))
        });
        return ChatMessage {
            role: message.role.clone(),
            content,
            tool_calls: message.tool_calls.clone(),
            tool_call_id: message.tool_call_id.clone(),
        };
    }

    message.clone()
}

fn scratchpad_candidates(summary: &str) -> Vec<TurnScratchpad> {
    let mut candidates = Vec::new();
    candidates.push(TurnScratchpad {
        summary: summary.to_string(),
    });
    for limit in [4000, 2000, 1000, 500, 250, 100] {
        let candidate = TurnScratchpad {
            summary: truncate(summary, limit),
        };
        if candidates.last() != Some(&candidate) {
            candidates.push(candidate);
        }
    }
    let empty = TurnScratchpad {
        summary: String::new(),
    };
    if candidates.last() != Some(&empty) {
        candidates.push(empty);
    }
    candidates
}

fn is_tool_image_attachment_message(message: &ChatMessage) -> bool {
    message.role == "user"
        && message
            .content_text()
            .is_some_and(|content| content.starts_with("Attached image(s) from read_image:"))
}

fn tool_round_compaction_source(messages: &[ChatMessage], batch_number: usize) -> String {
    let mut tools = Vec::new();
    let mut results = Vec::new();
    for message in messages {
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                tools.push(format!(
                    "{}({})",
                    call.function.name,
                    truncate(&call.function.arguments, 800)
                ));
            }
        } else if message.role == "tool" {
            let result = message.content_text().unwrap_or_default();
            results.push(truncate(result, 1600).replace('\n', " "));
        } else if is_tool_image_attachment_message(message) {
            results.push(message.content_text().unwrap_or_default().to_string());
        }
    }

    let mut line = format!("batch {batch_number}\ntools: {}", tools.join(", "));
    if !results.is_empty() {
        line.push_str("\nresults:\n- ");
        line.push_str(&results.join("\n- "));
    }
    truncate(&line, 2200)
}

fn assistant_response_source(message: &ChatMessage) -> String {
    let mut lines = Vec::new();
    if let Some(text) = message
        .content_text()
        .filter(|text| !text.trim().is_empty())
    {
        lines.push(format!("assistant response: {}", truncate(text, 1200)));
    }
    if let Some(tool_calls) = &message.tool_calls {
        let calls = tool_calls
            .iter()
            .map(|call| {
                format!(
                    "{}({})",
                    call.function.name,
                    truncate(&call.function.arguments, 400)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        if !calls.is_empty() {
            lines.push(format!("assistant requested next tools: {calls}"));
        }
    }
    if lines.is_empty() {
        "assistant response: none".to_string()
    } else {
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ToolCall, types::ToolCallFunction};

    #[test]
    fn turn_messages_include_scratchpad_and_current_batch_only() {
        let base = vec![
            ChatMessage::system("system"),
            ChatMessage::user("inspect files"),
        ];
        let scratchpad = TurnScratchpad {
            summary: "- README.md changed".to_string(),
        };
        let current_batch = vec![
            tool_call_message(2),
            ChatMessage::tool("call_2", "current result"),
        ];

        let messages = build_turn_messages(&base, &scratchpad, &current_batch);

        assert_eq!(messages.len(), 5);
        assert!(messages.iter().any(|message| {
            message
                .content_text()
                .is_some_and(|text| text.contains("[turn scratchpad]"))
        }));
        assert!(
            messages
                .iter()
                .any(|message| message.tool_call_id.as_deref() == Some("call_2"))
        );
        assert!(
            !messages
                .iter()
                .any(|message| message.tool_call_id.as_deref() == Some("call_1"))
        );
    }

    #[test]
    fn live_steering_message_is_an_immediate_human_instruction() {
        let message = live_steering_message("stop and inspect the tests");

        assert_eq!(message.role, "user");
        let content = message.content_text().unwrap();
        assert!(content.contains("live steering from the human"));
        assert!(content.contains("stop and inspect the tests"));
        assert!(content.contains("Apply this immediately"));
    }

    #[test]
    fn live_steering_completes_interrupted_tool_protocol_before_user_message() {
        let call = tool_call("call_write", "write_file");
        let mut batch = vec![ChatMessage::assistant_tool_calls(
            String::new(),
            vec![call.clone()],
        )];

        apply_live_steering_to_tool_batch(&mut batch, &[call], "do not write the file");

        assert_eq!(batch.len(), 3);
        assert_eq!(batch[1].role, "tool");
        assert_eq!(batch[1].tool_call_id.as_deref(), Some("call_write"));
        assert_eq!(batch[2].role, "user");
        assert!(
            batch[2]
                .content_text()
                .unwrap()
                .contains("do not write the file")
        );
    }

    #[test]
    fn next_turn_context_compacts_large_latest_tool_output_to_fit_budget() {
        let base = vec![
            ChatMessage::system("system"),
            ChatMessage::user("read large"),
        ];
        let scratchpad = TurnScratchpad {
            summary: "- large file contained marker\n- read count is 2\n- next step is answer now"
                .to_string(),
        };
        let tool_batch = vec![
            ChatMessage::assistant_tool_calls(
                String::new(),
                vec![tool_call("call_read", "read_file")],
            ),
            ChatMessage::tool("call_read", "large output ".repeat(2000)),
        ];

        let next = prepare_next_turn_context(&base, &scratchpad, &tool_batch, &[], 900).unwrap();

        assert!(next.preparation.before_tokens > next.preparation.after_tokens);
        assert!(next.preparation.after_tokens <= 900);
        assert!(next.scratchpad.summary.contains("read count is 2"));
        let messages = build_turn_messages(&base, &next.scratchpad, &next.tool_batch);
        let tool_output = messages
            .iter()
            .find(|message| message.role == "tool")
            .and_then(ChatMessage::content_text)
            .unwrap();
        assert!(tool_output.chars().count() < 2500);
    }

    #[test]
    fn turn_scratchpad_source_includes_tool_batch_and_assistant_response() {
        let batch = vec![
            tool_call_message(0),
            ChatMessage::tool("call_0", "found README.md and src/tui/repl.rs changes"),
        ];
        let response = ChatMessage::assistant_tool_calls(
            "Next I will inspect tests.".to_string(),
            vec![tool_call("call_1", "batch")],
        );

        let source = turn_scratchpad_update_source(&batch, &response);

        assert!(source.contains("found README.md"));
        assert!(source.contains("Next I will inspect tests."));
        assert!(source.contains("batch"));
    }

    #[test]
    fn scratchpad_update_prompt_drops_raw_output_by_instruction() {
        let messages =
            build_turn_scratchpad_update_messages("- existing fact", "tool output with details");

        let prompt = messages
            .iter()
            .filter_map(ChatMessage::content_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(prompt.contains("Drop raw tool JSON"));
        assert!(prompt.contains("existing fact"));
        assert!(prompt.contains("tool output with details"));
    }

    fn tool_call_message(index: usize) -> ChatMessage {
        ChatMessage::assistant_tool_calls(
            String::new(),
            vec![ToolCall {
                id: format!("call_{index}"),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: "read_file".to_string(),
                    arguments: format!(r#"{{"path":"fixture_{index}.txt"}}"#),
                },
            }],
        )
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            kind: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }
}
