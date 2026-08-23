use crate::agent::tokens::estimate_chat_request_breakdown;
use crate::agent::transcript::{truncate, truncate_ends};
use crate::llm::{ChatMessage, LlmError, MessageContent, ToolCall};

const TOOL_CONTEXT_COMPACTION_PERCENT: usize = 70;
const COMPACTED_TOOL_RESULT_CONTENT: &str = "[tool output compacted into turn scratchpad]";
const COMPACTED_TOOL_IMAGE_CONTENT: &str =
    "Attached image(s) from read_image compacted into turn scratchpad.";
const TURN_SCRATCHPAD_MAX_CHARS: usize = 1800;
const TOOL_BATCH_CHECKPOINT_MAX_CHARS: usize = 1400;

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
    if before_tokens <= threshold {
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

    for target in [threshold, max_tokens] {
        for candidate_scratchpad in scratchpad_candidates(&scratchpad.summary) {
            for candidate_batch in tool_batch_candidates(tool_batch) {
                let messages =
                    build_turn_messages(base_messages, &candidate_scratchpad, &candidate_batch);
                let after_tokens = estimate_chat_request_breakdown(&messages, tools).total();
                if after_tokens <= target {
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
    }

    Err(LlmError::Input(format!(
        "context budget exceeded: next tool request estimates {before_tokens} tokens before compaction and could not fit configured {max_tokens}"
    )))
}

pub fn update_turn_scratchpad(
    current: &TurnScratchpad,
    consumed_tool_batch: &[ChatMessage],
) -> TurnScratchpad {
    let checkpoint = turn_scratchpad_update_source(consumed_tool_batch);
    if checkpoint.trim().is_empty() {
        return current.clone();
    }
    let combined = if current.summary.trim().is_empty() {
        checkpoint
    } else {
        format!("{}\n{}", current.summary.trim(), checkpoint.trim())
    };
    TurnScratchpad {
        summary: truncate_ends(&combined, TURN_SCRATCHPAD_MAX_CHARS),
    }
}

pub fn turn_scratchpad_update_source(consumed_tool_batch: &[ChatMessage]) -> String {
    if consumed_tool_batch.is_empty() {
        String::new()
    } else {
        tool_round_compaction_source(consumed_tool_batch)
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
            summary: truncate_ends(summary, limit),
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

fn tool_round_compaction_source(messages: &[ChatMessage]) -> String {
    let mut tools = Vec::new();
    let mut results = Vec::new();
    for message in messages {
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                tools.push(format!(
                    "{}({})",
                    call.function.name,
                    truncate_ends(&compact_value(&call.function.arguments), 360)
                ));
            }
        } else if message.role == "tool" {
            let result = message.content_text().unwrap_or_default();
            results.push(truncate_ends(&compact_value(result), 700));
        } else if is_tool_image_attachment_message(message) {
            results.push(
                message
                    .content_text()
                    .unwrap_or_default()
                    .trim()
                    .to_string(),
            );
        }
    }

    let mut lines = Vec::new();
    if !tools.is_empty() {
        lines.push(format!("- tools: {}", tools.join(", ")));
    }
    if !results.is_empty() {
        lines.push(format!("- results: {}", results.join(" | ")));
    }
    truncate_ends(&lines.join("\n"), TOOL_BATCH_CHECKPOINT_MAX_CHARS)
}

fn compact_value(value: &str) -> String {
    serde_json::from_str::<serde_json::Value>(value)
        .ok()
        .and_then(|value| serde_json::to_string(&value).ok())
        .unwrap_or_else(|| value.trim().to_string())
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
    fn next_turn_context_compacts_at_safety_threshold_before_hard_limit() {
        let base = vec![
            ChatMessage::system("system"),
            ChatMessage::user("read medium output"),
        ];
        let scratchpad = TurnScratchpad {
            summary: "- exact marker VIOLET_7319".to_string(),
        };
        let tool_batch = vec![
            ChatMessage::assistant_tool_calls(
                String::new(),
                vec![tool_call("call_read", "read_file")],
            ),
            ChatMessage::tool("call_read", "x".repeat(2400)),
        ];

        let next = prepare_next_turn_context(&base, &scratchpad, &tool_batch, &[], 900).unwrap();

        assert!(next.preparation.before_tokens > next.preparation.threshold);
        assert!(next.preparation.before_tokens <= next.preparation.max_tokens);
        assert!(next.preparation.after_tokens <= next.preparation.threshold);
        assert!(next.preparation.after_tokens < next.preparation.before_tokens);
        assert!(next.scratchpad.summary.contains("VIOLET_7319"));
    }

    #[test]
    fn deterministic_scratchpad_keeps_exact_tool_facts_without_duplicate_request() {
        let batch = vec![
            ChatMessage::assistant_tool_calls(
                "Inspecting the requested file.".to_string(),
                vec![ToolCall {
                    id: "call_0".to_string(),
                    kind: "function".to_string(),
                    function: ToolCallFunction {
                        name: "read_file".to_string(),
                        arguments: r#"{"path":"/Users/bv-mac/project/README.md"}"#.to_string(),
                    },
                }],
            ),
            ChatMessage::tool("call_0", "found README.md and src/tui/repl.rs changes"),
        ];

        let scratchpad = update_turn_scratchpad(&TurnScratchpad::default(), &batch);

        assert!(
            scratchpad
                .summary
                .contains("/Users/bv-mac/project/README.md")
        );
        assert!(scratchpad.summary.contains("found README.md"));
        assert_eq!(scratchpad.summary.matches("read_file(").count(), 1);
        assert!(!scratchpad.summary.contains("requested next tools"));
    }

    #[test]
    fn deterministic_scratchpad_is_bounded_and_preserves_old_and_new_edges() {
        let current = TurnScratchpad {
            summary: format!("OLD_EXACT_PATH /Users/bv-mac/project\n{}", "a".repeat(1400)),
        };
        let batch = vec![
            tool_call_message(0),
            ChatMessage::tool(
                "call_0",
                format!("{}\nNEW_EXACT_MARKER VIOLET_7319", "b".repeat(1400)),
            ),
        ];

        let scratchpad = update_turn_scratchpad(&current, &batch);

        assert!(scratchpad.summary.chars().count() <= TURN_SCRATCHPAD_MAX_CHARS);
        assert!(scratchpad.summary.contains("OLD_EXACT_PATH"));
        assert!(scratchpad.summary.contains("NEW_EXACT_MARKER"));
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
