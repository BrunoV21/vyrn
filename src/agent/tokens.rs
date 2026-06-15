use crate::llm::{ChatMessage, ContentPart, MessageContent};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenEstimate {
    pub tokens: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenLedger {
    pub session_sent: usize,
    pub session_would_be: usize,
    pub session_saved: isize,
    pub turns: Vec<TurnUsage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnUsage {
    pub sent: usize,
    pub would_be: usize,
    pub saved: isize,
    pub context_tokens: usize,
    pub breakdown: TokenBreakdown,
    pub calls: Vec<CallUsage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CallUsage {
    pub label: String,
    pub sent: usize,
    pub would_be: usize,
    pub breakdown: TokenBreakdown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenBreakdown {
    pub system_prompt: usize,
    pub summaries: usize,
    pub user_requests: usize,
    pub images: usize,
    pub skills: usize,
    pub tool_schemas: usize,
    pub tool_call_inputs: usize,
    pub tool_call_outputs: usize,
    pub assistant_context: usize,
    pub overhead: usize,
    pub other: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BreakdownItem {
    pub label: &'static str,
    pub tokens: usize,
}

impl TokenLedger {
    pub fn push_turn(&mut self, mut usage: TurnUsage) {
        usage.saved = usage.would_be as isize - usage.sent as isize;
        self.session_sent += usage.sent;
        self.session_would_be += usage.would_be;
        self.session_saved += usage.saved;
        self.turns.push(usage);
    }
}

impl TurnUsage {
    pub fn add_call(&mut self, label: impl Into<String>, sent: usize, would_be: usize) {
        self.add_call_with_breakdown(label, sent, would_be, TokenBreakdown::other(sent));
    }

    pub fn add_call_with_breakdown(
        &mut self,
        label: impl Into<String>,
        sent: usize,
        would_be: usize,
        breakdown: TokenBreakdown,
    ) {
        let breakdown = breakdown.scaled_to_total(sent);
        self.sent += sent;
        self.would_be += would_be;
        self.breakdown.add(breakdown);
        self.calls.push(CallUsage {
            label: label.into(),
            sent,
            would_be,
            breakdown,
        });
    }
}

impl TokenBreakdown {
    pub fn other(tokens: usize) -> Self {
        Self {
            other: tokens,
            ..Self::default()
        }
    }

    pub fn total(&self) -> usize {
        self.system_prompt
            + self.summaries
            + self.user_requests
            + self.images
            + self.skills
            + self.tool_schemas
            + self.tool_call_inputs
            + self.tool_call_outputs
            + self.assistant_context
            + self.overhead
            + self.other
    }

    pub fn add(&mut self, other: Self) {
        self.system_prompt += other.system_prompt;
        self.summaries += other.summaries;
        self.user_requests += other.user_requests;
        self.images += other.images;
        self.skills += other.skills;
        self.tool_schemas += other.tool_schemas;
        self.tool_call_inputs += other.tool_call_inputs;
        self.tool_call_outputs += other.tool_call_outputs;
        self.assistant_context += other.assistant_context;
        self.overhead += other.overhead;
        self.other += other.other;
    }

    pub fn items(&self) -> Vec<BreakdownItem> {
        let mut items = vec![
            BreakdownItem {
                label: "system prompt",
                tokens: self.system_prompt,
            },
            BreakdownItem {
                label: "summaries",
                tokens: self.summaries,
            },
            BreakdownItem {
                label: "user requests",
                tokens: self.user_requests,
            },
            BreakdownItem {
                label: "images",
                tokens: self.images,
            },
            BreakdownItem {
                label: "skills",
                tokens: self.skills,
            },
            BreakdownItem {
                label: "tools",
                tokens: self.tool_schemas,
            },
            BreakdownItem {
                label: "tool call input",
                tokens: self.tool_call_inputs,
            },
            BreakdownItem {
                label: "tool call output",
                tokens: self.tool_call_outputs,
            },
            BreakdownItem {
                label: "assistant context",
                tokens: self.assistant_context,
            },
            BreakdownItem {
                label: "message overhead",
                tokens: self.overhead,
            },
            BreakdownItem {
                label: "other",
                tokens: self.other,
            },
        ];
        items.retain(|item| item.tokens > 0);
        items.sort_by(|a, b| b.tokens.cmp(&a.tokens).then_with(|| a.label.cmp(b.label)));
        items
    }

    pub fn scaled_to_total(self, target_total: usize) -> Self {
        let current_total = self.total();
        if current_total == target_total {
            return self;
        }
        if current_total == 0 {
            return Self::other(target_total);
        }

        let scale = |tokens: usize| tokens.saturating_mul(target_total) / current_total;
        let mut scaled = Self {
            system_prompt: scale(self.system_prompt),
            summaries: scale(self.summaries),
            user_requests: scale(self.user_requests),
            images: scale(self.images),
            skills: scale(self.skills),
            tool_schemas: scale(self.tool_schemas),
            tool_call_inputs: scale(self.tool_call_inputs),
            tool_call_outputs: scale(self.tool_call_outputs),
            assistant_context: scale(self.assistant_context),
            overhead: scale(self.overhead),
            other: scale(self.other),
        };
        let assigned = scaled.total();
        if assigned < target_total {
            scaled.other += target_total - assigned;
        }
        scaled
    }
}

pub fn estimate_text_tokens(text: &str) -> usize {
    // Endpoint tokenizers differ. This conservative heuristic keeps accounting local
    // and replaceable while still making token savings visible.
    text.chars().count().div_ceil(4).max(1)
}

pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    estimate_messages_breakdown(messages).total()
}

pub fn estimate_chat_request_breakdown(
    messages: &[ChatMessage],
    tools: &[Value],
) -> TokenBreakdown {
    let mut breakdown = estimate_messages_breakdown(messages);
    breakdown.tool_schemas += estimate_tool_schema_tokens(tools);
    breakdown
}

pub fn estimate_chat_request_tokens(messages: &[ChatMessage], tools: &[Value]) -> usize {
    estimate_chat_request_breakdown(messages, tools).total()
}

pub fn estimate_messages_breakdown(messages: &[ChatMessage]) -> TokenBreakdown {
    let mut breakdown = TokenBreakdown::default();
    let mut skill_tool_call_ids = BTreeSet::new();
    for message in messages {
        breakdown.overhead += estimate_text_tokens(&message.role) + 4;
        if let Some(content) = &message.content {
            if message.role == "tool"
                && message
                    .tool_call_id
                    .as_ref()
                    .is_some_and(|id| skill_tool_call_ids.contains(id))
            {
                add_skill_content_tokens(&mut breakdown, content);
            } else {
                add_content_tokens(&mut breakdown, &message.role, content);
            }
        }
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                breakdown.tool_call_inputs += estimate_text_tokens(&call.function.name);
                breakdown.tool_call_inputs += estimate_text_tokens(&call.function.arguments);
                if is_skill_read_tool_call(call) {
                    skill_tool_call_ids.insert(call.id.clone());
                }
            }
        }
    }
    breakdown
}

pub fn estimate_tool_schema_tokens(tools: &[Value]) -> usize {
    let mut total = 0;
    for tool in tools {
        let schema = serde_json::to_string(tool).unwrap_or_else(|_| tool.to_string());
        total += estimate_text_tokens(&schema);
    }
    total
}

fn add_content_tokens(breakdown: &mut TokenBreakdown, role: &str, content: &MessageContent) {
    match content {
        MessageContent::Text(text) => add_text_tokens(breakdown, role, text),
        MessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text { text } => add_text_tokens(breakdown, role, text),
                    // Image token accounting is endpoint-specific. Use a bounded placeholder
                    // instead of counting base64 bytes as prompt text.
                    ContentPart::ImageUrl { .. } => breakdown.images += 256,
                }
            }
        }
    }
}

fn add_skill_content_tokens(breakdown: &mut TokenBreakdown, content: &MessageContent) {
    match content {
        MessageContent::Text(text) => breakdown.skills += estimate_text_tokens(text),
        MessageContent::Parts(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text { text } => breakdown.skills += estimate_text_tokens(text),
                    ContentPart::ImageUrl { .. } => breakdown.images += 256,
                }
            }
        }
    }
}

fn add_text_tokens(breakdown: &mut TokenBreakdown, role: &str, text: &str) {
    let tokens = estimate_text_tokens(text);
    match role {
        "system" if text.starts_with("[summary]") => breakdown.summaries += tokens,
        "system" => add_system_text_tokens(breakdown, text),
        "user" => breakdown.user_requests += tokens,
        "assistant" => breakdown.assistant_context += tokens,
        "tool" => breakdown.tool_call_outputs += tokens,
        _ => breakdown.other += tokens,
    }
}

fn add_system_text_tokens(breakdown: &mut TokenBreakdown, text: &str) {
    const AVAILABLE_SKILLS_MARKER: &str = "[available_skills]";
    let Some(marker_start) = text.find(AVAILABLE_SKILLS_MARKER) else {
        add_system_manifest_text_tokens(breakdown, text);
        return;
    };

    let (system_text, skills_text) = text.split_at(marker_start);
    if !system_text.trim().is_empty() {
        add_system_manifest_text_tokens(breakdown, system_text);
    }
    if !skills_text.trim().is_empty() {
        breakdown.skills += estimate_text_tokens(skills_text);
    }
}

fn add_system_manifest_text_tokens(breakdown: &mut TokenBreakdown, text: &str) {
    for line in text.split_inclusive('\n') {
        if line.trim_start().starts_with("[skills]") {
            breakdown.skills += estimate_text_tokens(line);
        } else if !line.trim().is_empty() {
            breakdown.system_prompt += estimate_text_tokens(line);
        }
    }
}

fn is_skill_read_tool_call(call: &crate::llm::ToolCall) -> bool {
    if call.function.name != "read_file" {
        return false;
    }

    let Ok(arguments) = serde_json::from_str::<Value>(&call.function.arguments) else {
        return false;
    };
    arguments
        .get("path")
        .and_then(Value::as_str)
        .is_some_and(|path| path.replace('\\', "/").ends_with("/SKILL.md") || path == "SKILL.md")
}
