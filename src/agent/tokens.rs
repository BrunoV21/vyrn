use crate::llm::ChatMessage;

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
    pub calls: Vec<CallUsage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CallUsage {
    pub label: String,
    pub sent: usize,
    pub would_be: usize,
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
        self.sent += sent;
        self.would_be += would_be;
        self.calls.push(CallUsage {
            label: label.into(),
            sent,
            would_be,
        });
    }
}

pub fn estimate_text_tokens(text: &str) -> usize {
    // Endpoint tokenizers differ. This conservative heuristic keeps accounting local
    // and replaceable while still making token savings visible.
    text.chars().count().div_ceil(4).max(1)
}

pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    let mut total = 0;
    for message in messages {
        total += estimate_text_tokens(&message.role) + 4;
        if let Some(content) = &message.content {
            total += estimate_text_tokens(content);
        }
        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                total += estimate_text_tokens(&call.function.name);
                total += estimate_text_tokens(&call.function.arguments);
            }
        }
    }
    total
}
