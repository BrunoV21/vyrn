use crate::llm::types::ToolCall;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    TextDelta(String),
    ToolCallDone(ToolCall),
    Finished,
}
