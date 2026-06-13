use crate::llm::ToolCall;
use crate::tools::ToolResult;

#[derive(Debug, Clone, Default)]
pub struct Exchange {
    pub user_input: String,
    pub assistant_text: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

impl Exchange {
    pub fn compact(&self, include_tool_results: bool) -> String {
        let mut out = String::new();
        out.push_str("user: ");
        out.push_str(&self.user_input);
        out.push('\n');
        if !self.assistant_text.is_empty() {
            out.push_str("assistant: ");
            out.push_str(&self.assistant_text);
            out.push('\n');
        }
        if !self.tool_calls.is_empty() {
            out.push_str("tool_calls: ");
            out.push_str(
                &self
                    .tool_calls
                    .iter()
                    .map(|call| call.function.name.as_str())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            out.push('\n');
        }
        if include_tool_results && !self.tool_results.is_empty() {
            out.push_str("tool_results:\n");
            for result in &self.tool_results {
                out.push_str("- ");
                out.push_str(&result.name);
                out.push_str(": ");
                out.push_str(&truncate(&result.content, 2000));
                out.push('\n');
            }
        }
        out
    }
}

pub fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    truncated.push_str("\n[truncated]");
    truncated
}
