use crate::llm::ToolCall;
use crate::tools::ToolResult;

#[derive(Debug, Clone, Default)]
pub struct Exchange {
    pub user_input: String,
    pub assistant_text: String,
    pub turn_scratchpad: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
}

impl Exchange {
    pub fn compact(&self, include_tool_results: bool) -> String {
        let mut out = String::new();
        out.push_str("user: ");
        out.push_str(&self.user_input);
        out.push('\n');
        if !self.turn_scratchpad.trim().is_empty() {
            out.push_str("turn_scratchpad:\n");
            out.push_str(&truncate(self.turn_scratchpad.trim(), 2400));
            out.push('\n');
        }
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

pub fn truncate_ends(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars == 0 {
        return String::new();
    }

    const MARKER: &str = "\n[... compacted ...]\n";
    let marker_chars = MARKER.chars().count();
    if max_chars <= marker_chars + 2 {
        return value.chars().take(max_chars).collect();
    }

    let retained = max_chars - marker_chars;
    let head_chars = retained.div_ceil(2);
    let tail_chars = retained / 2;
    let head = value.chars().take(head_chars).collect::<String>();
    let tail = value
        .chars()
        .rev()
        .take(tail_chars)
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>();
    format!("{head}{MARKER}{tail}")
}
