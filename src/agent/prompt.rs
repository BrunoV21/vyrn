use crate::agent::tokens::{TokenEstimate, estimate_messages_tokens};
use crate::llm::ChatMessage;
use crate::tools::{MachineManifest, ToolRegistry};

#[derive(Debug, Clone)]
pub struct PromptBundle {
    pub system: String,
    pub messages: Vec<ChatMessage>,
    pub estimated_tokens: TokenEstimate,
}

pub fn build_agent_prompt(
    tools: &ToolRegistry,
    manifest: &MachineManifest,
    summary: Option<&str>,
    user_input: &str,
) -> PromptBundle {
    let system = compact_system_prompt(tools, manifest);
    let mut messages = vec![ChatMessage::system(system.clone())];
    if let Some(summary) = summary.filter(|summary| !summary.trim().is_empty()) {
        messages.push(ChatMessage::system(format!("[summary]\n{summary}")));
    }
    messages.push(ChatMessage::user(user_input));

    let estimated_tokens = TokenEstimate {
        tokens: estimate_messages_tokens(&messages),
    };

    PromptBundle {
        system,
        messages,
        estimated_tokens,
    }
}

pub fn compact_system_prompt(tools: &ToolRegistry, manifest: &MachineManifest) -> String {
    let mut prompt = String::new();
    prompt.push_str("[role] terminal coding agent. conserve tokens.\n");
    prompt
        .push_str("[rules] use tools when needed. prefer batch for shell. keep outputs compact.\n");
    prompt.push_str("[tools] ");
    prompt.push_str(&tools.compact_descriptions());
    let compact_manifest = manifest.compact();
    if !compact_manifest.is_empty() {
        prompt.push('\n');
        prompt.push_str(&compact_manifest);
    }
    prompt
}
