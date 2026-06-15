use crate::agent::tokens::{TokenEstimate, estimate_messages_tokens};
use crate::agent::transcript::Exchange;
use crate::config::SummaryAggressiveness;
use crate::llm::{ChatMessage, ImageAttachment};
use crate::tools::{MachineManifest, ToolRegistry};

pub const AGENT_SYSTEM_PROMPT_TEMPLATE: &str = "\
[role] terminal coding agent. conserve tokens.
[rules] use tools when needed. use read_image for image files. prefer batch for shell. keep outputs compact.
[style] no markdown headings. prefer plain short paragraphs/lists. use only inline **bold**, *italic*, ~~struck~~, or `code` when emphasis helps.
[tools] {{tools}}
{{manifest}}
{{available_skills}}";

pub const SUMMARY_SYSTEM_PROMPT_TEMPLATE: &str = "You rewrite compact agent session summaries.";

pub const SUMMARY_USER_PROMPT_TEMPLATE: &str = "\
Update the session summary for the next turn.
Keep the user's original high-level goal, constraints, decisions, paths touched, and open tasks.
Drop raw tool output that is no longer needed.
Aggressiveness: {{aggressiveness}}.

Current summary:
{{current_summary}}

Last exchange:
{{last_exchange}}";

#[derive(Debug, Clone)]
pub struct PromptBundle {
    pub system: String,
    pub messages: Vec<ChatMessage>,
    pub estimated_tokens: TokenEstimate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSystemPromptSections {
    pub tools: String,
    pub manifest: String,
    pub available_skills: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryPromptSections {
    pub aggressiveness: SummaryAggressiveness,
    pub current_summary: String,
    pub last_exchange: String,
}

pub fn build_agent_prompt(
    tools: &ToolRegistry,
    manifest: &MachineManifest,
    summary: Option<&str>,
    user_input: &str,
    images: &[ImageAttachment],
) -> PromptBundle {
    let system = agent_system_prompt(tools, manifest);
    let mut messages = vec![ChatMessage::system(system.clone())];
    if let Some(summary) = summary.filter(|summary| !summary.trim().is_empty()) {
        messages.push(ChatMessage::system(format!("[summary]\n{summary}")));
    }
    if images.is_empty() {
        messages.push(ChatMessage::user(user_input));
    } else {
        messages.push(ChatMessage::user_with_images(user_input, images));
    }

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
    agent_system_prompt(tools, manifest)
}

pub fn agent_system_prompt(tools: &ToolRegistry, manifest: &MachineManifest) -> String {
    render_agent_system_prompt(&AgentSystemPromptSections {
        tools: tools.compact_descriptions(),
        manifest: manifest.compact(),
        available_skills: available_skills_prompt(manifest),
    })
}

pub fn render_agent_system_prompt(sections: &AgentSystemPromptSections) -> String {
    render_template(
        AGENT_SYSTEM_PROMPT_TEMPLATE,
        &[
            ("tools", sections.tools.as_str()),
            ("manifest", sections.manifest.as_str()),
            ("available_skills", sections.available_skills.as_str()),
        ],
    )
}

pub fn available_skills_prompt(manifest: &MachineManifest) -> String {
    if manifest.skills.is_empty() {
        return String::new();
    }

    let skills = manifest
        .skills
        .iter()
        .map(|skill| skill.prompt_line())
        .collect::<Vec<_>>()
        .join("\n");
    format!("[available_skills]\n{skills}")
}

pub fn build_summary_refresh_messages(
    aggressiveness: SummaryAggressiveness,
    current_summary: Option<&str>,
    exchange: &Exchange,
    include_tool_results: bool,
) -> Vec<ChatMessage> {
    let sections = SummaryPromptSections {
        aggressiveness,
        current_summary: current_summary.unwrap_or("none").to_string(),
        last_exchange: exchange.compact(include_tool_results),
    };

    vec![
        ChatMessage::system(render_summary_system_prompt()),
        ChatMessage::user(render_summary_user_prompt(&sections)),
    ]
}

pub fn render_summary_system_prompt() -> String {
    render_template(SUMMARY_SYSTEM_PROMPT_TEMPLATE, &[])
}

pub fn render_summary_user_prompt(sections: &SummaryPromptSections) -> String {
    let aggressiveness = sections.aggressiveness.to_string();
    render_template(
        SUMMARY_USER_PROMPT_TEMPLATE,
        &[
            ("aggressiveness", aggressiveness.as_str()),
            ("current_summary", sections.current_summary.as_str()),
            ("last_exchange", sections.last_exchange.as_str()),
        ],
    )
}

fn render_template(template: &str, slots: &[(&str, &str)]) -> String {
    let mut rendered = template.to_string();
    for (name, value) in slots {
        rendered = rendered.replace(&format!("{{{{{name}}}}}"), value);
    }
    debug_assert!(
        !rendered.contains("{{"),
        "unfilled prompt template slot in {rendered:?}"
    );
    rendered.trim_end_matches('\n').to_string()
}
