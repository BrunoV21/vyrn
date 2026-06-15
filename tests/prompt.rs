use std::path::PathBuf;
use vyrn::agent::prompt::{
    AgentSystemPromptSections, SummaryPromptSections, agent_system_prompt,
    build_summary_refresh_messages, render_agent_system_prompt, render_summary_system_prompt,
    render_summary_user_prompt,
};
use vyrn::agent::transcript::Exchange;
use vyrn::config::SummaryAggressiveness;
use vyrn::skills::{SkillSource, SkillSummary};
use vyrn::tools::{MachineManifest, ToolRegistry, ToolResult};

#[test]
fn agent_system_prompt_renders_template_slots_explicitly() {
    let prompt = render_agent_system_prompt(&AgentSystemPromptSections {
        tools: "batch:run shell in cwd by default; refresh_manifest:rescan host manifest"
            .to_string(),
        manifest: "[machine] macos/aarch64\n[env] git,rg".to_string(),
        available_skills:
            "[available_skills]\n- code-review | project .vyrn | .vyrn/skills/code-review/SKILL.md | Review code changes."
                .to_string(),
    });

    assert_eq!(
        prompt,
        "\
[role] terminal coding agent. conserve tokens.
[rules] use tools when needed. use read_image for image files. prefer batch for shell. keep outputs compact.
[style] no markdown headings. prefer plain short paragraphs/lists. use only inline **bold**, *italic*, ~~struck~~, or `code` when emphasis helps.
[tools] batch:run shell in cwd by default; refresh_manifest:rescan host manifest
[machine] macos/aarch64
[env] git,rg
[available_skills]
- code-review | project .vyrn | .vyrn/skills/code-review/SKILL.md | Review code changes."
    );
    assert!(!prompt.contains("{{"));
}

#[test]
fn agent_system_prompt_includes_available_skill_sources_and_paths() {
    let manifest = MachineManifest {
        skills: vec![SkillSummary {
            name: "code-review".to_string(),
            description: "Review code changes.".to_string(),
            path: PathBuf::from("/repo/.agents/skills/code-review/SKILL.md"),
            source: SkillSource::ProjectAgents,
        }],
        ..MachineManifest::default()
    };

    let prompt = agent_system_prompt(&ToolRegistry::default(), &manifest);

    assert!(prompt.contains("[available_skills]"));
    assert!(prompt.contains("code-review"));
    assert!(prompt.contains("project .agents"));
    assert!(prompt.contains("/repo/.agents/skills/code-review/SKILL.md"));
    assert!(prompt.contains("Review code changes."));
}

#[test]
fn skill_display_line_includes_source_and_path_for_skills_command() {
    let skill = SkillSummary {
        name: "docs".to_string(),
        description: "Write project docs.".to_string(),
        path: PathBuf::from("/home/user/.vyrn/skills/docs/SKILL.md"),
        source: SkillSource::GlobalVyrn,
    };

    assert_eq!(
        skill.display_line(),
        "docs - Write project docs. [global ~/.vyrn: /home/user/.vyrn/skills/docs/SKILL.md]"
    );
}

#[test]
fn summary_prompts_render_from_templates() {
    let system_prompt = render_summary_system_prompt();
    let user_prompt = render_summary_user_prompt(&SummaryPromptSections {
        aggressiveness: SummaryAggressiveness::High,
        current_summary: "Need refactor prompt assembly.".to_string(),
        last_exchange: "user: make prompts observable".to_string(),
    });

    assert_eq!(
        system_prompt,
        "You rewrite compact agent session summaries."
    );
    assert_eq!(
        user_prompt,
        "\
Update the session summary for the next turn.
Keep the user's original high-level goal, constraints, decisions, paths touched, and open tasks.
Drop raw tool output that is no longer needed.
Aggressiveness: high.

Current summary:
Need refactor prompt assembly.

Last exchange:
user: make prompts observable"
    );
    assert!(!user_prompt.contains("{{"));
}

#[test]
fn summary_refresh_messages_insert_exchange_content() {
    let exchange = Exchange {
        user_input: "inspect prompt code".to_string(),
        assistant_text: "found inline strings".to_string(),
        tool_results: vec![ToolResult {
            name: "batch".to_string(),
            content: "src/agent/prompt.rs".to_string(),
            refresh_manifest: false,
            images: Vec::new(),
        }],
        ..Exchange::default()
    };

    let messages = build_summary_refresh_messages(
        SummaryAggressiveness::Low,
        Some("working on prompt refactor"),
        &exchange,
        true,
    );

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(
        messages[0].content_text(),
        Some("You rewrite compact agent session summaries.")
    );
    let user_prompt = messages[1].content_text().unwrap();
    assert!(user_prompt.contains("Current summary:\nworking on prompt refactor"));
    assert!(user_prompt.contains("user: inspect prompt code"));
    assert!(user_prompt.contains("assistant: found inline strings"));
    assert!(user_prompt.contains("- batch: src/agent/prompt.rs"));
}
