use vyrn::agent::tokens::TokenLedger;
use vyrn::agent::tokens::{TokenBreakdown, TurnUsage, estimate_chat_request_breakdown};
use vyrn::config::SummaryAggressiveness;
use vyrn::llm::types::ToolCallFunction;
use vyrn::llm::{ChatMessage, ImageAttachment, ToolCall};

#[test]
fn context_aggressiveness_escalates_near_budget() {
    let context = vyrn::agent::context::ContextManager::new(100, SummaryAggressiveness::Low);

    assert_eq!(
        context.effective_aggressiveness(50),
        SummaryAggressiveness::Low
    );
    assert_eq!(
        context.effective_aggressiveness(75),
        SummaryAggressiveness::Medium
    );
    assert_eq!(
        context.effective_aggressiveness(95),
        SummaryAggressiveness::High
    );
}

#[test]
fn token_ledger_accumulates_savings() {
    let mut ledger = TokenLedger::default();
    let mut turn = TurnUsage::default();
    turn.add_call_with_breakdown(
        "agent",
        100,
        250,
        TokenBreakdown {
            system_prompt: 20,
            user_requests: 30,
            tool_schemas: 50,
            ..TokenBreakdown::default()
        },
    );

    ledger.push_turn(turn);

    assert_eq!(ledger.session_sent, 100);
    assert_eq!(ledger.session_would_be, 250);
    assert_eq!(ledger.session_saved, 150);
    assert_eq!(ledger.turns[0].breakdown.system_prompt, 20);
    assert_eq!(ledger.turns[0].breakdown.user_requests, 30);
    assert_eq!(ledger.turns[0].breakdown.tool_schemas, 50);
}

#[test]
fn token_breakdown_tracks_prompt_contributors() {
    let messages = vec![
        ChatMessage::system(
            "[role] terminal agent\n[skills] docs\n[available_skills]\n- docs | project .vyrn | .vyrn/skills/docs/SKILL.md | Write docs.",
        ),
        ChatMessage::system("[summary]\nEdited src/lib.rs"),
        ChatMessage::user_with_images(
            "inspect this screenshot",
            &[ImageAttachment::new("screen.png", "image/png", "abc")],
        ),
        ChatMessage::assistant_tool_calls(
            String::new(),
            vec![ToolCall {
                id: "call_1".to_string(),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":"src/lib.rs"}"#.to_string(),
                },
            }],
        ),
        ChatMessage::tool("call_1", "file contents"),
    ];
    let tools = vec![serde_json::json!({
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "read a file",
            "parameters": { "type": "object" }
        }
    })];

    let breakdown = estimate_chat_request_breakdown(&messages, &tools);

    assert!(breakdown.system_prompt > 0);
    assert!(breakdown.summaries > 0);
    assert!(breakdown.user_requests > 0);
    assert_eq!(breakdown.images, 256);
    assert!(breakdown.skills > 0);
    assert!(breakdown.tool_schemas > 0);
    assert!(breakdown.tool_call_inputs > 0);
    assert!(breakdown.tool_call_outputs > 0);
    assert!(breakdown.overhead > 0);
}

#[test]
fn token_breakdown_tracks_skill_file_reads_as_skill_tokens() {
    let messages = vec![
        ChatMessage::assistant_tool_calls(
            String::new(),
            vec![ToolCall {
                id: "call_skill".to_string(),
                kind: "function".to_string(),
                function: ToolCallFunction {
                    name: "read_file".to_string(),
                    arguments: r#"{"path":".vyrn/skills/docs/SKILL.md"}"#.to_string(),
                },
            }],
        ),
        ChatMessage::tool("call_skill", "# Instructions\nWrite compact docs."),
    ];

    let breakdown = estimate_chat_request_breakdown(&messages, &[]);

    assert!(breakdown.skills > 0);
    assert!(breakdown.tool_call_inputs > 0);
    assert_eq!(breakdown.tool_call_outputs, 0);
}
