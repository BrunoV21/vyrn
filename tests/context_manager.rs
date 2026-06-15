use vyrn::agent::tokens::TokenLedger;
use vyrn::agent::tokens::{
    TokenBreakdown, TurnUsage, estimate_assistant_output_tokens, estimate_chat_request_breakdown,
    estimate_messages_breakdown, estimate_unpruned_request_tokens,
};
use vyrn::agent::transcript::Exchange;
use vyrn::config::SummaryAggressiveness;
use vyrn::llm::types::ToolCallFunction;
use vyrn::llm::{ChatMessage, ImageAttachment, ToolCall};
use vyrn::tools::ToolResult;

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
fn context_tracks_raw_history_tokens_until_clear() {
    let mut context = vyrn::agent::context::ContextManager::new(1000, SummaryAggressiveness::Low);
    assert_eq!(context.raw_history_tokens(), 0);

    context.set_previous_exchange(Exchange {
        user_input: "read the file".to_string(),
        assistant_text: "I read it.".to_string(),
        tool_calls: Vec::new(),
        tool_results: vec![ToolResult::text("read_file", "important file contents")],
    });

    assert!(context.raw_history_tokens() > 0);
    assert!(context.previous_exchange().is_some());

    context.clear();

    assert_eq!(context.raw_history_tokens(), 0);
    assert!(context.previous_exchange().is_none());
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
fn token_ledger_tracks_summary_input_and_output_without_creating_savings() {
    let mut ledger = TokenLedger::default();
    let mut turn = TurnUsage::default();
    turn.add_call_with_breakdown(
        "summary",
        125,
        125,
        TokenBreakdown {
            summary_inputs: 100,
            summary_outputs: 25,
            ..TokenBreakdown::default()
        },
    );

    ledger.push_turn(turn);

    assert_eq!(ledger.session_sent, 125);
    assert_eq!(ledger.session_would_be, 125);
    assert_eq!(ledger.session_saved, 0);
    assert_eq!(ledger.turns[0].breakdown.summary_inputs, 100);
    assert_eq!(ledger.turns[0].breakdown.summary_outputs, 25);
}

#[test]
fn unpruned_request_tokens_replace_summary_with_raw_history() {
    let messages = vec![
        ChatMessage::system("[role] terminal agent"),
        ChatMessage::system("[summary]\nEdited src/lib.rs"),
        ChatMessage::user("continue"),
    ];
    let breakdown = estimate_messages_breakdown(&messages);
    let raw_history_tokens = breakdown.summaries + 25;

    let would_be = estimate_unpruned_request_tokens(&breakdown, raw_history_tokens);

    assert_eq!(
        would_be,
        breakdown.total() - breakdown.summaries + raw_history_tokens
    );
    assert!(would_be > breakdown.total());
}

#[test]
fn unpruned_request_tokens_keep_current_turn_tool_history() {
    let first_round = vec![
        ChatMessage::system("[role] terminal agent"),
        ChatMessage::system("[summary]\nPrevious summarized work"),
        ChatMessage::user("inspect src/lib.rs"),
    ];
    let second_round = vec![
        ChatMessage::system("[role] terminal agent"),
        ChatMessage::system("[summary]\nPrevious summarized work"),
        ChatMessage::user("inspect src/lib.rs"),
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
        ChatMessage::tool("call_1", "large tool output that belongs to this turn"),
    ];
    let first_breakdown = estimate_messages_breakdown(&first_round);
    let second_breakdown = estimate_messages_breakdown(&second_round);
    let raw_history_tokens = first_breakdown.summaries + 10;

    let first_would_be = estimate_unpruned_request_tokens(&first_breakdown, raw_history_tokens);
    let second_would_be = estimate_unpruned_request_tokens(&second_breakdown, raw_history_tokens);

    assert!(second_breakdown.total() > first_breakdown.total());
    assert!(second_would_be > first_would_be);
    assert_eq!(
        second_would_be,
        second_breakdown.total() - second_breakdown.summaries + raw_history_tokens
    );
}

#[test]
fn assistant_output_tokens_are_added_to_both_sent_and_would_be() {
    let request = vec![
        ChatMessage::system("[role] terminal agent"),
        ChatMessage::system("[summary]\nPrevious summarized work"),
        ChatMessage::user("inspect src/lib.rs"),
    ];
    let request_breakdown = estimate_messages_breakdown(&request);
    let raw_history_tokens = request_breakdown.summaries + 10;
    let output_tokens = estimate_assistant_output_tokens(&ChatMessage::assistant("Done."));

    let mut turn = TurnUsage::default();
    turn.add_call_with_breakdown(
        "agent",
        request_breakdown.total() + output_tokens,
        estimate_unpruned_request_tokens(&request_breakdown, raw_history_tokens) + output_tokens,
        {
            let mut breakdown = request_breakdown;
            breakdown.assistant_outputs = output_tokens;
            breakdown
        },
    );

    assert_eq!(turn.breakdown.assistant_outputs, output_tokens);
    assert_eq!(
        turn.would_be as isize - turn.sent as isize,
        estimate_unpruned_request_tokens(&request_breakdown, raw_history_tokens) as isize
            - request_breakdown.total() as isize
    );
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
