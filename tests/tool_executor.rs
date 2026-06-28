use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};
use tempfile::tempdir;
use vyrn::tools::{
    ASK_USER_TOOL_NAME, AskUserAnswer, AskUserRequest, AskUserResponse, ToolRegistry,
};

#[tokio::test]
async fn edit_file_requires_exactly_one_match() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("sample.txt");
    std::fs::write(&path, "alpha beta gamma").unwrap();
    let tools = ToolRegistry::core();

    let result = tools
        .execute(
            "edit_file",
            json!({
                "path": path,
                "old": "beta",
                "new": "delta"
            }),
        )
        .await
        .unwrap();

    assert_eq!(result.name, "edit_file");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "alpha delta gamma");
}

#[tokio::test]
async fn edit_file_rejects_ambiguous_matches() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("sample.txt");
    std::fs::write(&path, "same same").unwrap();
    let tools = ToolRegistry::core();

    let error = tools
        .execute(
            "edit_file",
            json!({
                "path": path,
                "old": "same",
                "new": "other"
            }),
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("matched 2 times"));
}

#[tokio::test]
async fn batch_continues_after_failed_command() {
    let tools = ToolRegistry::core();
    let result = tools
        .execute(
            "batch",
            json!({
                "commands": [
                    "printf first",
                    "exit 7",
                    "printf third"
                ]
            }),
        )
        .await
        .unwrap();

    assert!(result.content.contains("\"status\": 7"));
    assert!(result.content.contains("first"));
    assert!(result.content.contains("third"));
}

#[tokio::test]
async fn batch_trims_huge_stdout_and_stderr() {
    let tools = ToolRegistry::core();
    let result = tools
        .execute(
            "batch",
            json!({
                "commands": [
                    "python3 -c 'import sys; print(\"A\" * 12000); print(\"B\" * 12000, file=sys.stderr)'"
                ]
            }),
        )
        .await
        .unwrap();

    assert!(result.content.contains("[trimmed "));
    assert!(result.content.contains("chars from batch output"));
    assert!(result.content.len() < 18_000, "{}", result.content.len());
}

#[tokio::test]
async fn batch_shares_output_budget_across_many_noisy_commands() {
    let tools = ToolRegistry::core();
    let noisy =
        "python3 -c 'import sys; print(\"A\" * 12000); print(\"B\" * 12000, file=sys.stderr)'";
    let result = tools
        .execute(
            "batch",
            json!({
                "commands": [noisy, noisy, noisy, noisy]
            }),
        )
        .await
        .unwrap();

    assert_eq!(result.content.matches("chars from batch output").count(), 8);
    assert!(result.content.len() < 24_000, "{}", result.content.len());
}

#[tokio::test]
async fn read_image_attaches_base64_images() {
    let temp = tempdir().unwrap();
    let path = temp.path().join("sample.png");
    std::fs::write(&path, [137, 80, 78, 71]).unwrap();
    let tools = ToolRegistry::core();

    let result = tools
        .execute(
            "read_image",
            json!({
                "paths": [path]
            }),
        )
        .await
        .unwrap();

    assert_eq!(result.name, "read_image");
    assert!(result.content.contains("attached 1 image"));
    assert_eq!(result.images.len(), 1);
    assert_eq!(result.images[0].mime_type, "image/png");
    assert_eq!(
        result.images[0].base64_data,
        STANDARD.encode([137, 80, 78, 71])
    );
}

#[test]
fn ask_user_is_exposed_as_core_tool_schema() {
    let tools = ToolRegistry::core();
    let schemas = tools.schemas();
    let ask_user = schemas
        .iter()
        .find(|schema| schema["function"]["name"] == ASK_USER_TOOL_NAME)
        .expect("ask_user schema should be exposed");

    assert_eq!(
        ask_user["function"]["description"],
        "ask human clarification"
    );
    assert_eq!(
        ask_user["function"]["parameters"]["required"],
        json!(["questions"])
    );
}

#[test]
fn ask_user_request_validates_question_shape() {
    let request = AskUserRequest::parse(json!({
        "questions": [{
            "id": "scope",
            "header": "Scope",
            "question": "Which path should I take?",
            "options": [
                { "label": "Core tool", "description": "Always available" },
                { "label": "Slash command" }
            ]
        }]
    }))
    .unwrap();

    assert_eq!(request.questions[0].id, "scope");
    assert_eq!(request.questions[0].options.len(), 2);

    let error = AskUserRequest::parse(json!({
        "questions": [
            { "id": "scope", "question": "One?" },
            { "id": "scope", "question": "Two?" }
        ]
    }))
    .unwrap_err();
    assert!(error.to_string().contains("duplicate question id"));
}

#[test]
fn ask_user_response_serializes_tool_payload() {
    let result = AskUserResponse {
        answers: vec![
            AskUserAnswer::Option {
                id: "scope".to_string(),
                answer: "Core tool".to_string(),
                option_index: 0,
                option_label: "Core tool".to_string(),
            },
            AskUserAnswer::Freeform {
                id: "notes".to_string(),
                answer: "Use the smallest API.".to_string(),
            },
        ],
    }
    .into_tool_result()
    .unwrap();

    let value: Value = serde_json::from_str(&result.content).unwrap();
    assert_eq!(value["answers"][0]["kind"], "option");
    assert_eq!(value["answers"][0]["option_index"], 0);
    assert_eq!(value["answers"][1]["kind"], "freeform");
    assert_eq!(value["answers"][1]["answer"], "Use the smallest API.");
}
