use serde_json::json;
use tempfile::tempdir;
use vyrn::tools::ToolRegistry;

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
