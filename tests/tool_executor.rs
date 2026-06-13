use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
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
