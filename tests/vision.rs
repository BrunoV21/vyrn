use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::Value;
use tempfile::tempdir;
use vyrn::agent::prompt::build_agent_prompt;
use vyrn::llm::{ChatMessage, ImageAttachment};
use vyrn::tools::{MachineManifest, ToolRegistry};
use vyrn::vision::attachments_from_text;

#[test]
fn user_message_serializes_multiple_images_as_openai_content_parts() {
    let message = ChatMessage::user_with_images(
        "compare these",
        &[
            ImageAttachment::new("one.png", "image/png", "AAA="),
            ImageAttachment::new("two.jpg", "image/jpeg", "BBB="),
        ],
    );

    let json = serde_json::to_value(message).unwrap();
    let content = json["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "compare these");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAA=");
    assert_eq!(
        content[2]["image_url"]["url"],
        "data:image/jpeg;base64,BBB="
    );
}

#[tokio::test]
async fn image_paths_are_loaded_as_base64_attachments() {
    let temp = tempdir().unwrap();
    let image = temp.path().join("sample.png");
    std::fs::write(&image, [137, 80, 78, 71]).unwrap();

    let attachments = attachments_from_text(&format!("describe {}", image.display()))
        .await
        .unwrap();

    assert_eq!(attachments.len(), 1);
    assert_eq!(attachments[0].mime_type, "image/png");
    assert_eq!(
        attachments[0].base64_data,
        STANDARD.encode([137, 80, 78, 71])
    );
}

#[tokio::test]
async fn prompt_includes_loaded_image_parts_in_current_user_message() {
    let temp = tempdir().unwrap();
    let image = temp.path().join("sample.jpg");
    std::fs::write(&image, [1, 2, 3]).unwrap();
    let attachments = attachments_from_text(&format!("what is in {}", image.display()))
        .await
        .unwrap();

    let prompt = build_agent_prompt(
        &ToolRegistry::default(),
        &MachineManifest::default(),
        None,
        "what is in this image?",
        &attachments,
    );
    let request = serde_json::json!({
        "model": "vision-model",
        "messages": prompt.messages,
        "tools": Vec::<Value>::new(),
        "stream": true
    });

    let content = request["messages"][1]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    assert!(
        content[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,")
    );
}
