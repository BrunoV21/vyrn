use clap::Parser;
use std::process::{Command, Stdio};
use tempfile::tempdir;
use vyrn::cli::Cli;

#[test]
fn model_flag_alias_enables_startup_model_selection() {
    let plural = Cli::try_parse_from(["vyrn", "--models"]).unwrap();
    assert!(plural.models);

    let singular = Cli::try_parse_from(["vyrn", "--model"]).unwrap();
    assert!(singular.models);
}

#[test]
fn prompt_flag_accepts_short_and_long_forms_with_debug() {
    let short = Cli::try_parse_from(["vyrn", "-p", "inspect src", "--debug"]).unwrap();
    assert_eq!(short.prompt.as_deref(), Some("inspect src"));
    assert!(short.debug);

    let long = Cli::try_parse_from(["vyrn", "--prompt", "run tests"]).unwrap();
    assert_eq!(long.prompt.as_deref(), Some("run tests"));
}

#[test]
fn debug_viewer_writes_static_html() {
    let temp = tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("debug-viewer")
        .current_dir(temp.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let viewer_path = temp.path().join(".vyrn/debug/viewer.html");
    assert!(viewer_path.exists());
    let html = std::fs::read_to_string(viewer_path).unwrap();
    assert!(html.contains("vyrn debug trace viewer"), "{html}");
    assert!(html.contains("--vy-tech"), "{html}");
    assert!(html.contains("human + agent"), "{html}");
    assert!(html.contains("harness internals"), "{html}");
}

#[test]
fn debug_viewer_can_embed_a_trace_path() {
    let temp = tempdir().unwrap();
    let trace_path = temp.path().join("trace.json");
    std::fs::write(
        &trace_path,
        r#"{"schema_version":2,"run_kind":"interactive","session_id":"embedded","calls":[],"note":"</script>"}"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("debug-viewer")
        .arg(&trace_path)
        .current_dir(temp.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let html = std::fs::read_to_string(temp.path().join(".vyrn/debug/viewer.html")).unwrap();
    assert!(html.contains("embedded"), "{html}");
    assert!(html.contains(r#"\u003c/script\u003e"#), "{html}");
}

#[test]
fn init_creates_project_local_vyrn_scaffold() {
    let temp = tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("init")
        .current_dir(temp.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let project_vyrn = temp.path().join(".vyrn");
    let config_path = project_vyrn.join("config.toml");
    let models_path = project_vyrn.join("models.toml");
    let skills_dir = project_vyrn.join("skills");

    assert!(project_vyrn.is_dir());
    assert!(config_path.exists());
    assert!(models_path.exists());
    assert!(skills_dir.is_dir());

    let config = std::fs::read_to_string(config_path).unwrap();
    assert!(config.contains("max_tokens = 4096"), "{config}");
    assert!(config.contains("default_model = \"llama3\""), "{config}");

    let models = std::fs::read_to_string(models_path).unwrap();
    assert!(models.contains("[models.llama3]"), "{models}");
    assert!(models.contains("http://localhost:11434/v1"), "{models}");
}

#[test]
fn startup_without_models_prompts_and_writes_default_global_profile() {
    let temp = tempdir().unwrap();
    let home = temp.path().join("home");
    let project = temp.path().join("project");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(&project).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(&project)
        .env("HOME", &home)
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let models_path = home.join(".vyrn/models.toml");
    let models = std::fs::read_to_string(models_path).unwrap();
    assert!(models.contains("[models.llama3]"), "{models}");
    assert!(
        models.contains("base_url = \"http://localhost:11434/v1\""),
        "{models}"
    );
    assert!(models.contains("model = \"llama3.2\""), "{models}");
}
