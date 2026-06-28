use clap::Parser;
use std::process::Command;
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
}
