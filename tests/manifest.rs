use std::path::PathBuf;
use tempfile::tempdir;
use vyrn::config::ConfigSources;
use vyrn::mcp::McpRegistry;
use vyrn::skills::SkillRegistry;
use vyrn::tools::MachineManifest;

#[test]
fn skill_and_mcp_metadata_render_in_compact_manifest() {
    let temp = tempdir().unwrap();
    let project = temp.path().join("project");
    let global = temp.path().join("home/.vyrn");
    let skill_dir = project.join(".vyrn/skills/code-review");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::create_dir_all(project.join(".agents")).unwrap();
    std::fs::create_dir_all(&global).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        r#"---
name: code-review
description: Review code changes.
---

# Instructions
"#,
    )
    .unwrap();
    std::fs::write(
        project.join(".agents/.mcp.json"),
        r#"{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["server"],
      "eager": true
    }
  }
}"#,
    )
    .unwrap();

    let sources = sources(project, global);
    let skills = SkillRegistry::discover(&sources).unwrap();
    let mcp = McpRegistry::load(&sources).unwrap();
    let manifest = MachineManifest::scan(&skills, &mcp);
    let compact = manifest.compact();

    assert!(compact.contains(&format!(
        "[machine] {}/{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )));
    assert!(compact.contains("[skills] code-review"));
    assert!(compact.contains("[mcp] filesystem(eager)"));
}

fn sources(project_root: PathBuf, global_vyrn: PathBuf) -> ConfigSources {
    let project_vyrn = project_root.join(".vyrn");
    let project_agents = project_root.join(".agents");
    ConfigSources {
        project_config: project_vyrn.join("config.toml"),
        global_config: global_vyrn.join("config.toml"),
        project_models: project_vyrn.join("models.toml"),
        global_models: global_vyrn.join("models.toml"),
        project_state: project_vyrn.join("state.toml"),
        project_vyrn_mcp: project_vyrn.join(".mcp.json"),
        project_agents_mcp: project_agents.join(".mcp.json"),
        project_root,
        project_vyrn,
        project_agents,
        global_vyrn,
    }
}
