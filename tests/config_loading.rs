use std::path::PathBuf;
use tempfile::tempdir;
use vyrn::config::{
    ConfigSources, EffectiveConfig, ModelState, SummaryAggressiveness, load_model_profiles,
};

#[test]
fn global_config_overrides_project_config_and_models() {
    let temp = tempdir().unwrap();
    let project = temp.path().join("project");
    let global = temp.path().join("home/.vyrn");
    std::fs::create_dir_all(project.join(".vyrn")).unwrap();
    std::fs::create_dir_all(&global).unwrap();

    std::fs::write(
        project.join(".vyrn/config.toml"),
        r#"
[context]
max_tokens = 2048
summary_aggressiveness = "low"

[agent]
default_model = "local"
"#,
    )
    .unwrap();
    std::fs::write(
        global.join("config.toml"),
        r#"
[context]
max_tokens = 8192

[agent]
default_model = "global"
"#,
    )
    .unwrap();
    std::fs::write(
        project.join(".vyrn/models.toml"),
        r#"
[models.local]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
"#,
    )
    .unwrap();
    std::fs::write(
        global.join("models.toml"),
        r#"
[models.global]
base_url = "https://example.test/v1"
model = "small"
api_key = "secret"
"#,
    )
    .unwrap();

    let sources = sources(project, global);
    let config = EffectiveConfig::load(&sources).unwrap();
    assert_eq!(config.context.max_tokens, 8192);
    assert_eq!(
        config.context.summary_aggressiveness,
        SummaryAggressiveness::Low
    );
    assert_eq!(config.agent.default_model, "global");

    let models = load_model_profiles(&sources).unwrap();
    let selected = models.resolve_default(&config.agent.default_model).unwrap();
    assert_eq!(selected.name, "global");
    assert_eq!(selected.api_key, "secret");
}

#[test]
fn model_startup_resolution_uses_last_selected_then_default_then_first() {
    let temp = tempdir().unwrap();
    let project = temp.path().join("project");
    let global = temp.path().join("home/.vyrn");
    std::fs::create_dir_all(project.join(".vyrn")).unwrap();
    std::fs::create_dir_all(&global).unwrap();
    std::fs::write(
        project.join(".vyrn/models.toml"),
        r#"
[models.alpha]
base_url = "http://alpha/v1"
model = "alpha-model"

[models.beta]
base_url = "http://beta/v1"
model = "beta-model"
"#,
    )
    .unwrap();

    let sources = sources(project, global);
    let models = load_model_profiles(&sources).unwrap();

    assert_eq!(
        models
            .resolve_startup("missing", Some("beta"))
            .unwrap()
            .name,
        "beta"
    );
    assert_eq!(
        models
            .resolve_startup("alpha", Some("missing"))
            .unwrap()
            .name,
        "alpha"
    );
    assert_eq!(
        models.resolve_startup("missing", None).unwrap().name,
        "alpha"
    );

    ModelState::save_last_selected(&sources, "beta").unwrap();
    assert_eq!(
        ModelState::load(&sources).last_selected_model.as_deref(),
        Some("beta")
    );
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
