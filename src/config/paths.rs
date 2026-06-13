use directories::BaseDirs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ConfigSources {
    pub project_root: PathBuf,
    pub project_vyrn: PathBuf,
    pub project_agents: PathBuf,
    pub global_vyrn: PathBuf,
    pub project_config: PathBuf,
    pub global_config: PathBuf,
    pub project_models: PathBuf,
    pub global_models: PathBuf,
    pub project_state: PathBuf,
    pub project_vyrn_mcp: PathBuf,
    pub project_agents_mcp: PathBuf,
}

impl ConfigSources {
    pub fn discover(project_root: PathBuf) -> std::io::Result<Self> {
        let global_vyrn = BaseDirs::new()
            .map(|dirs| dirs.home_dir().join(".vyrn"))
            .unwrap_or_else(|| PathBuf::from(".vyrn"));
        let project_vyrn = project_root.join(".vyrn");
        let project_agents = project_root.join(".agents");

        Ok(Self {
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
        })
    }

    pub fn skill_roots(&self) -> Vec<PathBuf> {
        vec![
            self.project_vyrn.join("skills"),
            self.global_vyrn.join("skills"),
            self.project_agents.join("skills"),
        ]
    }
}
