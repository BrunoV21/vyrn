use std::fs;
use std::path::Path;

use crate::config::ConfigSources;

const PROJECT_CONFIG: &str = r#"[context]
max_tokens = 4096
summary_aggressiveness = "medium"

[agent]
default_model = "llama3"
stream = true

[manifest]
auto_refresh = false
"#;

const PROJECT_MODELS: &str = r#"# Project-local model profiles for vyrn.
# Add OpenAI-compatible endpoints here.

[models.llama3]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
api_key = ""
"#;

pub fn run(sources: &ConfigSources) -> anyhow::Result<()> {
    fs::create_dir_all(&sources.project_vyrn)?;
    let skills_dir = sources.project_vyrn.join("skills");
    fs::create_dir_all(&skills_dir)?;

    let created_config = write_if_missing(&sources.project_config, PROJECT_CONFIG)?;
    let created_models = write_if_missing(&sources.project_models, PROJECT_MODELS)?;

    println!("initialized {}", sources.project_vyrn.display());
    if created_config {
        println!("created {}", sources.project_config.display());
    }
    if created_models {
        println!("created {}", sources.project_models.display());
    }
    println!("created {}", skills_dir.display());

    Ok(())
}

fn write_if_missing(path: &Path, contents: &str) -> anyhow::Result<bool> {
    if path.exists() {
        return Ok(false);
    }

    fs::write(path, contents)?;
    Ok(true)
}
