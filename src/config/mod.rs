pub mod models;
pub mod paths;

pub use models::{ModelProfile, ModelRegistry, ModelState, load_model_profiles};
pub use paths::ConfigSources;

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    ParseToml {
        path: String,
        source: toml::de::Error,
    },
    #[error("no model profiles found; create .vyrn/models.toml or ~/.vyrn/models.toml")]
    NoModelProfiles,
    #[error("model profile '{0}' was not found")]
    MissingModel(String),
    #[error("default model '{0}' was not found")]
    MissingDefaultModel(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SummaryAggressiveness {
    Low,
    Medium,
    High,
}

impl std::fmt::Display for SummaryAggressiveness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Low => f.write_str("low"),
            Self::Medium => f.write_str("medium"),
            Self::High => f.write_str("high"),
        }
    }
}

impl Default for SummaryAggressiveness {
    fn default() -> Self {
        Self::Medium
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct EffectiveConfig {
    pub context: ContextConfig,
    pub agent: AgentConfig,
    pub manifest: ManifestConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct ContextConfig {
    pub max_tokens: usize,
    pub summary_aggressiveness: SummaryAggressiveness,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct AgentConfig {
    pub default_model: String,
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default)]
pub struct ManifestConfig {
    pub auto_refresh: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct PartialConfig {
    context: PartialContextConfig,
    agent: PartialAgentConfig,
    manifest: PartialManifestConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PartialContextConfig {
    max_tokens: Option<usize>,
    summary_aggressiveness: Option<SummaryAggressiveness>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PartialAgentConfig {
    default_model: Option<String>,
    stream: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PartialManifestConfig {
    auto_refresh: Option<bool>,
}

impl Default for EffectiveConfig {
    fn default() -> Self {
        Self {
            context: ContextConfig::default(),
            agent: AgentConfig::default(),
            manifest: ManifestConfig::default(),
        }
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 4096,
            summary_aggressiveness: SummaryAggressiveness::Medium,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            default_model: "llama3".to_string(),
            stream: true,
        }
    }
}

impl Default for ManifestConfig {
    fn default() -> Self {
        Self {
            auto_refresh: false,
        }
    }
}

impl EffectiveConfig {
    pub fn load(sources: &ConfigSources) -> Result<Self, ConfigError> {
        let mut effective = Self::default();
        merge_config_file(&mut effective, &sources.project_config)?;
        merge_config_file(&mut effective, &sources.global_config)?;
        Ok(effective)
    }
}

fn merge_config_file(config: &mut EffectiveConfig, path: &Path) -> Result<(), ConfigError> {
    if !path.exists() {
        return Ok(());
    }

    let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.display().to_string(),
        source,
    })?;
    let partial: PartialConfig = toml::from_str(&raw).map_err(|source| ConfigError::ParseToml {
        path: path.display().to_string(),
        source,
    })?;

    if let Some(max_tokens) = partial.context.max_tokens {
        config.context.max_tokens = max_tokens;
    }
    if let Some(summary_aggressiveness) = partial.context.summary_aggressiveness {
        config.context.summary_aggressiveness = summary_aggressiveness;
    }
    if let Some(default_model) = partial.agent.default_model {
        config.agent.default_model = default_model;
    }
    if let Some(stream) = partial.agent.stream {
        config.agent.stream = stream;
    }
    if let Some(auto_refresh) = partial.manifest.auto_refresh {
        config.manifest.auto_refresh = auto_refresh;
    }

    Ok(())
}
