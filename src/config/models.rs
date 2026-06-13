use super::{ConfigError, ConfigSources};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelProfile {
    pub name: String,
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRegistry {
    profiles: BTreeMap<String, ModelProfile>,
}

#[derive(Debug, Deserialize)]
struct ModelsFile {
    #[serde(default)]
    models: BTreeMap<String, ModelProfileFile>,
}

#[derive(Debug, Deserialize)]
struct ModelProfileFile {
    base_url: String,
    model: String,
    #[serde(default)]
    api_key: String,
}

impl ModelRegistry {
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn list(&self) -> impl Iterator<Item = &ModelProfile> {
        self.profiles.values()
    }

    pub fn get(&self, name: &str) -> Option<ModelProfile> {
        self.profiles.get(name).cloned()
    }

    pub fn first(&self) -> Option<ModelProfile> {
        self.profiles.values().next().cloned()
    }

    pub fn resolve_default(&self, default_model: &str) -> Result<ModelProfile, ConfigError> {
        if self.profiles.is_empty() {
            return Err(ConfigError::NoModelProfiles);
        }

        self.get(default_model)
            .ok_or_else(|| ConfigError::MissingDefaultModel(default_model.to_string()))
    }

    pub fn resolve_startup(
        &self,
        default_model: &str,
        last_selected: Option<&str>,
    ) -> Result<ModelProfile, ConfigError> {
        if self.profiles.is_empty() {
            return Err(ConfigError::NoModelProfiles);
        }
        if let Some(last_selected) = last_selected
            && let Some(profile) = self.get(last_selected)
        {
            return Ok(profile);
        }
        if let Some(profile) = self.get(default_model) {
            return Ok(profile);
        }
        self.first().ok_or(ConfigError::NoModelProfiles)
    }

    fn merge_file(&mut self, path: &Path) -> Result<(), ConfigError> {
        if !path.exists() {
            return Ok(());
        }

        let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let parsed: ModelsFile = toml::from_str(&raw).map_err(|source| ConfigError::ParseToml {
            path: path.display().to_string(),
            source,
        })?;

        for (name, profile) in parsed.models {
            self.profiles.insert(
                name.clone(),
                ModelProfile {
                    name,
                    base_url: profile.base_url,
                    model: profile.model,
                    api_key: profile.api_key,
                },
            );
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ModelState {
    pub last_selected_model: Option<String>,
}

impl ModelState {
    pub fn load(sources: &ConfigSources) -> Self {
        if !sources.project_state.exists() {
            return Self::default();
        }
        let Ok(raw) = std::fs::read_to_string(&sources.project_state) else {
            return Self::default();
        };
        toml::from_str(&raw).unwrap_or_default()
    }

    pub fn save_last_selected(
        sources: &ConfigSources,
        model_name: &str,
    ) -> Result<(), std::io::Error> {
        if let Some(parent) = sources.project_state.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let state = Self {
            last_selected_model: Some(model_name.to_string()),
        };
        let raw = toml::to_string_pretty(&state).unwrap_or_default();
        std::fs::write(&sources.project_state, raw)
    }
}

pub fn load_model_profiles(sources: &ConfigSources) -> Result<ModelRegistry, ConfigError> {
    let mut registry = ModelRegistry::default();
    registry.merge_file(&sources.project_models)?;
    registry.merge_file(&sources.global_models)?;
    if registry.is_empty() {
        return Err(ConfigError::NoModelProfiles);
    }
    Ok(registry)
}
