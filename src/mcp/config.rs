use crate::config::ConfigSources;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum McpConfigError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: String,
        source: serde_json::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub eager: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerSummary {
    pub name: String,
    pub eager: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct McpRegistry {
    servers: BTreeMap<String, McpServerConfig>,
}

#[derive(Debug, Deserialize)]
struct McpFile {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, McpServerFile>,
}

#[derive(Debug, Deserialize)]
struct McpServerFile {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
    #[serde(default)]
    eager: bool,
}

impl McpRegistry {
    pub fn load(sources: &ConfigSources) -> Result<Self, McpConfigError> {
        let mut registry = Self::default();
        registry.merge_file(&sources.project_agents_mcp)?;
        registry.merge_file(&sources.project_vyrn_mcp)?;
        Ok(registry)
    }

    pub fn list(&self) -> impl Iterator<Item = &McpServerConfig> {
        self.servers.values()
    }

    pub fn summaries(&self) -> Vec<McpServerSummary> {
        self.servers
            .values()
            .map(|server| McpServerSummary {
                name: server.name.clone(),
                eager: server.eager,
            })
            .collect()
    }

    fn merge_file(&mut self, path: &Path) -> Result<(), McpConfigError> {
        if !path.exists() {
            return Ok(());
        }

        let raw = std::fs::read_to_string(path).map_err(|source| McpConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let parsed: McpFile =
            serde_json::from_str(&raw).map_err(|source| McpConfigError::Parse {
                path: path.display().to_string(),
                source,
            })?;

        for (name, server) in parsed.mcp_servers {
            self.servers.insert(
                name.clone(),
                McpServerConfig {
                    name,
                    command: server.command,
                    args: server.args,
                    env: server.env,
                    eager: server.eager,
                },
            );
        }

        Ok(())
    }
}
