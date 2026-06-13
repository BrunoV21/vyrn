use crate::mcp::{McpRegistry, McpServerSummary};
use crate::skills::{SkillRegistry, SkillSummary};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const KNOWN_BINARIES: &[&str] = &[
    "git",
    "curl",
    "wget",
    "python3",
    "node",
    "npm",
    "pnpm",
    "cargo",
    "rustc",
    "docker",
    "chrome",
    "google-chrome",
    "ffmpeg",
    "rg",
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MachineManifest {
    pub machine: MachineInfo,
    pub binaries: Vec<String>,
    pub skills: Vec<SkillSummary>,
    pub mcp_servers: Vec<McpServerSummary>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MachineInfo {
    pub os: String,
    pub arch: String,
    pub username: Option<String>,
    pub shell: Option<String>,
}

impl MachineManifest {
    pub fn scan(skills: &SkillRegistry, mcp: &McpRegistry) -> Self {
        Self {
            machine: MachineInfo::scan(),
            binaries: scan_binaries(),
            skills: skills.list().cloned().collect(),
            mcp_servers: mcp.summaries(),
        }
    }

    pub fn compact(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("[machine] {}", self.machine.compact()));
        if !self.binaries.is_empty() {
            lines.push(format!("[env] {}", self.binaries.join(",")));
        }
        if !self.skills.is_empty() {
            lines.push(format!(
                "[skills] {}",
                self.skills
                    .iter()
                    .map(|skill| skill.name.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        if !self.mcp_servers.is_empty() {
            lines.push(format!(
                "[mcp] {}",
                self.mcp_servers
                    .iter()
                    .map(|server| {
                        let mode = if server.eager { "eager" } else { "lazy" };
                        format!("{}({mode})", server.name)
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        lines.join("\n")
    }

    pub fn display_line(&self) -> String {
        if self.binaries.is_empty() {
            "none".to_string()
        } else {
            self.binaries.join(", ")
        }
    }
}

impl MachineInfo {
    fn scan() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            username: first_nonempty_env(&["USER", "USERNAME"]),
            shell: first_nonempty_env(&["SHELL", "ComSpec"])
                .and_then(|shell| file_name(&PathBuf::from(shell))),
        }
    }

    fn compact(&self) -> String {
        let mut parts = vec![format!("{}/{}", self.os, self.arch)];
        if let Some(username) = &self.username {
            parts.push(format!("user={username}"));
        }
        if let Some(shell) = &self.shell {
            parts.push(format!("shell={shell}"));
        }
        parts.join(" ")
    }
}

fn scan_binaries() -> Vec<String> {
    KNOWN_BINARIES
        .iter()
        .filter(|binary| binary_exists(binary))
        .map(|binary| normalize_binary_name(binary))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn binary_exists(binary: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };

    std::env::split_paths(&paths).any(|path| Path::new(&path).join(binary).is_file())
}

fn normalize_binary_name(binary: &str) -> String {
    match binary {
        "google-chrome" => "chrome".to_string(),
        other => other.to_string(),
    }
}

fn first_nonempty_env(keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|key| std::env::var(key).ok())
        .map(|value| value.trim().to_string())
        .find(|value| !value.is_empty())
}

fn file_name(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .or_else(|| path.to_str().map(str::to_string))
        .filter(|value| !value.is_empty())
}
