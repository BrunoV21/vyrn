use crate::config::ConfigSources;
use crate::skills::{SkillSource, SkillSummary};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct SkillRegistry {
    skills: BTreeMap<String, SkillSummary>,
}

impl SkillRegistry {
    pub fn discover(sources: &ConfigSources) -> std::io::Result<Self> {
        let mut registry = Self::default();
        let roots = [
            (
                sources.project_vyrn.join("skills"),
                SkillSource::ProjectVyrn,
            ),
            (sources.global_vyrn.join("skills"), SkillSource::GlobalVyrn),
            (
                sources.project_agents.join("skills"),
                SkillSource::ProjectAgents,
            ),
        ];

        for (root, source) in roots {
            registry.discover_root(&root, source)?;
        }

        Ok(registry)
    }

    pub fn list(&self) -> impl Iterator<Item = &SkillSummary> {
        self.skills.values()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    fn discover_root(&mut self, root: &Path, source: SkillSource) -> std::io::Result<()> {
        if !root.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let path = entry.path().join("SKILL.md");
            if !path.exists() {
                continue;
            }
            if let Some(summary) = parse_skill_summary(&path, source.clone())? {
                self.skills.entry(summary.name.clone()).or_insert(summary);
            }
        }

        Ok(())
    }
}

fn parse_skill_summary(
    path: &PathBuf,
    source: SkillSource,
) -> std::io::Result<Option<SkillSummary>> {
    let raw = std::fs::read_to_string(path)?;
    let Some(frontmatter) = raw.strip_prefix("---") else {
        return Ok(None);
    };
    let Some((frontmatter, _body)) = frontmatter.split_once("---") else {
        return Ok(None);
    };

    let mut name = None;
    let mut description = None;
    for line in frontmatter.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value
            .trim()
            .trim_matches('"')
            .trim_matches('\'')
            .to_string();
        match key.trim() {
            "name" => name = Some(value),
            "description" => description = Some(value),
            _ => {}
        }
    }

    match (name, description) {
        (Some(name), Some(description)) => Ok(Some(SkillSummary {
            name,
            description,
            path: path.clone(),
            source,
        })),
        _ => Ok(None),
    }
}
