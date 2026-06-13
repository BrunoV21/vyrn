use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillSource {
    ProjectVyrn,
    GlobalVyrn,
    ProjectAgents,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillSummary {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub source: SkillSource,
}
