use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkillSource {
    ProjectVyrn,
    GlobalVyrn,
    ProjectAgents,
}

impl SkillSource {
    pub fn label(&self) -> &'static str {
        match self {
            Self::ProjectVyrn => "project .vyrn",
            Self::GlobalVyrn => "global ~/.vyrn",
            Self::ProjectAgents => "project .agents",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillSummary {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub source: SkillSource,
}

impl SkillSummary {
    pub fn prompt_line(&self) -> String {
        format!(
            "- {} | {} | {} | {}",
            self.name,
            self.source.label(),
            self.path.display(),
            self.description
        )
    }

    pub fn display_line(&self) -> String {
        format!(
            "{} - {} [{}: {}]",
            self.name,
            self.description,
            self.source.label(),
            self.path.display()
        )
    }
}
