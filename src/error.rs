use thiserror::Error;

#[derive(Debug, Error)]
pub enum VyrnError {
    #[error(transparent)]
    Config(#[from] crate::config::ConfigError),
    #[error(transparent)]
    Llm(#[from] crate::llm::LlmError),
    #[error(transparent)]
    Tool(#[from] crate::tools::ToolError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
