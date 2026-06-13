use clap::Parser;

#[derive(Debug, Clone, Parser)]
#[command(name = "vyrn")]
#[command(about = "Token-efficient CLI agent for OpenAI-compatible local and small LLMs.")]
pub struct Cli {
    /// Select a configured model profile before starting the session.
    #[arg(long, alias = "model")]
    pub models: bool,

    /// Override context budget for this session.
    #[arg(long)]
    pub context: Option<usize>,

    /// Show full token counts and raw summaries.
    #[arg(long)]
    pub verbose: bool,

    /// Show provider URLs, HTTP status/body, and request-level debug details on errors.
    #[arg(long)]
    pub debug: bool,
}
