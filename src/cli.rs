use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Clone, Parser)]
#[command(name = "vyrn")]
#[command(about = "Token-efficient CLI agent for OpenAI-compatible local and small LLMs.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Select a configured model profile before starting the session.
    #[arg(long, alias = "model")]
    pub models: bool,

    /// Override context budget for this session.
    #[arg(long)]
    pub context: Option<usize>,

    /// Show full token counts and raw summaries.
    #[arg(long)]
    pub verbose: bool,

    /// Show provider/network details and write structured session trace JSON.
    #[arg(long)]
    pub debug: bool,

    /// Run one prompt non-interactively, then exit.
    #[arg(short = 'p', long, value_name = "PROMPT")]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Commands {
    /// Start the opt-in full-screen terminal interface.
    Tui(TuiArgs),
    /// Initialize the local .vyrn directory and seed project config files.
    Init,
    /// Run JSON-defined live agent evals.
    Eval(EvalArgs),
    /// Write the local static debug trace viewer and print its path.
    DebugViewer(DebugViewerArgs),
}

#[derive(Debug, Clone, Args)]
pub struct TuiArgs {
    /// Select a configured model profile before opening the interface.
    #[arg(long, alias = "model")]
    pub models: bool,

    /// Override context budget for this session.
    #[arg(long)]
    pub context: Option<usize>,

    /// Show full token counts and raw summaries.
    #[arg(long)]
    pub verbose: bool,

    /// Show provider/network details and write structured session trace JSON.
    #[arg(long)]
    pub debug: bool,
}

#[derive(Debug, Clone, Args)]
pub struct DebugViewerArgs {
    /// Embed this trace JSON so the generated viewer opens it immediately.
    pub trace: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct EvalArgs {
    /// JSON eval suite to run.
    pub suite: PathBuf,

    /// Run one case by id.
    #[arg(long)]
    pub case: Option<String>,

    /// Override the suite or case model profile.
    #[arg(long)]
    pub model: Option<String>,

    /// Directory for trace output.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Print the run summary as JSON.
    #[arg(long)]
    pub json: bool,

    /// Validate the suite without calling a model or running tools.
    #[arg(long)]
    pub dry_run: bool,

    /// Do not write per-case debug events.
    #[arg(long)]
    pub no_debug: bool,
}
