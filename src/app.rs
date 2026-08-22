use crate::agent::context::ContextManager;
use crate::agent::tokens::TokenLedger;
use crate::cli::Cli;
use crate::config::{
    ConfigError, ConfigSources, EffectiveConfig, ModelProfile, ModelRegistry, ModelState,
};
use crate::debug_trace::TraceRecorder;
use crate::llm::OpenAiClient;
use crate::mcp::McpRegistry;
use crate::skills::SkillRegistry;
use crate::tools::{MachineManifest, ToolRegistry};
use crate::tui::Repl;

pub struct App {
    pub config: EffectiveConfig,
    pub sources: ConfigSources,
    pub model: ModelProfile,
    pub models: ModelRegistry,
    pub client: OpenAiClient,
    pub tools: ToolRegistry,
    pub manifest: MachineManifest,
    pub skills: SkillRegistry,
    pub mcp: McpRegistry,
    pub context: ContextManager,
    pub stats: TokenLedger,
    pub verbose: bool,
    pub debug: bool,
    pub prompt: Option<String>,
    pub trace: Option<TraceRecorder>,
}

impl App {
    pub async fn build(args: Cli) -> anyhow::Result<Self> {
        let cwd = std::env::current_dir()?;
        let sources = ConfigSources::discover(cwd)?;
        let mut config = EffectiveConfig::load(&sources)?;
        if let Some(max_tokens) = args.context {
            config.context.max_tokens = max_tokens;
        }

        let mut models = match crate::config::load_model_profiles(&sources) {
            Ok(models) => models,
            Err(ConfigError::NoModelProfiles) => ModelRegistry::default(),
            Err(error) => return Err(error.into()),
        };
        let model_state = ModelState::load(&sources);
        let model = if models.is_empty() || args.models {
            let model = crate::tui::select_model(&sources, &mut models).await?;
            let _ = ModelState::save_last_selected(&sources, &model.name);
            model
        } else {
            models.resolve_startup(
                &config.agent.default_model,
                model_state.last_selected_model.as_deref(),
            )?
        };

        let skills = SkillRegistry::discover(&sources)?;
        let mcp = McpRegistry::load(&sources)?;
        let manifest = MachineManifest::scan(&skills, &mcp);
        let tools = ToolRegistry::core();
        let context = ContextManager::new(
            config.context.max_tokens,
            config.context.summary_aggressiveness,
        );

        let client = OpenAiClient::new(model.clone());
        let trace = if args.debug {
            Some(if args.prompt.is_some() {
                TraceRecorder::programmatic(&sources, &client)?
            } else {
                TraceRecorder::interactive(&sources, &client)?
            })
        } else {
            None
        };

        Ok(Self {
            client,
            model,
            models,
            sources,
            config,
            tools,
            manifest,
            skills,
            mcp,
            context,
            stats: TokenLedger::default(),
            verbose: args.verbose,
            debug: args.debug,
            prompt: args.prompt,
            trace,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        Repl::new(self).run().await
    }
}
