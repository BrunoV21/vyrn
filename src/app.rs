use crate::agent::context::ContextManager;
use crate::agent::tokens::TokenLedger;
use crate::cli::Cli;
use crate::config::{ConfigSources, EffectiveConfig, ModelProfile, ModelRegistry, ModelState};
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
}

impl App {
    pub async fn build(args: Cli) -> anyhow::Result<Self> {
        let cwd = std::env::current_dir()?;
        let sources = ConfigSources::discover(cwd)?;
        let mut config = EffectiveConfig::load(&sources)?;
        if let Some(max_tokens) = args.context {
            config.context.max_tokens = max_tokens;
        }

        let models = crate::config::load_model_profiles(&sources)?;
        let model_state = ModelState::load(&sources);
        let model = if args.models {
            let model = crate::tui::select_model(&models).await?;
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

        Ok(Self {
            client: OpenAiClient::new(model.clone()),
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
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        Repl::new(self).run().await
    }
}
