use clap::Parser;
use vyrn::{app, cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    if let Some(cli::Commands::Eval(eval_args)) = args.command.clone() {
        let exit_code = vyrn::eval::run(eval_args, args.context).await?;
        if exit_code != 0 {
            std::process::exit(exit_code);
        }
        return Ok(());
    }
    if let Some(cli::Commands::DebugViewer) = args.command.clone() {
        let sources = vyrn::config::ConfigSources::discover(std::env::current_dir()?)?;
        let path = vyrn::debug_trace::write_viewer(&sources)?;
        println!("{}", path.display());
        return Ok(());
    }
    app::App::build(args).await?.run().await
}
