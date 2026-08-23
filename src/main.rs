use clap::Parser;
use vyrn::{app, cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = cli::Cli::parse();
    if let Some(cli::Commands::Init) = args.command.clone() {
        let sources = vyrn::config::ConfigSources::discover(std::env::current_dir()?)?;
        vyrn::init::run(&sources)?;
        return Ok(());
    }
    if let Some(cli::Commands::Eval(eval_args)) = args.command.clone() {
        let exit_code = vyrn::eval::run(eval_args, args.context).await?;
        if exit_code != 0 {
            std::process::exit(exit_code);
        }
        return Ok(());
    }
    if let Some(cli::Commands::DebugViewer(viewer_args)) = args.command.clone() {
        let sources = vyrn::config::ConfigSources::discover(std::env::current_dir()?)?;
        let path =
            vyrn::debug_trace::write_viewer_for_trace(&sources, viewer_args.trace.as_deref())?;
        println!("{}", path.display());
        return Ok(());
    }
    if let Some(cli::Commands::Tui(tui_args)) = args.command.clone() {
        if args.prompt.is_some() {
            anyhow::bail!("vyrn tui is interactive and cannot be combined with --prompt");
        }
        args.models |= tui_args.models;
        args.context = tui_args.context.or(args.context);
        args.verbose |= tui_args.verbose;
        args.debug |= tui_args.debug;
        return app::App::build(args).await?.run_tui().await;
    }
    app::App::build(args).await?.run().await
}
