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
    app::App::build(args).await?.run().await
}
