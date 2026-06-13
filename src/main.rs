use clap::Parser;
use vyrn::{app, cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cli::Cli::parse();
    app::App::build(args).await?.run().await
}
