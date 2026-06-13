use clap::Parser;
use vyrn::cli::Cli;

#[test]
fn model_flag_alias_enables_startup_model_selection() {
    let plural = Cli::try_parse_from(["vyrn", "--models"]).unwrap();
    assert!(plural.models);

    let singular = Cli::try_parse_from(["vyrn", "--model"]).unwrap();
    assert!(singular.models);
}
