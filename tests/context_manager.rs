use vyrn::agent::tokens::TokenLedger;
use vyrn::agent::tokens::TurnUsage;
use vyrn::config::SummaryAggressiveness;

#[test]
fn context_aggressiveness_escalates_near_budget() {
    let context = vyrn::agent::context::ContextManager::new(100, SummaryAggressiveness::Low);

    assert_eq!(
        context.effective_aggressiveness(50),
        SummaryAggressiveness::Low
    );
    assert_eq!(
        context.effective_aggressiveness(75),
        SummaryAggressiveness::Medium
    );
    assert_eq!(
        context.effective_aggressiveness(95),
        SummaryAggressiveness::High
    );
}

#[test]
fn token_ledger_accumulates_savings() {
    let mut ledger = TokenLedger::default();
    let mut turn = TurnUsage::default();
    turn.add_call("agent", 100, 250);

    ledger.push_turn(turn);

    assert_eq!(ledger.session_sent, 100);
    assert_eq!(ledger.session_would_be, 250);
    assert_eq!(ledger.session_saved, 150);
}
