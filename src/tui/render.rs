use crate::agent::tokens::TokenLedger;
use crate::tools::MachineManifest;

pub fn startup(model_name: &str, base_url: &str, manifest: &MachineManifest, max_tokens: usize) {
    println!("> using {model_name} @ {base_url}");
    println!("> manifest: {}", manifest.display_line());
    println!("> context budget: {max_tokens} tokens");
    println!();
}

pub fn stats(ledger: &TokenLedger) {
    println!(
        "turn sent: {} | turn saved: {} | session saved: {}",
        format_number(
            ledger
                .turns
                .last()
                .map(|turn| turn.sent)
                .unwrap_or_default() as isize
        ),
        format_number(
            ledger
                .turns
                .last()
                .map(|turn| turn.saved)
                .unwrap_or_default()
        ),
        format_number(ledger.session_saved),
    );
}

pub fn format_number(value: isize) -> String {
    let negative = value < 0;
    let digits = value.abs().to_string();
    let mut out = String::new();
    for (idx, ch) in digits.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    let mut out = out.chars().rev().collect::<String>();
    if negative {
        out.insert(0, '-');
    }
    out
}
