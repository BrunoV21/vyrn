use crate::agent::prompt::build_summary_refresh_messages;
use crate::agent::tokens::{estimate_messages_breakdown, estimate_text_tokens};
use crate::agent::transcript::Exchange;
use crate::config::SummaryAggressiveness;
use crate::llm::{ChatCompletionRequest, LlmError, OpenAiClient};

#[derive(Debug, Clone)]
pub struct ContextManager {
    summary: Option<String>,
    previous_exchange: Option<Exchange>,
    raw_history_tokens: usize,
    configured_aggressiveness: SummaryAggressiveness,
    max_tokens: usize,
}

impl ContextManager {
    pub fn new(max_tokens: usize, configured_aggressiveness: SummaryAggressiveness) -> Self {
        Self {
            summary: None,
            previous_exchange: None,
            raw_history_tokens: 0,
            configured_aggressiveness,
            max_tokens,
        }
    }

    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    pub fn previous_exchange(&self) -> Option<&Exchange> {
        self.previous_exchange.as_ref()
    }

    pub fn raw_history_tokens(&self) -> usize {
        self.raw_history_tokens
    }

    pub fn set_previous_exchange(&mut self, exchange: Exchange) {
        self.raw_history_tokens += estimate_text_tokens(&exchange.compact(true));
        self.previous_exchange = Some(exchange);
    }

    pub fn clear(&mut self) {
        self.summary = None;
        self.previous_exchange = None;
        self.raw_history_tokens = 0;
    }

    pub fn effective_aggressiveness(
        &self,
        estimated_prompt_tokens: usize,
    ) -> SummaryAggressiveness {
        let ratio = estimated_prompt_tokens as f64 / self.max_tokens.max(1) as f64;
        if ratio > 0.9 {
            SummaryAggressiveness::High
        } else if ratio > 0.7 {
            match self.configured_aggressiveness {
                SummaryAggressiveness::Low => SummaryAggressiveness::Medium,
                other => other,
            }
        } else {
            self.configured_aggressiveness
        }
    }

    pub async fn refresh_summary(
        &mut self,
        client: &OpenAiClient,
        estimated_next_prompt_tokens: usize,
    ) -> Result<Option<SummaryRefreshUsage>, LlmError> {
        let Some(exchange) = &self.previous_exchange else {
            return Ok(None);
        };

        let aggressiveness = self.effective_aggressiveness(estimated_next_prompt_tokens);
        let include_tool_results = matches!(aggressiveness, SummaryAggressiveness::Low);
        let messages = build_summary_refresh_messages(
            aggressiveness,
            self.summary.as_deref(),
            exchange,
            include_tool_results,
        );
        let input_breakdown = estimate_messages_breakdown(&messages);
        let input_tokens = input_breakdown.total();
        let response = client
            .complete_chat(ChatCompletionRequest {
                model: String::new(),
                messages,
                tools: Vec::new(),
                tool_choice: None,
                stream: false,
            })
            .await?;
        let summary = response
            .choices
            .first()
            .and_then(|choice| choice.message.content_text().map(str::to_string))
            .unwrap_or_default();
        let estimated_output_tokens = if summary.trim().is_empty() {
            0
        } else {
            estimate_text_tokens(&summary)
        };
        let usage = response.usage;
        let input_tokens = usage
            .map(|usage| usage.prompt_tokens)
            .filter(|tokens| *tokens > 0)
            .unwrap_or(input_tokens);
        let output_tokens = usage
            .map(|usage| usage.completion_tokens)
            .filter(|tokens| *tokens > 0)
            .unwrap_or(estimated_output_tokens);
        self.summary = Some(summary.trim().to_string());
        Ok(Some(SummaryRefreshUsage {
            input_tokens,
            output_tokens,
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryRefreshUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}
