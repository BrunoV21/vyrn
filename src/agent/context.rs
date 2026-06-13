use crate::agent::tokens::{estimate_messages_tokens, estimate_text_tokens};
use crate::agent::transcript::Exchange;
use crate::config::SummaryAggressiveness;
use crate::llm::{ChatCompletionRequest, ChatMessage, LlmError, OpenAiClient};

#[derive(Debug, Clone)]
pub struct ContextManager {
    summary: Option<String>,
    previous_exchange: Option<Exchange>,
    configured_aggressiveness: SummaryAggressiveness,
    max_tokens: usize,
}

impl ContextManager {
    pub fn new(max_tokens: usize, configured_aggressiveness: SummaryAggressiveness) -> Self {
        Self {
            summary: None,
            previous_exchange: None,
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

    pub fn set_previous_exchange(&mut self, exchange: Exchange) {
        self.previous_exchange = Some(exchange);
    }

    pub fn clear(&mut self) {
        self.summary = None;
        self.previous_exchange = None;
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
    ) -> Result<Option<usize>, LlmError> {
        let Some(exchange) = &self.previous_exchange else {
            return Ok(None);
        };

        let aggressiveness = self.effective_aggressiveness(estimated_next_prompt_tokens);
        let include_tool_results = matches!(aggressiveness, SummaryAggressiveness::Low);
        let current_summary = self.summary.as_deref().unwrap_or("none");
        let prompt = format!(
            "Update the session summary for the next turn.\n\
Keep the user's original high-level goal, constraints, decisions, paths touched, and open tasks.\n\
Drop raw tool output that is no longer needed.\n\
Aggressiveness: {aggressiveness}.\n\n\
Current summary:\n{current_summary}\n\n\
Last exchange:\n{}",
            exchange.compact(include_tool_results)
        );

        let messages = vec![
            ChatMessage::system("You rewrite compact agent session summaries."),
            ChatMessage::user(prompt),
        ];
        let sent = estimate_messages_tokens(&messages);
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
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default();
        self.summary = Some(summary.trim().to_string());
        Ok(Some(
            response
                .usage
                .map(|usage| usage.prompt_tokens)
                .unwrap_or(sent),
        ))
    }

    pub fn estimate_would_be_tokens(&self, system: &str, user_input: &str) -> usize {
        let mut total = estimate_text_tokens(system) + estimate_text_tokens(user_input);
        if let Some(summary) = &self.summary {
            total += estimate_text_tokens(summary);
        }
        if let Some(exchange) = &self.previous_exchange {
            total += estimate_text_tokens(&exchange.compact(true));
        }
        total
    }
}
