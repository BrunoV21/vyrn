use crate::agent::prompt::build_summary_refresh_messages;
use crate::agent::tokens::{TokenSource, estimate_messages_breakdown, estimate_text_tokens};
use crate::agent::transcript::Exchange;
use crate::config::SummaryAggressiveness;
use crate::debug_trace::{TraceMetadata, TraceRecorder};
use crate::llm::{ChatCompletionRequest, LlmError, OpenAiClient};

#[derive(Debug, Clone)]
pub struct ContextManager {
    summary: Option<String>,
    previous_exchange: Option<Exchange>,
    session_goal: Option<String>,
    raw_history_tokens: usize,
    configured_aggressiveness: SummaryAggressiveness,
    max_tokens: usize,
}

impl ContextManager {
    pub fn new(max_tokens: usize, configured_aggressiveness: SummaryAggressiveness) -> Self {
        Self {
            summary: None,
            previous_exchange: None,
            session_goal: None,
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

    pub fn begin_turn(&mut self, user_input: &str) {
        if self.session_goal.is_none() && !user_input.trim().is_empty() {
            self.session_goal = Some(crate::agent::transcript::truncate(user_input.trim(), 1600));
        }
    }

    pub fn prompt_memory(&self) -> Option<String> {
        if self.summary.is_none() && self.previous_exchange.is_none() {
            return None;
        }

        let mut sections = Vec::new();
        if let Some(goal) = self.session_goal.as_deref() {
            sections.push(format!("session goal (verbatim):\n{goal}"));
        }
        if let Some(summary) = self
            .summary
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            sections.push(format!("rolling summary:\n{}", summary.trim()));
        }
        if let Some(exchange) = &self.previous_exchange {
            sections.push(format!(
                "most recent exchange (verbatim anchor):\n{}",
                crate::agent::transcript::truncate(&exchange.compact(false), 3600).trim()
            ));
        }
        (!sections.is_empty()).then(|| sections.join("\n\n"))
    }

    pub fn set_previous_exchange(&mut self, exchange: Exchange) {
        if self.session_goal.is_none() && !exchange.user_input.trim().is_empty() {
            self.session_goal = Some(crate::agent::transcript::truncate(
                exchange.user_input.trim(),
                1600,
            ));
        }
        self.raw_history_tokens += estimate_text_tokens(&exchange.compact(true));
        self.previous_exchange = Some(exchange);
    }

    pub fn clear(&mut self) {
        self.summary = None;
        self.previous_exchange = None;
        self.session_goal = None;
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
        trace: Option<&mut TraceRecorder>,
        mut trace_metadata: TraceMetadata,
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
        trace_metadata.estimated_input_tokens = Some(input_tokens);
        let request = ChatCompletionRequest {
            model: String::new(),
            messages,
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            stream_options: None,
            max_tokens: Some(384),
        };
        let pending = trace
            .as_ref()
            .map(|recorder| recorder.begin_call(client, &request, false, trace_metadata));
        let response_result = client.complete_chat(request).await;
        if let (Some(recorder), Some(pending)) = (trace, pending) {
            let _ = recorder.finish_call(pending, &response_result);
        }
        let response = response_result?;
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
        let provider_usage = response.usage;
        let input_tokens = provider_usage
            .map(|usage| usage.prompt_tokens)
            .filter(|tokens| *tokens > 0)
            .unwrap_or(input_tokens);
        let output_tokens = provider_usage
            .map(|usage| usage.completion_tokens)
            .filter(|tokens| *tokens > 0)
            .unwrap_or(estimated_output_tokens);
        if !summary.trim().is_empty() {
            self.summary = Some(summary.trim().to_string());
        }
        Ok(Some(SummaryRefreshUsage {
            input_tokens,
            output_tokens,
            input_source: if provider_usage.is_some_and(|usage| usage.prompt_tokens > 0) {
                TokenSource::Provider
            } else {
                TokenSource::Estimate
            },
            output_source: if provider_usage.is_some_and(|usage| usage.completion_tokens > 0) {
                TokenSource::Provider
            } else {
                TokenSource::Estimate
            },
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryRefreshUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub input_source: TokenSource,
    pub output_source: TokenSource,
}
