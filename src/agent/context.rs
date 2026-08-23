use crate::agent::prompt::build_summary_refresh_messages;
use crate::agent::tokens::{TokenSource, estimate_messages_breakdown, estimate_text_tokens};
use crate::agent::transcript::{Exchange, truncate_ends};
use crate::config::SummaryAggressiveness;
use crate::debug_trace::{TraceMetadata, TraceRecorder};
use crate::llm::types::ChatCompletionResponse;
use crate::llm::{ChatCompletionRequest, LlmError, OpenAiClient};

const SUMMARY_MAX_OUTPUT_TOKENS: usize = 384;

#[derive(Debug, Clone)]
pub struct ContextManager {
    summary: Option<String>,
    previous_exchange: Option<Exchange>,
    session_goal: Option<String>,
    exact_tool_memory: Option<String>,
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
            exact_tool_memory: None,
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
        if self.session_goal.is_none() && is_meaningful_goal_candidate(user_input) {
            self.session_goal = Some(crate::agent::transcript::truncate(user_input.trim(), 1600));
        }
    }

    pub fn prompt_memory(&self) -> Option<String> {
        if self.summary.is_none() && self.previous_exchange.is_none() {
            return None;
        }

        let mut sections = Vec::new();
        let memory_chars = self.max_tokens.max(1);
        let goal_chars = memory_chars.saturating_mul(20).div_ceil(100).max(80);
        let summary_chars = memory_chars.saturating_mul(25).div_ceil(100).max(100);
        let exact_chars = memory_chars.saturating_mul(25).div_ceil(100).max(100);
        let recent_chars = memory_chars.saturating_mul(30).div_ceil(100).max(120);
        if let Some(goal) = self.session_goal.as_deref() {
            sections.push(format!(
                "session goal (bounded verbatim):\n{}",
                truncate_ends(goal, goal_chars)
            ));
        }
        if let Some(summary) = self
            .summary
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            sections.push(format!(
                "rolling summary:\n{}",
                truncate_ends(summary.trim(), summary_chars)
            ));
        }
        if let Some(exact_tool_memory) = self
            .exact_tool_memory
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            sections.push(format!(
                "exact tool memory (verbatim, authoritative):\n{}",
                truncate_ends(exact_tool_memory.trim(), exact_chars)
            ));
        }
        if let Some(exchange) = &self.previous_exchange {
            sections.push(format!(
                "most recent exchange (bounded verbatim anchor):\n{}",
                truncate_ends(&exchange.compact(false), recent_chars).trim()
            ));
        }
        (!sections.is_empty()).then(|| sections.join("\n\n"))
    }

    pub fn set_previous_exchange(&mut self, exchange: Exchange) {
        if self.session_goal.is_none() && is_meaningful_goal_candidate(&exchange.user_input) {
            self.session_goal = Some(crate::agent::transcript::truncate(
                exchange.user_input.trim(),
                1600,
            ));
        }
        if !exchange.turn_scratchpad.trim().is_empty() {
            let next = match self.exact_tool_memory.as_deref() {
                Some(current) if !current.trim().is_empty() => {
                    format!("{}\n{}", current.trim(), exchange.turn_scratchpad.trim())
                }
                _ => exchange.turn_scratchpad.trim().to_string(),
            };
            self.exact_tool_memory = Some(truncate_ends(&next, 1800));
        }
        self.raw_history_tokens += estimate_text_tokens(&exchange.compact(true));
        self.previous_exchange = Some(exchange);
    }

    pub fn clear(&mut self) {
        self.summary = None;
        self.previous_exchange = None;
        self.session_goal = None;
        self.exact_tool_memory = None;
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
            max_tokens: Some(SUMMARY_MAX_OUTPUT_TOKENS),
        };
        let pending = trace
            .as_ref()
            .map(|recorder| recorder.begin_call(client, &request, false, trace_metadata));
        let response_result = client.complete_chat(request).await;
        if let (Some(recorder), Some(pending)) = (trace, pending) {
            let _ = recorder.finish_call(pending, &response_result);
        }
        let response = response_result?;
        let candidate = response
            .choices
            .first()
            .and_then(|choice| choice.message.content_text().map(str::to_string))
            .unwrap_or_default();
        let estimated_output_tokens = if candidate.trim().is_empty() {
            0
        } else {
            estimate_text_tokens(&candidate)
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
        let hit_output_limit = summary_response_hit_output_limit(&response);
        self.summary = Some(if candidate.trim().is_empty() || hit_output_limit {
            fallback_summary(self.summary.as_deref(), exchange)
        } else {
            candidate.trim().to_string()
        });
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

fn summary_response_hit_output_limit(response: &ChatCompletionResponse) -> bool {
    response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
        .is_some_and(|reason| reason.eq_ignore_ascii_case("length"))
        || response
            .usage
            .is_some_and(|usage| usage.completion_tokens >= SUMMARY_MAX_OUTPUT_TOKENS)
}

fn fallback_summary(current_summary: Option<&str>, exchange: &Exchange) -> String {
    let mut sections = Vec::new();
    if let Some(current) = current_summary.filter(|summary| !summary.trim().is_empty()) {
        sections.push(current.trim().to_string());
    }
    sections.push(format!(
        "Recent exchange checkpoint:\n{}",
        exchange.compact(false).trim()
    ));
    truncate_ends(&sections.join("\n\n"), 3000)
}

fn is_meaningful_goal_candidate(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|character: char| character.is_ascii_punctuation())
        .to_ascii_lowercase();
    !normalized.is_empty()
        && !matches!(
            normalized.as_str(),
            "hi" | "hi there" | "hello" | "hello there" | "hey" | "hey there" | "greetings"
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryRefreshUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub input_source: TokenSource,
    pub output_source: TokenSource,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;
    use crate::llm::types::{ChatChoice, Usage};

    #[test]
    fn summary_limit_detection_uses_finish_reason_or_provider_count() {
        let response = |finish_reason: Option<&str>, completion_tokens| ChatCompletionResponse {
            choices: vec![ChatChoice {
                message: ChatMessage::assistant("partial summary"),
                finish_reason: finish_reason.map(str::to_string),
            }],
            usage: Some(Usage {
                prompt_tokens: 20,
                completion_tokens,
                total_tokens: 20 + completion_tokens,
            }),
        };

        assert!(summary_response_hit_output_limit(&response(
            Some("length"),
            20
        )));
        assert!(summary_response_hit_output_limit(&response(None, 385)));
        assert!(!summary_response_hit_output_limit(&response(
            Some("stop"),
            80
        )));
    }

    #[test]
    fn fallback_summary_preserves_existing_and_recent_exact_context() {
        let exchange = Exchange {
            user_input: "continue".to_string(),
            assistant_text: "done".to_string(),
            turn_scratchpad: "- results: /Users/bv-mac/project".to_string(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
        };

        let fallback = fallback_summary(Some("existing marker VIOLET_1"), &exchange);

        assert!(fallback.contains("existing marker VIOLET_1"));
        assert!(fallback.contains("/Users/bv-mac/project"));
        assert!(fallback.contains("user: continue"));
    }
}
