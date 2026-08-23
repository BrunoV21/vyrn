use crate::agent::tokens::{
    TokenBreakdown, TokenSource, estimate_assistant_output_tokens, estimate_messages_breakdown,
};
use crate::config::ConfigSources;
use crate::llm::types::{ChatCompletionResponse, Usage};
use crate::llm::{ChatCompletionRequest, LlmError, OpenAiClient};
use anyhow::Context as _;
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TRACE_SCHEMA_VERSION: u32 = 2;
const VIEWER_HTML: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>vyrn debug trace viewer</title>
  <style>
    :root {
      --vy-bg: #06070A;
      --vy-surface: #0D1016;
      --vy-surface-raised: #151A22;
      --vy-border: #273142;
      --vy-border-strong: #3A475E;
      --vy-text-primary: #F3F7FB;
      --vy-text-muted: #98A3B3;
      --vy-text-dim: #677287;
      --vy-violet: #8B5CF6;
      --vy-tech: #7DA2C2;
      --vy-tech-strong: #A9BDD3;
      --vy-success: #9FE870;
      --vy-amber: #F5A524;
      --vy-red: #F43F5E;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, rgba(125, 162, 194, .14), transparent 34rem), var(--vy-bg);
      color: var(--vy-text-primary);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 8px; color: var(--vy-tech-strong); font-size: 28px; }
    p { color: var(--vy-text-muted); line-height: 1.5; }
    .panel {
      background: linear-gradient(180deg, var(--vy-surface-raised), var(--vy-surface));
      border: 1px solid var(--vy-border);
      border-radius: 14px;
      padding: 18px;
      margin: 18px 0;
      box-shadow: 0 18px 48px rgba(0, 0, 0, .32);
    }
    .drop {
      border: 1px dashed var(--vy-border-strong);
      color: var(--vy-text-muted);
      padding: 18px;
      border-radius: 12px;
    }
    input[type=file], input[type=search] {
      width: 100%;
      margin-top: 10px;
      background: #06070A;
      color: var(--vy-text-primary);
      border: 1px solid var(--vy-border);
      border-radius: 10px;
      padding: 12px;
      font: inherit;
    }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--vy-border); border-radius: 10px; padding: 12px; background: rgba(6, 7, 10, .55); }
    .metric b { display: block; color: var(--vy-tech); margin-bottom: 6px; }
    .lanes { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }
    .lane { min-width: 0; }
    .lane > h2 { position: sticky; top: 0; z-index: 2; margin: 0; padding: 12px; background: rgba(6, 7, 10, .94); border-bottom: 1px solid var(--vy-border); }
    .lane.interactions > h2 { color: var(--vy-violet); }
    .lane.harness > h2 { color: var(--vy-tech); }
    .lane-empty { color: var(--vy-text-dim); padding: 18px 4px; }
    details { border: 1px solid var(--vy-border); border-radius: 12px; padding: 12px; margin: 12px 0; background: rgba(6, 7, 10, .46); }
    summary { cursor: pointer; color: var(--vy-tech-strong); }
    .tag { color: var(--vy-bg); background: var(--vy-tech); padding: 2px 7px; border-radius: 999px; margin-right: 8px; }
    .interactions .tag { background: var(--vy-violet); color: var(--vy-text-primary); }
    .error { color: var(--vy-red); }
    .ok { color: var(--vy-success); }
    .muted { color: var(--vy-text-dim); }
    .token-line { margin: 10px 0 0; color: var(--vy-text-muted); line-height: 1.6; }
    .token-line strong { color: var(--vy-tech-strong); }
    .source-provider { color: var(--vy-success); }
    .source-estimate { color: var(--vy-amber); }
    .messages { display: grid; gap: 8px; margin: 12px 0; }
    .message { border-left: 3px solid var(--vy-border-strong); background: var(--vy-surface); padding: 10px; }
    .message.user { border-left-color: var(--vy-violet); }
    .message.assistant { border-left-color: var(--vy-tech); }
    .message.system, .message.tool { border-left-color: var(--vy-border-strong); }
    .message-head { display: flex; justify-content: space-between; gap: 8px; color: var(--vy-text-muted); margin-bottom: 7px; }
    .message-body { color: var(--vy-text-primary); white-space: pre-wrap; overflow-wrap: anywhere; max-height: 260px; overflow: auto; }
    pre {
      overflow: auto;
      max-height: 520px;
      background: #06070A;
      color: var(--vy-text-primary);
      border: 1px solid var(--vy-border);
      border-radius: 10px;
      padding: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    button {
      background: var(--vy-violet);
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 7px 10px;
      margin: 8px 8px 0 0;
      font: inherit;
      cursor: pointer;
    }
    .path-list { display: grid; gap: 8px; }
    .path-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--vy-border);
      border-radius: 10px;
      padding: 10px;
      background: rgba(6, 7, 10, .55);
    }
    .path-row code { color: var(--vy-tech-strong); overflow-wrap: anywhere; }
    @media (max-width: 860px) { .lanes { grid-template-columns: 1fr; } .lane > h2 { position: static; } }
  </style>
</head>
<body>
  <main>
    <h1>vyrn debug trace viewer</h1>
    <p>Load a session trace to inspect human/agent interactions beside harness compaction and scratchpad work. Provider counts and estimates stay visibly separate. Data stays in this browser tab.</p>
    <section class="panel drop" id="drop">
      Drop a trace JSON file here, or choose one:
      <input id="file" type="file" accept="application/json,.json">
    </section>
    <section class="panel">
      <h2>default trace locations</h2>
      <p>Browsers do not allow this static page to force the file picker to start in a specific directory. Use these generated paths as the default places to choose from.</p>
      <div id="hints" class="path-list"></div>
    </section>
    <section id="content"></section>
  </main>
  <script>
    const TRACE_HINTS = __VYRN_TRACE_HINTS__;
    const EMBEDDED_TRACE = __VYRN_EMBEDDED_TRACE__;
    const EMBEDDED_TRACE_NAME = __VYRN_EMBEDDED_TRACE_NAME__;
    const file = document.getElementById('file');
    const drop = document.getElementById('drop');
    const content = document.getElementById('content');
    const hints = document.getElementById('hints');
    file.addEventListener('change', () => file.files[0] && load(file.files[0]));
    drop.addEventListener('dragover', event => { event.preventDefault(); drop.style.borderColor = 'var(--vy-tech)'; });
    drop.addEventListener('dragleave', () => drop.style.borderColor = 'var(--vy-border-strong)');
    drop.addEventListener('drop', event => {
      event.preventDefault();
      drop.style.borderColor = 'var(--vy-border-strong)';
      if (event.dataTransfer.files[0]) load(event.dataTransfer.files[0]);
    });
    renderHints();
    if (EMBEDDED_TRACE) render(EMBEDDED_TRACE, EMBEDDED_TRACE_NAME);

    function renderHints() {
      const rows = [
        ['interactive sessions', TRACE_HINTS.session_dir],
        ['eval runs', TRACE_HINTS.eval_dir],
        ...TRACE_HINTS.recent_files.map((path, index) => [`recent ${index + 1}`, path])
      ];
      hints.innerHTML = rows.map(([label, path]) => `
        <div class="path-row">
          <div><span class="muted">${escapeHtml(label)}</span><br><code>${escapeHtml(path)}</code></div>
          <button data-copy="${escapeAttr(path)}">copy path</button>
        </div>
      `).join('');
      hints.querySelectorAll('button[data-copy]').forEach(button => {
        button.addEventListener('click', () => navigator.clipboard.writeText(button.dataset.copy));
      });
    }

    async function load(blob) {
      try {
        render(JSON.parse(await blob.text()), blob.name);
      } catch (error) {
        content.innerHTML = `<section class="panel error">Failed to load JSON: ${escapeHtml(error.message)}</section>`;
      }
    }

    function render(trace, name) {
      const calls = trace.calls || [];
      content.innerHTML = `
        <section class="panel">
          <h2>${escapeHtml(name)}</h2>
          <div class="summary">
            ${metric('kind', trace.run_kind || 'unknown')}
            ${metric('session', trace.session_id || trace.eval_case_id || 'unknown')}
            ${metric('model', `${trace.model_name || ''} ${trace.model || ''}`.trim())}
            ${metric('calls', calls.length)}
            ${metric('provider tokens', providerTotal(trace))}
            ${metric('estimated fallback', estimateFallbackTotal(trace))}
            ${metric('started', formatTime(trace.started_at_ms))}
            ${metric('ended', formatTime(trace.ended_at_ms))}
          </div>
          <input id="search" type="search" placeholder="filter by action, text, tool, call id">
        </section>
        <section class="lanes">
          <div class="lane interactions"><h2>human + agent</h2><div id="interaction-calls"></div></div>
          <div class="lane harness"><h2>harness internals</h2><div id="harness-calls"></div></div>
        </section>
      `;
      const search = document.getElementById('search');
      search.addEventListener('input', () => renderCalls(calls, search.value));
      renderCalls(calls, '');
    }

    function renderCalls(calls, query) {
      const needle = query.trim().toLowerCase();
      const filtered = needle ? calls.filter(call => JSON.stringify(call).toLowerCase().includes(needle)) : calls;
      const interactions = filtered.filter(call => (call.action_scope || scopeFor(call)) === 'interaction');
      const harness = filtered.filter(call => (call.action_scope || scopeFor(call)) !== 'interaction');
      renderLane(document.getElementById('interaction-calls'), interactions);
      renderLane(document.getElementById('harness-calls'), harness);
      document.querySelectorAll('.lanes button[data-copy]').forEach(button => {
        button.addEventListener('click', () => navigator.clipboard.writeText(button.dataset.copy));
      });
    }

    function renderLane(target, calls) {
      target.innerHTML = calls.length
        ? calls.map(call => callPanel(call)).join('')
        : '<div class="lane-empty">no matching actions</div>';
    }

    function callPanel(call) {
      const title = `${call.call_id || ''} ${call.action_type || ''}`.trim();
      const state = call.error ? '<span class="error">error</span>' : '<span class="ok">ok</span>';
      return `<details class="panel">
        <summary><span class="tag">${escapeHtml(call.action_type || 'call')}</span>${escapeHtml(title)} ${state} <span class="muted">${call.duration_ms || 0}ms</span></summary>
        ${tokenLine(call)}
        ${contextLine(call)}
        ${messageTimeline(call)}
        ${jsonBlock('metadata', metadata(call))}
        ${jsonBlock('request', call.request)}
        ${jsonBlock('response', call.response)}
        ${call.error ? jsonBlock('error', call.error) : ''}
      </details>`;
    }

    function metadata(call) {
      const { request, response, error, ...rest } = call;
      return rest;
    }

    function tokenLine(call) {
      const accounting = normalizedAccounting(call);
      const effective = accounting.effective;
      const provider = accounting.provider;
      const estimate = accounting.estimate;
      const promptSource = sourceBadge(effective.prompt_source);
      const completionSource = sourceBadge(effective.completion_source);
      const providerText = provider
        ? `provider ${number(provider.prompt_tokens)} in / ${number(provider.completion_tokens)} out / ${number(provider.total_tokens || provider.prompt_tokens + provider.completion_tokens)} total`
        : 'provider usage not returned';
      return `<div class="token-line"><strong>effective</strong> ${number(effective.prompt_tokens)} in ${promptSource} · ${number(effective.completion_tokens)} out ${completionSource} · ${number(effective.total_tokens)} total<br><span class="muted">${providerText} · estimate ${number(estimate.prompt_tokens)} in / ${number(estimate.completion_tokens)} out</span></div>`;
    }

    function contextLine(call) {
      const context = normalizedAccounting(call).context;
      if (!context) return '';
      return `<div class="token-line"><strong>context</strong> ${number(context.used_input_tokens)}/${number(context.limit_tokens)} · ${number(context.available_input_tokens)} available</div>`;
    }

    function messageTimeline(call) {
      const messages = call.request?.messages || [];
      const estimates = call.message_token_estimates || [];
      const rows = messages.map((message, index) => {
        const estimate = estimates.find(item => item.index === index)?.estimated_tokens;
        return messageCard(message.role || 'unknown', messageText(message), estimate);
      });
      const responseMessage = call.response?.choices?.[0]?.message;
      if (responseMessage) {
        rows.push(messageCard('assistant', messageText(responseMessage), normalizedAccounting(call).estimate.completion_tokens));
      }
      return rows.length ? `<div class="messages">${rows.join('')}</div>` : '';
    }

    function messageCard(role, body, estimatedTokens) {
      return `<div class="message ${escapeAttr(role)}"><div class="message-head"><strong>${escapeHtml(role)}</strong><span class="source-estimate">${number(estimatedTokens || 0)} tokens est.</span></div><div class="message-body">${escapeHtml(body)}</div></div>`;
    }

    function messageText(message) {
      const content = message?.content;
      let text = typeof content === 'string' ? content : Array.isArray(content)
        ? content.map(part => part.type === 'text' ? part.text : `[${part.type || 'attachment'}]`).join('\n')
        : '';
      const calls = message?.tool_calls || [];
      if (calls.length) text += `${text ? '\n\n' : ''}${calls.map(call => `tool ${call.function?.name || 'unknown'}(${call.function?.arguments || ''})`).join('\n')}`;
      return text || '(empty message)';
    }

    function normalizedAccounting(call) {
      if (call.token_accounting) return call.token_accounting;
      const provider = call.usage || null;
      const estimate = {
        prompt_tokens: call.estimated_input_tokens || 0,
        completion_tokens: call.estimated_output_tokens || 0,
        total_tokens: (call.estimated_input_tokens || 0) + (call.estimated_output_tokens || 0)
      };
      const effective = {
        prompt_tokens: provider?.prompt_tokens || estimate.prompt_tokens,
        prompt_source: provider?.prompt_tokens ? 'provider' : 'estimate',
        completion_tokens: provider?.completion_tokens || estimate.completion_tokens,
        completion_source: provider?.completion_tokens ? 'provider' : 'estimate'
      };
      effective.total_tokens = effective.prompt_tokens + effective.completion_tokens;
      return { provider, estimate, effective, context: call.context_limit ? {
        limit_tokens: call.context_limit,
        used_input_tokens: effective.prompt_tokens,
        available_input_tokens: Math.max(0, call.context_limit - effective.prompt_tokens)
      } : null };
    }

    function sourceBadge(source) {
      const value = source || 'estimate';
      return `<span class="source-${escapeAttr(value)}">(${escapeHtml(value)})</span>`;
    }

    function providerTotal(trace) {
      const totals = trace.token_totals || {};
      const recorded = (totals.provider_prompt_tokens || 0) + (totals.provider_completion_tokens || 0);
      const legacy = (trace.calls || []).reduce((sum, call) => sum + (call.usage?.prompt_tokens || 0) + (call.usage?.completion_tokens || 0), 0);
      return number(recorded || legacy);
    }

    function estimateFallbackTotal(trace) {
      const totals = trace.token_totals || {};
      const recorded = (totals.estimated_fallback_prompt_tokens || 0) + (totals.estimated_fallback_completion_tokens || 0);
      const legacy = (trace.calls || []).reduce((sum, call) => {
        const prompt = call.usage?.prompt_tokens ? 0 : (call.estimated_input_tokens || 0);
        const completion = call.usage?.completion_tokens ? 0 : (call.estimated_output_tokens || 0);
        return sum + prompt + completion;
      }, 0);
      return number(recorded || legacy);
    }

    function scopeFor(call) { return call.action_type === 'agent_turn' ? 'interaction' : 'harness'; }
    function number(value) { return Number(value || 0).toLocaleString(); }

    function jsonBlock(label, value) {
      const text = JSON.stringify(value ?? null, null, 2);
      return `<details><summary>${label}</summary><button data-copy="${escapeAttr(text)}">copy ${label}</button><pre>${escapeHtml(text)}</pre></details>`;
    }

    function metric(label, value) {
      return `<div class="metric"><b>${escapeHtml(label)}</b>${escapeHtml(String(value || '-'))}</div>`;
    }

    function formatTime(ms) {
      return ms ? new Date(ms).toLocaleString() : '-';
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
    }

    function escapeAttr(value) {
      return escapeHtml(value).replace(/`/g, '&#96;');
    }
  </script>
</body>
</html>
"##;

#[derive(Debug, Clone, Serialize)]
struct TraceFile {
    schema_version: u32,
    run_kind: String,
    session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_case_id: Option<String>,
    started_at_ms: u128,
    #[serde(skip_serializing_if = "Option::is_none")]
    ended_at_ms: Option<u128>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_reason: Option<String>,
    model_name: String,
    model: String,
    base_url: String,
    token_totals: TraceTokenTotals,
    calls: Vec<TraceCall>,
}

#[derive(Debug, Clone, Default, Serialize)]
struct TraceTokenTotals {
    provider_prompt_tokens: usize,
    provider_completion_tokens: usize,
    estimated_fallback_prompt_tokens: usize,
    estimated_fallback_completion_tokens: usize,
    effective_total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceCall {
    call_id: String,
    action_type: String,
    action_scope: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    turn_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    round_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_index: Option<usize>,
    started_at_ms: u128,
    ended_at_ms: u128,
    duration_ms: u128,
    model_name: String,
    model: String,
    base_url: String,
    endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    estimated_input_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    estimated_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    token_breakdown: Option<TokenBreakdown>,
    token_accounting: TraceTokenAccounting,
    message_token_estimates: Vec<TraceMessageEstimate>,
    request: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<TraceError>,
}

#[derive(Debug, Clone, Serialize)]
struct TraceTokenAccounting {
    #[serde(skip_serializing_if = "Option::is_none")]
    provider: Option<Usage>,
    estimate: TraceUsageCounts,
    effective: TraceEffectiveUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<TraceContextUsage>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct TraceUsageCounts {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct TraceEffectiveUsage {
    prompt_tokens: usize,
    prompt_source: TokenSource,
    completion_tokens: usize,
    completion_source: TokenSource,
    total_tokens: usize,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct TraceContextUsage {
    limit_tokens: usize,
    used_input_tokens: usize,
    available_input_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct TraceMessageEstimate {
    index: usize,
    role: String,
    estimated_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
struct TraceError {
    kind: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TraceMetadata {
    pub action_type: &'static str,
    pub label: Option<String>,
    pub turn_index: Option<usize>,
    pub round_index: Option<usize>,
    pub tool_index: Option<usize>,
    pub estimated_input_tokens: Option<usize>,
    pub estimated_output_tokens: Option<usize>,
    pub context_limit: Option<usize>,
    pub token_breakdown: Option<TokenBreakdown>,
}

pub struct PendingTraceCall {
    call_id: String,
    metadata: TraceMetadata,
    started_at_ms: u128,
    started: Instant,
    model_name: String,
    model: String,
    base_url: String,
    endpoint: String,
    request: Value,
    message_token_estimates: Vec<TraceMessageEstimate>,
}

#[derive(Debug, Clone)]
pub struct TraceRecorder {
    path: PathBuf,
    file: TraceFile,
}

impl TraceRecorder {
    pub fn interactive(sources: &ConfigSources, client: &OpenAiClient) -> anyhow::Result<Self> {
        Self::session(sources, client, "interactive")
    }

    pub fn programmatic(sources: &ConfigSources, client: &OpenAiClient) -> anyhow::Result<Self> {
        Self::session(sources, client, "prompt")
    }

    fn session(
        sources: &ConfigSources,
        client: &OpenAiClient,
        run_kind: &'static str,
    ) -> anyhow::Result<Self> {
        let session_id = unix_timestamp_millis().to_string();
        let path = sources
            .project_vyrn
            .join("debug")
            .join("sessions")
            .join(format!("{session_id}.json"));
        Self::new(path, run_kind, session_id, None, client)
    }

    pub fn eval_case(
        path: PathBuf,
        case_id: impl Into<String>,
        client: &OpenAiClient,
    ) -> anyhow::Result<Self> {
        let case_id = case_id.into();
        Self::new(path, "eval", case_id.clone(), Some(case_id), client)
    }

    fn new(
        path: PathBuf,
        run_kind: impl Into<String>,
        session_id: String,
        eval_case_id: Option<String>,
        client: &OpenAiClient,
    ) -> anyhow::Result<Self> {
        let profile = client.profile();
        let mut recorder = Self {
            path,
            file: TraceFile {
                schema_version: TRACE_SCHEMA_VERSION,
                run_kind: run_kind.into(),
                session_id,
                eval_case_id,
                started_at_ms: unix_timestamp_millis(),
                ended_at_ms: None,
                end_reason: None,
                model_name: profile.name.clone(),
                model: profile.model.clone(),
                base_url: profile.base_url.clone(),
                token_totals: TraceTokenTotals::default(),
                calls: Vec::new(),
            },
        };
        recorder.persist()?;
        Ok(recorder)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn begin_call(
        &self,
        client: &OpenAiClient,
        request: &ChatCompletionRequest,
        stream: bool,
        metadata: TraceMetadata,
    ) -> PendingTraceCall {
        let profile = client.profile();
        let mut finalized = request.clone();
        finalized.model = profile.model.clone();
        finalized.stream = stream;
        let message_token_estimates = finalized
            .messages
            .iter()
            .enumerate()
            .map(|(index, message)| TraceMessageEstimate {
                index,
                role: message.role.clone(),
                estimated_tokens: estimate_messages_breakdown(std::slice::from_ref(message))
                    .total(),
            })
            .collect();
        PendingTraceCall {
            call_id: format!("call-{}", self.file.calls.len() + 1),
            metadata,
            started_at_ms: unix_timestamp_millis(),
            started: Instant::now(),
            model_name: profile.name.clone(),
            model: profile.model.clone(),
            base_url: profile.base_url.clone(),
            endpoint: chat_completions_url(&profile.base_url),
            request: serde_json::to_value(finalized).unwrap_or(Value::Null),
            message_token_estimates,
        }
    }

    pub fn finish_call(
        &mut self,
        pending: PendingTraceCall,
        result: &Result<ChatCompletionResponse, LlmError>,
    ) -> anyhow::Result<()> {
        let ended_at_ms = unix_timestamp_millis();
        let (response, usage, estimated_response_tokens, error) = match result {
            Ok(response) => {
                let estimated_response_tokens = response
                    .choices
                    .first()
                    .map(|choice| estimate_assistant_output_tokens(&choice.message));
                (
                    Some(serde_json::to_value(response).unwrap_or(Value::Null)),
                    response.usage,
                    estimated_response_tokens,
                    None,
                )
            }
            Err(error) => (None, None, None, Some(trace_error(error))),
        };
        let metadata = pending.metadata;
        let estimated_input_tokens = metadata.estimated_input_tokens.unwrap_or_default();
        let estimated_output_tokens = metadata
            .estimated_output_tokens
            .or(estimated_response_tokens)
            .unwrap_or_default();
        let provider_input = usage
            .map(|usage| usage.prompt_tokens)
            .filter(|tokens| *tokens > 0);
        let provider_output = usage
            .map(|usage| usage.completion_tokens)
            .filter(|tokens| *tokens > 0);
        let effective = TraceEffectiveUsage {
            prompt_tokens: provider_input.unwrap_or(estimated_input_tokens),
            prompt_source: if provider_input.is_some() {
                TokenSource::Provider
            } else {
                TokenSource::Estimate
            },
            completion_tokens: provider_output.unwrap_or(estimated_output_tokens),
            completion_source: if provider_output.is_some() {
                TokenSource::Provider
            } else {
                TokenSource::Estimate
            },
            total_tokens: provider_input
                .unwrap_or(estimated_input_tokens)
                .saturating_add(provider_output.unwrap_or(estimated_output_tokens)),
        };
        let token_accounting = TraceTokenAccounting {
            provider: usage,
            estimate: TraceUsageCounts {
                prompt_tokens: estimated_input_tokens,
                completion_tokens: estimated_output_tokens,
                total_tokens: estimated_input_tokens.saturating_add(estimated_output_tokens),
            },
            effective,
            context: metadata
                .context_limit
                .map(|limit_tokens| TraceContextUsage {
                    limit_tokens,
                    used_input_tokens: effective.prompt_tokens,
                    available_input_tokens: limit_tokens.saturating_sub(effective.prompt_tokens),
                }),
        };
        if effective.prompt_source == TokenSource::Provider {
            self.file.token_totals.provider_prompt_tokens += effective.prompt_tokens;
        } else {
            self.file.token_totals.estimated_fallback_prompt_tokens += effective.prompt_tokens;
        }
        if effective.completion_source == TokenSource::Provider {
            self.file.token_totals.provider_completion_tokens += effective.completion_tokens;
        } else {
            self.file.token_totals.estimated_fallback_completion_tokens +=
                effective.completion_tokens;
        }
        self.file.token_totals.effective_total_tokens += effective.total_tokens;
        self.file.calls.push(TraceCall {
            call_id: pending.call_id,
            action_type: metadata.action_type.to_string(),
            action_scope: action_scope(metadata.action_type).to_string(),
            label: metadata.label,
            turn_index: metadata.turn_index,
            round_index: metadata.round_index,
            tool_index: metadata.tool_index,
            started_at_ms: pending.started_at_ms,
            ended_at_ms,
            duration_ms: pending.started.elapsed().as_millis(),
            model_name: pending.model_name,
            model: pending.model,
            base_url: pending.base_url,
            endpoint: pending.endpoint,
            estimated_input_tokens: metadata.estimated_input_tokens,
            estimated_output_tokens: Some(estimated_output_tokens),
            context_limit: metadata.context_limit,
            token_breakdown: metadata.token_breakdown,
            token_accounting,
            message_token_estimates: pending.message_token_estimates,
            request: pending.request,
            response,
            usage,
            error,
        });
        self.persist()
    }

    pub fn finish(&mut self, reason: impl Into<String>) -> anyhow::Result<()> {
        self.file.ended_at_ms = Some(unix_timestamp_millis());
        self.file.end_reason = Some(reason.into());
        self.persist()
    }

    fn persist(&mut self) -> anyhow::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let raw = serde_json::to_string_pretty(&self.file)?;
        std::fs::write(&self.path, raw)
            .with_context(|| format!("failed to write {}", self.path.display()))
    }
}

pub fn write_viewer(sources: &ConfigSources) -> anyhow::Result<PathBuf> {
    write_viewer_for_trace(sources, None)
}

pub fn write_viewer_for_trace(
    sources: &ConfigSources,
    trace_path: Option<&Path>,
) -> anyhow::Result<PathBuf> {
    let path = sources.project_vyrn.join("debug").join("viewer.html");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::create_dir_all(sources.project_vyrn.join("debug").join("sessions"))
        .with_context(|| "failed to create debug sessions directory".to_string())?;
    std::fs::create_dir_all(sources.project_vyrn.join("eval-runs"))
        .with_context(|| "failed to create eval runs directory".to_string())?;
    let hints = trace_hints(sources);
    let hints_json = serde_json::to_string(&hints)?;
    let (embedded_trace, embedded_name) = if let Some(trace_path) = trace_path {
        let raw = std::fs::read_to_string(trace_path)
            .with_context(|| format!("failed to read {}", trace_path.display()))?;
        let trace: Value = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse {} as JSON", trace_path.display()))?;
        (
            script_safe_json(&trace)?,
            serde_json::to_string(
                &trace_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("embedded trace"),
            )?,
        )
    } else {
        ("null".to_string(), "\"embedded trace\"".to_string())
    };
    let html = VIEWER_HTML
        .replace("__VYRN_TRACE_HINTS__", &hints_json)
        .replace("__VYRN_EMBEDDED_TRACE__", &embedded_trace)
        .replace("__VYRN_EMBEDDED_TRACE_NAME__", &embedded_name);
    std::fs::write(&path, html).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

fn script_safe_json(value: &Value) -> anyhow::Result<String> {
    Ok(serde_json::to_string(value)?
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026"))
}

#[derive(Debug, Serialize)]
struct TraceHints {
    session_dir: String,
    eval_dir: String,
    recent_files: Vec<String>,
}

fn trace_hints(sources: &ConfigSources) -> TraceHints {
    let session_dir = sources.project_vyrn.join("debug").join("sessions");
    let eval_dir = sources.project_vyrn.join("eval-runs");
    let mut recent_files = Vec::new();
    collect_json_files(&session_dir, 1, &mut recent_files);
    collect_named_files(&eval_dir, "llm-trace.json", 6, &mut recent_files);
    recent_files.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    TraceHints {
        session_dir: session_dir.display().to_string(),
        eval_dir: eval_dir.display().to_string(),
        recent_files: recent_files
            .into_iter()
            .take(8)
            .map(|(_, path)| path.display().to_string())
            .collect(),
    }
}

fn collect_json_files(dir: &Path, depth: usize, files: &mut Vec<(u128, PathBuf)>) {
    collect_files_matching(dir, depth, files, &|path| {
        path.extension().and_then(|ext| ext.to_str()) == Some("json")
    });
}

fn collect_named_files(dir: &Path, name: &str, depth: usize, files: &mut Vec<(u128, PathBuf)>) {
    collect_files_matching(dir, depth, files, &|path| {
        path.file_name().and_then(|file_name| file_name.to_str()) == Some(name)
    });
}

fn collect_files_matching(
    dir: &Path,
    depth: usize,
    files: &mut Vec<(u128, PathBuf)>,
    matches: &dyn Fn(&Path) -> bool,
) {
    if depth == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_matching(&path, depth.saturating_sub(1), files, matches);
        } else if matches(&path) {
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis())
                .unwrap_or_default();
            files.push((modified, path));
        }
    }
}

fn trace_error(error: &LlmError) -> TraceError {
    match error {
        LlmError::Request { url, source } => TraceError {
            kind: "request".to_string(),
            message: source.to_string(),
            url: Some(url.clone()),
            status: None,
            body: None,
        },
        LlmError::HttpStatus { url, status, body } => TraceError {
            kind: "http_status".to_string(),
            message: error.to_string(),
            url: Some(url.clone()),
            status: Some(status.as_u16()),
            body: Some(body.clone()),
        },
        LlmError::ParseStream(message) => TraceError {
            kind: "parse_stream".to_string(),
            message: message.clone(),
            url: None,
            status: None,
            body: None,
        },
        LlmError::Input(message) => TraceError {
            kind: "input".to_string(),
            message: message.clone(),
            url: None,
            status: None,
            body: None,
        },
        LlmError::MissingChoice => TraceError {
            kind: "missing_choice".to_string(),
            message: error.to_string(),
            url: None,
            status: None,
            body: None,
        },
        LlmError::ToolRoundLimit { .. } => TraceError {
            kind: "tool_round_limit".to_string(),
            message: error.to_string(),
            url: None,
            status: None,
            body: None,
        },
        LlmError::Canceled => TraceError {
            kind: "canceled".to_string(),
            message: error.to_string(),
            url: None,
            status: None,
            body: None,
        },
    }
}

fn action_scope(action_type: &str) -> &'static str {
    if action_type == "agent_turn" {
        "interaction"
    } else {
        "harness"
    }
}

fn chat_completions_url(base_url: &str) -> String {
    format!("{}/chat/completions", base_url.trim_end_matches('/'))
}

fn unix_timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelProfile;
    use crate::llm::ChatMessage;
    use crate::llm::types::{ChatChoice, Usage};

    fn client() -> OpenAiClient {
        OpenAiClient::new(ModelProfile {
            name: "test".to_string(),
            base_url: "http://127.0.0.1:9999/v1".to_string(),
            model: "fake".to_string(),
            api_key: "secret".to_string(),
        })
    }

    #[test]
    fn trace_records_finalized_request_without_headers() {
        let temp = tempfile::tempdir().unwrap();
        let client = client();
        let mut recorder = TraceRecorder::new(
            temp.path().join("trace.json"),
            "eval",
            "case".to_string(),
            None,
            &client,
        )
        .unwrap();
        let request = ChatCompletionRequest {
            model: String::new(),
            messages: vec![ChatMessage::user("hello")],
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            stream_options: None,
            max_tokens: None,
        };

        let pending = recorder.begin_call(
            &client,
            &request,
            true,
            TraceMetadata {
                action_type: "agent_turn",
                estimated_input_tokens: Some(12),
                ..TraceMetadata::default()
            },
        );
        let result = Ok(ChatCompletionResponse {
            choices: vec![ChatChoice {
                message: ChatMessage::assistant("hi"),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(Usage {
                prompt_tokens: 8,
                completion_tokens: 2,
                total_tokens: 10,
            }),
        });
        recorder.finish_call(pending, &result).unwrap();

        let raw = std::fs::read_to_string(temp.path().join("trace.json")).unwrap();
        assert!(raw.contains(r#""model": "fake""#), "{raw}");
        assert!(raw.contains(r#""stream": true"#), "{raw}");
        assert!(raw.contains(r#""action_type": "agent_turn""#), "{raw}");
        assert!(!raw.contains("secret"), "{raw}");
        let trace: Value = serde_json::from_str(&raw).unwrap();
        let call = &trace["calls"][0];
        assert_eq!(trace["schema_version"], 2);
        assert_eq!(call["action_scope"], "interaction");
        assert_eq!(call["token_accounting"]["provider"]["total_tokens"], 10);
        assert_eq!(call["token_accounting"]["estimate"]["prompt_tokens"], 12);
        assert_eq!(
            call["token_accounting"]["effective"]["prompt_source"],
            "provider"
        );
        assert_eq!(call["message_token_estimates"][0]["role"], "user");
    }
}
