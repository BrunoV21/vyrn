use crate::agent::tokens::{TokenBreakdown, estimate_assistant_output_tokens};
use crate::config::ConfigSources;
use crate::llm::types::{ChatCompletionResponse, Usage};
use crate::llm::{ChatCompletionRequest, LlmError, OpenAiClient};
use anyhow::Context as _;
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TRACE_SCHEMA_VERSION: u32 = 1;
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
    details { border: 1px solid var(--vy-border); border-radius: 12px; padding: 12px; margin: 12px 0; background: rgba(6, 7, 10, .46); }
    summary { cursor: pointer; color: var(--vy-tech-strong); }
    .tag { color: var(--vy-bg); background: var(--vy-tech); padding: 2px 7px; border-radius: 999px; margin-right: 8px; }
    .error { color: var(--vy-red); }
    .ok { color: var(--vy-success); }
    .muted { color: var(--vy-text-dim); }
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
  </style>
</head>
<body>
  <main>
    <h1>vyrn debug trace viewer</h1>
    <p>Load a <code>llm-trace.json</code> or session trace from <code>.vyrn/debug/sessions/</code>. Data stays in this browser tab.</p>
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
            ${metric('started', formatTime(trace.started_at_ms))}
            ${metric('ended', formatTime(trace.ended_at_ms))}
          </div>
          <input id="search" type="search" placeholder="filter by action, text, tool, call id">
        </section>
        <section id="calls"></section>
      `;
      const search = document.getElementById('search');
      search.addEventListener('input', () => renderCalls(calls, search.value));
      renderCalls(calls, '');
    }

    function renderCalls(calls, query) {
      const target = document.getElementById('calls');
      const needle = query.trim().toLowerCase();
      const filtered = needle ? calls.filter(call => JSON.stringify(call).toLowerCase().includes(needle)) : calls;
      target.innerHTML = filtered.map(call => callPanel(call)).join('');
      target.querySelectorAll('button[data-copy]').forEach(button => {
        button.addEventListener('click', () => navigator.clipboard.writeText(button.dataset.copy));
      });
    }

    function callPanel(call) {
      const title = `${call.call_id || ''} ${call.action_type || ''}`.trim();
      const state = call.error ? '<span class="error">error</span>' : '<span class="ok">ok</span>';
      return `<details class="panel">
        <summary><span class="tag">${escapeHtml(call.action_type || 'call')}</span>${escapeHtml(title)} ${state} <span class="muted">${call.duration_ms || 0}ms</span></summary>
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
    calls: Vec<TraceCall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceCall {
    call_id: String,
    action_type: String,
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
    request: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    response: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<TraceError>,
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
}

#[derive(Debug, Clone)]
pub struct TraceRecorder {
    path: PathBuf,
    file: TraceFile,
}

impl TraceRecorder {
    pub fn interactive(sources: &ConfigSources, client: &OpenAiClient) -> anyhow::Result<Self> {
        let session_id = unix_timestamp_millis().to_string();
        let path = sources
            .project_vyrn
            .join("debug")
            .join("sessions")
            .join(format!("{session_id}.json"));
        Self::new(path, "interactive", session_id, None, client)
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
        }
    }

    pub fn finish_call(
        &mut self,
        pending: PendingTraceCall,
        result: &Result<ChatCompletionResponse, LlmError>,
    ) -> anyhow::Result<()> {
        let ended_at_ms = unix_timestamp_millis();
        let (response, usage, derived_output_tokens, error) = match result {
            Ok(response) => {
                let derived_output_tokens = response
                    .usage
                    .map(|usage| usage.completion_tokens)
                    .filter(|tokens| *tokens > 0)
                    .or_else(|| {
                        response
                            .choices
                            .first()
                            .map(|choice| estimate_assistant_output_tokens(&choice.message))
                    });
                (
                    Some(serde_json::to_value(response).unwrap_or(Value::Null)),
                    response.usage,
                    derived_output_tokens,
                    None,
                )
            }
            Err(error) => (None, None, None, Some(trace_error(error))),
        };
        let metadata = pending.metadata;
        let estimated_output_tokens = metadata.estimated_output_tokens.or(derived_output_tokens);
        self.file.calls.push(TraceCall {
            call_id: pending.call_id,
            action_type: metadata.action_type.to_string(),
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
            estimated_output_tokens,
            context_limit: metadata.context_limit,
            token_breakdown: metadata.token_breakdown,
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
    let html = VIEWER_HTML.replace("__VYRN_TRACE_HINTS__", &hints_json);
    std::fs::write(&path, html).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
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
    recent_files.sort_by(|left, right| right.0.cmp(&left.0));
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
    }
}
