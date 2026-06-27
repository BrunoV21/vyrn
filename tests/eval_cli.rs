use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::tempdir;

#[test]
fn eval_runs_json_suite_and_writes_traces() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for _ in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            if body.contains("Current scratchpad")
                || body.contains("New consumed tool batch and assistant response")
            {
                write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- read_file saw fixture.txt: hello from eval."}}],"usage":{"prompt_tokens":80,"completion_tokens":16,"total_tokens":96}}"#,
                );
            } else if body.contains("[turn scratchpad]") {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"fixture.txt says hello from eval."}}]}"#,
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_read","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"fixture.txt\"}"}}]}}]}"#,
                );
            }
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::create_dir_all(temp.path().join("evals")).unwrap();
    std::fs::write(temp.path().join("fixture.txt"), "hello from eval").unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.llama3]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#
        ),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("evals/basic.json"),
        r#"{
  "name": "basic",
  "default_model": "llama3",
  "cases": [
    {
      "id": "read-fixture",
      "prompt": "read fixture.txt",
      "assertions": [
        { "type": "assistant_contains", "value": "hello from eval" },
        { "type": "tool_called", "name": "read_file" },
        { "type": "file_contains", "path": "fixture.txt", "value": "hello from eval" }
      ]
    }
  ]
}
"#,
    )
    .unwrap();

    let output_dir = temp.path().join(".vyrn/eval-runs/test");
    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .arg("eval")
        .arg("evals/basic.json")
        .arg("--output")
        .arg(&output_dir)
        .arg("--json")
        .output()
        .unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("read-fixture"), "{stdout}");
    assert!(stdout.contains("\"passed\": 1"), "{stdout}");
    let trace = std::fs::read_to_string(output_dir.join("read-fixture/trace.json")).unwrap();
    assert!(trace.contains("\"name\": \"read_file\""), "{trace}");
    assert!(trace.contains("hello from eval"), "{trace}");
    let transcript =
        std::fs::read_to_string(output_dir.join("read-fixture/transcript.md")).unwrap();
    assert!(transcript.contains("Final Assistant"), "{transcript}");
    let debug_log = std::fs::read_to_string(output_dir.join("read-fixture/debug.log")).unwrap();
    assert!(debug_log.contains("eval_case_start"), "{debug_log}");
    assert!(debug_log.contains("agent_request round=0"), "{debug_log}");
}

#[test]
fn eval_exits_nonzero_when_assertion_fails() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _body = read_http_body(&mut stream);
        write_sse(
            &mut stream,
            r#"data: {"choices":[{"delta":{"content":"wrong text"}}]}"#,
        );
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::create_dir_all(temp.path().join("evals")).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.llama3]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#
        ),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("evals/fail.json"),
        r#"{
  "name": "fail",
  "cases": [
    {
      "id": "bad-output",
      "prompt": "say anything",
      "assertions": [
        { "type": "assistant_contains", "value": "expected" }
      ]
    }
  ]
}
"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .arg("eval")
        .arg("evals/fail.json")
        .output()
        .unwrap();
    server.join().unwrap();

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("FAIL bad-output"), "{stdout}");
}

#[test]
fn eval_compacts_large_intermediate_tool_outputs_under_context_budget() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        let mut agent_round = 0;
        while agent_round < 3 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body.clone());
            if body.contains("Current scratchpad")
                || body.contains("New consumed tool batch and assistant response")
            {
                write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- large.txt was read and contained LARGE_CONTEXT_MARKER."}}],"usage":{"prompt_tokens":90,"completion_tokens":18,"total_tokens":108}}"#,
                );
                continue;
            }
            if agent_round < 2 {
                write_sse(
                    &mut stream,
                    &format!(
                        r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call_read_{agent_round}","type":"function","function":{{"name":"read_file","arguments":"{{\"path\":\"large.txt\"}}"}}}}]}}}}]}}"#
                    ),
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"LARGE_CONTEXT_DONE"}}]}"#,
                );
            }
            agent_round += 1;
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::create_dir_all(temp.path().join("evals")).unwrap();
    std::fs::write(
        temp.path().join("large.txt"),
        format!("LARGE_CONTEXT_MARKER\n{}", "0123456789abcdef\n".repeat(900)),
    )
    .unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.llama3]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#
        ),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("evals/context.json"),
        r#"{
  "name": "context",
  "cases": [
    {
      "id": "large-tools",
      "prompt": "Read large.txt twice, then reply LARGE_CONTEXT_DONE.",
      "max_turns": 5,
      "assertions": [
        { "type": "assistant_contains", "value": "LARGE_CONTEXT_DONE" },
        { "type": "tool_called", "name": "read_file" }
      ]
    }
  ]
}
"#,
    )
    .unwrap();

    let output_dir = temp.path().join(".vyrn/eval-runs/context");
    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .arg("--context")
        .arg("1200")
        .arg("eval")
        .arg("evals/context.json")
        .arg("--output")
        .arg(&output_dir)
        .arg("--json")
        .output()
        .unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let trace_path = output_dir.join("large-tools/trace.json");
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&trace_path).unwrap()).unwrap();
    let requests = trace["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 3, "{trace}");
    for request in requests {
        let tokens = request["estimated_input_tokens"].as_u64().unwrap();
        assert!(tokens <= 1200, "tokens={tokens}\n{request}");
        let tool_messages = request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|message| message["role"] == "tool")
            .collect::<Vec<_>>();
        assert!(tool_messages.len() <= 1, "{request}");
        for message in tool_messages {
            let content = message["content"].as_str().unwrap_or_default();
            assert!(
                content.chars().count() < 2500,
                "tool output was not compacted: {} chars",
                content.chars().count()
            );
        }
    }
    let debug_log = std::fs::read_to_string(output_dir.join("large-tools/debug.log")).unwrap();
    assert!(debug_log.contains("tool_chain_prepare"), "{debug_log}");
    assert!(debug_log.contains("before_tokens="), "{debug_log}");
    assert!(debug_log.contains("after_tokens="), "{debug_log}");
}

#[test]
fn eval_processes_many_large_tool_calls_incrementally() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        listener.set_nonblocking(true).unwrap();
        let mut agent_requests = 0;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let Ok((mut stream, _)) = listener.accept() else {
                if Instant::now() > deadline {
                    break;
                }
                thread::sleep(Duration::from_millis(20));
                continue;
            };
            stream.set_nonblocking(false).unwrap();
            let body = read_http_body(&mut stream);
            if body.is_empty() {
                continue;
            }
            server_bodies.lock().unwrap().push(body.clone());
            if body.contains("Current scratchpad")
                || body.contains("New consumed tool batch and assistant response")
            {
                write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- one large file result was processed and compacted."}}],"usage":{"prompt_tokens":90,"completion_tokens":18,"total_tokens":108}}"#,
                );
                continue;
            }
            if agent_requests == 0 {
                let event = serde_json::json!({
                    "choices": [{
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"large-a.txt\"}"
                                    }
                                },
                                {
                                    "index": 1,
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"large-b.txt\"}"
                                    }
                                },
                                {
                                    "index": 2,
                                    "id": "call_c",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"large-c.txt\"}"
                                    }
                                }
                            ]
                        }
                    }]
                });
                write_sse(
                    &mut stream,
                    &format!("data: {}", serde_json::to_string(&event).unwrap()),
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"MANY_TOOLS_DONE"}}]}"#,
                );
                break;
            }
            agent_requests += 1;
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::create_dir_all(temp.path().join("evals")).unwrap();
    for name in ["large-a.txt", "large-b.txt", "large-c.txt"] {
        std::fs::write(temp.path().join(name), "large payload ".repeat(900)).unwrap();
    }
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.llama3]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#
        ),
    )
    .unwrap();
    std::fs::write(
        temp.path().join("evals/many-tools.json"),
        r#"{
  "name": "many-tools",
  "cases": [
    {
      "id": "many-large-tools",
      "prompt": "Read three large files, then reply MANY_TOOLS_DONE.",
      "max_turns": 4,
      "assertions": [
        { "type": "assistant_contains", "value": "MANY_TOOLS_DONE" },
        { "type": "tool_called_at_least", "name": "read_file", "count": 3 }
      ]
    }
  ]
}
"#,
    )
    .unwrap();

    let output_dir = temp.path().join(".vyrn/eval-runs/many-tools");
    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .arg("--context")
        .arg("1500")
        .arg("eval")
        .arg("evals/many-tools.json")
        .arg("--output")
        .arg(&output_dir)
        .arg("--json")
        .output()
        .unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let trace_path = output_dir.join("many-large-tools/trace.json");
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&trace_path).unwrap()).unwrap();
    let requests = trace["requests"].as_array().unwrap();
    assert_eq!(requests.len(), 2, "{trace}");
    for request in requests {
        let tokens = request["estimated_input_tokens"].as_u64().unwrap();
        assert!(tokens <= 1500, "tokens={tokens}\n{request}");
    }
    let final_messages = requests[1]["messages"].as_array().unwrap();
    assert_eq!(
        final_messages
            .iter()
            .filter(|message| message["role"] == "tool")
            .count(),
        3
    );
    assert!(
        std::fs::read_to_string(output_dir.join("many-large-tools/debug.log"))
            .unwrap()
            .matches("tool_chain_prepare")
            .count()
            >= 3
    );
}

fn read_http_body(stream: &mut TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut temp = [0; 1024];
    loop {
        let read = stream.read(&mut temp).unwrap();
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&temp[..read]);
        if buffer.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }

    let header_end = buffer
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap()
        + 4;
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);

    while buffer.len() < header_end + content_length {
        let read = stream.read(&mut temp).unwrap();
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&temp[..read]);
    }

    String::from_utf8_lossy(&buffer[header_end..header_end + content_length]).to_string()
}

fn write_sse(stream: &mut TcpStream, event: &str) {
    let body = format!("{event}\n\ndata: [DONE]\n\n");
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).unwrap();
}

fn write_json(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).unwrap();
}
