use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tempfile::tempdir;
use vyrn::agent::tokens::estimate_chat_request_breakdown;
use vyrn::llm::ChatMessage;

#[test]
fn visibility_commands_work_before_the_first_turn() {
    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        r#"[models.local]
base_url = "http://127.0.0.1:9/v1"
model = "offline"
api_key = ""
"#,
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("--debug")
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"/help\n/context\n/scratchpad\n/debug\n/exit\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("/scratchpad  show the last evolving turn scratchpad"),
        "{stdout}"
    );
    assert!(stdout.contains("context (estimated): 0/4096"), "{stdout}");
    assert!(stdout.contains("available: 4096"), "{stdout}");
    assert!(stdout.contains("turn scratchpad: none"), "{stdout}");
    assert!(stdout.contains("debug trace:"), "{stdout}");
    assert!(stdout.contains(".vyrn/debug/sessions/"), "{stdout}");
}

#[test]
fn repl_runs_against_openai_compatible_streaming_server() {
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
                    r#"{"choices":[{"message":{"role":"assistant","content":"- read_file saw fixture.txt: hello from e2e."}}],"usage":{"prompt_tokens":80,"completion_tokens":16,"total_tokens":96}}"#,
                );
            } else if body.contains("[turn scratchpad]") {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"I read fixture.txt: hello from e2e."}}]}"#,
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
    std::fs::write(temp.path().join("fixture.txt"), "hello from e2e").unwrap();
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

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(b"read fixture.txt\n/scratchpad\n/exit\n")
            .unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("> using llama3 @"));
    assert!(stdout.contains("[read_file ok]"));
    assert!(stdout.contains("I read fixture.txt: hello from e2e."));
    assert!(stdout.contains("turn scratchpad ("), "{stdout}");
    assert!(stdout.contains("read_file saw fixture.txt"), "{stdout}");
    assert!(stdout.contains("turn spent:"));
}

#[test]
fn parallel_tool_calls_share_one_bounded_scratchpad_update() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body);
            match index {
                0 => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}},{"index":1,"id":"call_b","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"b.txt\"}"}}]}}]}"#,
                ),
                1 => write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- Read a.txt and b.txt in one parallel batch."}}],"usage":{"prompt_tokens":70,"completion_tokens":12,"total_tokens":82}}"#,
                ),
                _ => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"Both files were read."}}]}"#,
                ),
            }
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(temp.path().join("a.txt"), "alpha").unwrap();
    std::fs::write(temp.path().join("b.txt"), "beta").unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.local]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#
        ),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"read both files\n/exit\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = bodies.lock().unwrap();
    assert_eq!(requests.len(), 3, "{requests:#?}");
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.contains("Current scratchpad"))
            .count(),
        1,
        "{requests:#?}"
    );
    assert!(requests[1].contains("alpha"), "{}", requests[1]);
    assert!(requests[1].contains("beta"), "{}", requests[1]);
    assert!(
        requests[1].contains(r#""max_tokens":384"#),
        "{}",
        requests[1]
    );
    assert_eq!(requests[2].matches(r#""role":"tool""#).count(), 2);
}

#[test]
fn repl_plain_mode_answers_ask_user_tool_and_continues_turn() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body.clone());
            match index {
                0 => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ask","type":"function","function":{"name":"ask_user","arguments":"{\"questions\":[{\"id\":\"scope\",\"header\":\"Scope\",\"question\":\"Which approach should I use?\",\"options\":[{\"label\":\"Core tool\",\"description\":\"Always available\"},{\"label\":\"Slash command\"}]}]}"}}]}}]}"#,
                ),
                1 => write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- User chose Core tool for scope clarification."}}],"usage":{"prompt_tokens":90,"completion_tokens":16,"total_tokens":106}}"#,
                ),
                _ => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"Continuing with the core tool approach."}}]}"#,
                ),
            }
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
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

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(b"clarify before continuing\n1\n/exit\n")
            .unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[ask_user 1/1] Scope"), "{stdout}");
    assert!(
        stdout.contains("1. Core tool - Always available"),
        "{stdout}"
    );
    assert!(stdout.contains("[ask_user ok]"), "{stdout}");
    assert!(
        stdout.contains("Continuing with the core tool approach."),
        "{stdout}"
    );

    let bodies = bodies.lock().unwrap();
    assert!(bodies[0].contains("\"name\":\"ask_user\""), "{}", bodies[0]);
    assert!(
        bodies[2].contains("\"tool_call_id\":\"call_ask\""),
        "{}",
        bodies[2]
    );
    assert!(bodies[2].contains("Core tool"), "{}", bodies[2]);
}

#[test]
fn stats_command_prints_token_contributors() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for index in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let _body = read_http_body(&mut stream);
            if index == 1 {
                write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"Previous request completed."}}],"usage":{"prompt_tokens":91,"completion_tokens":17,"total_tokens":108}}"#,
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"Done."}}]}"#,
                );
            }
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn/skills/docs")).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/skills/docs/SKILL.md"),
        r#"---
name: docs
description: Write terse terminal-native docs.
---

# Instructions

Keep examples compact.
"#,
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

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin
            .write_all(
                b"say done with enough user request detail for stats accounting\ncontinue with enough user request detail for stats accounting\n/stats\n/exit\n",
            )
            .unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("session spent"), "{stdout}");
    assert!(stdout.contains("contributors:"), "{stdout}");
    assert!(stdout.contains("summary input:"), "{stdout}");
    assert!(stdout.contains("summary output:"), "{stdout}");
    assert!(stdout.contains("tools:"), "{stdout}");
    assert!(stdout.contains("skills:"), "{stdout}");
    assert!(stdout.contains("user requests:"), "{stdout}");
}

#[test]
fn debug_mode_writes_token_accounting_log() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let _body = read_http_body(&mut stream);
        write_sse(
            &mut stream,
            r#"data: {"choices":[{"delta":{"content":"Logged."}}]}"#,
        );
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
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

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("--debug")
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"say logged\n/exit\n").unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let log = std::fs::read_to_string(temp.path().join(".vyrn/debug.log")).unwrap();
    assert!(log.contains("session_start"), "{log}");
    assert!(log.contains("turn_start"), "{log}");
    assert!(log.contains("agent_request round=0"), "{log}");
    assert!(log.contains("agent_response round=0"), "{log}");
    assert!(log.contains("turn_complete"), "{log}");

    let sessions_dir = temp.path().join(".vyrn/debug/sessions");
    let trace_path = std::fs::read_dir(&sessions_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .unwrap();
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(trace_path).unwrap()).unwrap();
    assert_eq!(trace["run_kind"], "interactive");
    assert_eq!(trace["calls"][0]["action_type"], "agent_turn");
    assert_eq!(trace["calls"][0]["request"]["model"], "fake-small");
    assert_eq!(trace["calls"][0]["request"]["stream"], true);
    assert_eq!(
        trace["calls"][0]["response"]["choices"][0]["message"]["content"],
        "Logged."
    );
}

#[test]
fn prompt_mode_runs_once_and_records_provider_and_estimated_tokens() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let body = Arc::new(Mutex::new(String::new()));
    let server_body = Arc::clone(&body);
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        *server_body.lock().unwrap() = read_http_body(&mut stream);
        write_sse(
            &mut stream,
            "data: {\"choices\":[{\"delta\":{\"content\":\"One shot complete.\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":41,\"completion_tokens\":5,\"total_tokens\":46}}",
        );
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
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

    let output = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("-p")
        .arg("say one shot")
        .arg("--debug")
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::null())
        .output()
        .unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("One shot complete."), "{stdout}");
    assert!(!stdout.contains("> using"), "{stdout}");
    assert!(!stdout.contains("you:"), "{stdout}");

    let request = body.lock().unwrap();
    assert!(request.contains(r#""include_usage":true"#), "{request}");

    let trace_path = std::fs::read_dir(temp.path().join(".vyrn/debug/sessions"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .unwrap();
    let trace: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(trace_path).unwrap()).unwrap();
    let call = &trace["calls"][0];
    assert_eq!(trace["schema_version"], 2);
    assert_eq!(trace["run_kind"], "prompt");
    assert_eq!(trace["end_reason"], "prompt_complete");
    assert_eq!(call["action_scope"], "interaction");
    assert_eq!(call["token_accounting"]["provider"]["total_tokens"], 46);
    assert_eq!(
        call["token_accounting"]["effective"]["prompt_source"],
        "provider"
    );
    assert_eq!(
        call["token_accounting"]["effective"]["completion_source"],
        "provider"
    );
    assert!(call["message_token_estimates"].as_array().unwrap().len() >= 2);
}

#[test]
fn repl_compacts_tool_history_and_continues_past_eight_rounds() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        let mut agent_index = 0;
        while agent_index < 11 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body.clone());
            if body.contains("Current scratchpad")
                || body.contains("New consumed tool batch and assistant response")
            {
                write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- Summarized consumed fixture read batch and retained that fixture.txt returned repeated x characters."}}],"usage":{"prompt_tokens":120,"completion_tokens":24,"total_tokens":144}}"#,
                );
                continue;
            }
            if agent_index < 10 {
                write_sse(
                    &mut stream,
                    &format!(
                        r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call_read_{agent_index}","type":"function","function":{{"name":"read_file","arguments":"{{\"path\":\"fixture.txt\"}}"}}}}]}}}}]}}"#
                    ),
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"Finished after many tools."}}]}"#,
                );
            }
            agent_index += 1;
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(temp.path().join("fixture.txt"), "x".repeat(4000)).unwrap();
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

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("--context")
        .arg("1200")
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"inspect repeatedly\n/exit\n").unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Finished after many tools."), "{stdout}");
    let requests = bodies.lock().unwrap();
    let agent_requests = requests
        .iter()
        .filter(|request| request.contains(r#""stream":true"#))
        .collect::<Vec<_>>();
    assert_eq!(agent_requests.len(), 11);
    assert!(
        agent_requests
            .iter()
            .any(|request| request.contains("[turn scratchpad]")),
        "{requests:#?}"
    );
    assert!(
        !agent_requests[0].contains("[turn scratchpad]"),
        "{}",
        agent_requests[0]
    );
    assert!(
        agent_requests
            .iter()
            .skip(1)
            .all(|request| request.matches(r#""role":"tool""#).count() <= 1),
        "{requests:#?}"
    );
    assert!(
        requests
            .iter()
            .any(|request| request.contains("New consumed tool batch and assistant response")),
        "{requests:#?}"
    );
    for request in agent_requests.into_iter().skip(1) {
        let estimate = estimate_request_tokens(request);
        assert!(estimate <= 1200, "estimate={estimate}\n{request}");
    }
}

#[test]
fn repl_sends_image_paths_as_openai_image_parts() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let body = read_http_body(&mut stream);
        server_bodies.lock().unwrap().push(body);
        write_sse(
            &mut stream,
            r#"data: {"choices":[{"delta":{"content":"I can see the image."}}]}"#,
        );
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(temp.path().join("sample.png"), [137, 80, 78, 71]).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.vision]
base_url = "http://{addr}/v1"
model = "fake-vision"
api_key = ""
"#
        ),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"describe sample.png\n/exit\n").unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let request = bodies.lock().unwrap().pop().unwrap();
    assert!(request.contains(r#""type":"image_url""#), "{request}");
    assert!(
        request.contains("data:image/png;base64,iVBORw=="),
        "{request}"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("I can see the image."));
}

#[test]
fn repl_sends_read_image_tool_results_as_vision_content() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body.clone());
            if body.contains("Current scratchpad")
                || body.contains("New consumed tool batch and assistant response")
            {
                write_json(
                    &mut stream,
                    r#"{"choices":[{"message":{"role":"assistant","content":"- read_image attached sample.png for inspection."}}],"usage":{"prompt_tokens":90,"completion_tokens":18,"total_tokens":108}}"#,
                );
            } else if index == 0 {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_image","type":"function","function":{"name":"read_image","arguments":"{\"paths\":[\"sample.png\"]}"}}]}}]}"#,
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"The image tool result was visible."}}]}"#,
                );
            }
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(temp.path().join("sample.png"), [137, 80, 78, 71]).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.vision]
base_url = "http://{addr}/v1"
model = "fake-vision"
api_key = ""
"#
        ),
    )
    .unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"inspect sample.png\n/exit\n").unwrap();
    }

    let output = child.wait_with_output().unwrap();
    server.join().unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = bodies.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests[0].contains(r#""name":"read_image""#),
        "{}",
        requests[0]
    );
    assert!(
        !requests[1].contains("data:image/png;base64"),
        "{}",
        requests[1]
    );
    assert!(
        requests[1].contains("Attached image(s) from read_image"),
        "{}",
        requests[1]
    );
    assert!(requests[2].contains("[turn scratchpad]"), "{}", requests[2]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("The image tool result was visible."));
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

fn estimate_request_tokens(body: &str) -> usize {
    let value: serde_json::Value = serde_json::from_str(body).unwrap();
    let messages = serde_json::from_value::<Vec<ChatMessage>>(value["messages"].clone()).unwrap();
    let tools = value
        .get("tools")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    estimate_chat_request_breakdown(&messages, &tools).total()
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
