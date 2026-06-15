use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use tempfile::tempdir;

#[test]
fn repl_runs_against_openai_compatible_streaming_server() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            if body.contains(r#""role":"tool""#) {
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
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"read fixture.txt\n/exit\n").unwrap();
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
    assert!(stdout.contains("turn spent:"));
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
fn repl_compacts_tool_history_and_continues_past_eight_rounds() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..11 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body);
            if index < 10 {
                write_sse(
                    &mut stream,
                    &format!(
                        r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"call_read_{index}","type":"function","function":{{"name":"read_file","arguments":"{{\"path\":\"fixture.txt\"}}"}}}}]}}}}]}}"#
                    ),
                );
            } else {
                write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"Finished after many tools."}}]}"#,
                );
            }
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
    assert_eq!(requests.len(), 11);
    assert!(
        requests
            .iter()
            .any(|request| request.contains("[compacted tool history]")),
        "{requests:#?}"
    );
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
        for index in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body);
            if index == 0 {
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
    assert_eq!(requests.len(), 2);
    assert!(
        requests[0].contains(r#""name":"read_image""#),
        "{}",
        requests[0]
    );
    assert!(
        requests[1].contains("data:image/png;base64,iVBORw=="),
        "{}",
        requests[1]
    );
    assert!(
        requests[1].contains("Attached image(s) from read_image"),
        "{}",
        requests[1]
    );
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
