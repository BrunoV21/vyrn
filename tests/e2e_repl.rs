use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
#[cfg(unix)]
use std::{
    fs::File,
    os::fd::FromRawFd,
    sync::mpsc,
    time::{Duration, Instant},
};
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

#[cfg(unix)]
#[test]
fn inline_tui_submits_live_steering_during_an_active_model_request() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (first_request_tx, first_request_rx) = mpsc::channel();
    let server = thread::spawn(move || {
        let (mut first, _) = listener.accept().unwrap();
        let first_body = read_http_body(&mut first);
        assert!(first_body.contains("Start the long task"), "{first_body}");
        first
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"Working on the old direction...\"}}]}\n\n",
            )
            .unwrap();
        first.flush().unwrap();
        first_request_tx.send(()).unwrap();
        thread::sleep(Duration::from_millis(500));

        let (mut second, _) = listener.accept().unwrap();
        let second_body = read_http_body(&mut second);
        assert!(
            second_body.contains("live steering from the human"),
            "{second_body}"
        );
        assert!(
            second_body.contains("Change direction now"),
            "{second_body}"
        );
        write_sse(
            &mut second,
            r#"data: {"choices":[{"delta":{"content":"VYRN_INLINE_STEERING_APPLIED"}}]}"#,
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
"#
        ),
    )
    .unwrap();

    let (mut master, slave) = open_pty();
    let stdin = slave.try_clone().unwrap();
    let stdout = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let output = Arc::new(Mutex::new(Vec::new()));
    let reader_output = Arc::clone(&output);
    let mut reader = master.try_clone().unwrap();
    let reader_handle = thread::spawn(move || {
        let mut buffer = [0_u8; 2048];
        while let Ok(read) = reader.read(&mut buffer) {
            if read == 0 {
                break;
            }
            if buffer[..read].windows(4).any(|bytes| bytes == b"\x1b[6n") {
                reader.write_all(b"\x1b[11;1R").unwrap();
                reader.flush().unwrap();
            }
            reader_output
                .lock()
                .unwrap()
                .extend_from_slice(&buffer[..read]);
        }
    });

    wait_for_pty_output(&output, "type / or press Ctrl+O for commands", 5);
    master.write_all(b"Start the long task\r").unwrap();
    master.flush().unwrap();
    first_request_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("first model request did not start");
    master.write_all(b"Change direction now\r").unwrap();
    master.flush().unwrap();

    wait_for_pty_output(&output, "VYRN_INLINE_STEERING_APPLIED", 5);
    wait_for_pty_output(&output, "turn spent:", 5);
    master.write_all(b"/exit\r").unwrap();
    master.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "inline vyrn exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("inline vyrn did not exit after /exit");
        }
        thread::sleep(Duration::from_millis(20));
    }
    drop(master);
    reader_handle.join().unwrap();
    server.join().unwrap();
}

#[cfg(unix)]
#[test]
fn inline_command_palette_runs_a_mouse_clicked_command() {
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

    let (mut master, slave) = open_pty();
    let stdin = slave.try_clone().unwrap();
    let stdout = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let output = Arc::new(Mutex::new(Vec::new()));
    let reader_output = Arc::clone(&output);
    let mut reader = master.try_clone().unwrap();
    let reader_handle = thread::spawn(move || {
        let mut buffer = [0_u8; 2048];
        while let Ok(read) = reader.read(&mut buffer) {
            if read == 0 {
                break;
            }
            if buffer[..read].windows(4).any(|bytes| bytes == b"\x1b[6n") {
                reader.write_all(b"\x1b[11;1R").unwrap();
                reader.flush().unwrap();
            }
            reader_output
                .lock()
                .unwrap()
                .extend_from_slice(&buffer[..read]);
        }
    });

    wait_for_pty_output(&output, "type / or press Ctrl+O for commands", 5);
    master.write_all(b"/").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "exit vyrn", 5);

    // SGR mouse coordinates are one-based. With the fixed 120x30 PTY, /exit is
    // the twelfth palette row beneath the composer anchored at row 11.
    master.write_all(b"\x1b[<0;5;23m").unwrap();
    master.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "inline vyrn exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("inline vyrn did not execute the mouse-clicked /exit command");
        }
        thread::sleep(Duration::from_millis(20));
    }
    drop(master);
    reader_handle.join().unwrap();
}

#[cfg(unix)]
#[test]
fn fullscreen_header_controls_and_model_picker_support_mouse_clicks() {
    let temp = tempdir().unwrap();
    let project = temp
        .path()
        .join("this-is-a-deliberately-long-project-directory-for-header");
    std::fs::create_dir_all(project.join(".vyrn")).unwrap();
    std::fs::write(
        project.join(".vyrn/models.toml"),
        r#"[models.alpha]
base_url = "http://127.0.0.1:9/v1"
model = "offline-alpha"
api_key = ""

[models.beta]
base_url = "http://127.0.0.1:9/v1"
model = "offline-beta"
api_key = ""
"#,
    )
    .unwrap();

    let (mut master, slave) = open_pty();
    let stdin = slave.try_clone().unwrap();
    let stdout = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("tui")
        .current_dir(&project)
        .env("HOME", temp.path())
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let output = Arc::new(Mutex::new(Vec::new()));
    let reader_output = Arc::clone(&output);
    let mut reader = master.try_clone().unwrap();
    let reader_handle = thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        while let Ok(read) = reader.read(&mut buffer) {
            if read == 0 {
                break;
            }
            reader_output
                .lock()
                .unwrap()
                .extend_from_slice(&buffer[..read]);
        }
    });

    wait_for_pty_output(&output, "message vyrn", 5);

    // Header controls are stable because the deliberately long cwd is compacted
    // to 30 columns in the fixed 120x30 viewport.
    master.write_all(b"\x1b[<0;75;1M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output_count(&output, "collapsed", 1, 5);

    master.write_all(b"\x1b[<0;75;1M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output_count(&output, "expanded", 1, 5);

    // The boxed scratchpad button toggles its already-open panel closed, then open.
    master.write_all(b"\x1b[<0;35;8M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output_count(&output, "collapsed", 2, 5);

    master.write_all(b"\x1b[<0;35;8M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output_count(&output, "expanded", 2, 5);

    master.write_all(b"\x1b[<0;83;1M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "cleared", 5);

    master.write_all(b"\x1b[<0;55;1M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "model profiles", 5);

    // The centered picker is 56x7; beta is its second inner row.
    master.write_all(b"\x1b[<0;35;14M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "switched → beta", 5);

    master.write_all(b"\x03").unwrap();
    master.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "fullscreen vyrn exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("fullscreen vyrn did not execute the clicked header controls");
        }
        thread::sleep(Duration::from_millis(20));
    }
    drop(master);
    reader_handle.join().unwrap();
}

#[cfg(unix)]
#[test]
fn fullscreen_tui_stays_clickable_while_streaming_a_real_turn() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (request_received_tx, request_received_rx) = mpsc::channel();
    let (release_response_tx, release_response_rx) = mpsc::channel();
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let body = read_http_body(&mut stream);
        assert!(body.contains("Show the fullscreen marker"), "{body}");
        request_received_tx.send(()).unwrap();
        release_response_rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
        write_sse(
            &mut stream,
            r#"data: {"choices":[{"delta":{"content":"VYRN_FULLSCREEN_STREAM_OK"}}]}"#,
        );
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.local]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#,
        ),
    )
    .unwrap();

    let (mut master, slave) = open_pty();
    let stdin = slave.try_clone().unwrap();
    let stdout = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("tui")
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let output = Arc::new(Mutex::new(Vec::new()));
    let reader_output = Arc::clone(&output);
    let mut reader = master.try_clone().unwrap();
    let reader_handle = thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        while let Ok(read) = reader.read(&mut buffer) {
            if read == 0 {
                break;
            }
            reader_output
                .lock()
                .unwrap()
                .extend_from_slice(&buffer[..read]);
        }
    });

    wait_for_pty_output(&output, "message vyrn", 5);
    master.write_all(b"Show the fullscreen marker\r").unwrap();
    master.flush().unwrap();
    request_received_rx
        .recv_timeout(Duration::from_secs(5))
        .unwrap();
    wait_for_pty_output(&output, "thinking", 5);

    // The selected boxed scratchpad control remains clickable while the model
    // response is deliberately held open by the test server.
    master.write_all(b"\x1b[<0;35;8M").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "inspector collaps", 5);

    release_response_tx.send(()).unwrap();
    wait_for_pty_output(&output, "VYRN_FULLSCREEN_STREAM_OK", 5);
    wait_for_pty_output(&output, "turn spent:", 5);
    master.write_all(b"/exit\r").unwrap();
    master.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "fullscreen vyrn exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("fullscreen vyrn did not exit after /exit");
        }
        thread::sleep(Duration::from_millis(20));
    }
    drop(master);
    reader_handle.join().unwrap();
    server.join().unwrap();
}

#[cfg(unix)]
#[test]
fn fullscreen_tui_answers_ask_user_in_a_modal_and_continues_the_turn() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body);
            match index {
                0 => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ask","type":"function","function":{"name":"ask_user","arguments":"{\"questions\":[{\"id\":\"scope\",\"header\":\"Scope\",\"question\":\"Which approach should I use?\",\"options\":[{\"label\":\"Core tool\",\"description\":\"Always available\"},{\"label\":\"Slash command\"}]}]}"}}]}}]}"#,
                ),
                _ => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"content":"VYRN_FULLSCREEN_ASK_USER_OK"}}]}"#,
                ),
            }
        }
    });

    let temp = tempdir().unwrap();
    std::fs::create_dir_all(temp.path().join(".vyrn")).unwrap();
    std::fs::write(
        temp.path().join(".vyrn/models.toml"),
        format!(
            r#"[models.local]
base_url = "http://{addr}/v1"
model = "fake-small"
api_key = ""
"#,
        ),
    )
    .unwrap();

    let (mut master, slave) = open_pty();
    let stdin = slave.try_clone().unwrap();
    let stdout = slave.try_clone().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_vyrn"))
        .arg("tui")
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .stdin(Stdio::from(stdin))
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(slave))
        .spawn()
        .unwrap();

    let output = Arc::new(Mutex::new(Vec::new()));
    let reader_output = Arc::clone(&output);
    let mut reader = master.try_clone().unwrap();
    let reader_handle = thread::spawn(move || {
        let mut buffer = [0_u8; 4096];
        while let Ok(read) = reader.read(&mut buffer) {
            if read == 0 {
                break;
            }
            reader_output
                .lock()
                .unwrap()
                .extend_from_slice(&buffer[..read]);
        }
    });

    wait_for_pty_output(&output, "message vyrn", 5);
    master.write_all(b"clarify before continuing\r").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "Which approach should I use?", 5);
    master.write_all(b"\r").unwrap();
    master.flush().unwrap();
    wait_for_pty_output(&output, "VYRN_FULLSCREEN_ASK_USER_OK", 5);
    wait_for_pty_output(&output, "turn spent:", 5);
    master.write_all(b"/exit\r").unwrap();
    master.flush().unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "fullscreen vyrn exited with {status}");
            break;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("fullscreen vyrn did not exit after ask_user");
        }
        thread::sleep(Duration::from_millis(20));
    }
    drop(master);
    reader_handle.join().unwrap();
    server.join().unwrap();

    let bodies = bodies.lock().unwrap();
    assert!(bodies[1].contains("Core tool"), "{}", bodies[1]);
    assert!(bodies[1].contains("call_ask"), "{}", bodies[1]);
}

#[test]
fn repl_runs_against_openai_compatible_streaming_server() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = thread::spawn(move || {
        for _ in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            if body.contains("[turn scratchpad]") {
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
    assert!(stdout.contains("[updating turn memory...]"), "{stdout}");
    assert!(stdout.contains("I read fixture.txt: hello from e2e."));
    assert!(
        stdout.contains("estimated tokens, deterministic checkpoint"),
        "{stdout}"
    );
    assert!(stdout.contains("hello from e2e"), "{stdout}");
    assert!(stdout.contains("turn spent:"));
}

#[test]
fn parallel_tool_calls_share_one_bounded_deterministic_checkpoint() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body);
            match index {
                0 => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.txt\"}"}},{"index":1,"id":"call_b","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"b.txt\"}"}}]}}]}"#,
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
    assert_eq!(requests.len(), 2, "{requests:#?}");
    assert_eq!(
        requests
            .iter()
            .filter(|request| request.contains("Current scratchpad"))
            .count(),
        0,
        "{requests:#?}"
    );
    assert!(requests[1].contains("alpha"), "{}", requests[1]);
    assert!(requests[1].contains("beta"), "{}", requests[1]);
    assert!(requests[1].contains("[turn scratchpad]"), "{}", requests[1]);
    assert_eq!(requests[1].matches(r#""role":"tool""#).count(), 2);
}

#[test]
fn repl_plain_mode_answers_ask_user_tool_and_continues_turn() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let server_bodies = Arc::clone(&bodies);
    let server = thread::spawn(move || {
        for index in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body.clone());
            match index {
                0 => write_sse(
                    &mut stream,
                    r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_ask","type":"function","function":{"name":"ask_user","arguments":"{\"questions\":[{\"id\":\"scope\",\"header\":\"Scope\",\"question\":\"Which approach should I use?\",\"options\":[{\"label\":\"Core tool\",\"description\":\"Always available\"},{\"label\":\"Slash command\"}]}]}"}}]}}]}"#,
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
        bodies[1].contains("\"tool_call_id\":\"call_ask\""),
        "{}",
        bodies[1]
    );
    assert!(bodies[1].contains("Core tool"), "{}", bodies[1]);
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
    assert!(stdout.contains("memory overhead"), "{stdout}");
    assert!(stdout.contains("net vs raw history"), "{stdout}");
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
            .all(|request| !request.contains("New consumed tool batch and assistant response")),
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
        for index in 0..2 {
            let (mut stream, _) = listener.accept().unwrap();
            let body = read_http_body(&mut stream);
            server_bodies.lock().unwrap().push(body.clone());
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
    assert_eq!(requests.len(), 2);
    assert!(
        requests[0].contains(r#""name":"read_image""#),
        "{}",
        requests[0]
    );
    assert!(
        requests[1].contains("data:image/png;base64"),
        "{}",
        requests[1]
    );
    assert!(
        requests[1].contains("Attached image(s) from read_image"),
        "{}",
        requests[1]
    );
    assert!(requests[1].contains("[turn scratchpad]"), "{}", requests[1]);
    let request: serde_json::Value = serde_json::from_str(&requests[1]).unwrap();
    let checkpoint = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|message| {
            message["content"]
                .as_str()
                .filter(|content| content.starts_with("[turn scratchpad]"))
        })
        .unwrap();
    assert!(
        !checkpoint.contains("data:image/png;base64"),
        "{checkpoint}"
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

#[cfg(unix)]
fn open_pty() -> (File, File) {
    let mut master_fd = -1;
    let mut slave_fd = -1;
    let mut window = libc::winsize {
        ws_row: 30,
        ws_col: 120,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    let result = unsafe {
        libc::openpty(
            &mut master_fd,
            &mut slave_fd,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut window,
        )
    };
    assert_eq!(
        result,
        0,
        "openpty failed: {}",
        std::io::Error::last_os_error()
    );
    let master = unsafe { File::from_raw_fd(master_fd) };
    let slave = unsafe { File::from_raw_fd(slave_fd) };
    (master, slave)
}

#[cfg(unix)]
fn wait_for_pty_output(output: &Arc<Mutex<Vec<u8>>>, needle: &str, timeout_seconds: u64) {
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
    loop {
        let snapshot = output.lock().unwrap().clone();
        if String::from_utf8_lossy(&snapshot).contains(needle) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {needle:?}; output:\n{}",
            String::from_utf8_lossy(&snapshot)
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[cfg(unix)]
fn wait_for_pty_output_count(
    output: &Arc<Mutex<Vec<u8>>>,
    needle: &str,
    count: usize,
    timeout_seconds: u64,
) {
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
    loop {
        let snapshot = output.lock().unwrap().clone();
        let rendered = String::from_utf8_lossy(&snapshot);
        if rendered.matches(needle).count() >= count {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {count} occurrences of {needle:?}; output:\n{rendered}"
        );
        thread::sleep(Duration::from_millis(20));
    }
}
