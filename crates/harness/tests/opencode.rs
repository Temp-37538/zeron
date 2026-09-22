//! Native opencode driver against a scripted in-test HTTP/SSE server.
//!
//! The fake speaks the v1 surface (`/global/health`, `/session`,
//! `/session/{id}/prompt_async`, `/session/{id}/abort`, `/provider`,
//! `/command`, `/global/event`) or, in `start_v2()` mode, the 2.x `/api/*`
//! surface (`/api/info`, `/api/session`, `/api/session/{id}/prompt`,
//! `/api/session/{id}/interrupt`, `/api/model`, `/api/command`,
//! `/api/event`) — every shape captured live on a 2.0.10 server. Either way
//! the TEST owns bus timing via `emit()` / `emit_v2()`: the premature-done
//! class is exactly about what happens between events, so the fixtures must
//! own the clock.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{broadcast, mpsc, oneshot};
use zeron_harness::{
    CancellationToken, Harness, HarnessError, OpencodeHarness, RunControls, SteerMessage,
};
use zeron_proto::{
    AgentEvent, DoneStatus, ReasoningLevel, RunRequest, SandboxLevel, ToolCall, UserInputAnswer,
};

// ---------------------------------------------------------------------------
// Fake server
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FakeOpencode {
    base: String,
    events: broadcast::Sender<(u64, String)>,
    /// Every emitted frame, sequence-stamped — replayed to late SSE
    /// subscribers so tests may emit before the driver's stream connects.
    backlog: Arc<Mutex<Vec<(u64, String)>>>,
    /// Recorded `(path, body)` of every POST.
    posts: Arc<Mutex<Vec<(String, Value)>>>,
    /// Recorded `(method, path, body)` of every non-GET call.
    calls: Arc<Mutex<Vec<(String, String, Value)>>>,
    providers: Arc<Mutex<Value>>,
    /// Pending forms served on `GET /api/session/{id}/form` (2.x).
    forms: Arc<Mutex<Value>>,
    /// `Accept` header of the 2.x `/api/event` request (the route serves
    /// nothing without `text/event-stream`).
    sse_accept: Arc<Mutex<Option<String>>>,
    /// Answers left on `GET /api/model` before the 2.x catalog turns
    /// non-empty (the real server serves an empty list while models.dev
    /// syncs and the driver must poll through it).
    catalog_empty_left: Arc<Mutex<u32>>,
    /// Latest `session.status` per session, so the status-poll route can
    /// answer an ambiguous idle.
    statuses: Arc<Mutex<serde_json::Map<String, Value>>>,
    /// Commands served on the 1.x `/command` route.
    commands: Arc<Mutex<Value>>,
    /// Whether the fake speaks the 2.x `/api/*` wire.
    v2: bool,
    /// Whether an SSE subscriber existed when the FIRST prompt landed (the
    /// no-replay bus makes prompting before the subscription a real
    /// event-loss race — observed live on fast-failing turns).
    first_prompt_had_subscriber: Arc<Mutex<Option<bool>>>,
    /// Leading 500s to answer `POST /session` with (the opencode
    /// lazy-migration crash class: first access 500s, retry succeeds).
    fail_session_creates: Arc<Mutex<u32>>,
}

impl FakeOpencode {
    /// The 1.18 wire.
    async fn start() -> Self {
        Self::start_proto(false).await
    }

    /// The 2.x wire, shaped like 2.0.8+ as captured live on 2.0.10:
    /// `/api/info` carries the version, `/api/health` is gone, the web UI
    /// answers `/global/health` with HTML, and the bus is `/api/event`.
    async fn start_v2() -> Self {
        Self::start_proto(true).await
    }

    async fn start_proto(v2: bool) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let (events, _) = broadcast::channel::<(u64, String)>(256);
        let fake = Self {
            base,
            events: events.clone(),
            backlog: Arc::new(Mutex::new(Vec::new())),
            posts: Arc::new(Mutex::new(Vec::new())),
            calls: Arc::new(Mutex::new(Vec::new())),
            providers: Arc::new(Mutex::new(json!({ "all": [], "default": {} }))),
            forms: Arc::new(Mutex::new(json!([]))),
            sse_accept: Arc::new(Mutex::new(None)),
            catalog_empty_left: Arc::new(Mutex::new(0)),
            statuses: Arc::default(),
            commands: Arc::new(Mutex::new(
                json!([{ "name": "init", "description": "Create AGENTS.md" }]),
            )),
            v2,
            first_prompt_had_subscriber: Arc::new(Mutex::new(None)),
            fail_session_creates: Arc::new(Mutex::new(0)),
        };
        let accept = fake.clone();
        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    return;
                };
                let fake = accept.clone();
                tokio::spawn(async move { fake.serve(stream).await });
            }
        });
        fake
    }

    fn push_frame(&self, framed: String) {
        let mut backlog = self.backlog.lock().unwrap();
        let seq = backlog.len() as u64;
        backlog.push((seq, framed.clone()));
        let _ = self.events.send((seq, framed));
    }

    /// Push one 1.x bus event (the driver accepts both the bare and the
    /// `/global/event` envelope; the fake uses the enveloped form). Status
    /// frames are also recorded so the status-poll route can answer them.
    fn emit(&self, payload: Value) {
        if payload["type"] == "session.status"
            && let Some(id) = payload["properties"]["sessionID"].as_str()
        {
            self.statuses
                .lock()
                .unwrap()
                .insert(id.to_owned(), payload["properties"]["status"].clone());
        }
        let framed = format!(
            "data: {}\n\n",
            json!({ "directory": "/", "payload": payload })
        );
        self.push_frame(framed);
    }

    /// Push one raw 2.x `/api/event` frame — the shape captured live on
    /// 2.0.10 (`{id, created, type, data}`; normalized by the driver).
    fn emit_v2(&self, kind: &str, data: Value) {
        let framed = format!(
            "data: {}\n\n",
            json!({ "id": format!("evt_{kind}"), "created": 0, "type": kind, "data": data })
        );
        self.push_frame(framed);
    }

    fn set_providers(&self, providers: Value) {
        *self.providers.lock().unwrap() = providers;
    }

    /// Pending forms the 2.x `GET /api/session/{id}/form` route answers.
    fn set_forms(&self, forms: Value) {
        *self.forms.lock().unwrap() = forms;
    }

    fn posts_to(&self, path: &str) -> Vec<Value> {
        self.posts
            .lock()
            .unwrap()
            .iter()
            .filter(|(p, _)| p == path)
            .map(|(_, b)| b.clone())
            .collect()
    }

    fn calls_to(&self, method: &str, path: &str) -> Vec<Value> {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, p, _)| m == method && p == path)
            .map(|(_, _, b)| b.clone())
            .collect()
    }

    async fn serve(self, mut stream: tokio::net::TcpStream) {
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        // One request per connection is enough for reqwest's default pool
        // behavior in these tests; keep-alive requests re-enter here.
        loop {
            let header_end = loop {
                if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                    break pos + 4;
                }
                match stream.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            };
            let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
            let mut lines = head.lines();
            let start = lines.next().unwrap_or_default().to_owned();
            let headers: Vec<(String, String)> = lines
                .filter_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    Some((k.trim().to_ascii_lowercase(), v.trim().to_owned()))
                })
                .collect();
            let content_length = headers
                .iter()
                .find(|(k, _)| k == "content-length")
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(0);
            while buf.len() < header_end + content_length {
                match stream.read(&mut chunk).await {
                    Ok(0) | Err(_) => return,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
            let body: Value = serde_json::from_slice(&buf[header_end..header_end + content_length])
                .unwrap_or(Value::Null);
            buf.drain(..header_end + content_length);

            let mut parts = start.split_whitespace();
            let method = parts.next().unwrap_or_default().to_owned();
            let target = parts.next().unwrap_or_default().to_owned();
            let path = target.split('?').next().unwrap_or_default().to_owned();

            let sse_path = if self.v2 { "/api/event" } else { "/global/event" };
            if method == "GET" && path == sse_path {
                if self.v2 {
                    // The 2.x bus serves nothing without this header
                    // (observed live on 2.0.3) — record it for assertions.
                    *self.sse_accept.lock().unwrap() = headers
                        .iter()
                        .find(|(k, _)| k == "accept")
                        .map(|(_, v)| v.clone());
                }
                // Subscribe FIRST, then snapshot the backlog: frames landing
                // in between arrive on both channels and dedupe by sequence.
                let mut rx = self.events.subscribe();
                let replay = self.backlog.lock().unwrap().clone();
                let mut next_seq = replay.len() as u64;
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                          cache-control: no-cache\r\nconnection: close\r\n\r\n",
                    )
                    .await;
                let _ = stream
                    .write_all(b"data: {\"payload\":{\"type\":\"server.connected\",\"properties\":{}}}\n\n")
                    .await;
                for (_, frame) in &replay {
                    if stream.write_all(frame.as_bytes()).await.is_err() {
                        return;
                    }
                }
                let _ = stream.flush().await;
                while let Ok((seq, frame)) = rx.recv().await {
                    if seq < next_seq {
                        continue;
                    }
                    next_seq = seq + 1;
                    if stream.write_all(frame.as_bytes()).await.is_err() {
                        return;
                    }
                    let _ = stream.flush().await;
                }
                return;
            }

            // 2.0.8 serves its web UI on `/global/health`: HTML, not JSON —
            // the version guard must reject it and fall through to /api/info.
            if self.v2 && method == "GET" && path == "/global/health" {
                let html = "<!doctype html><html><body>opencode</body></html>";
                let resp = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\n\
                     content-length: {}\r\n\r\n{html}",
                    html.len()
                );
                if stream.write_all(resp.as_bytes()).await.is_err() {
                    return;
                }
                continue;
            }

            if method != "GET" {
                self.calls
                    .lock()
                    .unwrap()
                    .push((method.clone(), path.clone(), body.clone()));
                if method == "POST" {
                    // 2.x prompts land on `/prompt`; the gated first send is
                    // the same no-replay race as 1.x `prompt_async`.
                    if path.ends_with("/prompt_async") || path.ends_with("/prompt") {
                        let mut first = self.first_prompt_had_subscriber.lock().unwrap();
                        if first.is_none() {
                            *first = Some(self.events.receiver_count() > 0);
                        }
                    }
                    self.posts.lock().unwrap().push((path.clone(), body));
                }
            }
            let (status, payload) = self.route(&method, &path);
            let body = payload.to_string();
            let resp = format!(
                "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n\
                 content-length: {}\r\n\r\n{body}",
                body.len()
            );
            if stream.write_all(resp.as_bytes()).await.is_err() {
                return;
            }
        }
    }

    fn route(&self, method: &str, path: &str) -> (&'static str, Value) {
        if self.v2 {
            return self.route_v2(method, path);
        }
        match (method, path) {
            ("GET", "/global/health") => ("200 OK", json!({ "healthy": true })),
            ("GET", "/provider") => ("200 OK", self.providers.lock().unwrap().clone()),
            ("GET", "/command") => ("200 OK", self.commands.lock().unwrap().clone()),
            ("POST", "/session") => {
                let mut fails = self.fail_session_creates.lock().unwrap();
                if *fails > 0 {
                    *fails -= 1;
                    (
                        "500 Internal Server Error",
                        json!({
                            "name": "UnknownError",
                            "data": {
                                "message": "Unexpected server error. Check server logs for details.",
                                "ref": "err_test",
                            },
                        }),
                    )
                } else {
                    ("200 OK", json!({ "id": "ses_test" }))
                }
            }
            ("GET", "/session/status") => (
                "200 OK",
                Value::Object(self.statuses.lock().unwrap().clone()),
            ),
            ("GET", "/session/ses_resume") => ("200 OK", json!({ "id": "ses_resume" })),
            ("GET", p) if p.starts_with("/session/") => ("404 Not Found", json!({})),
            ("POST", p) if p.ends_with("/prompt_async") => ("204 No Content", json!({})),
            ("POST", p) if p.ends_with("/abort") => ("200 OK", json!(true)),
            ("POST", p) if p.contains("/permission/") || p.contains("/question/") => {
                ("200 OK", json!(true))
            }
            _ => ("404 Not Found", json!({ "missing": path })),
        }
    }

    /// The 2.x `/api/*` surface (2.0.8+ shape). `/global/health` is served
    /// as HTML in `serve` and `/api/health` 404s, so only `/api/info`
    /// carries a version.
    fn route_v2(&self, method: &str, path: &str) -> (&'static str, Value) {
        match (method, path) {
            ("GET", "/api/info") => (
                "200 OK",
                json!({
                    "version": "2.0.10",
                    "pid": 4242,
                    "urls": ["http://127.0.0.1:4096"],
                    "paths": { "tmp": "/tmp" },
                }),
            ),
            ("GET", "/api/health") => ("404 Not Found", json!({})),
            ("GET", "/api/session/active") => ("200 OK", json!({ "data": {} })),
            ("GET", "/api/model") => {
                let mut left = self.catalog_empty_left.lock().unwrap();
                if *left > 0 {
                    *left -= 1;
                    return ("200 OK", json!({ "data": [] }));
                }
                (
                    "200 OK",
                    json!({ "data": [{
                        "providerID": "opencode",
                        "id": "test-model",
                        "name": "Test Model",
                        "limit": { "context": 100_000, "output": 8000 },
                        "variants": [{ "id": "high" }],
                        "enabled": true,
                    }]}),
                )
            }
            ("GET", "/api/command") => (
                "200 OK",
                json!({ "data": [{ "name": "init", "description": "Create AGENTS.md" }] }),
            ),
            ("POST", "/api/session") => ("200 OK", json!({ "data": { "id": "ses_test" } })),
            ("GET", "/api/session/ses_resume") => {
                ("200 OK", json!({ "data": { "id": "ses_resume" } }))
            }
            // The pending list is the authoritative form shape (`form.created`
            // nests the same record under `data.form` — captured live).
            ("GET", p) if p.ends_with("/form") => {
                ("200 OK", json!({ "data": self.forms.lock().unwrap().clone() }))
            }
            ("POST", p) if p.ends_with("/model") => ("204 No Content", json!({})),
            ("POST", p) if p.ends_with("/prompt") => ("200 OK", json!({ "data": {} })),
            ("POST", p) if p.ends_with("/interrupt") => {
                ("200 OK", json!({ "interrupted": true }))
            }
            ("POST", p) if p.ends_with("/command") => ("200 OK", json!({ "data": {} })),
            ("POST", p) if p.contains("/permission/") => ("200 OK", json!({ "data": {} })),
            // Answering or cancelling settles the pending form server-side.
            ("POST", p) if p.contains("/form/") && p.ends_with("/reply") => {
                *self.forms.lock().unwrap() = json!([]);
                ("200 OK", json!({ "data": {} }))
            }
            ("DELETE", p) if p.contains("/form/") => {
                *self.forms.lock().unwrap() = json!([]);
                ("200 OK", json!({ "data": {} }))
            }
            ("GET", p) if p.starts_with("/api/session/") => ("404 Not Found", json!({})),
            _ => ("404 Not Found", json!({ "missing": path })),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn request(prompt: &str) -> RunRequest {
    RunRequest {
        prompt: prompt.into(),
        harness: None,
        model: None,
        reasoning: None,
        model_options: serde_json::Map::new(),
        cwd: "/tmp".into(),
        sandbox: SandboxLevel::DangerFullAccess,
        auto_approve: true,
        attachments: Vec::new(),
        resume: None,
        worktree: None,
    }
}

#[allow(clippy::type_complexity)]
fn controls() -> (RunControls, mpsc::Sender<SteerMessage>, CancellationToken) {
    let (steer_tx, steering) = mpsc::channel(8);
    let token = CancellationToken::new();
    let controls = RunControls {
        request_input: Box::new(move |questions| {
            let (tx, rx) = oneshot::channel();
            let answers: Vec<UserInputAnswer> = questions
                .iter()
                .map(|q| UserInputAnswer {
                    question_id: q.id.clone(),
                    labels: q.options.first().cloned().into_iter().collect(),
                })
                .collect();
            let _ = tx.send(answers);
            rx
        }),
        steering,
        interrupt: token.clone(),
    };
    (controls, steer_tx, token)
}

fn harness(fake: &FakeOpencode) -> OpencodeHarness {
    OpencodeHarness::new().with_base_url(fake.base.clone())
}

/// Emit the standard opening frames of an assistant turn.
fn assistant_message(fake: &FakeOpencode, session: &str, message: &str) {
    fake.emit(json!({
        "type": "session.status",
        "properties": { "sessionID": session, "status": { "type": "busy" } },
    }));
    fake.emit(json!({
        "type": "message.updated",
        "properties": { "info": { "id": message, "role": "assistant", "sessionID": session } },
    }));
}

fn idle(fake: &FakeOpencode, session: &str) {
    fake.emit(json!({
        "type": "session.status",
        "properties": { "sessionID": session, "status": { "type": "idle" } },
    }));
}

/// Emit the captured 2.x opening frames of an assistant step.
fn v2_assistant_message(fake: &FakeOpencode, session: &str, message: &str) {
    fake.emit_v2("session.execution.started", json!({ "sessionID": session }));
    fake.emit_v2(
        "session.step.started",
        json!({
            "sessionID": session,
            "agent": "build",
            "model": { "id": "test-model", "providerID": "opencode", "variant": "default" },
            "assistantMessageID": message,
        }),
    );
}

/// The captured 2.x turn end.
fn v2_idle(fake: &FakeOpencode, session: &str) {
    fake.emit_v2("session.execution.succeeded", json!({ "sessionID": session }));
}

/// Consume the two opening events every run emits (session + commands).
async fn opening(
    stream: &mut (impl futures::Stream<Item = Result<AgentEvent, HarnessError>> + Unpin),
) {
    let started = next_event(stream).await;
    assert!(matches!(started, AgentEvent::SessionStarted { .. }));
    let commands = next_event(stream).await;
    assert!(matches!(commands, AgentEvent::AvailableCommands { .. }));
}

async fn next_event(
    stream: &mut (impl futures::Stream<Item = Result<AgentEvent, HarnessError>> + Unpin),
) -> AgentEvent {
    tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("event within budget")
        .expect("stream open")
        .expect("ok event")
}

/// Poll until `(method, path)` has received `n` calls; returns their bodies.
async fn wait_calls(fake: &FakeOpencode, method: &str, path: &str, n: usize) -> Vec<Value> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let calls = fake.calls_to(method, path);
            if calls.len() >= n {
                return calls;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{method} {path} never saw {n} calls"))
}

/// Poll until `path` has received `n` POSTs; returns their bodies.
async fn wait_posts(fake: &FakeOpencode, path: &str, n: usize) -> Vec<Value> {
    wait_calls(fake, "POST", path, n).await
}

/// Drain until a Done arrives; returns everything seen (Done last).
async fn drain_to_done(
    stream: &mut (impl futures::Stream<Item = Result<AgentEvent, HarnessError>> + Unpin),
) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    loop {
        let ev = next_event(stream).await;
        let done = matches!(&ev, AgentEvent::Done { .. });
        events.push(ev);
        if done {
            return events;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thinking_streams_and_the_turn_settles_only_on_idle() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");

    let started = next_event(&mut stream).await;
    assert!(matches!(
        &started,
        AgentEvent::SessionStarted { session_id, .. } if session_id == "ses_test"
    ));
    // AvailableCommands from /command.
    let commands = next_event(&mut stream).await;
    assert!(matches!(
        &commands,
        AgentEvent::AvailableCommands { commands } if commands.len() == 1
    ));

    assistant_message(&fake, "ses_test", "msg_1");
    // Reasoning part: open snapshot → deltas → closing snapshot (full text,
    // must dedup to nothing).
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_r", "messageID": "msg_1", "sessionID": "ses_test",
            "type": "reasoning", "text": "",
        }},
    }));
    fake.emit(json!({
        "type": "message.part.delta",
        "properties": {
            "sessionID": "ses_test", "messageID": "msg_1", "partID": "prt_r",
            "field": "text", "delta": "let me think",
        },
    }));
    let thinking = next_event(&mut stream).await;
    assert!(matches!(
        &thinking,
        AgentEvent::ReasoningDelta { text } if text == "let me think"
    ));
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_r", "messageID": "msg_1", "sessionID": "ses_test",
            "type": "reasoning", "text": "let me think",
        }},
    }));

    // Text streams; the turn must NOT settle during the quiet gap after it —
    // only idle ends the turn (the premature-done regression).
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_t", "messageID": "msg_1", "sessionID": "ses_test",
            "type": "text", "text": "Hello",
        }},
    }));
    let text = next_event(&mut stream).await;
    assert!(matches!(&text, AgentEvent::TextDelta { text } if text == "Hello"));
    let quiet = tokio::time::timeout(Duration::from_millis(600), stream.next()).await;
    assert!(quiet.is_err(), "nothing may settle a quiet-but-live turn");

    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.first(),
        Some(AgentEvent::AssistantMessageCompleted { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            session_id: Some(sid),
            ..
        }) if sid == "ses_test"
    ));
}

#[tokio::test]
async fn session_create_retries_once_through_the_lazy_migration_500() {
    // opencode 1.18.x's first directory-scoped request can 500 inside
    // Project.migrateProjectId while still committing the project row, so
    // the retried create succeeds — the turn must not die (field report:
    // `POST /session: 500 UnknownError`, ref-keyed `no such column:
    // project_id` in the server log).
    let fake = FakeOpencode::start().await;
    *fake.fail_session_creates.lock().unwrap() = 1;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");

    let started = next_event(&mut stream).await;
    assert!(matches!(
        &started,
        AgentEvent::SessionStarted { session_id, .. } if session_id == "ses_test"
    ));
    let commands = next_event(&mut stream).await;
    assert!(matches!(&commands, AgentEvent::AvailableCommands { .. }));

    // Exactly one retry: the 500 must not surface as a chip or a dead turn.
    let creates = wait_posts(&fake, "/session", 2).await;
    assert_eq!(creates.len(), 2);
    assistant_message(&fake, "ses_test", "msg_1");
    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(
        !events
            .iter()
            .any(|ev| matches!(ev, AgentEvent::Error { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn foreign_session_idle_never_settles_our_turn() {
    // The exact bug in opencode's own ACP layer: the first idle observed —
    // any session's — settled the turn.
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await; // SessionStarted
    let _ = next_event(&mut stream).await; // AvailableCommands

    assistant_message(&fake, "ses_test", "msg_1");
    idle(&fake, "ses_OTHER");
    let quiet = tokio::time::timeout(Duration::from_millis(600), stream.next()).await;
    assert!(quiet.is_err(), "a foreign session's idle settled our turn");

    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn model_and_advertised_variant_ride_the_prompt() {
    let fake = FakeOpencode::start().await;
    fake.set_providers(json!({
        "all": [{
            "id": "anthropic",
            "name": "Anthropic",
            "models": { "opus": { "name": "Opus", "variants": { "high": {}, "max": {} } } },
        }],
    }));
    let (controls, _steer, _token) = controls();
    let mut req = request("hi");
    req.model = Some("anthropic/opus".into());
    req.reasoning = Some(ReasoningLevel::XHigh);
    let mut stream = harness(&fake).run(req, controls).await.expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    let prompts = wait_posts(&fake, "/session/ses_test/prompt_async", 1).await;
    assert_eq!(prompts[0]["model"]["providerID"], "anthropic");
    assert_eq!(prompts[0]["model"]["modelID"], "opus");
    // XHigh isn't advertised: the ladder clamps to "high".
    assert_eq!(prompts[0]["variant"], "high");
    assert_eq!(prompts[0]["parts"][0]["text"], "hi");

    assistant_message(&fake, "ses_test", "msg_1");
    idle(&fake, "ses_test");
    drain_to_done(&mut stream).await;
}

#[tokio::test]
async fn steer_queues_mid_turn_and_delivers_at_idle() {
    let fake = FakeOpencode::start().await;
    let (controls, steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    assistant_message(&fake, "ses_test", "msg_1");
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_t", "messageID": "msg_1", "sessionID": "ses_test",
            "type": "text", "text": "working",
        }},
    }));
    let _ = next_event(&mut stream).await; // TextDelta

    steer
        .send(SteerMessage {
            prompt: "also do this".into(),
            message_id: None,
        })
        .await
        .unwrap();
    // Give the steer time to land in the queue, then end turn 1.
    tokio::time::sleep(Duration::from_millis(100)).await;
    idle(&fake, "ses_test");

    let ev = next_event(&mut stream).await;
    assert!(
        matches!(&ev, AgentEvent::Steered { .. }),
        "queued steer must continue the run at the turn boundary, got {ev:?}"
    );
    // The steer went out as a second prompt on the SAME session.
    let prompts = wait_posts(&fake, "/session/ses_test/prompt_async", 2).await;
    assert_eq!(prompts[1]["parts"][0]["text"], "also do this");

    // Turn 2 settles normally.
    assistant_message(&fake, "ses_test", "msg_2");
    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn interrupt_aborts_and_settles_interrupted() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    assistant_message(&fake, "ses_test", "msg_1");
    token.cancel();
    wait_posts(&fake, "/session/ses_test/abort", 1).await;
    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Interrupted,
            ..
        })
    ));
}

#[tokio::test]
async fn provider_retries_surface_and_cap_out() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    let retry = |attempt: u64| {
        json!({
            "type": "session.status",
            "properties": { "sessionID": "ses_test", "status": {
                "type": "retry", "attempt": attempt,
                "message": "AI_APICallError: unreachable", "next": 0,
            }},
        })
    };
    fake.emit(retry(1));
    fake.emit(retry(3));
    let ev = next_event(&mut stream).await;
    let AgentEvent::Error { message } = &ev else {
        panic!("expected a retry error chip, got {ev:?}");
    };
    assert!(
        message.contains("retrying") && message.contains("attempt 3"),
        "{message}"
    );
    assert!(message.contains("unreachable"), "{message}");

    fake.emit(retry(8));
    let ev = next_event(&mut stream).await;
    let AgentEvent::Error { message } = &ev else {
        panic!("expected the give-up chip, got {ev:?}");
    };
    assert!(message.contains("Giving up"), "{message}");
    // The driver aborted the turn; the server answers with idle.
    wait_posts(&fake, "/session/ses_test/abort", 1).await;
    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Errored,
            error: Some(_),
            ..
        })
    ));
}

#[tokio::test]
async fn session_error_with_no_content_settles_errored() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    fake.emit(json!({
        "type": "session.error",
        "properties": { "sessionID": "ses_test", "error": {
            "name": "ProviderAuthError",
            "data": { "message": "no credentials for anthropic" },
        }},
    }));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Error { message } if message.contains("no credentials")
    ));
    // opencode re-emits the same failure with an exception-name prefix and a
    // stack — that must NOT mint a second chip (field report: every failure
    // rendered twice).
    fake.emit(json!({
        "type": "session.error",
        "properties": { "sessionID": "ses_test", "error": {
            "name": "UnknownError",
            "data": { "message": "ProviderAuthError: no credentials for anthropic\n    at stack" },
        }},
    }));
    let quiet = tokio::time::timeout(Duration::from_millis(400), stream.next()).await;
    assert!(
        quiet.is_err(),
        "duplicate error must not mint a second chip"
    );
    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Errored,
            error: Some(e),
            ..
        }) if e.contains("no credentials")
    ));
}

#[tokio::test]
async fn subagent_task_streams_tagged_and_settles_from_the_task_part() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("spawn"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    assistant_message(&fake, "ses_test", "msg_1");
    // The task tool part registers the chip and binds by metadata.
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_task", "messageID": "msg_1", "sessionID": "ses_test",
            "type": "tool", "tool": "task",
            "state": {
                "status": "running",
                "input": { "description": "Viz probe", "prompt": "run", "subagent_type": "general" },
                "metadata": { "sessionId": "ses_child", "parentSessionId": "ses_test" },
            },
        }},
    }));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::ToolCall { id, call: ToolCall::Unknown { name, .. } }
            if id == "prt_task" && name == "Agent: Viz probe"
    ));

    // Child comes up and streams: prompt in, assistant text out — tagged.
    fake.emit(json!({
        "type": "session.created",
        "properties": { "info": {
            "id": "ses_child", "parentID": "ses_test",
            "title": "Viz probe (@general subagent)",
        }},
    }));
    fake.emit(json!({
        "type": "message.updated",
        "properties": { "info": { "id": "msg_cu", "role": "user", "sessionID": "ses_child" } },
    }));
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_cu", "messageID": "msg_cu", "sessionID": "ses_child",
            "type": "text", "text": "run",
        }},
    }));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Subagent { parent_tool_use_id, event }
            if parent_tool_use_id == "prt_task"
                && matches!(&**event, AgentEvent::UserMessage { text } if text == "run")
    ));
    fake.emit(json!({
        "type": "message.updated",
        "properties": { "info": { "id": "msg_ca", "role": "assistant", "sessionID": "ses_child" } },
    }));
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_ca", "messageID": "msg_ca", "sessionID": "ses_child",
            "type": "text", "text": "finished",
        }},
    }));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Subagent { event, .. }
            if matches!(&**event, AgentEvent::TextDelta { text } if text == "finished")
    ));

    // The task part completing settles the chip: ToolResult + tagged Done.
    fake.emit(json!({
        "type": "message.part.updated",
        "properties": { "part": {
            "id": "prt_task", "messageID": "msg_1", "sessionID": "ses_test",
            "type": "tool", "tool": "task",
            "state": {
                "status": "completed",
                "input": { "description": "Viz probe" },
                "output": "<task_result>finished</task_result>",
                "title": "Viz probe",
                "metadata": { "sessionId": "ses_child" },
                "time": { "start": 1, "end": 2 },
            },
        }},
    }));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::ToolResult { id, is_error: false, .. } if id == "prt_task"
    ));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Subagent { parent_tool_use_id, event }
            if parent_tool_use_id == "prt_task"
                && matches!(&**event, AgentEvent::Done { status: DoneStatus::Completed, .. })
    ));

    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn resume_reuses_the_durable_session() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut req = request("continue");
    req.resume = Some("ses_resume".into());
    let mut stream = harness(&fake).run(req, controls).await.expect("run starts");
    let started = next_event(&mut stream).await;
    assert!(matches!(
        &started,
        AgentEvent::SessionStarted { session_id, .. } if session_id == "ses_resume"
    ));
    let _ = next_event(&mut stream).await;
    wait_posts(&fake, "/session/ses_resume/prompt_async", 1).await;

    assistant_message(&fake, "ses_resume", "msg_1");
    idle(&fake, "ses_resume");
    drain_to_done(&mut stream).await;
}

#[tokio::test]
async fn slash_command_routes_through_the_command_endpoint() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("/init the repo"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;

    let commands = wait_posts(&fake, "/session/ses_test/command", 1).await;
    assert_eq!(commands[0]["command"], "init");
    assert_eq!(commands[0]["arguments"], "the repo");
    assert!(fake.posts_to("/session/ses_test/prompt_async").is_empty());

    assistant_message(&fake, "ses_test", "msg_1");
    idle(&fake, "ses_test");
    drain_to_done(&mut stream).await;
}

#[tokio::test]
async fn first_prompt_waits_for_the_live_event_subscription() {
    // The v1 bus has no replay: a fast-failing turn (bad model id) emits
    // busy → session.error → idle within ~200ms of the prompt. Prompting
    // before the SSE stream exists loses the whole turn (observed live,
    // 1.18.21) — the driver must gate the first prompt on the connection.
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;
    wait_posts(&fake, "/session/ses_test/prompt_async", 1).await;
    assert_eq!(
        *fake.first_prompt_had_subscriber.lock().unwrap(),
        Some(true),
        "prompt must not be posted before the /global/event subscription exists"
    );

    // And the fast-failure lifecycle settles promptly (all three frames in
    // one burst), not via the stall watchdog.
    fake.emit(json!({
        "type": "session.status",
        "properties": { "sessionID": "ses_test", "status": { "type": "busy" } },
    }));
    fake.emit(json!({
        "type": "session.error",
        "properties": { "sessionID": "ses_test", "error": {
            "name": "UnknownError",
            "data": { "message": "Model not found: opencode/gone-model" },
        }},
    }));
    idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Errored,
            error: Some(e),
            ..
        }) if e.contains("Model not found")
    ));
}

#[tokio::test]
async fn models_discover_from_the_provider_catalog() {
    let fake = FakeOpencode::start().await;
    fake.set_providers(json!({
        "all": [
            {
                "id": "opencode",
                "name": "OpenCode Zen",
                "models": { "big-pickle": { "name": "Big Pickle" } },
            },
            {
                "id": "catalog-only",
                "name": "Needs A Key",
                "models": { "locked": { "name": "Locked" } },
            },
        ],
        "connected": ["opencode"],
    }));
    let harness = harness(&fake);
    let models = harness.models().await.expect("models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "opencode/big-pickle");
    assert!(models[0].options.is_empty(), "v1 must not advertise agents");
    // Commands were primed off the same probe.
    let commands = harness.commands().await.expect("commands");
    assert_eq!(commands[0].name, "init");
}

#[tokio::test]
async fn models_keep_large_catalog_on_empty_response_and_recover() {
    let fake = FakeOpencode::start().await;
    let harness = harness(&fake);
    let models: serde_json::Map<String, Value> = (0..512)
        .map(|i| (format!("model-{i}"), json!({"name": "x".repeat(2048)})))
        .collect();
    let catalog = json!({
        "all": [{"id": "provider", "models": models}],
        "connected": ["provider"],
    });
    fake.set_providers(catalog.clone());
    assert_eq!(harness.models().await.unwrap().len(), 512);

    fake.set_providers(json!({"all": [], "connected": []}));
    // An empty response without a credential-context change is a failed probe,
    // so the last successful catalog remains available.
    let retained = harness.models().await.unwrap();
    assert_eq!(retained.len(), 512);
    assert!(
        retained
            .iter()
            .all(|model| model.id.starts_with("provider/"))
    );

    fake.set_providers(json!({
        "all": [{"id": "new-account", "models": {"fresh": {"name": "Fresh"}}}],
        "connected": ["new-account"],
    }));
    let refreshed = harness.models().await.unwrap();
    assert_eq!(refreshed.len(), 1);
    assert_eq!(refreshed[0].id, "new-account/fresh");
}

#[tokio::test]
async fn repeated_session_create_failure_stops_after_one_retry() {
    let fake = FakeOpencode::start().await;
    *fake.fail_session_creates.lock().unwrap() = 10;
    let (controls, _, _) = controls();
    let mut stream = harness(&fake).run(request("hi"), controls).await.unwrap();
    let events = drain_to_done(&mut stream).await;
    assert_eq!(
        fake.posts
            .lock()
            .unwrap()
            .iter()
            .filter(|(path, _)| path == "/session")
            .count(),
        2
    );
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::SessionStarted { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Errored,
            ..
        })
    ));
}

// ---------------------------------------------------------------------------
// 2.x wire — every shape below was captured from a live 2.0.10 server
// ---------------------------------------------------------------------------

#[tokio::test]
async fn v2_detection_and_a_full_turn_ride_the_api_wire() {
    // 2.0.8+ serves its web UI on `/global/health` (HTML) and 404s
    // `/api/health`; only `/api/info` carries a version. A completed run
    // proves the V2 wire was resolved — everything below uses `/api/*`.
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut req = request("hi");
    req.model = Some("opencode/test-model".into());
    req.reasoning = Some(ReasoningLevel::XHigh);
    let mut stream = harness(&fake)
        .run(req, controls)
        .await
        .expect("run starts");
    let started = next_event(&mut stream).await;
    assert!(matches!(
        &started,
        AgentEvent::SessionStarted { session_id, .. } if session_id == "ses_test"
    ));
    let commands = next_event(&mut stream).await;
    assert!(matches!(
        &commands,
        AgentEvent::AvailableCommands { commands } if commands.len() == 1
    ));

    // 2.x sets the model (+ variant) once on the session; the catalog
    // advertises `high`, so XHigh clamps to it. The prompt carries text and
    // files only.
    let models = wait_calls(&fake, "POST", "/api/session/ses_test/model", 1).await;
    assert_eq!(
        models[0]["model"],
        json!({ "providerID": "opencode", "id": "test-model", "variant": "high" })
    );
    let prompts = wait_calls(&fake, "POST", "/api/session/ses_test/prompt", 1).await;
    assert_eq!(prompts[0], json!({ "text": "hi", "files": [] }));
    assert_eq!(
        *fake.first_prompt_had_subscriber.lock().unwrap(),
        Some(true),
        "the first prompt must not be posted before the /api/event subscription exists"
    );
    assert_eq!(
        fake.sse_accept.lock().unwrap().as_deref(),
        Some("text/event-stream"),
        "/api/event serves nothing without the SSE Accept header"
    );

    // The captured lifecycle: started → step.started → text delta → full-text
    // snapshot (dedups to nothing) → succeeded settles Completed.
    v2_assistant_message(&fake, "ses_test", "msg_1");
    fake.emit_v2(
        "session.text.started",
        json!({ "sessionID": "ses_test", "assistantMessageID": "msg_1", "ordinal": 0, "text": "" }),
    );
    fake.emit_v2(
        "session.text.delta",
        json!({ "sessionID": "ses_test", "assistantMessageID": "msg_1", "ordinal": 0, "delta": "Hello" }),
    );
    let text = next_event(&mut stream).await;
    assert!(matches!(&text, AgentEvent::TextDelta { text } if text == "Hello"));
    fake.emit_v2(
        "session.text.ended",
        json!({ "sessionID": "ses_test", "assistantMessageID": "msg_1", "ordinal": 0, "text": "Hello" }),
    );
    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.first(),
        Some(AgentEvent::AssistantMessageCompleted { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            session_id: Some(sid),
            ..
        }) if sid == "ses_test"
    ));
}

#[tokio::test]
async fn v2_slash_command_sends_name_and_text() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("/init the repo"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;

    let commands = wait_calls(&fake, "POST", "/api/session/ses_test/command", 1).await;
    assert_eq!(commands[0], json!({ "name": "init", "text": "the repo" }));
    assert!(
        fake.calls_to("POST", "/api/session/ses_test/prompt").is_empty(),
        "a known slash command must not also post a prompt"
    );

    v2_assistant_message(&fake, "ses_test", "msg_1");
    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_subagent_progress_binds_the_child_and_success_settles_the_chip() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("spawn"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    // 2.0.8 renamed the spawn tool `subagent` (the name rides input.started;
    // the args arrive on `called`).
    fake.emit_v2(
        "session.tool.input.started",
        json!({ "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1", "name": "subagent" }),
    );
    fake.emit_v2(
        "session.tool.called",
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1",
            "input": { "agent": "explore", "description": "Check the repo", "prompt": "go" },
            "executed": false,
        }),
    );
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::ToolCall { id, call: ToolCall::Unknown { name, .. } }
            if id == "ses_test:msg_1:call_1" && name == "Agent: Check the repo"
    ));

    // The child session and its late `progress` metadata bind the chip;
    // child traffic streams tagged with it.
    fake.emit_v2(
        "session.created",
        json!({
            "sessionID": "ses_child", "parentID": "ses_test",
            "title": "Check the repo", "agent": "explore",
        }),
    );
    fake.emit_v2(
        "session.tool.progress",
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1",
            "metadata": { "sessionID": "ses_child", "status": "running" },
        }),
    );
    fake.emit_v2(
        "session.step.started",
        json!({ "sessionID": "ses_child", "assistantMessageID": "msg_child" }),
    );
    fake.emit_v2(
        "session.text.ended",
        json!({ "sessionID": "ses_child", "assistantMessageID": "msg_child", "ordinal": 0, "text": "done" }),
    );
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Subagent { parent_tool_use_id, event }
            if parent_tool_use_id == "ses_test:msg_1:call_1"
                && matches!(&**event, AgentEvent::TextDelta { text } if text == "done")
    ));

    // The terminal `success` folds its content into the chip and settles the
    // tagged child.
    fake.emit_v2(
        "session.tool.success",
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1",
            "content": [{ "text": "done" }],
            "metadata": { "sessionID": "ses_child", "status": "completed", "truncated": false },
        }),
    );
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::ToolResult { id, is_error: false, output: Some(output), .. }
            if id == "ses_test:msg_1:call_1" && output == "done"
    ));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Subagent { parent_tool_use_id, event }
            if parent_tool_use_id == "ses_test:msg_1:call_1"
                && matches!(&**event, AgentEvent::Done { status: DoneStatus::Completed, .. })
    ));

    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_subagent_failure_reaches_the_chip_with_its_metadata() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("spawn"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    fake.emit_v2(
        "session.tool.input.started",
        json!({ "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1", "name": "subagent" }),
    );
    fake.emit_v2(
        "session.tool.called",
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1",
            "input": { "agent": "broken", "description": "Say hi", "prompt": "say hi" },
            "executed": false,
        }),
    );
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::ToolCall { call: ToolCall::Unknown { name, .. }, .. } if name == "Agent: Say hi"
    ));

    fake.emit_v2(
        "session.created",
        json!({ "sessionID": "ses_child", "parentID": "ses_test", "title": "Say hi", "agent": "broken" }),
    );
    fake.emit_v2(
        "session.tool.progress",
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1",
            "metadata": { "sessionID": "ses_child", "status": "running" },
        }),
    );
    // The captured failure frame: opencode wraps the child provider error in
    // its own tool-error message, and the metadata still names the child —
    // the chip binds on failure too.
    fake.emit_v2(
        "session.tool.failed",
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1", "id": "call_1",
            "error": {
                "type": "tool.execution",
                "message": "Subagent failed (sessionID: ses_child): capture stub: deliberate 500",
            },
            "metadata": { "sessionID": "ses_child", "status": "running" },
            "executed": false,
        }),
    );
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::ToolResult { id, is_error: true, output: Some(output), .. }
            if id == "ses_test:msg_1:call_1" && output.contains("Subagent failed")
    ));
    let ev = next_event(&mut stream).await;
    assert!(matches!(
        &ev,
        AgentEvent::Subagent { parent_tool_use_id, event }
            if parent_tool_use_id == "ses_test:msg_1:call_1"
                && matches!(&**event, AgentEvent::Done { status: DoneStatus::Errored, .. })
    ));

    // The parent turn itself was not poisoned by the child's failure.
    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_interrupt_uses_the_interrupt_route_and_settles_interrupted() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    token.cancel();
    wait_calls(&fake, "POST", "/api/session/ses_test/interrupt", 1).await;
    fake.emit_v2(
        "session.execution.interrupted",
        json!({ "sessionID": "ses_test" }),
    );
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Interrupted,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_steer_queues_mid_turn_and_delivers_at_idle() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;

    v2_assistant_message(&fake, "ses_test", "msg_1");
    fake.emit_v2(
        "session.text.delta",
        json!({ "sessionID": "ses_test", "assistantMessageID": "msg_1", "ordinal": 0, "delta": "working" }),
    );
    let _ = next_event(&mut stream).await; // TextDelta

    steer
        .send(SteerMessage {
            prompt: "also do this".into(),
            message_id: None,
        })
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    v2_idle(&fake, "ses_test");

    let ev = next_event(&mut stream).await;
    assert!(
        matches!(&ev, AgentEvent::Steered { .. }),
        "queued steer must continue the run at the turn boundary, got {ev:?}"
    );
    let prompts = wait_calls(&fake, "POST", "/api/session/ses_test/prompt", 2).await;
    assert_eq!(prompts[1], json!({ "text": "also do this", "files": [] }));

    v2_assistant_message(&fake, "ses_test", "msg_2");
    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_resume_reuses_the_durable_session() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut req = request("continue");
    req.resume = Some("ses_resume".into());
    let mut stream = harness(&fake)
        .run(req, controls)
        .await
        .expect("run starts");
    let started = next_event(&mut stream).await;
    assert!(matches!(
        &started,
        AgentEvent::SessionStarted { session_id, .. } if session_id == "ses_resume"
    ));
    let _ = next_event(&mut stream).await;
    wait_calls(&fake, "POST", "/api/session/ses_resume/prompt", 1).await;

    v2_assistant_message(&fake, "ses_resume", "msg_1");
    v2_idle(&fake, "ses_resume");
    drain_to_done(&mut stream).await;
}

#[tokio::test]
async fn v2_permission_reply_sends_the_decision_body() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    fake.emit_v2(
        "permission.asked",
        json!({
            "sessionID": "ses_test", "id": "per_1", "type": "external_directory",
            "pattern": "/tmp/**", "title": "Access outside the workspace",
        }),
    );
    let reply = wait_calls(
        &fake,
        "POST",
        "/api/session/ses_test/permission/per_1/reply",
        1,
    )
    .await;
    assert_eq!(reply[0], json!({ "decision": "once" }));

    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_forms_flow_through_the_input_bridge_by_field_key() {
    let fake = FakeOpencode::start_v2().await;
    let form = json!({
        "id": "frm_1", "sessionID": "ses_test", "title": "Capture form",
        "fields": [
            { "key": "environment", "title": "Environment",
              "description": "Which environment should this target?",
              "required": true, "type": "string",
              "options": [
                  { "value": "staging", "label": "Staging" },
                  { "value": "prod", "label": "Production" }] },
            { "key": "tags", "title": "Tags", "type": "multiselect",
              "options": [
                  { "value": "alpha", "label": "Alpha" },
                  { "value": "beta", "label": "Beta" }] },
            { "key": "confirm", "title": "Confirm?", "type": "boolean" },
        ],
    });
    fake.set_forms(json!([form.clone()]));
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("ask me"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    // The captured frame nests the whole record under `data.form`; the
    // pending list is fetched from the session it names.
    fake.emit_v2("form.created", json!({ "form": form.clone() }));
    let reply = wait_calls(&fake, "POST", "/api/session/ses_test/form/frm_1/reply", 1).await;
    assert_eq!(
        reply[0],
        json!({ "answer": {
            "environment": "staging",
            "tags": ["alpha"],
            "confirm": true,
        }})
    );
    let ev = next_event(&mut stream).await;
    assert!(matches!(&ev, AgentEvent::InputResolved { request_id } if request_id == "frm_1"));

    // The server's own `form.replied` echo (flat `{id, sessionID}`) closes
    // nothing the panel hasn't already released.
    fake.emit_v2(
        "form.replied",
        json!({
            "id": "frm_1", "sessionID": "ses_test",
            "answer": { "environment": "staging" },
        }),
    );
    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::InputResolved { request_id } if request_id == "frm_1")),
        "the form must close on our panel"
    );
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_unanswerable_form_is_deleted_and_chipped() {
    let fake = FakeOpencode::start_v2().await;
    let form = json!({
        "id": "frm_2", "sessionID": "ses_test", "title": "Sign in",
        "fields": [{
            "key": "login", "type": "external", "title": "Login",
            "url": "https://example.com/auth",
        }],
    });
    fake.set_forms(json!([form.clone()]));
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("ask me"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    fake.emit_v2("form.created", json!({ "form": form.clone() }));
    let ev = next_event(&mut stream).await;
    assert!(
        matches!(&ev, AgentEvent::Error { message } if message.contains("auth flow")),
        "an unanswerable form must surface a chip, got {ev:?}"
    );
    wait_calls(&fake, "DELETE", "/api/session/ses_test/form/frm_2", 1).await;
    let ev = next_event(&mut stream).await;
    assert!(matches!(&ev, AgentEvent::InputResolved { request_id } if request_id == "frm_2"));

    v2_idle(&fake, "ses_test");
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Completed,
            ..
        })
    ));
}

#[tokio::test]
async fn v2_provider_retries_feed_the_ladder_and_cap_out() {
    let fake = FakeOpencode::start_v2().await;
    let (controls, _steer, _token) = controls();
    let mut stream = harness(&fake)
        .run(request("hi"), controls)
        .await
        .expect("run starts");
    opening(&mut stream).await;
    v2_assistant_message(&fake, "ses_test", "msg_1");

    // The captured retry frame carries the provider error verbatim.
    let retry = |attempt: u64| {
        json!({
            "sessionID": "ses_test", "assistantMessageID": "msg_1",
            "attempt": attempt, "at": 0,
            "error": {
                "type": "provider.internal",
                "message": "capture stub: deliberate 500",
                "status": 500,
            },
        })
    };
    fake.emit_v2("session.retry.scheduled", retry(3));
    let ev = next_event(&mut stream).await;
    let AgentEvent::Error { message } = &ev else {
        panic!("expected a retry error chip, got {ev:?}");
    };
    assert!(
        message.contains("retrying") && message.contains("attempt 3"),
        "{message}"
    );
    assert!(message.contains("deliberate 500"), "{message}");

    fake.emit_v2("session.retry.scheduled", retry(8));
    let ev = next_event(&mut stream).await;
    let AgentEvent::Error { message } = &ev else {
        panic!("expected the give-up chip, got {ev:?}");
    };
    assert!(message.contains("Giving up"), "{message}");
    wait_calls(&fake, "POST", "/api/session/ses_test/interrupt", 1).await;

    // The captured terminal failure: error + idle in one frame. The raw
    // provider message lands in the Done error (the give-up chip above
    // already carried the retry summary).
    fake.emit_v2(
        "session.execution.failed",
        json!({
            "sessionID": "ses_test",
            "error": {
                "type": "provider.internal",
                "message": "capture stub: deliberate 500",
                "status": 500,
            },
        }),
    );
    let events = drain_to_done(&mut stream).await;
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done {
            status: DoneStatus::Errored,
            error: Some(e),
            ..
        }) if e.contains("deliberate 500")
    ));
}

#[tokio::test]
async fn v2_catalog_polls_through_the_empty_warmup() {
    // models.dev syncs after the health endpoint opens: the server answers
    // an empty `/api/model` until it lands. The driver must poll instead of
    // believing the first empty list.
    let fake = FakeOpencode::start_v2().await;
    *fake.catalog_empty_left.lock().unwrap() = 2;
    let models = harness(&fake).models().await.expect("models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "opencode/test-model");
}

#[tokio::test]
async fn slash_command_rejects_attachments_instead_of_dropping_them() {
    let fake = FakeOpencode::start().await;
    let (controls, _steer, _token) = controls();
    let mut req = request("/init the repo");
    req.attachments.push("/tmp/image.png".into());
    let mut stream = harness(&fake).run(req, controls).await.unwrap();
    let events = drain_to_done(&mut stream).await;
    assert!(events.iter().any(|event| matches!(event,
        AgentEvent::Done { status: DoneStatus::Errored, error: Some(message), .. }
        if message.contains("attachments")
    )));
    assert!(fake.posts_to("/session/ses_test/command").is_empty());
    assert!(fake.posts_to("/session/ses_test/prompt_async").is_empty());
}

#[tokio::test]
async fn dollar_selected_skill_uses_opencode_native_command_with_arguments() {
    use zeron_proto::{
        HarnessId,
        invocation::{Invocation, harness_prompt},
    };
    let fake = FakeOpencode::start().await;
    *fake.commands.lock().unwrap() = json!([
        {"name":"review","description":"Native skill","source":"skill"},
        {"name":"init","source":"command"}
    ]);
    let cwd = tempfile::tempdir().unwrap();
    std::fs::create_dir(cwd.path().join(".git")).unwrap();
    let h = harness(&fake);
    let skills = h.skills(cwd.path()).await.unwrap().unwrap();
    let skill = skills
        .into_iter()
        .find(|skill| skill.name == "review")
        .unwrap();
    let invocation = Invocation::Skill {
        name: skill.name,
        path: skill.path,
        command: skill.command,
    };
    let prompt = harness_prompt(
        &format!("\n  {} inspect tests", invocation.link()),
        HarnessId::Opencode,
    );
    assert_eq!(prompt, "\n  /review inspect tests");
    let (controls, _steer, _) = controls();
    let mut stream = h.run(request(&prompt), controls).await.unwrap();
    let _ = next_event(&mut stream).await;
    let _ = next_event(&mut stream).await;
    let commands = wait_posts(&fake, "/session/ses_test/command", 1).await;
    assert_eq!(commands[0]["command"], "review");
    assert_eq!(commands[0]["arguments"], "inspect tests");
    assert!(fake.posts_to("/session/ses_test/prompt_async").is_empty());
    assistant_message(&fake, "ses_test", "msg_1");
    idle(&fake, "ses_test");
    drain_to_done(&mut stream).await;
}
