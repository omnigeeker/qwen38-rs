//! HTTP surface: OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
//! and Anthropic (`/v1/messages`), both served by the same local engine.

use crate::anthropic::{ErrorEnvelope, MessagesRequest};
use crate::engine::{Engine, EngineEvent, Prompt};
use crate::openai::{
    ApiError, ChatCompletionRequest, CompletionRequest, Content, ModelCard, ModelList,
};
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, StreamExt};
use qw_engine::tokenizer::Message;
use std::convert::Infallible;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub model_id: String,
    pub model_dir: String,
    /// Set once the Metal engine is loaded; endpoints report 503 until then.
    pub ready: bool,
    /// Rows of key/value and delta-net state the engine was loaded with.  A
    /// request's prompt plus its completion has to fit inside this.
    pub max_ctx: usize,
    engine: Option<Engine>,
}

impl AppState {
    pub fn new(model_id: impl Into<String>, model_dir: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            model_dir: model_dir.into(),
            ready: false,
            max_ctx: 8192,
            engine: None,
        }
    }

    /// Tell the handlers how much context the engine actually has, so they stop
    /// clamping completions at an arbitrary 1024 that predates the flag.
    pub fn with_max_ctx(mut self, max_ctx: usize) -> Self {
        self.max_ctx = max_ctx;
        self
    }

    /// Attach a loaded engine.  This is what flips `/health` to `ok`.
    pub fn with_engine(mut self, engine: Engine) -> Self {
        self.engine = Some(engine);
        self.ready = true;
        self
    }
}

/// Bind `addr` and serve the router until shutdown.
pub async fn serve(state: AppState, addr: std::net::SocketAddr) -> anyhow::Result<()> {
    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("qwen38 listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn not_ready_openai(msg: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ApiError::new(
            msg,
            "engine_unavailable",
            "engine_not_loaded",
        )),
    )
        .into_response()
}

fn not_ready_anthropic(msg: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorEnvelope::new("overloaded_error", msg)),
    )
        .into_response()
}

fn error_openai(msg: &str, kind: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ApiError::new(msg, "engine_error", kind)),
    )
        .into_response()
}

fn error_anthropic(msg: &str) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorEnvelope::new("api_error", msg)),
    )
        .into_response()
}

pub fn build_router(state: AppState) -> Router {
    let shared = Arc::new(state);
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/messages", post(messages))
        .with_state(shared)
}

async fn health(State(st): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": if st.ready { "ok" } else { "loading" },
        "engine": "qwen38-rs",
        "model": st.model_id,
        "model_dir": st.model_dir,
    }))
}

async fn list_models(State(st): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: st.model_id.clone(),
            object: "model",
            created: now_secs(),
            owned_by: "local".to_string(),
        }],
    })
}

/// OpenAI message roles mapped onto the chat template's message type.
fn to_messages(msgs: &[crate::openai::ChatMessage]) -> Vec<Message> {
    msgs.iter()
        .map(|m| {
            let text = m.content.as_ref().map(Content::as_text).unwrap_or_default();
            match m.role.as_str() {
                "system" => Message::system(text),
                "assistant" => Message::assistant(text),
                _ => Message::user(text),
            }
        })
        .collect()
}

/// The engine reports the prompt length first (for `usage`); the SSE encoders
/// only care about text, so this adapter drops the metadata event.
fn text_only(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Result<EngineEvent, String>>,
) -> tokio::sync::mpsc::UnboundedReceiver<Result<String, String>> {
    let (tx, out) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Some(item) = rx.recv().await {
            let keep = match item {
                Ok(EngineEvent::Prompt(_)) => None,
                Ok(EngineEvent::Piece(p)) => Some(Ok(p)),
                Err(e) => Some(Err(e)),
            };
            if let Some(v) = keep {
                if tx.send(v).is_err() {
                    break;
                }
            }
        }
    });
    out
}

/// `data: ...` chunks in OpenAI's wire format, terminated by `data: [DONE]`.
fn openai_sse(
    rx: tokio::sync::mpsc::UnboundedReceiver<Result<String, String>>,
    id: String,
    model: String,
    created: u64,
) -> Response {
    let id_c = id.clone();
    let model_c = model.clone();
    let body = stream::unfold((rx, true), move |(mut rx, first)| {
        let id = id_c.clone();
        let model = model_c.clone();
        async move {
            match rx.recv().await {
                Some(Ok(piece)) => {
                    let delta = if first {
                        serde_json::json!({ "role": "assistant", "content": piece })
                    } else {
                        serde_json::json!({ "content": piece })
                    };
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
                    });
                    let ev: Result<Event, Infallible> =
                        Ok(Event::default().data(chunk.to_string()));
                    Some((ev, (rx, false)))
                }
                Some(Err(e)) => {
                    let chunk = serde_json::json!({
                        "error": {"message": e, "type": "engine_error"}
                    });
                    let ev: Result<Event, Infallible> =
                        Ok(Event::default().data(chunk.to_string()));
                    Some((ev, (rx, false)))
                }
                None => None,
            }
        }
    })
    .chain(stream::iter(vec![
        Ok(Event::default().data(
            serde_json::json!({
                "id": id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            })
            .to_string(),
        )),
        Ok(Event::default().data("[DONE]")),
    ]));
    Sse::new(body).into_response()
}

/// Anthropic's event stream: `message_start`, then deltas, then `message_stop`.
fn anthropic_sse(
    rx: tokio::sync::mpsc::UnboundedReceiver<Result<String, String>>,
    id: String,
    model: String,
    input_tokens: usize,
) -> Response {
    let start = serde_json::json!({
        "type": "message_start",
        "message": {
            "id": id, "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": serde_json::Value::Null, "stop_sequence": serde_json::Value::Null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
    });
    let block_start = serde_json::json!({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}
    });
    let head: Vec<Result<Event, Infallible>> = vec![
        Ok(Event::default()
            .event("message_start")
            .data(start.to_string())),
        Ok(Event::default()
            .event("content_block_start")
            .data(block_start.to_string())),
    ];
    let body = stream::iter(head)
        .chain(stream::unfold(rx, move |mut rx| async move {
            match rx.recv().await {
                Some(Ok(piece)) => {
                    let ev = serde_json::json!({
                        "type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": piece}
                    });
                    let item: Result<Event, Infallible> = Ok(Event::default()
                        .event("content_block_delta")
                        .data(ev.to_string()));
                    Some((item, rx))
                }
                Some(Err(e)) => {
                    let ev = serde_json::json!({
                        "type": "error", "error": {"type": "api_error", "message": e}
                    });
                    let item: Result<Event, Infallible> =
                        Ok(Event::default().event("error").data(ev.to_string()));
                    Some((item, rx))
                }
                None => None,
            }
        }))
        .chain(stream::iter(vec![
            Ok(Event::default()
                .event("content_block_stop")
                .data(serde_json::json!({"type": "content_block_stop", "index": 0}).to_string())),
            Ok(Event::default()
                .event("message_stop")
                .data(serde_json::json!({"type": "message_stop"}).to_string())),
        ]));
    Sse::new(body).into_response()
}

async fn collect(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Result<EngineEvent, String>>,
) -> Result<(usize, String, usize), String> {
    let mut prompt = 0usize;
    let mut text = String::new();
    let mut n = 0usize;
    while let Some(item) = rx.recv().await {
        match item {
            Ok(EngineEvent::Prompt(p)) => prompt = p,
            Ok(EngineEvent::Piece(piece)) => {
                text.push_str(&piece);
                n += 1;
            }
            Err(e) => return Err(e),
        }
    }
    Ok((prompt, text, n))
}

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let Some(engine) = st.engine.clone() else {
        return not_ready_openai("engine is still loading the 4-bit weights");
    };
    let messages = to_messages(&req.messages);
    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, st.max_ctx);
    let id = format!("chatcmpl-{}", now_secs());
    let created = now_secs();
    let rx = engine.submit(Prompt::Chat(messages), max_tokens);
    if req.stream.unwrap_or(false) {
        return openai_sse(text_only(rx), id, st.model_id.clone(), created);
    }
    let (prompt, text, n) = match collect(rx).await {
        Ok(v) => v,
        Err(e) => return error_openai(&e, "generation_failed"),
    };
    Json(serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": st.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": prompt, "completion_tokens": n, "total_tokens": prompt + n},
    }))
    .into_response()
}

async fn completions(
    State(st): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let Some(engine) = st.engine.clone() else {
        return not_ready_openai("engine is still loading the 4-bit weights");
    };
    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, st.max_ctx);
    let id = format!("cmpl-{}", now_secs());
    let created = now_secs();
    let rx = engine.submit(Prompt::Text(req.prompt), max_tokens);
    if req.stream.unwrap_or(false) {
        return openai_sse(text_only(rx), id, st.model_id.clone(), created);
    }
    let (prompt, text, n) = match collect(rx).await {
        Ok(v) => v,
        Err(e) => return error_openai(&e, "generation_failed"),
    };
    Json(serde_json::json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": st.model_id,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": n, "total_tokens": prompt + n},
    }))
    .into_response()
}

async fn messages(State(st): State<Arc<AppState>>, Json(req): Json<MessagesRequest>) -> Response {
    let Some(engine) = st.engine.clone() else {
        return not_ready_anthropic("engine is still loading the 4-bit weights");
    };
    let messages = to_messages(&req.to_chat_messages());
    let max_tokens = req.max_tokens.unwrap_or(256).clamp(1, st.max_ctx);
    let id = format!("msg_{}", now_secs());
    let rx = engine.submit(Prompt::Chat(messages), max_tokens);
    if req.stream.unwrap_or(false) {
        return anthropic_sse(text_only(rx), id, st.model_id.clone(), 0);
    }
    let (prompt, text, n) = match collect(rx).await {
        Ok(v) => v,
        Err(e) => return error_anthropic(&e),
    };
    Json(serde_json::json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": st.model_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": prompt, "output_tokens": n},
    }))
    .into_response()
}
