//! HTTP surface: OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/models`)
//! and Anthropic (`/v1/messages`).

use crate::anthropic::{ErrorEnvelope, MessagesRequest};
use crate::openai::{ApiError, ChatCompletionRequest, CompletionRequest, ModelCard, ModelList};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub model_id: String,
    pub model_dir: String,
    /// Set once the Metal engine is loaded; endpoints report 503 until then.
    pub ready: bool,
}

impl AppState {
    pub fn new(model_id: impl Into<String>, model_dir: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            model_dir: model_dir.into(),
            ready: false,
        }
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

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    Json(_req): Json<ChatCompletionRequest>,
) -> Response {
    if !st.ready {
        return not_ready_openai("engine is still loading the 4-bit weights");
    }
    not_ready_openai("generation backend not wired yet")
}

async fn completions(
    State(st): State<Arc<AppState>>,
    Json(_req): Json<CompletionRequest>,
) -> Response {
    if !st.ready {
        return not_ready_openai("engine is still loading the 4-bit weights");
    }
    not_ready_openai("generation backend not wired yet")
}

async fn messages(State(st): State<Arc<AppState>>, Json(req): Json<MessagesRequest>) -> Response {
    if !st.ready {
        return not_ready_anthropic("engine is still loading the 4-bit weights");
    }
    let _ = req.to_chat_messages();
    not_ready_anthropic("generation backend not wired yet")
}
