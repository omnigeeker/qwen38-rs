//! OpenAI-compatible wire types (the subset a local engine must serve).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Part {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl Content {
    pub fn as_text(&self) -> String {
        match self {
            Content::Text(t) => t.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSeq>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSeq {
    One(String),
    Many(Vec<String>),
}

impl StopSeq {
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StopSeq::One(s) => vec![s.clone()],
            StopSeq::Many(v) => v.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiError {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub code: String,
}

impl ApiError {
    pub fn new(message: impl Into<String>, kind: &str, code: &str) -> Self {
        Self {
            error: ErrorBody {
                message: message.into(),
                kind: kind.to_string(),
                code: code.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_text_and_parts_content() {
        let a: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#)
                .unwrap();
        assert_eq!(a.messages[0].content.as_ref().unwrap().as_text(), "hi");

        let b: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
        )
        .unwrap();
        assert_eq!(b.messages[0].content.as_ref().unwrap().as_text(), "ab");
    }

    #[test]
    fn reads_sampling_controls() {
        let r: ChatCompletionRequest = serde_json::from_str(
            r#"{"messages":[],"temperature":0.7,"top_p":0.9,"top_k":40,"max_tokens":128,"stream":true,"seed":3}"#,
        )
        .unwrap();
        assert_eq!(r.temperature, Some(0.7));
        assert_eq!(r.top_k, Some(40));
        assert_eq!(r.max_tokens, Some(128));
        assert_eq!(r.stream, Some(true));
    }
}
