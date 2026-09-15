//! Anthropic Messages API wire types + conversion to/from the internal form.

use crate::openai::{ChatMessage, Content};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<SystemPrompt>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<TextBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: Content,
}

impl SystemPrompt {
    pub fn as_text(&self) -> String {
        match self {
            SystemPrompt::Text(t) => t.clone(),
            SystemPrompt::Blocks(b) => b
                .iter()
                .filter_map(|x| x.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl MessagesRequest {
    /// Flatten to the OpenAI-style message list the engine consumes.
    pub fn to_chat_messages(&self) -> Vec<ChatMessage> {
        let mut out = Vec::new();
        if let Some(sys) = &self.system {
            let text = sys.as_text();
            if !text.is_empty() {
                out.push(ChatMessage {
                    role: "system".into(),
                    content: Some(Content::Text(text)),
                });
            }
        }
        for m in &self.messages {
            out.push(ChatMessage {
                role: m.role.clone(),
                content: Some(m.content.clone()),
            });
        }
        out
    }
}

// ---- responses ----

#[derive(Debug, Clone, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorEnvelope {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub error: AnthropicError,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}

impl ErrorEnvelope {
    pub fn new(kind: &str, message: impl Into<String>) -> Self {
        Self {
            kind: "error",
            error: AnthropicError {
                kind: kind.to_string(),
                message: message.into(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_system_and_messages() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"model":"claude","system":"be nice","max_tokens":64,
                "messages":[{"role":"user","content":"hello"}]}"#,
        )
        .unwrap();
        let msgs = req.to_chat_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content.as_ref().unwrap().as_text(), "be nice");
        assert_eq!(msgs[1].role, "user");
    }

    #[test]
    fn accepts_block_system_prompt() {
        let req: MessagesRequest = serde_json::from_str(
            r#"{"system":[{"type":"text","text":"a"},{"type":"text","text":"b"}],
                "messages":[{"role":"user","content":"x"}]}"#,
        )
        .unwrap();
        assert_eq!(req.system.unwrap().as_text(), "a\nb");
    }

    #[test]
    fn error_envelope_shape() {
        let e = serde_json::to_value(ErrorEnvelope::new("invalid_request_error", "nope")).unwrap();
        assert_eq!(e["type"], "error");
        assert_eq!(e["error"]["type"], "invalid_request_error");
    }
}
