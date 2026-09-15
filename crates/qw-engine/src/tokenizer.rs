//! Tokenizer + chat template.
//!
//! `tokenizer.json` (shipped with the checkpoint) drives the BPE; the chat
//! template is the Qwen ChatML form rendered directly in Rust (avoids pulling a
//! Jinja engine into the inference path).

use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
    pub eos_ids: Vec<u32>,
    pub bos_id: Option<u32>,
    pub im_start_id: u32,
    pub im_end_id: u32,
    pub think_start_id: Option<u32>,
    pub think_end_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
    fn role_str(&self) -> &'static str {
        match self.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl Tokenizer {
    pub fn from_file(p: &Path) -> Result<Self> {
        let inner = HfTokenizer::from_file(p).map_err(|e| anyhow!("load {}: {e}", p.display()))?;
        let find = |s: &str| inner.token_to_id(s);
        let im_start_id = find("<|im_start|>").unwrap_or(248045);
        let im_end_id = find("<|im_end|>").unwrap_or(248046);
        let bos_id = find("<|endoftext|>");
        let think_start_id = find(" thinking");
        let think_end_id = find("<｜end▁of▁thinking｜>");
        Ok(Self {
            inner,
            eos_ids: vec![im_end_id, 248044],
            bos_id,
            im_start_id,
            im_end_id,
            think_start_id,
            think_end_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow!("detokenize: {e}"))
    }

    /// Render the Qwen ChatML prompt for a conversation.
    pub fn apply_chat_template(&self, messages: &[Message]) -> String {
        let mut out = String::new();
        for m in messages {
            out.push_str("<|im_start|>");
            out.push_str(m.role_str());
            out.push('\n');
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
        out
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_ids.contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> std::path::PathBuf {
        Path::new("../../models/Qwen3.8-27B-4bit").to_path_buf()
    }

    #[test]
    fn roundtrips_text_when_model_present() {
        let p = model_dir().join("tokenizer.json");
        if !p.exists() {
            eprintln!("skipping: tokenizer not downloaded");
            return;
        }
        let t = Tokenizer::from_file(&p).unwrap();
        let ids = t.encode("Hello, 世界!", false).unwrap();
        assert!(!ids.is_empty());
        let back = t.decode(&ids, false).unwrap();
        assert_eq!(back, "Hello, 世界!");
        println!("vocab={} ids={:?}", t.vocab_size(), ids);
    }

    #[test]
    fn chat_template_has_chatml_markers() {
        let p = model_dir().join("tokenizer.json");
        if !p.exists() {
            return;
        }
        let t = Tokenizer::from_file(&p).unwrap();
        let prompt = t.apply_chat_template(&[Message::user("hi")]);
        assert!(prompt.starts_with("<|im_start|>user\nhi<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        let ids = t.encode(&prompt, false).unwrap();
        assert_eq!(ids[0], t.im_start_id);
    }
}
