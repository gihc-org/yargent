//! LLM backend abstraction.
//!
//! Defines [`LLMProvider`], the trait every backend implements, plus two
//! concrete implementations:
//!
//! - [`OpenAICompatProvider`] speaks the `/v1/chat/completions` dialect used by
//!   DeepSeek, OpenAI, Groq, OpenRouter, Together, Mistral, lm-studio and Ollama.
//! - [`AnthropicProvider`] speaks the Anthropic Messages API, which differs in
//!   several ways: `system` is a top-level field rather than a message,
//!   `max_tokens` is required, auth uses `x-api-key` instead of `Bearer`, and
//!   streaming uses a different event shape.

use std::pin::Pin;

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

/// A single chat message in a conversation.
///
/// `role` is one of `"system"`, `"user"`, or `"assistant"` — the same vocabulary
/// used by every LLM chat API we currently target. We keep it as a `String`
/// rather than an enum so we can serialize directly to JSON without a custom
/// `Serialize` impl, and so new roles ("tool", "function", "developer", …) don't
/// require a code change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
}

/// A stream of incremental text chunks produced by an LLM.
///
/// Boxed and `Pin`ned so it can be returned from a trait method as a runtime
/// type. Each item is either a piece of generated text or an error if the
/// stream encountered a problem mid-flight (network blip, malformed payload).
pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<String>> + Send>>;

/// Backend-agnostic interface for chat-completion LLMs.
///
/// Implementations are responsible for translating our generic [`Message`]
/// list into whatever wire format the upstream API expects, and for parsing
/// streaming responses back into a [`ChunkStream`].
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Send `messages` to the model and return a stream of text chunks.
    ///
    /// The returned stream completes when the model finishes generating. Errors
    /// from the HTTP handshake (auth, 4xx/5xx) surface as `Err(...)` from
    /// this method itself; errors that occur mid-stream surface as `Err` items
    /// inside the stream.
    async fn complete_stream(&self, messages: &[Message]) -> Result<ChunkStream>;
}

/// Backend that speaks the OpenAI `/v1/chat/completions` dialect.
///
/// Works unchanged with DeepSeek, OpenAI, Groq, OpenRouter, Together, Mistral,
/// lm-studio and Ollama — they differ only in base URL and API key.
pub struct OpenAICompatProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenAICompatProvider {
    /// Construct a provider against an arbitrary OpenAI-compatible endpoint.
    pub fn new(base_url: impl Into<String>, api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key,
            model,
        }
    }

}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
}

#[async_trait]
impl LLMProvider for OpenAICompatProvider {
    async fn complete_stream(&self, messages: &[Message]) -> Result<ChunkStream> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = ChatRequest { model: &self.model, messages, stream: true };

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("HTTP request to LLM provider failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("provider returned HTTP {status}: {body}");
        }

        let mut byte_stream = resp.bytes_stream();

        // The OpenAI streaming wire format is Server-Sent Events:
        //     data: {"choices":[{"delta":{"content":"Hel"}}]}\n
        //     data: {"choices":[{"delta":{"content":"lo"}}]}\n
        //     data: [DONE]\n
        // TCP doesn't guarantee that one read = one line, so we buffer raw
        // bytes and split on '\n' ourselves, parsing each `data:` payload.
        let stream = async_stream::try_stream! {
            let mut buffer = String::new();
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes.context("stream read failed")?;
                let s = std::str::from_utf8(&bytes).context("non-utf8 in SSE chunk")?;
                buffer.push_str(s);

                while let Some(nl) = buffer.find('\n') {
                    let line: String = buffer.drain(..=nl).collect();
                    let line = line.trim_end();
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    if data == "[DONE]" { return; }

                    let parsed: StreamChunk = match serde_json::from_str(data) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    if let Some(text) = parsed
                        .choices
                        .into_iter()
                        .next()
                        .and_then(|c| c.delta.content)
                    {
                        if !text.is_empty() {
                            yield text;
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

/// Backend for Anthropic's Messages API (Claude models).
///
/// The wire format differs from OpenAI in several ways the trait hides from
/// callers: system prompts are extracted from the message list into a top-level
/// `system` field, `max_tokens` is required, and authentication uses the
/// `x-api-key` header rather than `Authorization: Bearer ...`.
pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    /// Construct a provider against an Anthropic-compatible endpoint.
    ///
    /// `base_url` is typically `https://api.anthropic.com`; passing a different
    /// URL lets you point at a proxy that speaks the same protocol.
    /// `max_tokens` defaults to 8192, generous enough for code-shaped responses
    /// while still bounded. Anthropic requires the field on every request.
    pub fn new(base_url: impl Into<String>, api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key,
            model,
            max_tokens: 8192,
        }
    }
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<&'a Message>,
    stream: bool,
}

/// Pull every `role == "system"` entry out of `messages` and concatenate their
/// contents (with blank lines between) into a single Anthropic `system` field.
/// The remaining `user`/`assistant` messages are returned in order.
fn split_system(messages: &[Message]) -> (Option<String>, Vec<&Message>) {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut rest: Vec<&Message> = Vec::new();
    for m in messages {
        if m.role == "system" {
            system_parts.push(&m.content);
        } else {
            rest.push(m);
        }
    }
    let system = (!system_parts.is_empty()).then(|| system_parts.join("\n\n"));
    (system, rest)
}

#[derive(Deserialize)]
struct AnthropicEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<AnthropicEventDelta>,
    #[serde(default)]
    error: Option<AnthropicErrorBody>,
}

#[derive(Deserialize)]
struct AnthropicEventDelta {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicErrorBody {
    message: String,
}

#[async_trait]
impl LLMProvider for AnthropicProvider {
    async fn complete_stream(&self, messages: &[Message]) -> Result<ChunkStream> {
        let (system, msgs) = split_system(messages);

        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            system,
            messages: msgs,
            stream: true,
        };

        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .context("HTTP request to Anthropic failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic returned HTTP {status}: {body}");
        }

        let mut byte_stream = resp.bytes_stream();

        // Anthropic's SSE format carries one JSON envelope per `data:` line with
        // a `type` field tagging the event. We only care about `content_block_delta`
        // (whose `delta.text` is the next chunk of generated text), `message_stop`
        // (end of stream), and `error` (mid-stream failure). All other event types
        // — message_start, content_block_start/stop, message_delta, ping — carry
        // metadata we don't need yet and are silently ignored.
        let stream = async_stream::try_stream! {
            let mut buffer = String::new();
            while let Some(bytes) = byte_stream.next().await {
                let bytes = bytes.context("stream read failed")?;
                let s = std::str::from_utf8(&bytes).context("non-utf8 in SSE chunk")?;
                buffer.push_str(s);

                while let Some(nl) = buffer.find('\n') {
                    let line: String = buffer.drain(..=nl).collect();
                    let line = line.trim_end();
                    let Some(data) = line.strip_prefix("data: ") else { continue };

                    let event: AnthropicEvent = match serde_json::from_str(data) {
                        Ok(e) => e,
                        Err(_) => continue,
                    };

                    match event.event_type.as_str() {
                        "content_block_delta" => {
                            if let Some(d) = event.delta {
                                if d.delta_type == "text_delta" {
                                    if let Some(text) = d.text {
                                        if !text.is_empty() {
                                            yield text;
                                        }
                                    }
                                }
                            }
                        }
                        "message_stop" => return,
                        "error" => {
                            let msg = event
                                .error
                                .map(|e| e.message)
                                .unwrap_or_else(|| "unknown anthropic error".into());
                            Err(anyhow::anyhow!("anthropic stream error: {msg}"))?;
                        }
                        _ => {}
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}
