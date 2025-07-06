// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Stateful Chat
//!
//! This module provides the `Chat` struct, a high-level API for managing
//! stateful, multi-turn conversations. It is the Rust equivalent of `chat.ts`.

use crate::document::Part;
use crate::generate::{
    generate, generate_stream, BaseGenerateOptions, GenerateOptions, GenerateResponse,
    GenerateStreamResponse,
};
use crate::message::{MessageData, Role};
use crate::session::{run_with_session, Session, SessionUpdater};
use genkit_core::error::Result;
use genkit_core::tracing;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// Constants for tracing attributes.
pub const MAIN_THREAD: &str = "main";
pub const SESSION_ID_ATTR: &str = "genkit:sessionId";
pub const THREAD_NAME_ATTR: &str = "genkit:threadName";

/// Represents the shared, mutable state of a `Chat` instance.
struct ChatState {
    thread_name: String,
    /// Holds the base configuration for the chat, including the model, tools, etc.
    /// This is updated as the conversation progresses (e.g., if a tool changes config).
    request_base: BaseGenerateOptions,
    /// The current, full history of messages in this chat thread.
    messages: Vec<MessageData>,
}

/// Represents an ongoing, stateful chat conversation.
///
/// `Chat` is designed to be cloneable and safe to share across threads.
/// It uses interior mutability (`Arc<Mutex<>>`) to manage its state.
#[derive(Clone)]
pub struct Chat<S> {
    session: Arc<Session<S>>,
    state: Arc<Mutex<ChatState>>,
}

/// Represents the various ways a user can provide input to the `send` method.
pub enum SendInput {
    Text(String),
    Parts(Vec<Part>),
    Options(Box<GenerateOptions>),
}

// From implementations to make `send` more ergonomic.
impl From<&str> for SendInput {
    fn from(s: &str) -> Self {
        SendInput::Text(s.to_string())
    }
}
impl From<String> for SendInput {
    fn from(s: String) -> Self {
        SendInput::Text(s)
    }
}
impl From<Vec<Part>> for SendInput {
    fn from(parts: Vec<Part>) -> Self {
        SendInput::Parts(parts)
    }
}
impl From<GenerateOptions> for SendInput {
    fn from(opts: GenerateOptions) -> Self {
        SendInput::Options(Box::new(opts))
    }
}

/// Normalizes a `SendInput` enum into a `GenerateOptions` struct.
fn resolve_send_options(input: SendInput) -> GenerateOptions {
    match input {
        SendInput::Text(text) => GenerateOptions {
            prompt: Some(vec![Part {
                text: Some(text),
                ..Default::default()
            }]),
            ..Default::default()
        },
        SendInput::Parts(parts) => GenerateOptions {
            prompt: Some(parts),
            ..Default::default()
        },
        SendInput::Options(opts) => *opts,
    }
}

impl<S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static> Chat<S> {
    /// Creates a new `Chat` instance.
    /// This is typically called via `Session::chat()` rather than directly.
    pub fn new(
        session: Arc<Session<S>>,
        mut request_base: BaseGenerateOptions,
        thread_name: String,
        history: Vec<MessageData>,
    ) -> Self {
        // This logic merges a "preamble" (e.g., from a system prompt) with the
        // existing chat history, ensuring the preamble always comes first.
        if !request_base.messages.is_empty() {
            let preamble = request_base
                .messages
                .iter()
                .filter(|m| {
                    m.metadata
                        .as_ref()
                        .and_then(|meta| meta.get("preamble"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                })
                .cloned()
                .collect::<Vec<_>>();

            if !preamble.is_empty() {
                let history_no_preamble = history
                    .into_iter()
                    .filter(|m| {
                        m.metadata
                            .as_ref()
                            .and_then(|meta| meta.get("preamble"))
                            .and_then(|v| v.as_bool())
                            .is_none()
                    })
                    .collect::<Vec<_>>();
                request_base.messages = preamble;
                request_base.messages.extend(history_no_preamble);
            } else {
                let mut full_history = history;
                full_history.extend(request_base.messages);
                request_base.messages = full_history;
            }
        } else {
            request_base.messages = history;
        }

        let state = ChatState {
            thread_name,
            request_base,
            messages: Vec::new(),
        };

        Chat {
            session,
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// Sends a message to the model and gets a response, maintaining history.
    pub async fn send(&self, input: impl Into<SendInput> + Send) -> Result<GenerateResponse> {
        run_with_session(self.session.clone(), async {
            let mut attrs = HashMap::new();
            attrs.insert(
                SESSION_ID_ATTR.to_string(),
                Value::String(self.session.id.clone()),
            );
            attrs.insert(
                THREAD_NAME_ATTR.to_string(),
                Value::String(self.state.lock().await.thread_name.clone()),
            );

            let (response, _telemetry) = tracing::in_new_span(
                "chat.send".to_string(),
                Some(attrs),
                |_trace_context| async {
                    let resolved_options = resolve_send_options(input.into());
                    let mut state = self.state.lock().await;

                    let mut generate_options_base = state.request_base.clone();
                    if let Some(prompt_parts) = resolved_options.prompt {
                        generate_options_base.messages.push(MessageData {
                            role: Role::User,
                            content: prompt_parts,
                            metadata: None,
                        });
                    }

                    // TODO: Wire up streaming callback from resolved_options.
                    let generate_opts = GenerateOptions {
                        model: generate_options_base.model,
                        docs: generate_options_base.docs,
                        messages: Some(generate_options_base.messages),
                        tools: generate_options_base.tools,
                        tool_choice: generate_options_base.tool_choice,
                        config: generate_options_base.config,
                        output: generate_options_base.output,
                        ..Default::default()
                    };

                    let response = generate(&self.session.registry, generate_opts).await?;

                    // Update state with any changes from the generation call.
                    if let Some(req) = &response.request {
                        if let Some(tools) = &req.tools {
                            state.request_base.tools =
                                Some(tools.iter().map(|t| t.name.clone().into()).collect());
                        }
                        // In a full implementation, toolChoice and config would also be updated.
                    }

                    let final_messages = response.messages()?;
                    drop(state); // Unlock before calling async update_messages

                    self.update_messages(&final_messages).await?;

                    Ok(response)
                },
            )
            .await?;
            Ok(response)
        })
        .await
    }

    /// Sends a message and returns a stream of response chunks.
    pub async fn send_stream(
        &self,
        input: impl Into<SendInput> + Send + 'static,
    ) -> Result<GenerateStreamResponse> {
        let resolved_options = resolve_send_options(input.into());
        let state = self.state.lock().await;
        let base_options = state.request_base.clone();

        let mut messages = base_options.messages;
        if let Some(prompt_parts) = resolved_options.prompt {
            messages.push(MessageData {
                role: Role::User,
                content: prompt_parts,
                metadata: None,
            });
        }

        // Convert BaseGenerateOptions to GenerateOptions
        let generate_options = GenerateOptions {
            model: base_options.model,
            messages: Some(messages),
            tools: base_options.tools,
            tool_choice: base_options.tool_choice,
            config: base_options.config,
            output: base_options.output,
            docs: base_options.docs,
            ..Default::default()
        };

        // NOTE: This doesn't yet handle state updates after the stream completes.
        // A more complete implementation would spawn a task to await the final
        // response from the stream and then update the chat's message history.
        generate_stream(&self.session.registry, generate_options).await
    }

    /// Adds a "preamble" message to the beginning of the chat history.
    /// Preamble messages are typically system messages that set the context for the model.
    pub async fn add_preamble(&mut self, message: &MessageData) -> Result<()> {
        let mut state = self.state.lock().await;
        // Mark the message as a preamble message.
        let mut preamble_message = message.clone();
        let mut metadata = preamble_message.metadata.unwrap_or_default();
        metadata.insert("preamble".to_string(), Value::Bool(true));
        preamble_message.metadata = Some(metadata);

        // Insert the preamble at the beginning of the base request messages.
        state.request_base.messages.insert(0, preamble_message);
        // Also update the main message history.
        state.messages.insert(0, message.clone());
        Ok(())
    }

    /// Retrieves the preamble messages from the chat's history.
    pub async fn preamble(&self) -> Vec<MessageData> {
        let state = self.state.lock().await;
        state
            .messages
            .iter()
            .filter(|m| {
                m.metadata
                    .as_ref()
                    .and_then(|meta| meta.get("preamble"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Gets the current list of messages in the chat history.
    pub async fn messages(&self) -> Vec<MessageData> {
        self.state.lock().await.messages.clone()
    }

    /// Updates the message history for the current thread.
    async fn update_messages(&self, messages: &[MessageData]) -> Result<()> {
        let mut state = self.state.lock().await;
        state.messages = messages.to_vec();
        self.session
            .update_messages(&state.thread_name, messages)
            .await
    }
}
