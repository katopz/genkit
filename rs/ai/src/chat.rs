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
        // Take the new messages for this instance.
        let new_messages = std::mem::take(&mut request_base.messages);

        // Mark them all as preamble for this chat instance.
        let new_preamble_messages: Vec<MessageData> = new_messages
            .into_iter()
            .map(|mut msg| {
                msg.metadata
                    .get_or_insert_with(Default::default)
                    .insert("preamble".to_string(), Value::Bool(true));
                msg
            })
            .collect();

        // Check if a new system prompt is being provided.
        let has_new_system_prompt = new_preamble_messages.iter().any(|m| m.role == Role::System);

        // Start the final message list with the new preamble.
        let mut final_messages = new_preamble_messages;

        // Append historical messages, filtering out any old system prompts if a new one was provided.
        final_messages.extend(history.into_iter().filter(|historical_msg| {
            if has_new_system_prompt {
                historical_msg.role != Role::System
            } else {
                true
            }
        }));

        // Set the combined list back on the request.
        request_base.messages = final_messages;

        let state = ChatState {
            thread_name,
            request_base: request_base.clone(),
            messages: request_base.messages.clone(),
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
                        // This is the case for a simple string or Part[] input.
                        generate_options_base.messages.push(MessageData {
                            role: Role::User,
                            content: prompt_parts,
                            metadata: None,
                        });
                    } else {
                        // This is the case for a rendered prompt (GenerateOptions).
                        // We need to merge its properties with the chat's base state.
                        if let Some(mut messages) = resolved_options.messages {
                            generate_options_base.messages.append(&mut messages);
                        }
                        if resolved_options.model.is_some() {
                            generate_options_base.model = resolved_options.model;
                        }
                        if resolved_options.tools.is_some() {
                            generate_options_base.tools = resolved_options.tools;
                        }
                        if resolved_options.tool_choice.is_some() {
                            generate_options_base.tool_choice = resolved_options.tool_choice;
                        }
                        if let Some(new_config) = resolved_options.config {
                            let base_config = generate_options_base
                                .config
                                .get_or_insert_with(|| serde_json::json!({}));
                            if let (Some(base), Some(new)) =
                                (base_config.as_object_mut(), new_config.as_object())
                            {
                                base.extend(new.clone());
                            }
                        }
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

                    println!(
                        "[Chat::send] Generating with messages: {}",
                        serde_json::to_string_pretty(&generate_opts.messages).unwrap_or_default()
                    );
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
                    state.request_base.messages = final_messages.clone();
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
        drop(state); // unlock early

        let mut messages = base_options.messages.clone();
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

        let stream_response = generate_stream(&self.session.registry, generate_options).await?;

        // Spawn a task to update the history once the stream is complete.
        let chat_clone = self.clone();
        let original_response_handle = stream_response.response;
        let new_response_handle = tokio::spawn(async move {
            // Wait for the original generation task to complete.
            let response_result = match original_response_handle.await {
                Ok(res) => res,
                Err(e) => {
                    return Err(genkit_core::error::Error::new_internal(format!(
                        "Stream response task failed: {}",
                        e
                    )))
                }
            };

            // If generation was successful, update the state.
            if let Ok(response) = &response_result {
                let mut state = chat_clone.state.lock().await;

                // Update state with any changes from the generation call (e.g., tools)
                if let Some(req) = &response.request {
                    if let Some(tools) = &req.tools {
                        state.request_base.tools =
                            Some(tools.iter().map(|t| t.name.clone().into()).collect());
                    }
                }

                // Update the message history in the state and persist it
                if let Ok(final_messages) = response.messages() {
                    state.request_base.messages = final_messages.clone();
                    drop(state); // Unlock before async call

                    if let Err(e) = chat_clone.update_messages(&final_messages).await {
                        // We have the final response already, but saving failed.
                        // We'll log the error and return the successful response.
                        // A more robust implementation might return an error here instead.
                        println!("[Chat::send_stream] Failed to update session store: {}", e);
                    }
                }
            }

            // Return the original result, regardless of whether the state update succeeded.
            response_result
        });

        Ok(GenerateStreamResponse {
            stream: stream_response.stream,
            response: new_response_handle,
        })
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

        // Strip transient preamble metadata before persisting.
        self.session
            .update_messages(&state.thread_name, messages)
            .await
    }
}
