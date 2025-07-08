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

//! # Session Management for Stateful Conversations
//!
//! This module provides the `Session` struct and related components for managing
//! stateful, multi-turn conversations. It is the Rust equivalent of `session.ts`.

use crate::chat::{Chat, MAIN_THREAD};
use crate::generate::{BaseGenerateOptions, GenerateOptions};
use crate::message::MessageData;
use crate::prompt::ExecutablePrompt;
use async_trait::async_trait;
use genkit_core::error::{Error, Result};
use genkit_core::registry::Registry;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

tokio::task_local! {
    /// Task-local storage for the current `Session`.
    ///
    /// The state type `S` is fixed to `Value` here to accommodate different
    /// state types across the application, similar to how TypeScript's `any` works.
    pub static CURRENT_SESSION: Arc<Session<Value>>;
}

/// A data structure representing the persisted state of a session.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionData<S = Value> {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<S>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub threads: HashMap<String, Vec<MessageData>>,
}

/// A trait for session storage backends.
#[async_trait]
pub trait SessionStore<S: Send + Sync>: Send + Sync {
    async fn get(&self, session_id: &str) -> Result<Option<SessionData<S>>>;
    async fn save(&self, session_id: &str, data: &SessionData<S>) -> Result<()>;
}

/// A trait for updating session data.
#[async_trait]
pub trait SessionUpdater<S: Send + Sync>: Send + Sync {
    /// Updates the message history for a given thread and persists it.
    async fn update_messages(&self, thread_name: &str, messages: &[MessageData]) -> Result<()>;
}

/// A simple, in-memory session store for testing and development.
#[derive(Debug, Default)]
pub struct InMemorySessionStore<S> {
    data: Mutex<HashMap<String, SessionData<S>>>,
}

impl<S> InMemorySessionStore<S> {
    pub fn new() -> Self {
        Self {
            data: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl<S: Clone + Send + Sync> SessionStore<S> for InMemorySessionStore<S> {
    async fn get(&self, session_id: &str) -> Result<Option<SessionData<S>>> {
        Ok(self.data.lock().await.get(session_id).cloned())
    }

    async fn save(&self, session_id: &str, data: &SessionData<S>) -> Result<()> {
        self.data
            .lock()
            .await
            .insert(session_id.to_string(), data.clone());
        Ok(())
    }
}

/// Manages the state and history of a multi-turn conversation.
pub struct Session<S = Value> {
    pub id: String,
    store: Arc<dyn SessionStore<S>>,
    data: Mutex<SessionData<S>>,
    pub registry: Arc<Registry>,
}

/// Options for creating a new chat session.
pub struct ChatOptions<'a, I, S> {
    pub thread_name: Option<String>,
    pub preamble: Option<&'a ExecutablePrompt<I>>,
    pub base_options: Option<BaseGenerateOptions>,
    pub prompt_render_input: Option<I>,
    _marker: std::marker::PhantomData<S>,
}

impl<'a, I, S> Default for ChatOptions<'a, I, S> {
    fn default() -> Self {
        Self {
            thread_name: None,
            preamble: None,
            base_options: None,
            prompt_render_input: None,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<S> Session<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    /// Creates a new session, either by creating a new one or loading from a store.
    pub async fn new(
        registry: Arc<Registry>,
        store: Option<Arc<dyn SessionStore<S>>>,
        session_id: Option<String>,
        initial_state: Option<S>,
    ) -> Result<Self> {
        let store = store.unwrap_or_else(|| Arc::new(InMemorySessionStore::new()));
        let id = session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

        let data = match store.get(&id).await? {
            Some(d) => d,
            None => SessionData {
                id: id.clone(),
                state: initial_state,
                threads: HashMap::new(),
            },
        };

        Ok(Self {
            id,
            store,
            data: Mutex::new(data),
            registry,
        })
    }

    /// Returns a clone of the current session state.
    pub async fn state(&self) -> Option<S> {
        self.data.lock().await.state.clone()
    }

    /// Updates the session state and persists it to the store.
    pub async fn update_state(&self, state: S) -> Result<()> {
        let mut data = self.data.lock().await;
        data.state = Some(state);
        self.store.save(&self.id, &data).await
    }

    /// Creates a new `Chat` instance for a specific conversation thread.
    pub async fn chat<'a, I>(
        self: &Arc<Self>,
        options: Option<ChatOptions<'a, I, S>>,
    ) -> Result<Chat<S>>
    where
        I: Serialize + DeserializeOwned + JsonSchema + Default + Send + Sync + Clone + 'static,
    {
        let options = options.unwrap_or_default();
        let thread_name = options
            .thread_name
            .unwrap_or_else(|| MAIN_THREAD.to_string());

        let base_options = if let Some(preamble) = options.preamble {
            preamble
                .render(options.prompt_render_input.unwrap_or_default(), None)
                .await?
        } else {
            let base = options.base_options.unwrap_or_default();
            GenerateOptions {
                model: base.model,
                messages: Some(base.messages),
                docs: base.docs,
                tools: base.tools,
                tool_choice: base.tool_choice,
                config: base.config,
                output: base.output,
                ..Default::default()
            }
        };

        let data = self.data.lock().await;
        let history = data.threads.get(&thread_name).cloned().unwrap_or_default();

        let chat_base_options = BaseGenerateOptions {
            model: base_options.model,
            docs: base_options.docs,
            messages: base_options.messages.unwrap_or_default(),
            tools: base_options.tools,
            tool_choice: base_options.tool_choice,
            config: base_options.config,
            output: base_options.output,
        };

        Ok(Chat::new(
            self.clone(),
            chat_base_options,
            thread_name,
            history,
        ))
    }

    /// Executes a future within the context of this session.
    pub async fn run<F, R>(self: &Arc<Self>, fut: F) -> R
    where
        F: std::future::Future<Output = R>,
    {
        run_with_session(self.clone(), fut).await
    }
}

#[async_trait]
impl<S> SessionUpdater<S> for Session<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    async fn update_messages(&self, thread_name: &str, messages: &[MessageData]) -> Result<()> {
        let mut data = self.data.lock().await;
        data.threads
            .insert(thread_name.to_string(), messages.to_vec());
        self.store.save(&self.id, &data).await
    }
}

/// Executes a future with a given session set as the current task-local session.
pub async fn run_with_session<S, F, R>(session: Arc<Session<S>>, fut: F) -> R
where
    S: Send + Sync + 'static,
    F: std::future::Future<Output = R>,
{
    // Transmute to `Session<Value>` to store in the task-local.
    // This is a trade-off for ergonomic session management without complex generics
    // in the task-local variable itself.
    let session_for_context =
        unsafe { std::mem::transmute::<Arc<Session<S>>, Arc<Session<Value>>>(session) };
    CURRENT_SESSION.scope(session_for_context, fut).await
}

/// Returns the current session from task-local storage, if one is set.
pub fn get_current_session() -> Result<Arc<Session<Value>>> {
    CURRENT_SESSION
        .try_with(|session| session.clone())
        .map_err(|_| {
            Error::new_internal("Not currently running within a session context.".to_string())
        })
}
