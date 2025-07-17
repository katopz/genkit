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

//! # Genkit Registry
//!
//! This module provides the `Registry`, a central component for managing and
//! accessing actions, plugins, and schemas. It is the Rust equivalent of
//! `registry.ts`.
//!
//! In Rust, the `Registry` is designed to be thread-safe using `Arc` for shared
//! ownership and `Mutex` for interior mutability, allowing it to be safely
//! used across asynchronous tasks.

/// A macro to associate a struct with a string identifier for registration purposes.
#[macro_export]
macro_rules! impl_register {
    ($t:ty, $name:tt) => {
        impl $t {
            pub fn type_name() -> &'static str {
                $name
            }
        }
    };
    // Handle generic structs like `MyStruct<T>`.
    ($t:ident < $($p:ident),+ >, $name:tt) => {
        impl<$($p),+> $t<$($p),+> {
            pub fn type_name() -> &'static str {
                $name
            }
        }
    };
}

use crate::action::{Action, ActionMetadata, ActionRunOptions, CloneableError, StreamingResponse};
use crate::context::ActionContext;
use crate::error::{Error, Result};
use crate::runtime;
use crate::schema::{self, parse_schema, ProvidedSchema};
use crate::status::StatusCode;
use async_trait::async_trait;
use futures::future::{try_join_all, BoxFuture, Shared};
use futures::{FutureExt, StreamExt};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::future::Future;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use strum::{Display, EnumString};

// Re-export the Plugin trait to make it accessible via `genkit_core::registry::Plugin`
pub use crate::plugin::Plugin;

/// The type of a runnable action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "kebab-case")]
#[strum(serialize_all = "kebab-case")]
pub enum ActionType {
    Custom,
    Embedder,
    Evaluator,
    ExecutablePrompt,
    Flow,
    Indexer,
    Model,
    BackgroundModel,
    CheckOperation,
    CancelOperation,
    Prompt,
    Reranker,
    Resource,
    Retriever,
    Tool,
    Util,
}

/// A type-erased `Action` that can be stored in the registry.
///
/// This trait allows the registry to hold actions with different generic
/// input and output types in a single collection.
#[async_trait]
pub trait ErasedAction: Send + Sync {
    /// Executes the action with a raw JSON value and optional context.
    async fn run_http_json(&self, input: Value, context: Option<ActionContext>) -> Result<Value>;

    /// Executes a streaming action with a raw JSON value.
    fn stream_http_json(
        &self,
        input: Value,
        context: Option<ActionContext>,
    ) -> Result<StreamingResponse<Value, Value>>;
    /// Returns the name of the action.
    fn name(&self) -> &str;
    /// Returns the metadata for the action.
    fn metadata(&self) -> &ActionMetadata;
    /// Provides a way to downcast to the concrete `Any` type for inspection.
    fn as_any(&self) -> &dyn Any;
}

// A concrete implementation of `ErasedAction` is needed to store our test
// actions in the registry.
#[async_trait]
impl<I, O, S> ErasedAction for Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + Serialize + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
{
    async fn run_http_json(&self, input: Value, context: Option<ActionContext>) -> Result<Value> {
        let parsed_input: I = match self.meta.input_schema.clone() {
            Some(schema_def) => parse_schema(input, ProvidedSchema::FromType(schema_def))?,
            None => serde_json::from_value(input).map_err(|e| {
                Error::new_user_facing(
                    StatusCode::InvalidArgument,
                    format!("Failed to parse input: {}", e),
                    None,
                )
            })?,
        };
        let result = self
            .run(
                parsed_input,
                Some(ActionRunOptions {
                    context,
                    ..Default::default()
                }),
            )
            .await?;
        serde_json::to_value(&result)
            .map_err(|e| Error::new_internal(format!("Failed to serialize action output: {}", e)))
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<ActionContext>,
    ) -> Result<StreamingResponse<Value, Value>> {
        let parsed_input: I = match self.meta.input_schema.clone() {
            Some(schema_def) => parse_schema(input, ProvidedSchema::FromType(schema_def))?,
            None => serde_json::from_value(input).map_err(|e| {
                Error::new_user_facing(
                    StatusCode::InvalidArgument,
                    format!("Failed to parse input: {}", e),
                    None,
                )
            })?,
        };
        let streaming_response = self.stream(
            parsed_input,
            Some(ActionRunOptions {
                context,
                ..Default::default()
            }),
        );

        let value_stream = streaming_response.stream.map(|chunk_result| {
            chunk_result.and_then(|chunk| {
                serde_json::to_value(chunk).map_err(|e| Error::new_internal(e.to_string()))
            })
        });

        let value_output = streaming_response.output.then(|output_result| async {
            output_result.and_then(|output| {
                serde_json::to_value(output).map_err(|e| Error::new_internal(e.to_string()))
            })
        });

        Ok(StreamingResponse {
            stream: Box::pin(value_stream),
            output: Box::pin(value_output),
        })
    }

    fn name(&self) -> &str {
        &self.meta.name
    }

    fn metadata(&self) -> &ActionMetadata {
        &self.meta
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The central registry for Genkit components.
///
/// The `Registry` is responsible for storing and providing access to all
/// registered actions, plugins, and schemas. It supports hierarchical
/// composition, allowing for child registries that inherit from a parent.
#[derive(Clone)]
pub struct Registry {
    /// Using `Arc<Mutex<...>>` allows for thread-safe interior mutability.
    state: Arc<Mutex<RegistryState>>,
}

/// Represents a slot in the registry's action map.
/// An action can either be fully loaded and ready, or represented by a
/// future that will be resolved on first lookup.
enum ActionSlot {
    Ready(Arc<dyn ErasedAction>),
    Lazy(Shared<BoxFuture<'static, Result<Arc<dyn ErasedAction>, CloneableError>>>),
}

impl Clone for ActionSlot {
    fn clone(&self) -> Self {
        match self {
            ActionSlot::Ready(action) => ActionSlot::Ready(action.clone()),
            ActionSlot::Lazy(fut) => ActionSlot::Lazy(fut.clone()),
        }
    }
}

#[derive(Default)]
struct RegistryState {
    actions: HashMap<String, ActionSlot>,
    plugins: HashMap<String, Arc<dyn Plugin>>,
    schemas: HashMap<String, schema::ProvidedSchema>,
    values: HashMap<String, Arc<dyn Any + Send + Sync>>,
    parent: Option<Registry>,
    /** Additional runtime context data for flows and tools. */
    context: Option<ActionContext>,
    default_model: Option<String>,
    // This replaces the old `plugins_initialized: bool`
    initialized_plugins: HashSet<String>,
}

impl Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("Registry")
            .field("actions", &state.actions.keys())
            .field("plugins", &state.plugins.keys())
            .field("schemas", &state.schemas.keys())
            .field("values", &state.values.keys())
            .field("parent", &state.parent)
            .finish()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parses a registry key into its constituent parts.
/// e.g., "/model/foo/something" -> ("model", "foo", "something")
fn parse_key(key: &str) -> Option<(&str, &str, &str)> {
    let mut parts = key.trim_start_matches('/').splitn(3, '/');
    let action_type = parts.next()?;
    let plugin_id = parts.next()?;
    let action_id = parts.next()?;
    if action_id.is_empty() {
        None
    } else {
        Some((action_type, plugin_id, action_id))
    }
}

impl Registry {
    /// Creates a new, empty `Registry`.
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(RegistryState::default()));
        Self { state }
    }

    /// Creates a new registry that inherits from a parent registry.
    pub fn with_parent(parent: &Registry) -> Self {
        let state = Arc::new(Mutex::new(RegistryState {
            parent: Some(parent.clone()),
            ..Default::default()
        }));
        Self { state }
    }

    /// Sets a global context on the registry.
    pub fn set_context(&self, context: ActionContext) {
        self.state.lock().unwrap().context = Some(context);
    }

    /// Gets the global context from the registry.
    pub fn get_context(&self) -> Option<ActionContext> {
        self.state.lock().unwrap().context.clone()
    }

    /// Initializes a single plugin by name, ensuring it only runs once.
    pub async fn initialize_plugin(&self, name: &str) -> Result<()> {
        let plugin_to_init = {
            let mut state = self.state.lock().unwrap();
            if state.initialized_plugins.contains(name) {
                None
            } else if let Some(plugin) = state.plugins.get(name).cloned() {
                state.initialized_plugins.insert(name.to_string());
                Some(plugin.clone())
            } else {
                None
            }
        }; // Lock is dropped here

        if let Some(plugin) = plugin_to_init {
            plugin.initialize(self).await?;
        }
        Ok(())
    }

    /// Initializes all registered plugins if they haven't been already.
    async fn initialize_all_plugins(&self) -> Result<()> {
        let plugin_names = {
            let state = self.state.lock().unwrap();
            state.plugins.keys().cloned().collect::<Vec<String>>()
        }; // Lock is dropped here

        for name in plugin_names {
            self.initialize_plugin(&name).await?;
        }
        Ok(())
    }

    pub fn set_default_model(&self, name: String) {
        self.state.lock().unwrap().default_model = Some(name);
    }

    pub fn get_default_model(&self) -> Option<String> {
        let (model, parent) = {
            let state = self.state.lock().unwrap();
            (state.default_model.clone(), state.parent.clone())
        };
        model.or_else(|| parent.and_then(|p| p.get_default_model()))
    }

    /// Returns the parent registry, if one exists.
    pub fn parent(&self) -> Option<Registry> {
        self.state.lock().unwrap().parent.clone()
    }

    /// Registers an action with the registry.
    pub fn register_action(
        &self,
        action_type: ActionType,
        action: impl ErasedAction + 'static,
    ) -> Result<()> {
        let key = format!("/{}/{}", action_type, action.name());
        self.state
            .lock()
            .unwrap()
            .actions
            .insert(key, ActionSlot::Ready(Arc::new(action)));
        Ok(())
    }

    /// Looks up an action by its full key (e.g., "/flow/myflow").
    pub async fn lookup_action(&self, key: &str) -> Option<Arc<dyn ErasedAction>> {
        let slot = { self.state.lock().unwrap().actions.get(key).cloned() };
        if let Some(slot) = slot {
            return match slot {
                ActionSlot::Ready(action) => Some(action),
                ActionSlot::Lazy(fut) => fut.await.ok(),
            };
        }

        if runtime::is_in_runtime_context() {
            // Per the test, the first lookup in a runtime context initializes ALL plugins.
            if self.initialize_all_plugins().await.is_ok() {
                let slot_after_init = { self.state.lock().unwrap().actions.get(key).cloned() };
                if let Some(slot) = slot_after_init {
                    return match slot {
                        ActionSlot::Ready(action) => Some(action),
                        ActionSlot::Lazy(fut) => fut.await.ok(),
                    };
                }
            }

            // If still not found, try dynamic resolution.
            if let Some((type_str, plugin_id, action_id)) = parse_key(key) {
                if let Some(plugin) = self.lookup_plugin(plugin_id).await {
                    if let Ok(action_type) = ActionType::from_str(type_str) {
                        if plugin
                            .resolve_action(action_type, action_id, self)
                            .await
                            .is_ok()
                        {
                            let slot_after_resolve =
                                { self.state.lock().unwrap().actions.get(key).cloned() };
                            if let Some(slot) = slot_after_resolve {
                                return match slot {
                                    ActionSlot::Ready(action) => Some(action),
                                    ActionSlot::Lazy(fut) => fut.await.ok(),
                                };
                            }
                        }
                    }
                }
            }
        }

        if let Some(parent) = self.parent() {
            return Box::pin(parent.lookup_action(key)).await;
        }

        None
    }

    /// Returns a map of all registered actions.
    pub async fn list_actions(&self) -> HashMap<String, Arc<dyn ErasedAction>> {
        if runtime::is_in_runtime_context() {
            self.initialize_all_plugins().await.unwrap();
        }

        let (mut ready_actions, lazy_keys) = {
            let state = self.state.lock().unwrap();
            let mut ready = HashMap::new();
            let mut lazy = Vec::new();
            for (key, slot) in &state.actions {
                if let ActionSlot::Ready(action) = slot {
                    ready.insert(key.clone(), action.clone());
                } else {
                    lazy.push(key.clone());
                }
            }
            (ready, lazy)
        };

        for key in lazy_keys {
            if let Some(action) = self.lookup_action(&key).await {
                ready_actions.insert(key, action);
            }
        }

        if let Some(parent) = self.parent() {
            let parent_actions = Box::pin(parent.list_actions()).await;
            for (key, action) in parent_actions {
                ready_actions.entry(key).or_insert(action);
            }
        }

        ready_actions
    }

    /// Registers a plugin with the registry.
    pub async fn register_plugin(&self, plugin: Arc<dyn Plugin>) -> Result<()> {
        let name = plugin.name();
        let mut state = self.state.lock().unwrap();
        if state.plugins.contains_key(name) {
            return Err(Error::new_internal(format!(
                "Plugin '{}' is already registered.",
                name
            )));
        }
        state.plugins.insert(name.to_string(), plugin);
        Ok(())
    }

    /// Looks up a plugin by name.
    pub async fn lookup_plugin(&self, name: &str) -> Option<Arc<dyn Plugin>> {
        let (plugin, parent) = {
            let state = self.state.lock().unwrap();
            (state.plugins.get(name).cloned(), state.parent.clone())
        };

        if plugin.is_some() {
            return plugin;
        }

        if let Some(parent) = parent {
            return Box::pin(parent.lookup_plugin(name)).await;
        }

        None
    }

    pub fn register_value<T: 'static + Send + Sync>(
        &mut self,
        value_type: &str,
        name: &str,
        value: T,
    ) {
        let key = format!("{}/{}", value_type, name);
        self.state
            .lock()
            .unwrap()
            .values
            .insert(key, Arc::new(value));
    }

    pub fn register_any(&self, value_type: &str, name: &str, value: Arc<dyn Any + Send + Sync>) {
        let key = format!("{}/{}", value_type, name);
        self.state.lock().unwrap().values.insert(key, value);
    }

    pub async fn lookup_any(
        &self,
        value_type: &str,
        name: &str,
    ) -> Option<Arc<dyn Any + Send + Sync>> {
        let key = format!("{}/{}", value_type, name);
        let (value, parent) = {
            let state = self.state.lock().unwrap();
            (state.values.get(&key).cloned(), state.parent.clone())
        };

        if let Some(v) = value {
            return Some(v);
        }

        if let Some(parent) = parent {
            return Box::pin(parent.lookup_any(value_type, name)).await;
        }

        None
    }

    pub async fn lookup_value<T: Any + Send + Sync>(
        &self,
        value_type: &str,
        name: &str,
    ) -> Option<Arc<T>> {
        self.lookup_any(value_type, name)
            .await
            .and_then(|v| v.downcast().ok())
    }

    pub fn list_values(&self, value_type: &str) -> HashMap<String, Arc<dyn Any + Send + Sync>> {
        let parent = self.state.lock().unwrap().parent.clone();
        let mut all_values = if let Some(parent) = parent {
            parent.list_values(value_type)
        } else {
            HashMap::new()
        };

        let state = self.state.lock().unwrap();
        let prefix = format!("{}/", value_type);
        for (key, value) in &state.values {
            if let Some(name) = key.strip_prefix(&prefix) {
                all_values.insert(name.to_string(), value.clone());
            }
        }
        all_values
    }

    pub async fn list_resolvable_actions(&self) -> Result<HashMap<String, ActionMetadata>> {
        // Run initializers only when in a runtime context.
        if runtime::is_in_runtime_context() {
            self.initialize_all_plugins().await?;
        }

        let parent_actions = if let Some(parent) = self.parent() {
            Box::pin(parent.list_resolvable_actions()).await?
        } else {
            HashMap::new()
        };
        let mut resolvable_actions = parent_actions;

        let plugins = {
            let state = self.state.lock().unwrap();
            state.plugins.values().cloned().collect::<Vec<_>>()
        };

        let plugin_futures = plugins.iter().map(|plugin| plugin.list_actions());
        let plugin_action_lists = try_join_all(plugin_futures).await?;

        for action_list in plugin_action_lists {
            for meta in action_list {
                let key = format!("/{}/{}", meta.action_type, meta.name);
                resolvable_actions.insert(key, meta);
            }
        }

        let registered_actions = self.list_actions().await;
        for (key, action) in registered_actions {
            resolvable_actions.insert(key, (*action.metadata()).clone());
        }

        Ok(resolvable_actions)
    }

    /// Registers a schema with the registry.
    pub fn register_schema(&self, name: &str, schema: schema::ProvidedSchema) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.schemas.contains_key(name) {
            return Err(Error::new_internal(format!(
                "Schema '{}' is already registered.",
                name
            )));
        }
        state.schemas.insert(name.to_string(), schema);
        Ok(())
    }

    /// Looks up a schema by name.
    pub fn lookup_schema(&self, name: &str) -> Option<schema::ProvidedSchema> {
        let (schema, parent) = {
            let state = self.state.lock().unwrap();
            (state.schemas.get(name).cloned(), state.parent.clone())
        };

        if schema.is_some() {
            return schema;
        }

        if let Some(parent) = parent {
            return parent.lookup_schema(name);
        }

        None
    }

    pub fn register_action_async<F>(&self, action_type: ActionType, name: String, fut: F)
    where
        F: Future<Output = Result<Arc<dyn ErasedAction>>> + Send + 'static,
    {
        let key = format!("/{}/{}", action_type, name);
        let mut state = self.state.lock().unwrap();
        if state.actions.contains_key(&key) {
            // In a real implementation, you might want to log a warning.
        }
        let future = async move { fut.await.map_err(CloneableError::from) };
        let box_future: BoxFuture<'static, Result<Arc<dyn ErasedAction>, CloneableError>> =
            Box::pin(future);
        let shared_future = box_future.shared();
        state.actions.insert(key, ActionSlot::Lazy(shared_future));
    }
}
