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

use crate::action::{Action, ActionMetadata, ActionRunOptions, StreamingResponse};
use crate::context::ActionContext;
use crate::error::{Error, Result};
use crate::runtime;
use crate::schema::{self, parse_schema, ProvidedSchema};
use crate::status::StatusCode;
use async_trait::async_trait;
use futures::{FutureExt, StreamExt};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use strum::{Display, EnumString};

// Re-export the Plugin trait to make it accessible via `genkit_core::registry::Plugin`
pub use crate::plugin::Plugin;

/// The type of a runnable action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Display, EnumString)]
#[serde(rename_all = "camelCase")]
#[strum(serialize_all = "lowercase")]
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

#[derive(Default)]
struct RegistryState {
    actions: HashMap<String, Arc<dyn ErasedAction>>,
    plugins: HashMap<String, Arc<dyn Plugin>>,
    schemas: HashMap<String, schema::ProvidedSchema>,
    values: HashMap<String, Arc<dyn Any + Send + Sync>>,
    parent: Option<Registry>,
    /** Additional runtime context data for flows and tools. */
    context: Option<ActionContext>,
    default_model: Option<String>,
    plugins_initialized: bool,
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

impl Registry {
    /// Creates a new, empty `Registry`.
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(RegistryState::default()));
        Self { state }
    }

    /// Creates a new registry that inherits from a parent registry.
    ///
    /// The new registry can override components from the parent or add new ones.
    /// Lookups will fall back to the parent if a component is not found in the child.
    pub fn with_parent(parent: &Registry) -> Self {
        let state = Arc::new(Mutex::new(RegistryState {
            parent: Some(parent.clone()),
            ..Default::default()
        }));
        Self { state }
    }

    /// Sets a global context on the registry. This context will be available to all
    /// actions unless overridden by a more specific context passed to a call.
    pub fn set_context(&self, context: ActionContext) {
        self.state.lock().unwrap().context = Some(context);
    }

    /// Gets the global context from the registry, if it has been set.
    pub fn get_context(&self) -> Option<ActionContext> {
        self.state.lock().unwrap().context.clone()
    }

    /// Initializes all registered plugins if they haven't been already.
    /// This is typically called lazily when an action is looked up within a runtime context.
    async fn initialize_all_plugins(&self) -> Result<()> {
        let plugins_to_init = {
            let mut state = self.state.lock().unwrap();
            if state.plugins_initialized {
                return Ok(());
            }
            // Clone plugins to avoid holding the lock during async operations.
            let plugins = state.plugins.values().cloned().collect::<Vec<_>>();
            // Eagerly set the flag to prevent concurrent initializations.
            state.plugins_initialized = true;
            plugins
        }; // Lock is released here.

        for plugin in plugins_to_init {
            // We need a separate `Registry` clone for each async call because `self` is consumed.
            plugin.initialize(&self.clone()).await?;
        }

        Ok(())
    }

    pub fn set_default_model(&self, name: String) {
        let mut state = self.state.lock().unwrap();
        state.default_model = Some(name);
    }

    pub fn get_default_model(&self) -> Option<String> {
        let state = self.state.lock().unwrap();
        if let Some(model_name) = state.default_model.as_ref() {
            return Some(model_name.clone());
        }
        if let Some(parent) = &state.parent {
            return parent.get_default_model();
        }
        None
    }

    /// Returns the parent registry, if one exists.
    pub fn parent(&self) -> Option<Registry> {
        self.state.lock().unwrap().parent.clone()
    }

    /// Registers an action with the registry.
    ///
    /// The action key is automatically generated from its type and name.
    pub fn register_action<A: ErasedAction + Send + Sync + 'static>(
        &self,
        action_type: ActionType,
        action: A,
    ) -> Result<()> {
        let action_arc: Arc<dyn ErasedAction> = Arc::new(action);
        let mut state = self.state.lock().unwrap();
        let meta = action_arc.metadata();

        if action_type != meta.action_type {
            return Err(Error::new_internal(format!(
                "action type ({:?}) does not match type on action ({:?})",
                action_type, meta.action_type
            )));
        }

        // Use serde_json to serialize the enum to respect `rename_all = "camelCase"`
        let type_str = serde_json::to_value(action_type)?
            .as_str()
            .unwrap_or(&action_type.to_string())
            .to_string();

        let key = format!("/{}/{}", type_str, meta.name);

        if state.actions.contains_key(&key) {
            // In a production framework, you might want to log a warning here.
        }

        state.actions.insert(key, action_arc);
        Ok(())
    }

    /// Looks up an action by its full key (e.g., "/flow/myflow").
    /// If in a runtime context, this will trigger plugin initialization and dynamic resolution.
    pub async fn lookup_action(&self, key: &str) -> Option<Arc<dyn ErasedAction>> {
        // If we're in a runtime context, ensure all plugins are initialized.
        if runtime::is_in_runtime_context() {
            if let Err(e) = self.initialize_all_plugins().await {
                // In a real scenario, you might want to log this error.
                println!("Failed to initialize plugins during lookup: {}", e);
                return None;
            }
        }

        // 1. Check if the action is already registered locally.
        {
            let state = self.state.lock().unwrap();
            if let Some(action) = state.actions.get(key) {
                return Some(action.clone());
            }
        } // Lock is released.

        // 2. If not found and in a runtime context, try to resolve it dynamically.
        if runtime::is_in_runtime_context() {
            let parts: Vec<&str> = key.trim_start_matches('/').splitn(3, '/').collect();
            if parts.len() == 3 {
                let action_type_str = parts[0];
                let plugin_id = parts[1];
                let action_id = parts[2];

                if let Ok(action_type) = ActionType::from_str(action_type_str) {
                    let plugin_to_resolve = {
                        let state = self.state.lock().unwrap();
                        state.plugins.get(plugin_id).cloned()
                    };

                    if let Some(plugin) = plugin_to_resolve {
                        // Attempt to resolve the action. We ignore errors because a plugin
                        // might not implement the resolver, which is a valid case.
                        if plugin
                            .resolve_action(action_type, action_id, &self.clone())
                            .await
                            .is_ok()
                        {
                            // If resolution was attempted, check again if the action now exists.
                            let state = self.state.lock().unwrap();
                            if let Some(action) = state.actions.get(key) {
                                return Some(action.clone());
                            }
                        }
                    }
                }
            }
        }

        // 3. If still not found, check the parent registry.
        let parent = {
            let state = self.state.lock().unwrap();
            state.parent.clone()
        };

        if let Some(parent) = parent {
            return Box::pin(parent.lookup_action(key)).await;
        }

        None
    }

    /// Returns a map of all registered actions, including those from parent registries.
    ///
    /// Child actions take precedence over parent actions with the same key.
    pub async fn list_actions(&self) -> HashMap<String, Arc<dyn ErasedAction>> {
        let (mut actions, parent) = {
            let state = self.state.lock().unwrap();
            (state.actions.clone(), state.parent.clone())
        };

        if let Some(parent) = parent {
            let parent_actions = Box::pin(parent.list_actions()).await;
            for (key, action) in parent_actions {
                // This ensures that child actions are not overwritten by parent actions.
                actions.entry(key).or_insert(action);
            }
        }

        actions
    }

    /// Registers a plugin with the registry.
    pub async fn register_plugin(&mut self, plugin: Arc<dyn Plugin>) -> Result<()> {
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

    /// Registers a generic value with the registry, keyed by type and name.
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

    /// Registers a type-erased value with the registry.
    pub fn register_any(&self, value_type: &str, name: &str, value: Arc<dyn Any + Send + Sync>) {
        let key = format!("{}/{}", value_type, name);
        self.state.lock().unwrap().values.insert(key, value);
    }

    /// Looks up a type-erased value from the registry.
    pub async fn lookup_any(
        &self,
        value_type: &str,
        name: &str,
    ) -> Option<Arc<dyn Any + Send + Sync>> {
        let key = format!("{}/{}", value_type, name);
        let parent = {
            let state = self.state.lock().unwrap();
            if let Some(value) = state.values.get(&key) {
                return Some(value.clone());
            }
            state.parent.clone()
        };

        if let Some(parent) = parent {
            return Box::pin(parent.lookup_any(value_type, name)).await;
        }

        None
    }

    /// Looks up a generic value from the registry.
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
}
