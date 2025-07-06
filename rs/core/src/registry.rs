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

use crate::action::{Action, ActionMetadata, ActionRunOptions, StreamingResponse};
use crate::error::{Error, Result};
use crate::schema::{self, parse_schema, ProvidedSchema};
use crate::status::StatusCode;
use async_trait::async_trait;
// #[cfg(feature = "dotprompt-private")]
// use dotprompt::Dotprompt;
use futures::{FutureExt, StreamExt};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// The type of a runnable action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
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
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<crate::context::ActionContext>,
    ) -> Result<Value>;

    /// Executes a streaming action with a raw JSON value.
    fn stream_http_json(
        &self,
        input: Value,
        context: Option<crate::context::ActionContext>,
    ) -> Result<StreamingResponse<Value, Value>>;
    /// Returns the name of the action.
    fn name(&self) -> &str;
    /// Returns the metadata for the action.
    fn metadata(&self) -> &ActionMetadata;
    /// Provides a way to downcast to the concrete `Any` type for inspection.
    fn as_any(&self) -> &dyn Any;
}

/// A trait for Genkit plugins.
///
/// Plugins are the primary mechanism for extending Genkit with new capabilities,
/// such as integrating with different model providers or services.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Returns the unique name of the plugin.
    fn name(&self) -> &'static str;
    /// Initializes the plugin, registering any actions or other components with the registry.
    async fn initialize(&self, registry: &mut Registry) -> Result<()>;
}

// A concrete implementation of `ErasedAction` is needed to store our test
// actions in the registry.
#[async_trait]
impl<I, O, S> ErasedAction for Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
{
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<crate::context::ActionContext>,
    ) -> Result<Value> {
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
        context: Option<crate::context::ActionContext>,
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
    // #[cfg(feature = "dotprompt-private")]
    // pub dotprompt: Arc<Dotprompt<'static>>,
}

#[derive(Default)]
struct RegistryState {
    actions: HashMap<String, Arc<dyn ErasedAction>>,
    plugins: HashMap<String, Arc<dyn Plugin>>,
    schemas: HashMap<String, schema::ProvidedSchema>,
    values: HashMap<String, Arc<dyn Any + Send + Sync>>,
    parent: Option<Registry>,
    default_model: Option<String>,
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
        // #[cfg(feature = "dotprompt-private")]
        // {
        //     let state_for_resolver = state.clone();
        //     let resolver = move |name: String| {
        //         let resolver_state = state_for_resolver.clone();
        //         async move {
        //             fn lookup_schema_from_state(
        //                 state_arc: &Arc<Mutex<RegistryState>>,
        //                 name: &str,
        //             ) -> Option<ProvidedSchema> {
        //                 let state = state_arc.lock().unwrap();
        //                 if let Some(schema) = state.schemas.get(name) {
        //                     return Some(schema.clone());
        //                 }
        //                 if let Some(parent) = &state.parent {
        //                     return lookup_schema_from_state(&parent.state, name);
        //                 }
        //                 None
        //             }

        //             match lookup_schema_from_state(&resolver_state, &name) {
        //                 Some(schema) => {
        //                     let json_val = match schema {
        //                         ProvidedSchema::FromType(s) => {
        //                             serde_json::to_value(s).map_err(|e| {
        //                                 dotprompt::Error::new_render_error(format!(
        //                                     "Failed to serialize schema: {}",
        //                                     e
        //                                 ))
        //                             })?
        //                         }
        //                         ProvidedSchema::Raw(v) => v,
        //                     };
        //                     Ok(json_val)
        //                 }
        //                 None => Err(dotprompt::Error::new_render_error(format!(
        //                     "Schema '{}' not found",
        //                     name
        //                 ))),
        //             }
        //         }
        //     };
        //     let dotprompt = Arc::new(Dotprompt::new_with_resolver(Box::new(resolver)));
        //     Self { state, dotprompt }
        // }
        // #[cfg(not(feature = "dotprompt-private"))]
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

        // #[cfg(feature = "dotprompt-private")]
        // {
        //     // NOTE: This follows the TS implementation by sharing the parent's dotprompt instance.
        //     // This means partials and helpers are shared, but it also means the schema resolver
        //     // will not see schemas defined only in the child registry. This may be revised later.
        //     let dotprompt = parent.dotprompt.clone();

        //     Self { state, dotprompt }
        // }
        // #[cfg(not(feature = "dotprompt-private"))]
        Self { state }
    }

    pub fn set_default_model(&mut self, name: String) {
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
    /// The action must be wrapped in an `Arc` to allow for shared ownership.
    pub fn register_action<A: ErasedAction + Send + Sync + 'static>(
        &mut self,
        name: &str,
        action: A,
    ) -> Result<()> {
        println!(
            "Registry::register_action: registering action named `{}` of type {:?}",
            name,
            std::any::Any::type_id(&action)
        );
        let action_arc: Arc<dyn ErasedAction> = Arc::new(action);
        let mut state = self.state.lock().unwrap();
        let meta = action_arc.metadata();
        let key = format!(
            "/{}/{}",
            format!("{:?}", meta.action_type).to_lowercase(),
            name
        );

        if state.actions.contains_key(&key) {
            // In a production framework, you might want to log a warning here.
        }

        state.actions.insert(key, action_arc);
        Ok(())
    }

    /// Looks up an action by its full key (e.g., "/flow/myflow").
    pub async fn lookup_action(&self, key: &str) -> Option<Arc<dyn ErasedAction>> {
        let parent = {
            let state = self.state.lock().unwrap();
            if let Some(action) = state.actions.get(key) {
                return Some(action.clone());
            }
            state.parent.clone()
        };

        if let Some(parent) = parent {
            // Box the future to break the recursive type definition.
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

    /// Looks up a generic value from the registry.
    pub async fn lookup_value<T: Any + Send + Sync>(
        &self,
        value_type: &str,
        name: &str,
    ) -> Option<Arc<T>> {
        let key = format!("{}/{}", value_type, name);
        let parent = {
            let state = self.state.lock().unwrap();
            if let Some(value) = state.values.get(&key) {
                // Attempt to downcast the `Arc<dyn Any>` to `Arc<T>`.
                return value.clone().downcast().ok();
            }
            state.parent.clone()
        };

        if let Some(parent) = parent {
            return Box::pin(parent.lookup_value(value_type, name)).await;
        }

        None
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionBuilder;
    use schemars::JsonSchema;

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
    struct TestInput {
        value: String,
    }
    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
    struct TestOutput {
        value: String,
    }

    #[tokio::test]
    async fn test_register_and_lookup_action() {
        // Define a local version of TestInput that derives Clone to satisfy the
        // ErasedAction trait bounds.
        #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
        struct TestInput {
            value: String,
        }

        let mut registry = Registry::new();
        let test_action = ActionBuilder::<TestInput, TestOutput, (), _>::new(
            ActionType::Flow,
            "myFlow",
            |i, _| async move {
                Ok(TestOutput {
                    value: format!("processed: {}", i.value),
                })
            },
        )
        .build();

        registry.register_action("myFlow", test_action).unwrap();

        let key = "/flow/myFlow";
        let looked_up = registry.lookup_action(key).await;
        assert!(looked_up.is_some());
        let looked_up_action = looked_up.unwrap();
        assert_eq!(looked_up_action.metadata().name, "myFlow");
        assert_eq!(looked_up_action.metadata().action_type, ActionType::Flow);
    }

    #[tokio::test]
    async fn test_child_registry_fallback() {
        let mut parent_registry = Registry::new();
        let parent_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Util, "parentUtil", |_, _| async {
                Ok(())
            })
            .build();
        parent_registry
            .register_action("parentUtil", parent_action)
            .unwrap();

        let mut child_registry = Registry::with_parent(&parent_registry);
        let child_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Util, "childUtil", |_, _| async {
                Ok(())
            })
            .build();
        child_registry
            .register_action("childUtil", child_action)
            .unwrap();

        // Child can find its own action
        assert!(child_registry
            .lookup_action("/util/childUtil")
            .await
            .is_some());
        // Child can find parent's action
        assert!(child_registry
            .lookup_action("/util/parentUtil")
            .await
            .is_some());
        // Parent cannot find child's action
        assert!(parent_registry
            .lookup_action("/util/childUtil")
            .await
            .is_none());
    }
}
