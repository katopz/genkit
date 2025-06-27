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

use crate::action::{Action, ActionMetadata};
use crate::error::{Error, Result};
use crate::schema;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
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
    /// Executes the action with a raw JSON value.
    async fn run_http(&self, input: Value) -> Result<Value>;
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

/// The central registry for Genkit components.
///
/// The `Registry` is responsible for storing and providing access to all
/// registered actions, plugins, and schemas. It supports hierarchical
/// composition, allowing for child registries that inherit from a parent.
#[derive(Clone, Default)]
pub struct Registry {
    /// Using `Arc<Mutex<...>>` allows for thread-safe interior mutability.
    state: Arc<Mutex<RegistryState>>,
}

#[derive(Default)]
struct RegistryState {
    actions: HashMap<String, Arc<dyn ErasedAction>>,
    plugins: HashMap<String, Arc<dyn Plugin>>,
    schemas: HashMap<String, schema::ProvidedSchema>,
    parent: Option<Registry>,
}

impl Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap();
        f.debug_struct("Registry")
            .field("actions", &state.actions.keys())
            .field("plugins", &state.plugins.keys())
            .field("schemas", &state.schemas.keys())
            .field("parent", &state.parent)
            .finish()
    }
}

impl Registry {
    /// Creates a new, empty `Registry`.
    pub fn new() -> Self {
        Registry::default()
    }

    /// Creates a new registry that inherits from a parent registry.
    ///
    /// The new registry can override components from the parent or add new ones.
    /// Lookups will fall back to the parent if a component is not found in the child.
    pub fn with_parent(parent: &Registry) -> Self {
        let state = RegistryState {
            parent: Some(parent.clone()),
            ..Default::default()
        };
        Registry {
            state: Arc::new(Mutex::new(state)),
        }
    }

    /// Registers an action with the registry.
    ///
    /// The action key is automatically generated from its type and name.
    /// The action must be wrapped in an `Arc` to allow for shared ownership.
    pub fn register_action<I, O, S>(&mut self, action: Action<I, O, S>) -> Result<()>
    where
        Action<I, O, S>: ErasedAction + 'static,
    {
        let mut state = self.state.lock().unwrap();
        let meta = action.metadata();
        let key = format!("/{:?}/{}", meta.action_type, meta.name).to_lowercase();

        if state.actions.contains_key(&key) {
            // In a production framework, you might want to log a warning here.
        }

        state.actions.insert(key, Arc::new(action));
        Ok(())
    }

    /// Looks up an action by its full key (e.g., "/flow/myflow").
    pub async fn lookup_action(&self, key: &str) -> Option<Arc<dyn ErasedAction>> {
        let state = self.state.lock().unwrap();
        if let Some(action) = state.actions.get(key) {
            return Some(action.clone());
        }
        if let Some(parent) = &state.parent {
            // Box the future to break the recursive type definition.
            return Box::pin(parent.lookup_action(key)).await;
        }
        None
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionBuilder;
    use crate::error::Result;
    use schemars::JsonSchema;
    use serde::de::DeserializeOwned;

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
    struct TestInput {
        value: String,
    }
    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
    struct TestOutput {
        value: String,
    }

    #[async_trait]
    impl<I, O, S> ErasedAction for Action<I, O, S>
    where
        I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
        O: Serialize + Send + Sync + 'static,
        S: Send + Sync + 'static,
    {
        async fn run_http(&self, _input: Value) -> Result<Value> {
            unimplemented!("run_http not implemented for tests");
        }

        fn metadata(&self) -> &ActionMetadata {
            &self.meta
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[tokio::test]
    async fn test_register_and_lookup_action() {
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
        .build(&mut registry);

        registry.register_action(test_action).unwrap();

        let key = "/flow/myflow";
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
            .build(&mut parent_registry);
        parent_registry.register_action(parent_action).unwrap();

        let mut child_registry = Registry::with_parent(&parent_registry);
        let child_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Util, "childUtil", |_, _| async {
                Ok(())
            })
            .build(&mut child_registry);
        child_registry.register_action(child_action).unwrap();

        // Child can find its own action
        assert!(child_registry
            .lookup_action("/util/childutil")
            .await
            .is_some());
        // Child can find parent's action
        assert!(child_registry
            .lookup_action("/util/parentutil")
            .await
            .is_some());
        // Parent cannot find child's action
        assert!(parent_registry
            .lookup_action("/util/childutil")
            .await
            .is_none());
    }
}
