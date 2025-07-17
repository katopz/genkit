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

//! # Genkit Plugins
//!
//! This module defines the `Plugin` trait, the primary mechanism for extending
//! the Genkit framework with new capabilities. It is the Rust equivalent of
//! `plugin.ts`.
//!
//! Plugins can provide new actions, models, retrievers, and other components
//! by implementing the `initialize` method and registering them with the
//! provided `Registry`.

use crate::action::ActionMetadata;
use crate::error::{Error, Result};
use crate::registry::{ActionType, Registry};
use async_trait::async_trait;
use std::future::Future;
use std::pin::Pin;

/// A trait for Genkit plugins.
///
/// Plugins are the primary mechanism for extending Genkit with new capabilities,
/// such as integrating with different model providers or services.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Returns the unique name of the plugin. This name is used to identify
    /// the plugin in the registry and in configuration.
    fn name(&self) -> &'static str;

    /// Initializes the plugin.
    ///
    /// This method is called by the framework when loading the plugin. The implementation
    /// should define and register all of its `Action`s and other components with the
    /// provided `registry`.
    async fn initialize(&self, registry: &Registry) -> Result<()>;

    /// (Optional) Lists all actions that can be resolved by this plugin.
    ///
    /// This method allows for dynamic discovery of actions, such as models available
    /// via an external API, without registering them all at startup.
    fn list_actions(&self) -> Pin<Box<dyn Future<Output = Result<Vec<ActionMetadata>>> + Send>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    /// Resolves and registers a single action dynamically.
    ///
    /// When the framework tries to look up an action that isn't already registered,
    /// it may call this method on relevant plugins to see if they can provide it.
    /// If the plugin can provide the action, it should define and register it
    /// with the provided `registry`.
    async fn resolve_action(
        &self,
        _action_type: ActionType,
        _target: &str,
        _registry: &Registry,
    ) -> Result<()> {
        Err(Error::new_internal(format!(
            "resolve_action is not implemented for plugin '{}'",
            self.name()
        )))
    }
}
