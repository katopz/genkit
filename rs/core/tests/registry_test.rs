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

//! # Registry Tests
//!
//! Integration tests for the registry, ported from `registry_test.ts`.

use async_trait::async_trait;
use genkit_core::action::ActionBuilder;
use genkit_core::error::Result;
use genkit_core::registry::{ActionType, Plugin, Registry};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// Helper structs for testing
#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
struct TestInput {}
#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
struct TestOutput {}

// A mock plugin for testing purposes.
struct MockPlugin {
    name: &'static str,
    initialized: Arc<AtomicBool>,
}

impl MockPlugin {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl Plugin for MockPlugin {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        self.initialized.store(true, Ordering::SeqCst);
        let action_name = format!("{}/model", self.name);
        let test_action = ActionBuilder::<TestInput, TestOutput, (), _>::new(
            ActionType::Model,
            action_name.clone(),
            |_, _| async { Ok(TestOutput {}) },
        )
        .build();
        registry.register_action(action_name.as_str(), test_action)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::*;
    #[tokio::test]
    async fn test_manual_plugin_initialization() {
        let mut registry = Registry::new();
        let plugin = Arc::new(MockPlugin::new("foo"));
        let was_initialized = plugin.initialized.clone();

        // Registering the plugin does not initialize it in this implementation.
        registry.register_plugin(plugin.clone()).await.unwrap();
        assert!(
            !was_initialized.load(Ordering::SeqCst),
            "Plugin should not be initialized after registration only."
        );

        // Manually initialize the plugin.
        plugin.initialize(&mut registry).await.unwrap();
        assert!(
            was_initialized.load(Ordering::SeqCst),
            "Plugin should be initialized after explicit call."
        );

        // Verify that the plugin registered its action.
        let looked_up = registry.lookup_action("/model/foo/model").await;
        assert!(
            looked_up.is_some(),
            "Action from plugin should be found after initialization."
        );
    }

    #[tokio::test]
    async fn test_parent_child_registry_lookup() {
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

        // Child can find parent's action by falling back
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

    #[tokio::test]
    async fn test_register_and_lookup_action() {
        let mut registry = Registry::new();
        let test_action = ActionBuilder::<TestInput, TestOutput, (), _>::new(
            ActionType::Flow,
            "myFlow",
            |_, _| async { Ok(TestOutput {}) },
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
}
