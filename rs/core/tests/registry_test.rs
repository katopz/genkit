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
//! Integration tests for the registry, refined from the TypeScript version.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use genkit_core::action::ActionBuilder;
use genkit_core::error::Result;
use genkit_core::plugin::Plugin;
use genkit_core::registry::{ActionType, Registry};
use rstest::*;

/// Fixture that provides a new, empty `Registry` for each test.
#[fixture]
fn registry() -> Registry {
    Registry::new()
}

#[cfg(test)]
/// 'listActions'
mod list_actions_test {
    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions'
    async fn returns_all_registered_actions(registry: Registry) {
        // Define and register the first action.
        let foo_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "foo_something", |_, _| async {
                Ok(())
            })
            .build();
        registry
            .register_action("foo_something", foo_action)
            .unwrap();

        // Define and register the second action.
        let bar_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "bar_something", |_, _| async {
                Ok(())
            })
            .build();
        registry
            .register_action("bar_something", bar_action)
            .unwrap();

        // Retrieve all actions from the registry.
        let actions = registry.list_actions().await;

        // Assert that the registry contains exactly the two actions we registered.
        assert_eq!(actions.len(), 2, "Should have two actions registered.");
        assert!(
            actions.contains_key("/model/foo_something"),
            "Registry should contain foo_something."
        );
        assert!(
            actions.contains_key("/model/bar_something"),
            "Registry should contain bar_something."
        );

        // Verify the names of the retrieved actions.
        assert_eq!(
            actions.get("/model/foo_something").unwrap().name(),
            "foo_something"
        );
        assert_eq!(
            actions.get("/model/bar_something").unwrap().name(),
            "bar_something"
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions by plugins'
    async fn returns_all_registered_actions_by_plugins(registry: Registry) {
        // Define a mock plugin 'foo'
        struct FooPlugin;
        #[async_trait]
        impl Plugin for FooPlugin {
            fn name(&self) -> &'static str {
                "foo"
            }

            async fn initialize(&self, registry: &Registry) -> Result<()> {
                let foo_something_action = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "foo/something",
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action("foo/something", foo_something_action)?;
                Ok(())
            }
        }

        // Define a mock plugin 'bar'
        struct BarPlugin;
        #[async_trait]
        impl Plugin for BarPlugin {
            fn name(&self) -> &'static str {
                "bar"
            }

            async fn initialize(&self, registry: &Registry) -> Result<()> {
                let bar_something_action = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "bar/something",
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action("bar/something", bar_something_action)?;

                let bar_sub_something_action = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "bar/sub/something",
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action("bar/sub/something", bar_sub_something_action)?;

                Ok(())
            }
        }

        // Simulate plugin initialization.
        let foo_plugin = FooPlugin;
        foo_plugin.initialize(&registry).await.unwrap();

        let bar_plugin = BarPlugin;
        bar_plugin.initialize(&registry).await.unwrap();

        // Retrieve all registered actions.
        let actions = registry.list_actions().await;

        // Assert that the actions from all plugins are present with the correct keys.
        assert_eq!(
            actions.len(),
            3,
            "Should have three actions registered from two plugins."
        );
        assert!(
            actions.contains_key("/model/foo/something"),
            "Registry should contain '/model/foo/something'."
        );
        assert!(
            actions.contains_key("/model/bar/something"),
            "Registry should contain '/model/bar/something'."
        );
        assert!(
            actions.contains_key("/model/bar/sub/something"),
            "Registry should contain '/model/bar/sub/something'."
        );
    }

    // A mock plugin to test initialization behavior.
    #[derive(Default, Clone)]
    struct FooPlugin {
        initialized: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Plugin for FooPlugin {
        fn name(&self) -> &'static str {
            "foo"
        }

        async fn initialize(&self, registry: &Registry) -> Result<()> {
            let action = ActionBuilder::<(), (), (), _>::new(
                ActionType::Model,
                "foo/something",
                |_, _| async { Ok(()) },
            )
            .build();
            registry.register_action("foo/something", action)?;
            self.initialized.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[rstest]
    #[tokio::test]
    /// 'should allow plugin initialization from runtime context'
    async fn should_allow_plugin_initialization_from_runtime_context(registry: Registry) {
        let foo_initialized = Arc::new(AtomicBool::new(false));
        let plugin = FooPlugin {
            initialized: foo_initialized.clone(),
        };

        // Before initialization, the flag is false and the action is not found.
        assert!(!foo_initialized.load(Ordering::SeqCst));
        let action_before_init = registry.lookup_action("/model/foo/something").await;
        assert!(action_before_init.is_none());

        // Manually initialize the plugin, which simulates the runtime context triggering it.
        plugin.initialize(&registry).await.unwrap();

        // After initialization, the flag is true and the action can be looked up.
        let action_after_init = registry.lookup_action("/model/foo/something").await;
        assert!(
            action_after_init.is_some(),
            "Action should be found after plugin initialization."
        );
        assert!(
            foo_initialized.load(Ordering::SeqCst),
            "Plugin's initialized flag should be true."
        );
    }
}
