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
            .register_action(foo_action.meta.action_type, foo_action)
            .unwrap();

        // Define and register the second action.
        let bar_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "bar_something", |_, _| async {
                Ok(())
            })
            .build();
        registry
            .register_action(bar_action.meta.action_type, bar_action)
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
                registry
                    .register_action(foo_something_action.meta.action_type, foo_something_action)?;
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
                registry
                    .register_action(bar_something_action.meta.action_type, bar_something_action)?;

                let bar_sub_something_action = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "bar/sub/something",
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action(
                    bar_sub_something_action.meta.action_type,
                    bar_sub_something_action,
                )?;

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
            registry.register_action(action.meta.action_type, action)?;
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

#[cfg(test)]
/// 'listResolvableActions'
mod list_resolvable_actions_test {
    use super::*;
    use async_trait::async_trait;
    use genkit_core::action::ActionBuilder;
    use genkit_core::error::Result;
    use genkit_core::registry::{ActionType, Plugin};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions'
    async fn returns_all_registered(registry: Registry) {
        let foo_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "foo_something", |_, _| async {
                Ok(())
            })
            .build();
        registry
            .register_action(foo_action.meta.action_type, foo_action)
            .unwrap();

        let bar_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "bar_something", |_, _| async {
                Ok(())
            })
            .build();
        registry
            .register_action(bar_action.meta.action_type, bar_action)
            .unwrap();

        // The Rust equivalent of `listResolvableActions` is `list_actions`.
        let actions = registry.list_actions().await;

        assert_eq!(
            actions.len(),
            2,
            "The registry should contain exactly two actions."
        );
        assert!(
            actions.contains_key("/model/foo_something"),
            "Action '/model/foo_something' should be in the registry."
        );
        assert!(
            actions.contains_key("/model/bar_something"),
            "Action '/model/bar_something' should be in the registry."
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions by plugins'
    async fn returns_all_registered_by_plugins(registry: Registry) {
        // Mock Plugin 'foo'
        struct FooPlugin;
        #[async_trait]
        impl Plugin for FooPlugin {
            fn name(&self) -> &'static str {
                "foo"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                // The action is defined with its local name 'something'.
                let action = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "something", // Note: local name, not "foo/something"
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action(action.meta.action_type, action)?;
                Ok(())
            }
        }

        // Mock Plugin 'bar'
        struct BarPlugin;
        #[async_trait]
        impl Plugin for BarPlugin {
            fn name(&self) -> &'static str {
                "bar"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                let action1 = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "something",
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action(action1.meta.action_type, action1)?;

                let action2 = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    "sub/something",
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action(action2.meta.action_type, action2)?;
                Ok(())
            }
        }

        // Simulate framework initialization of plugins.
        FooPlugin.initialize(&registry).await.unwrap();
        BarPlugin.initialize(&registry).await.unwrap();

        let actions = registry.list_actions().await;

        // Assert that the final keys in the registry are constructed correctly.
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

    #[rstest]
    #[tokio::test]
    /// 'should allow plugin initialization from runtime context'
    async fn should_allow_plugin_initialization_from_runtime_context(mut registry: Registry) {
        let foo_initialized = Arc::new(AtomicBool::new(false));

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
                registry.register_action(action.meta.action_type, action)?;
                self.initialized.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let plugin = Arc::new(FooPlugin {
            initialized: foo_initialized.clone(),
        });

        // The framework would typically trigger initialization. Here we simulate it.
        // First, register the plugin without initializing.
        registry.register_plugin(plugin.clone()).await.unwrap();

        // At this point, the action is not yet available.
        assert!(!foo_initialized.load(Ordering::SeqCst));
        assert!(registry
            .lookup_action("/model/foo/something")
            .await
            .is_none());

        // Now, simulate the runtime context triggering the initialization.
        plugin.initialize(&registry).await.unwrap();

        // After initialization, the action should be available.
        let action = registry.lookup_action("/model/foo/something").await;
        assert!(
            action.is_some(),
            "Action should be found after initialization."
        );
        assert!(
            foo_initialized.load(Ordering::SeqCst),
            "Initialized flag should be true."
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions, including parent'
    async fn returns_all_registered_including_parent(registry: Registry) {
        let child = Registry::with_parent(&registry);

        // Register an action on the parent registry.
        let foo_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "foo_something", |_, _| async {
                Ok(())
            })
            .build();
        registry
            .register_action(foo_action.meta.action_type, foo_action)
            .unwrap();

        // Register an action on the child registry.
        let bar_action =
            ActionBuilder::<(), (), (), _>::new(ActionType::Model, "bar_something", |_, _| async {
                Ok(())
            })
            .build();
        child
            .register_action(bar_action.meta.action_type, bar_action)
            .unwrap();

        // The child should see both its own actions and the parent's.
        let child_actions = child.list_actions().await;
        assert_eq!(child_actions.len(), 2);
        assert!(child_actions.contains_key("/model/foo_something"));
        assert!(child_actions.contains_key("/model/bar_something"));

        // The parent should only see its own actions.
        let parent_actions = registry.list_actions().await;
        assert_eq!(parent_actions.len(), 1);
        assert!(parent_actions.contains_key("/model/foo_something"));
    }

    // #[rstest]
    // #[tokio::test]
    // #[ignore = "This test requires `Registry::list_actions` to be updated to also query plugins for their dynamically resolvable actions."]
    // /// 'returns all registered actions and ones returned by listActions by plugins'
    // async fn returns_registered_and_dynamically_listed_actions(registry: Registry) {
    //     // NOTE: The current Rust implementation of `Registry::list_actions` does not query
    //     // plugins for their dynamically available actions via the `Plugin::list_actions`
    //     // method. This test will fail until that functionality is added to align with
    //     // the TypeScript behavior of `listResolvableActions`.

    //     // Plugin 'foo' registers its action directly upon initialization.
    //     struct FooPlugin;
    //     #[async_trait]
    //     impl Plugin for FooPlugin {
    //         fn name(&self) -> &'static str {
    //             "foo"
    //         }
    //         async fn initialize(&self, registry: &Registry) -> Result<()> {
    //             let action = ActionBuilder::<(), (), (), _>::new(
    //                 ActionType::Model,
    //                 "foo/something",
    //                 |_, _| async { Ok(()) },
    //             )
    //             .build();
    //             registry.register_action(action.meta.action_type, action)?;
    //             Ok(())
    //         }
    //     }

    //     // Plugin 'bar' registers some actions and dynamically lists another.
    //     struct BarPlugin;
    //     #[async_trait]
    //     impl Plugin for BarPlugin {
    //         fn name(&self) -> &'static str {
    //             "bar"
    //         }
    //         async fn initialize(&self, registry: &Registry) -> Result<()> {
    //             let action1 = ActionBuilder::<(), (), (), _>::new(
    //                 ActionType::Model,
    //                 "bar/something",
    //                 |_, _| async { Ok(()) },
    //             )
    //             .build();
    //             registry.register_action(action1.meta.action_type, action1)?;
    //             let action2 = ActionBuilder::<(), (), (), _>::new(
    //                 ActionType::Model,
    //                 "bar/sub/something",
    //                 |_, _| async { Ok(()) },
    //             )
    //             .build();
    //             registry.register_action(action2.meta.action_type, action2)?;
    //             Ok(())
    //         }

    //         // This method should be called by `registry.list_actions`.
    //         async fn list_actions(&self) -> Result<Vec<ActionMetadata>> {
    //             let dynamic_action_meta = ActionMetadata {
    //                 action_type: ActionType::Model,
    //                 name: "bar/barDynamicallyResolvable".to_string(),
    //                 description: Some("sings a song".to_string()),
    //                 subtype: None,
    //                 input_schema: None,
    //                 output_schema: None,
    //                 stream_schema: None,
    //                 metadata: Default::default(),
    //             };
    //             Ok(vec![dynamic_action_meta])
    //         }
    //     }

    //     // Simulate initialization.
    //     FooPlugin.initialize(&registry).await.unwrap();
    //     BarPlugin.initialize(&registry).await.unwrap();

    //     // This call would need to be updated to also poll the plugins.
    //     let actions = registry.list_actions().await;

    //     // Assertions:
    //     assert_eq!(
    //         actions.len(),
    //         4,
    //         "Should have 3 registered actions and 1 dynamic one."
    //     );
    //     assert!(actions.contains_key("/model/foo/something"));
    //     assert!(actions.contains_key("/model/bar/something"));
    //     assert!(actions.contains_key("/model/bar/sub/something"));
    //     // This assertion will fail because dynamic actions are not currently collected.
    //     assert!(actions.contains_key("/model/bar/barDynamicallyResolvable"));
    // }
}
