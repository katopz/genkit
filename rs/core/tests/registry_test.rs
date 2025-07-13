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

use async_trait::async_trait;
use genkit_core::action::{define_action, ActionBuilder, ActionFnArg, ActionName};
use genkit_core::error::Result;
use genkit_core::registry::{ActionType, Plugin, Registry};
use genkit_core::runtime;
use rstest::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    /// 'returns all registered actions by plugins'
    async fn returns_all_registered_actions_by_plugins(mut registry: Registry) {
        // Define mock plugin 'foo'
        struct FooPlugin;
        #[async_trait]
        impl Plugin for FooPlugin {
            // Implements genkit_core::registry::Plugin
            fn name(&self) -> &'static str {
                "foo"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                define_action(
                    registry,
                    ActionType::Model,
                    "foo/something",
                    |_: (), _: ActionFnArg<()>| async { Ok(()) },
                );
                Ok(())
            }
        }

        // Define mock plugin 'bar'
        struct BarPlugin;
        #[async_trait]
        impl Plugin for BarPlugin {
            // Implements genkit_core::registry::Plugin
            fn name(&self) -> &'static str {
                "bar"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                define_action(
                    registry,
                    ActionType::Model,
                    "bar/something",
                    |_: (), _: ActionFnArg<()>| async { Ok(()) },
                );
                define_action(
                    registry,
                    ActionType::Model,
                    "bar/sub/something",
                    |_: (), _: ActionFnArg<()>| async { Ok(()) },
                );
                Ok(())
            }
        }

        // Register and initialize plugins
        let foo_plugin = Arc::new(FooPlugin);
        registry.register_plugin(foo_plugin.clone()).await.unwrap();
        foo_plugin.initialize(&registry).await.unwrap();

        let bar_plugin = Arc::new(BarPlugin);
        registry.register_plugin(bar_plugin.clone()).await.unwrap();
        bar_plugin.initialize(&registry).await.unwrap();

        // Retrieve actions and check results
        let actions = registry.list_actions().await;
        assert_eq!(
            actions.len(),
            3,
            "Should have three actions from two plugins."
        );
        assert!(actions.contains_key("/model/foo/something"));
        assert!(actions.contains_key("/model/bar/something"));
        assert!(actions.contains_key("/model/bar/sub/something"));
    }

    #[rstest]
    #[tokio::test]
    /// 'should allow plugin initialization from runtime context'
    async fn should_allow_plugin_initialization_from_runtime_context(mut registry: Registry) {
        // A mock plugin to test runtime initialization behavior
        #[derive(Default, Clone)]
        struct RuntimePlugin {
            initialized: Arc<AtomicBool>,
        }

        #[async_trait]
        impl Plugin for RuntimePlugin {
            // Implements genkit_core::registry::Plugin
            fn name(&self) -> &'static str {
                "runtime"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                define_action(
                    registry,
                    ActionType::Model,
                    "runtime/something",
                    |_: (), _: ActionFnArg<()>| async { Ok(()) },
                );
                self.initialized.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let initialized_flag = Arc::new(AtomicBool::new(false));
        let plugin = Arc::new(RuntimePlugin {
            initialized: initialized_flag.clone(),
        });
        registry.register_plugin(plugin.clone()).await.unwrap();

        // Before initialization, the action should not exist
        assert!(!initialized_flag.load(Ordering::SeqCst));
        assert!(registry
            .lookup_action("/model/runtime/something")
            .await
            .is_none());

        // Simulate runtime initialization
        plugin.initialize(&registry).await.unwrap();

        // After initialization, the action should be present
        assert!(registry
            .lookup_action("/model/runtime/something")
            .await
            .is_some());
        assert!(initialized_flag.load(Ordering::SeqCst));
    }

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions, including parent'
    async fn returns_all_registered_including_parent(registry: Registry) {
        // 1. Create a child registry that inherits from the main one.
        let child = Registry::with_parent(&registry);

        // 2. Define an action on the parent registry.
        let foo_action = define_action(
            &registry,
            ActionType::Model,
            "foo_something",
            |_: (), _: ActionFnArg<()>| async { Ok(()) },
        );

        // 3. Define another action on the child registry.
        let bar_action = define_action(
            &child,
            ActionType::Model,
            "bar_something",
            |_: (), _: ActionFnArg<()>| async { Ok(()) },
        );

        // 4. Assert that the child registry contains both actions.
        let child_actions = child.list_actions().await;
        let child_actions_json = serde_json::Value::Object(
            child_actions
                .into_iter()
                .map(|(k, v)| (k, serde_json::json!(v.metadata())))
                .collect(),
        );

        let expected_child_json = serde_json::json!({
            "/model/foo_something": *foo_action.meta,
            "/model/bar_something": *bar_action.meta,
        });
        assert_eq!(child_actions_json, expected_child_json);

        // 5. Assert that the parent registry only contains its own action.
        let parent_actions = registry.list_actions().await;
        let parent_actions_json = serde_json::Value::Object(
            parent_actions
                .into_iter()
                .map(|(k, v)| (k, serde_json::json!(v.metadata())))
                .collect(),
        );

        let expected_parent_json = serde_json::json!({
            "/model/foo_something": *foo_action.meta,
        });
        assert_eq!(parent_actions_json, expected_parent_json);
    }
}

#[cfg(test)]
/// 'listResolvableActions'
mod list_resolvable_actions_test {
    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions'
    async fn returns_all_registered(registry: Registry) {
        define_action(
            &registry,
            ActionType::Model,
            "foo_something",
            |_: (), _: ActionFnArg<()>| async { Ok(()) },
        );
        define_action(
            &registry,
            ActionType::Model,
            "bar_something",
            |_: (), _: ActionFnArg<()>| async { Ok(()) },
        );

        let actions = registry.list_actions().await;
        assert_eq!(actions.len(), 2);
        assert!(actions.contains_key("/model/foo_something"));
        assert!(actions.contains_key("/model/bar_something"));
    }

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions by plugins'
    async fn returns_all_registered_by_plugins(registry: Registry) {
        // Mock Plugin 'foo'
        struct FooPlugin;
        #[async_trait]
        impl Plugin for FooPlugin {
            // Implements genkit_core::registry::Plugin
            fn name(&self) -> &'static str {
                "foo"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                let action = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    ActionName::Namespaced {
                        plugin_id: "foo".to_string(),
                        action_id: "something".to_string(),
                    },
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
            // Implements genkit_core::registry::Plugin
            fn name(&self) -> &'static str {
                "bar"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                let action1 = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    ActionName::Namespaced {
                        plugin_id: "bar".to_string(),
                        action_id: "something".to_string(),
                    },
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action(action1.meta.action_type, action1)?;

                let action2 = ActionBuilder::<(), (), (), _>::new(
                    ActionType::Model,
                    ActionName::Namespaced {
                        plugin_id: "bar".to_string(),
                        action_id: "sub/something".to_string(),
                    },
                    |_, _| async { Ok(()) },
                )
                .build();
                registry.register_action(action2.meta.action_type, action2)?;
                Ok(())
            }
        }

        // Initialize plugins and check actions
        FooPlugin.initialize(&registry).await.unwrap();
        BarPlugin.initialize(&registry).await.unwrap();
        let actions = registry.list_actions().await;

        assert_eq!(
            actions.len(),
            3,
            "Should have three actions registered from two plugins."
        );
        assert!(actions.contains_key("/model/foo/something"));
        assert!(actions.contains_key("/model/bar/something"));
        assert!(actions.contains_key("/model/bar/sub/something"));
    }

    #[rstest]
    #[tokio::test]
    /// 'should allow plugin initialization from runtime context'
    async fn should_allow_plugin_initialization_from_runtime_context(mut registry: Registry) {
        let initialized = Arc::new(AtomicBool::new(false));

        struct FooPlugin {
            initialized: Arc<AtomicBool>,
        }
        #[async_trait]
        impl Plugin for FooPlugin {
            fn name(&self) -> &'static str {
                "foo"
            }
            async fn initialize(&self, registry: &Registry) -> Result<()> {
                define_action(
                    registry,
                    ActionType::Model,
                    "foo/something",
                    |_: (), _: ActionFnArg<()>| async { Ok(()) },
                );
                self.initialized.store(true, Ordering::SeqCst);
                Ok(())
            }
        }

        let plugin = Arc::new(FooPlugin {
            initialized: initialized.clone(),
        });
        registry.register_plugin(plugin).await.unwrap();

        assert!(
            !initialized.load(Ordering::SeqCst),
            "Plugin should not be initialized yet."
        );
        let action_before = registry.lookup_action("/model/foo/something").await;
        assert!(
            action_before.is_none(),
            "Action should not exist before runtime lookup."
        );
        assert!(
            !initialized.load(Ordering::SeqCst),
            "Plugin should still not be initialized."
        );

        // Correctly call the function from the `runtime` module.
        let action_after = runtime::run_in_action_runtime_context(async {
            registry.lookup_action("/model/foo/something").await
        })
        .await;

        assert!(
            action_after.is_some(),
            "Action should be found after runtime lookup."
        );
        assert!(
            initialized.load(Ordering::SeqCst),
            "Plugin initializer should have been called."
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'returns all registered actions, including parent'
    async fn returns_all_registered_including_parent(registry: Registry) {
        let child = Registry::with_parent(&registry);

        // Register one action on the parent and one on the child
        define_action(
            &registry,
            ActionType::Model,
            "parent_action",
            |_: (), _: ActionFnArg<()>| async { Ok(()) },
        );
        define_action(
            &child,
            ActionType::Model,
            "child_action",
            |_: (), _: ActionFnArg<()>| async { Ok(()) },
        );

        // The child should see both actions
        let child_actions = child.list_actions().await;
        assert_eq!(child_actions.len(), 2);
        assert!(child_actions.contains_key("/model/parent_action"));
        assert!(child_actions.contains_key("/model/child_action"));

        // The parent should only see its own action
        let parent_actions = registry.list_actions().await;
        assert_eq!(parent_actions.len(), 1);
        assert!(parent_actions.contains_key("/model/parent_action"));
    }
}
