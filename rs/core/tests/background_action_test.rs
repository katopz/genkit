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

//! # Action Tests
//!
//! Integration tests for the action system.

#[cfg(test)]
mod background_action_test {
    use genkit_core::action::ActionFnArg;
    use genkit_core::background_action::BackgroundActionParams;
    use genkit_core::registry::{ActionType, Registry};
    use genkit_core::{define_background_action, lookup_background_action, Operation};
    use rstest::{fixture, rstest};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[fixture]
    fn registry() -> Registry {
        Registry::new()
    }

    #[derive(Serialize, Deserialize, JsonSchema, Clone, Default)]
    struct TestInput {}

    #[derive(Serialize, Deserialize, JsonSchema, Clone, Default)]
    struct TestOutput {}

    #[rstest]
    #[tokio::test]
    /// 'should lookup a defined background action and its components'
    async fn test_lookup_background_action(registry: Registry) {
        // 1. Define a background action with start, check, and cancel handlers.
        define_background_action(
            &registry,
            BackgroundActionParams {
                name: "lookupTest".to_string(),
                action_type: ActionType::Flow,
                description: None,
                metadata: None,
                start: |_: TestInput, _args: ActionFnArg<()>| async move {
                    Ok(Operation {
                        id: Uuid::new_v4().to_string(),
                        done: false,
                        ..Default::default()
                    })
                },
                check: |op: Operation<TestOutput>| async move { Ok(op) },
                cancel: Some(|op: Operation<TestOutput>| async move {
                    Ok(Operation { done: true, ..op })
                }),
                _marker: std::marker::PhantomData,
            },
        );

        // 2. Look up the action using its primary key.
        let looked_up_result =
            lookup_background_action::<TestInput, TestOutput>(&registry, "/flow/lookupTest").await;

        // 3. Assert that the lookup was successful and returned a valid action.
        assert!(
            looked_up_result.is_ok(),
            "Lookup should not produce an error"
        );
        let bg_action = looked_up_result
            .unwrap()
            .expect("BackgroundAction should be found in the registry");

        // 4. Verify that the components of the looked-up action are correct.
        assert_eq!(bg_action.start_action.meta.name, "lookupTest");
        assert_eq!(bg_action.check_action.meta.name, "lookupTest/check");
        assert!(
            bg_action.cancel_action.is_some(),
            "Cancel action should be present"
        );
        assert_eq!(
            bg_action.cancel_action.unwrap().meta.name,
            "lookupTest/cancel"
        );

        // 5. Verify that looking up a non-existent action returns None.
        let not_found_result =
            lookup_background_action::<TestInput, TestOutput>(&registry, "/flow/nonExistent").await;
        assert!(not_found_result.is_ok());
        assert!(
            not_found_result.unwrap().is_none(),
            "Lookup for a non-existent action should return None"
        );
    }
}
