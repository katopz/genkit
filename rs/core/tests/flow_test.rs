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

//! # Flow Tests
//!
//! Integration tests for the flow system, ported from `flow_test.ts`.

use genkit_core::action::ActionFnArg;
use genkit_core::async_utils::channel;
use genkit_core::context::{run_with_context, ActionContext};
use genkit_core::error::Result;
use genkit_core::flow::{define_flow, run};
use genkit_core::tracing::TraceContext;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
struct TestInput {
    name: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
struct TestOutput {
    message: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
struct TestStreamChunk {}

async fn run_test_flow(
    flow: &genkit_core::flow::Flow<TestInput, TestOutput, TestStreamChunk>,
    input: TestInput,
    context: Option<ActionContext>,
) -> Result<TestOutput> {
    let (chunk_tx, _chunk_rx) = channel();
    let args = ActionFnArg {
        streaming_requested: false,
        chunk_sender: chunk_tx,
        context,
        trace: TraceContext {
            trace_id: "test-trace-id".to_string(),
            span_id: "test-span-id".to_string(),
        },
        abort_signal: CancellationToken::new(),
    };
    flow.func.run(input, args).await
}

#[cfg(test)]
mod test {
    use crate::*;
    use genkit_core::registry::Registry;
    #[tokio::test]
    async fn test_run_simple_flow() {
        let registry = Registry::default();
        let test_flow = define_flow(&registry, "testFlow", |input: TestInput, _| async move {
            Ok(TestOutput {
                message: format!("bar {}", input.name),
            })
        });

        let result = run_test_flow(
            &test_flow,
            TestInput {
                name: "foo".to_string(),
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(result.message, "bar foo");
    }

    #[tokio::test]
    async fn test_run_flow_with_steps() {
        let registry = Registry::default();
        let test_flow = define_flow(
            &registry,
            "flowWithSteps",
            |input: TestInput, _| async move {
                let step1_result = run("step1", || async { Ok(input.name.to_uppercase()) }).await?;
                let step2_result =
                    run("step2", || async { Ok(format!("Hello, {}", step1_result)) }).await?;
                Ok(TestOutput {
                    message: step2_result,
                })
            },
        );

        let result = run_test_flow(
            &test_flow,
            TestInput {
                name: "world".to_string(),
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(result.message, "Hello, WORLD");
    }

    #[tokio::test]
    async fn test_flow_context_inheritance() {
        let registry = Registry::default();
        let child_flow = define_flow(
            &registry,
            "childFlow",
            |input: TestInput, args: ActionFnArg<TestStreamChunk>| async move {
                let auth_user = args
                    .context
                    .and_then(|c| c.auth)
                    .and_then(|a| a.get("user").and_then(|u| u.as_str().map(String::from)))
                    .unwrap_or_else(|| "no_user".to_string());
                Ok(TestOutput {
                    message: format!("child saw {} for {}", auth_user, input.name),
                })
            },
        );

        let parent_flow = define_flow(&registry, "parentFlow", move |input: TestInput, _| {
            let child_flow = child_flow.clone();
            async move {
                // In a real scenario, we wouldn't manually construct the args.
                // This simulates the framework propagating context.
                let child_context = genkit_core::context::get_context();
                run_test_flow(&child_flow, input, child_context).await
            }
        });

        let context = ActionContext {
            auth: Some(json!({ "user": "test-user" })),
            ..Default::default()
        };
        let result = run_with_context(context, async {
            run_test_flow(
                &parent_flow,
                TestInput {
                    name: "parent".to_string(),
                },
                None, // The context is picked up from the task-local storage.
            )
            .await
        })
        .await
        .unwrap();

        assert_eq!(result.message, "child saw test-user for parent");
    }
}
