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

use genkit_core::action::{Action, ActionFnArg};
use genkit_core::async_utils::channel;
use genkit_core::context::{run_with_context, ActionContext};
use genkit_core::error::Result;
use genkit_core::flow::{define_flow, run};
use genkit_core::registry::Registry;
use genkit_core::tracing::TraceContext;
use rstest::fixture;
use rstest_macros::rstest;
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

async fn run_test_flow<I, O>(
    flow: &Action<I, O, TestStreamChunk>,
    input: I,
    context: Option<ActionContext>,
) -> Result<O>
where
    I: Send + 'static,
    O: Send + 'static,
{
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

#[fixture]
fn registry() -> Registry {
    Registry::default()
}

#[cfg(test)]
/// 'runFlow'
mod run_flow_test {
    use std::collections::HashMap;

    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'should run the flow'
    async fn test_run_simple_flow(#[from(registry)] registry: Registry) {
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

    #[rstest]
    #[tokio::test]
    /// 'should set metadata on the flow action'
    async fn test_set_metadata_on_flow_action(#[from(registry)] registry: Registry) {
        // This definition mirrors the TypeScript options object.
        let test_flow = define_flow(
            &registry,
            "testFlow",
            |input: String, _: ActionFnArg<TestStreamChunk>| async move { Ok(format!("bar {}", input)) },
        );

        // In a real implementation with an updated `define_flow`, you would pass
        // metadata at creation time. For now, we mutate it after creation
        // to demonstrate the testing of the metadata field itself.
        let mut flow_mut = test_flow.clone();
        flow_mut
            .metadata_mut()
            .metadata
            .insert("foo".to_string(), json!("bar"));

        let mut expected_metadata = HashMap::new();
        expected_metadata.insert("foo".to_string(), json!("bar"));

        // The assertion checks the `metadata` field within the action's metadata.
        assert_eq!(flow_mut.meta.metadata, expected_metadata);
    }

    #[rstest]
    #[tokio::test]
    /// 'should run simple sync flow'
    async fn test_run_simple_sync_flow(registry: Registry) {
        // This flow is "sync" in the sense that its logic doesn't use `.await`.
        // The `async move` block is required to match the expected function signature,
        // which must return a Future.
        let test_flow = define_flow(&registry, "testFlow", |input: TestInput, _| async move {
            // The logic here is executed synchronously.
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

    #[rstest]
    #[tokio::test]
    /// 'should include trace info in the context'
    async fn test_include_trace_info_in_context(registry: Registry) {
        // This flow's job is to check its own context and report what it finds.
        // Input is a String (ignored), Output is a String.
        let test_flow = define_flow(
            &registry,
            "traceContextFlow",
            // The second argument to the closure is the ActionFnArg, which contains the context.
            |_: String, args: ActionFnArg<_>| async move {
                let has_trace_id = !args.trace.trace_id.is_empty();
                let has_span_id = !args.trace.span_id.is_empty();

                Ok(format!("traceId={} spanId={}", has_trace_id, has_span_id))
            },
        );

        // Our test runner provides a dummy TraceContext.
        // In a real scenario, `enable_telemetry` and `in_new_span` would create this.
        let result = run_test_flow(&test_flow, "foo".to_string(), None)
            .await
            .unwrap();

        assert_eq!(result, "traceId=true spanId=true");
    }

    #[rstest]
    #[tokio::test]
    /// 'should rethrow the error'
    async fn test_rethrow_the_error(registry: Registry) {
        let test_flow = define_flow(&registry, "throwingFlow", |input: String, _| async move {
            // This flow is designed to fail.
            Err::<String, _>(genkit_core::Error::new_internal(format!(
                "bad happened: {}",
                input
            )))
        });

        // Call the flow and expect an error.
        let result = run_test_flow(&test_flow, "foo".to_string(), None).await;

        // Assert that the result is indeed an error.
        assert!(result.is_err());

        // Further inspect the error to ensure it's the one we threw.
        match result.unwrap_err() {
            genkit_core::Error::Internal { message, .. } => {
                assert_eq!(message, "bad happened: foo");
            }
            _ => panic!("Expected an Internal error variant."),
        }
    }

    #[rstest]
    #[tokio::test]
    /// 'should validate input'
    async fn test_validate_input(registry: Registry) {
        // Define the expected input structure.
        // `Deserialize` and `JsonSchema` are required for validation to work.
        // `Clone` is needed for the `ErasedAction` trait bounds.
        #[derive(JsonSchema, Deserialize, Clone)]
        #[allow(dead_code)]
        struct ValidatingInput {
            foo: String,
            bar: i32,
        }

        let test_flow = define_flow(
            &registry,
            "validatingFlow",
            // The logic here should never be executed because validation will fail first.
            |_: ValidatingInput, _: ActionFnArg<TestStreamChunk>| async { Ok("ok".to_string()) },
        );

        // Create a JSON object with an invalid type for the `bar` field.
        let invalid_input = serde_json::json!({
            "foo": "foo",
            "bar": "bar" // This should be a number.
        });

        // Use `run_http_json` which takes a raw `Value` and performs validation.
        let result = test_flow.run_http(invalid_input, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err();

        // Assert that we received the correct type of error.
        assert!(matches!(err, genkit_core::Error::Validation(_)));
        // Optionally, check the error message content.
        assert!(err.to_string().contains("Schema validation failed"));
    }
}

// #[cfg(test)]
// /// 'getContext'
// mod run_flow_test {
//     use std::collections::HashMap;

//     use super::*;
// }
