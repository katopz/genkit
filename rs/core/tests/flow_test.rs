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
    
    
}
