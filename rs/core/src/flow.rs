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

//! # Genkit Flows
//!
//! This module defines `Flow`, a primary orchestration primitive in the Genkit
//! framework. It is the Rust equivalent of `flow.ts`. Flows are specialized
//! actions that are designed to coordinate multiple steps, including calls to
//! other actions, models, or external services.

use crate::action::{Action, ActionBuilder, ActionFnArg};
use crate::context::{run_with_flow_context, FlowContext};
use crate::error::Result;
use crate::registry::{ActionType, Registry};
use crate::tracing as genkit_tracing;

use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Serialize};
use std::future::Future;
use std::sync::Arc;

/// A `Flow` is a specialized `Action` designed for orchestrating multiple steps.
///
/// It provides a structured way to define complex, stateful operations with
/// built-in observability through tracing.
pub type Flow<I, O, S> = Action<I, O, S>;

/// Defines a new flow and registers it with the Genkit framework.
///
/// # Arguments
///
/// * `registry` - A mutable reference to the `Registry` where the flow will be registered.
/// * `name` - A unique name for the flow.
/// * `func` - The asynchronous function that implements the flow's logic.
///
/// # Returns
///
/// An `Action` instance representing the defined flow.
pub fn define_flow<I, O, S, F, Fut>(
    registry: &mut Registry,
    name: impl Into<String>,
    func: F,
) -> Flow<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
    Action<I, O, S>: crate::registry::ErasedAction + 'static,
{
    let name = name.into();
    let flow_name = name.clone();
    let func = Arc::new(func);
    let wrapped_func = move |input: I, args: ActionFnArg<S>| {
        let func = func.clone();
        let flow_name = flow_name.clone();
        async move {
            let flow_context = FlowContext {
                flow_id: format!("{}-{}", flow_name, args.trace.trace_id),
            };
            run_with_flow_context(flow_context, (*func)(input, args)).await
        }
    };
    let action = ActionBuilder::new(ActionType::Flow, name.clone(), wrapped_func).build();
    registry
        .register_action(name.as_str(), action.clone())
        .expect("Failed to register flow");
    action
}

/// Executes a given function as a distinct step within a flow.
///
/// Each `run` call creates a new child span in the current trace, making it
/// easier to visualize and debug the flow's execution path. This is the primary
/// mechanism for instrumenting custom logic within a flow.
///
/// # Arguments
///
/// * `name` - A descriptive name for this step, used as the span name.
/// * `func` - The asynchronous closure to execute for this step.
pub async fn run<F, Fut, T>(name: &str, func: F) -> Result<T>
where
    F: FnOnce() -> Fut + Send,
    Fut: Future<Output = Result<T>> + Send,
    T: Serialize + Send + 'static,
{
    // This function wraps the core tracing logic to provide a simple API for
    // defining instrumented steps. The `map` at the end unwraps the
    // (T, TelemetryInfo) tuple from `in_new_span`, as `run`'s public API
    // only exposes the business logic result `T`.
    genkit_tracing::in_new_span(name.to_string(), None, |_trace_context| async {
        // TODO: Add attributes to the span, e.g., genkit:type = "flowStep".
        // This would require access to the current OpenTelemetry span.
        let result = func().await;
        // TODO: Record the output of the step as a span attribute.
        result
    })
    .await
    .map(|(result, _telemetry)| result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_utils::channel;
    use crate::registry::Registry;
    use crate::tracing::TraceContext;
    use serde::Deserialize;
    use tokio_util::sync::CancellationToken;

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
    struct MyInput {
        name: String,
    }

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
    struct MyOutput {
        message: String,
    }

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
    struct MyStreamChunk {}

    #[tokio::test]
    async fn test_define_and_run_flow() {
        let mut registry = Registry::default();
        // Define a flow with two instrumented steps.
        let my_flow = define_flow(
            &mut registry,
            "testFlow",
            |input: MyInput, _args: ActionFnArg<MyStreamChunk>| async move {
                let upper_name = run("step1: uppercase", || async {
                    Ok(input.name.to_uppercase())
                })
                .await?;

                run("step2: format_message", move || async move {
                    Ok(MyOutput {
                        message: format!("Hello, {}", upper_name),
                    })
                })
                .await
            },
        );

        // Simulate an action invocation by manually creating the arguments.
        // In a real application, the framework would handle this.
        let (chunk_tx, _chunk_rx) = channel();
        let args = ActionFnArg {
            streaming_requested: false,
            chunk_sender: chunk_tx,
            context: None,
            trace: TraceContext {
                trace_id: "test-trace-id".to_string(),
                span_id: "test-span-id".to_string(),
            },
            abort_signal: CancellationToken::new(),
        };

        let input = MyInput {
            name: "world".to_string(),
        };

        // Execute the flow's logic.
        let result = my_flow.func.run(input, args).await.unwrap();

        assert_eq!(
            result,
            MyOutput {
                message: "Hello, WORLD".to_string()
            }
        );
    }
}
