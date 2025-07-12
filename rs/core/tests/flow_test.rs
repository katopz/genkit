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

use genkit_core::action::{Action, ActionFnArg, ActionRunOptions};
use genkit_core::async_utils::channel;
use genkit_core::context::{get_context, run_with_context, ActionContext};
use genkit_core::error::{Error as GenkitError, Result};
use genkit_core::flow::{define_flow, run, Flow};
use genkit_core::registry::{ErasedAction, Registry};
use genkit_core::tracing::TraceContext;
use rstest::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
struct TestInput {
    name: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
struct TestOutput {
    message: String,
}

/// Test helper to invoke a flow's function with basic arguments.
async fn run_test_flow<I, O, S>(
    flow: &Action<I, O, S>,
    input: I,
    context: Option<ActionContext>,
    abort_signal: CancellationToken,
) -> Result<O>
where
    I: Send + 'static,
    O: Send + 'static,
    S: Send + 'static,
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
        abort_signal,
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
    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'should run the flow'
    async fn test_run_simple_flow(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "testFlow",
            |input: TestInput, _: ActionFnArg<()>| async move {
                Ok(TestOutput {
                    message: format!("bar {}", input.name),
                })
            },
        );

        let result = run_test_flow(
            &test_flow,
            TestInput {
                name: "foo".to_string(),
            },
            None,
            CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(result.message, "bar foo");
    }

    #[rstest]
    #[tokio::test]
    /// 'should set metadata on the flow action'
    async fn test_set_metadata_on_flow_action(registry: Registry) {
        let mut test_flow = define_flow(
            &registry,
            "testFlow",
            |input: String, _: ActionFnArg<()>| async move { Ok(format!("bar {}", input)) },
        );

        test_flow
            .metadata_mut()
            .metadata
            .insert("foo".to_string(), json!("bar"));

        let mut expected_metadata = HashMap::new();
        expected_metadata.insert("foo".to_string(), json!("bar"));

        assert_eq!(test_flow.meta.metadata, expected_metadata);
    }

    #[rstest]
    #[tokio::test]
    /// 'should run simple sync flow'
    async fn test_run_simple_sync_flow(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "testFlow",
            |input: TestInput, _: ActionFnArg<()>| async move {
                Ok(TestOutput {
                    message: format!("bar {}", input.name),
                })
            },
        );

        let result = run_test_flow(
            &test_flow,
            TestInput {
                name: "foo".to_string(),
            },
            None,
            CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(result.message, "bar foo");
    }

    #[rstest]
    #[tokio::test]
    /// 'should include trace info in the context'
    async fn test_include_trace_info_in_context(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "traceContextFlow",
            |_: String, args: ActionFnArg<()>| async move {
                let has_trace_id = !args.trace.trace_id.is_empty();
                let has_span_id = !args.trace.span_id.is_empty();

                Ok(format!("traceId={} spanId={}", has_trace_id, has_span_id))
            },
        );

        let result = run_test_flow(
            &test_flow,
            "foo".to_string(),
            None,
            CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(result, "traceId=true spanId=true");
    }

    #[rstest]
    #[tokio::test]
    /// 'should rethrow the error'
    async fn test_rethrow_the_error(registry: Registry) {
        #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
        struct TestStreamChunk {}

        let test_flow = define_flow(
            &registry,
            "throwingFlow",
            |input: String, _: ActionFnArg<TestStreamChunk>| async move {
                Err(GenkitError::new_internal(format!(
                    "bad happened: {}",
                    input
                )))
            },
        );

        let result: Result<TestStreamChunk> = run_test_flow(
            &test_flow,
            "foo".to_string(),
            None,
            CancellationToken::new(),
        )
        .await;

        assert!(result.is_err());
        match result.unwrap_err() {
            GenkitError::Internal { message, .. } => {
                assert_eq!(message, "bad happened: foo");
            }
            _ => panic!("Expected an Internal error variant."),
        }
    }

    #[rstest]
    #[tokio::test]
    /// 'should validate input'
    async fn test_validate_input(registry: Registry) {
        #[derive(JsonSchema, Deserialize, Clone, Serialize)]
        #[allow(dead_code)]
        struct ValidatingInput {
            foo: String,
            bar: i32,
        }

        let test_flow = define_flow(
            &registry,
            "validatingFlow",
            |_: ValidatingInput, _: ActionFnArg<()>| async { Ok("ok".to_string()) },
        );

        let invalid_input = json!({ "foo": "foo", "bar": "bar" });
        let result = test_flow.run_http_json(invalid_input, None).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, GenkitError::Validation(_)));
        assert!(err.to_string().contains("Schema validation failed"));
    }
}

#[cfg(test)]
/// 'getContext'
mod get_context_test {
    use super::*;
    use futures::stream::TryStreamExt;
    use genkit_core::action::StreamingResponse;

    #[rstest]
    #[tokio::test]
    /// 'should run the flow' (with context)
    async fn test_run_with_context(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "contextFlow",
            |input: String, args: ActionFnArg<()>| async move {
                let context_str = if let Some(ctx) = args.context {
                    serde_json::to_string(&ctx.additional_context).unwrap()
                } else {
                    "null".to_string()
                };
                Ok(format!("bar {} {}", input, context_str))
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("user".to_string(), json!("test-user"));
        let context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let result = run_test_flow(
            &test_flow,
            "foo".to_string(),
            Some(context),
            CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(result, r#"bar foo {"user":"test-user"}"#);
    }

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
    struct StreamCount {
        count: i32,
    }

    #[rstest]
    #[tokio::test]
    /// 'should streams the flow' (with context)
    async fn test_streaming_with_context(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "streamingContextFlow",
            |input: i32, args: ActionFnArg<StreamCount>| async move {
                if args.streaming_requested {
                    for i in 0..input {
                        let _ = args.chunk_sender.send(StreamCount { count: i });
                    }
                }
                let context_str = if let Some(ctx) = args.context {
                    serde_json::to_string(&ctx.additional_context).unwrap()
                } else {
                    "null".to_string()
                };
                Ok(format!(
                    "bar {} {} {}",
                    input, args.streaming_requested, context_str
                ))
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("user".to_string(), json!("test-user"));
        let context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let streaming_response: StreamingResponse<String, StreamCount> = test_flow.stream(
            3,
            Some(ActionRunOptions {
                context: Some(context),
                ..Default::default()
            }),
        );

        let (chunks_result, output_result) = tokio::join!(
            streaming_response.stream.try_collect::<Vec<_>>(),
            streaming_response.output
        );

        let received_chunks = chunks_result.unwrap();
        let final_output = output_result.unwrap();

        assert_eq!(final_output, r#"bar 3 true {"user":"test-user"}"#);
        assert_eq!(
            received_chunks,
            vec![
                StreamCount { count: 0 },
                StreamCount { count: 1 },
                StreamCount { count: 2 }
            ]
        );
    }
}

#[cfg(test)]
/// 'context'
mod context_test {
    use super::*;
    use futures::stream::TryStreamExt;
    use genkit_core::action::ActionRunOptions;
    use tokio::time::{sleep, Duration};

    #[rstest]
    #[tokio::test]
    /// 'should run the flow with context (old way)'
    async fn test_run_with_context_old_way(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "testFlowOldWay",
            |input: String, _: ActionFnArg<()>| async move {
                let context = get_context().unwrap();
                let context_str = serde_json::to_string(&context.additional_context).unwrap();
                Ok(format!("bar {} {}", input, context_str))
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("user".to_string(), json!("test-user"));
        let test_context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let result = run_with_context(test_context, async {
            run_test_flow(
                &test_flow,
                "foo".to_string(),
                None,
                CancellationToken::new(),
            )
            .await
            .unwrap()
        })
        .await;

        assert_eq!(result, r#"bar foo {"user":"test-user"}"#);
    }

    #[rstest]
    #[tokio::test]
    /// 'should run the flow with context (new way)'
    async fn test_run_with_context_new_way(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "testFlowNewWay",
            |input: String, args: ActionFnArg<()>| async move {
                let context_str = if let Some(ctx) = args.context {
                    serde_json::to_string(&ctx.additional_context).unwrap()
                } else {
                    "null".to_string()
                };
                Ok(format!("bar {} {}", input, context_str))
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("user".to_string(), json!("test-user"));
        let context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let result = run_test_flow(
            &test_flow,
            "foo".to_string(),
            Some(context),
            CancellationToken::new(),
        )
        .await
        .unwrap();

        assert_eq!(result, r#"bar foo {"user":"test-user"}"#);
    }

    #[rstest]
    #[tokio::test]
    /// 'should inherit context from the parent'
    async fn test_inherit_context_from_parent(registry: Registry) {
        let child_flow = define_flow(
            &registry,
            "childFlow",
            |input: String, args: ActionFnArg<()>| async move {
                let context_str = if let Some(ctx) = args.context {
                    serde_json::to_string(&ctx.additional_context).unwrap()
                } else {
                    "null".to_string()
                };
                Ok(format!("bar {} {}", input, context_str))
            },
        );

        let parent_flow: Flow<String, String, ()> = define_flow(
            &registry,
            "parentFlow",
            move |input: String, _: ActionFnArg<()>| {
                let child_flow = child_flow.clone();
                async move {
                    let inherited_context = get_context();
                    run_test_flow(
                        &child_flow,
                        input,
                        inherited_context,
                        CancellationToken::new(),
                    )
                    .await
                }
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("user".to_string(), json!("test-user"));
        let test_context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let result = run_with_context(test_context, async {
            run_test_flow(
                &parent_flow,
                "foo".to_string(),
                None,
                CancellationToken::new(),
            )
            .await
            .unwrap()
        })
        .await;

        assert_eq!(result, r#"bar foo {"user":"test-user"}"#);
    }

    #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
    struct StreamCount {
        count: i32,
    }

    #[rstest]
    #[tokio::test]
    /// 'should streams the flow with context'
    async fn test_streaming_the_flow_with_context(registry: Registry) {
        let test_flow: Flow<i32, String, StreamCount> = define_flow(
            &registry,
            "streamingFlowWithContext",
            |input: i32, args| async move {
                for i in 0..input {
                    let _ = args.chunk_sender.send(StreamCount { count: i });
                }
                let context_str = if let Some(ctx) = args.context {
                    serde_json::to_string(&ctx.additional_context).unwrap()
                } else {
                    "null".to_string()
                };
                Ok(format!(
                    "bar {} {} {}",
                    input, args.streaming_requested, context_str
                ))
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("user".to_string(), json!("test-user"));
        let context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };
        let options = ActionRunOptions {
            context: Some(context),
            ..Default::default()
        };

        let streaming_response = test_flow.stream(3, Some(options));

        let (chunks_result, output_result) = tokio::join!(
            streaming_response.stream.try_collect::<Vec<_>>(),
            streaming_response.output
        );

        let received_chunks = chunks_result.unwrap();
        let final_output = output_result.unwrap();

        assert_eq!(final_output, r#"bar 3 true {"user":"test-user"}"#);
        assert_eq!(
            received_chunks,
            vec![
                StreamCount { count: 0 },
                StreamCount { count: 1 },
                StreamCount { count: 2 }
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'aborts flow via signal'
    async fn test_aborts_flow_via_signal(registry: Registry) {
        let test_flow = define_flow(
            &registry,
            "abortableFlow",
            |input: String, args: ActionFnArg<()>| async move {
                loop {
                    if args.abort_signal.is_cancelled() {
                        break;
                    }
                    sleep(Duration::from_millis(1)).await;
                }
                Ok(format!("done {}", input))
            },
        );

        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn({
            let flow = test_flow.clone();
            let token = cancellation_token.clone();
            async move { run_test_flow(&flow, "foo".to_string(), None, token).await }
        });

        sleep(Duration::from_millis(10)).await;
        cancellation_token.cancel();

        let result = handle.await.unwrap().unwrap();
        assert_eq!(result, "done foo");
    }
}

#[cfg(test)]
/// 'telemetry'
mod telemetry_test {
    use super::*;
    use genkit_core::action::define_action;
    use genkit_core::registry::ActionType;
    use genkit_core::telemetry::TelemetryConfig;
    use once_cell::sync::{Lazy, OnceCell};
    use opentelemetry::Value as OTelValue;
    use opentelemetry_sdk::{
        error::{OTelSdkError, OTelSdkResult},
        trace::{self, SpanData, SpanExporter, SpanProcessor},
    };
    use serde_json::Value as JsonValue;
    use std::sync::{Arc, Mutex, MutexGuard};

    // A single mutex to ensure tests run serially, as they modify global state.
    static TELEMETRY_TEST_MUTEX: Mutex<()> = Mutex::new(());

    // Type aliases to simplify the complex nested types.
    type SharedSpans = Arc<Mutex<Vec<SpanData>>>;
    type SwappableSharedSpans = Arc<Mutex<SharedSpans>>;

    // The global state that our swappable exporter will use. It holds a pointer
    // to the Vec<SpanData> of the currently running test.
    static CURRENT_SPANS: Lazy<SwappableSharedSpans> =
        Lazy::new(|| Arc::new(Mutex::new(Arc::new(Mutex::new(Vec::new())))));

    // A SpanExporter that can have its destination swapped out.
    #[derive(Debug, Clone)]
    struct SwappableSpanExporter {
        // A pointer to the Arc that holds the current test's span vector.
        spans_arc: SwappableSharedSpans,
    }

    impl SpanExporter for SwappableSpanExporter {
        fn export(
            &self,
            batch: Vec<SpanData>,
        ) -> impl futures::Future<Output = std::result::Result<(), OTelSdkError>> + std::marker::Send
        {
            // Get the Arc for the currently active test's spans.
            let current_spans = self.spans_arc.lock().unwrap().clone();
            // Lock that Arc's Mutex to write the spans.
            current_spans.lock().unwrap().extend(batch);
            Box::pin(async { Ok(()) })
        }
        fn shutdown(&mut self) -> OTelSdkResult {
            Ok(())
        }
    }

    // Global flag to ensure telemetry is only initialized once.
    static TELEMETRY_INIT: OnceCell<()> = OnceCell::new();

    // Initializes the global telemetry with our swappable exporter. Only runs once.
    fn init_test_telemetry() {
        TELEMETRY_INIT.get_or_init(|| {
            let exporter = SwappableSpanExporter {
                spans_arc: CURRENT_SPANS.clone(),
            };
            let processor: Box<dyn SpanProcessor> =
                Box::new(trace::SimpleSpanProcessor::new(exporter));
            genkit_core::tracing::enable_telemetry(TelemetryConfig {
                span_processors: vec![processor],
                ..Default::default()
            })
            .unwrap();
        });
    }

    // The test harness that each test will use.
    struct TestHarness {
        // Locks the test execution to be serial.
        _guard: MutexGuard<'static, ()>,
        // Holds the spans collected *only* for this test instance.
        spans_for_this_test: SharedSpans,
    }

    impl TestHarness {
        fn new() -> Self {
            // Ensure serial execution.
            let guard = TELEMETRY_TEST_MUTEX.lock().unwrap();

            // Make sure the global telemetry is set up.
            init_test_telemetry();

            // Create a new clean slate for this test's spans.
            let spans_for_this_test: SharedSpans = Arc::new(Mutex::new(Vec::new()));

            // Point the global exporter to this test's span vector.
            *CURRENT_SPANS.lock().unwrap() = spans_for_this_test.clone();

            TestHarness {
                _guard: guard,
                spans_for_this_test,
            }
        }

        // Get the spans collected during this test run.
        fn get_spans(&self) -> Vec<SpanData> {
            self.spans_for_this_test.lock().unwrap().clone()
        }
    }

    #[fixture]
    fn harness() -> TestHarness {
        TestHarness::new()
    }

    fn attributes_to_json_value(attrs: &[opentelemetry::KeyValue]) -> JsonValue {
        let map: serde_json::Map<String, JsonValue> = attrs
            .iter()
            .map(|kv| {
                let value = match &kv.value {
                    OTelValue::String(s) => JsonValue::String(s.to_string()),
                    OTelValue::Bool(b) => JsonValue::Bool(*b),
                    OTelValue::I64(i) => JsonValue::Number(serde_json::Number::from(*i)),
                    OTelValue::F64(f) => serde_json::Number::from_f64(*f)
                        .map(JsonValue::Number)
                        .unwrap_or(JsonValue::Null),
                    other => JsonValue::String(other.to_string()),
                };
                (kv.key.to_string(), value)
            })
            .collect();
        JsonValue::Object(map)
    }

    fn create_test_flow(registry: &Registry) -> Flow<String, String, ()> {
        define_flow(
            registry,
            "testFlow",
            |input: String, _: ActionFnArg<()>| async move { Ok(format!("bar {}", input)) },
        )
    }

    #[rstest]
    #[tokio::test]
    /// 'should create a trace'
    async fn test_create_a_trace(registry: Registry, harness: TestHarness) {
        let test_flow = create_test_flow(&registry);
        let mut labels = HashMap::new();
        labels.insert("custom".to_string(), "label".to_string());
        let mut context = ActionContext::default();
        context
            .additional_context
            .insert("user".to_string(), json!("pavel"));
        let options = ActionRunOptions {
            telemetry_labels: Some(labels),
            context: Some(context),
            ..Default::default()
        };

        let result = test_flow
            .run("foo".to_string(), Some(options))
            .await
            .unwrap();
        assert_eq!(result.result, "bar foo");

        let spans = harness.get_spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name.as_ref(), "testFlow");

        let expected_attrs = json!({
            "genkit:input": "\"foo\"",
            "genkit:isRoot": true,
            "genkit:metadata:subtype": "flow",
            "genkit:metadata:context": "{\"user\":\"pavel\"}",
            "genkit:name": "testFlow",
            "genkit:output": "\"bar foo\"",
            "genkit:path": "/{testFlow,t:flow}",
            "genkit:state": "success",
            "genkit:type": "action",
            "custom": "label"
        });
        assert_eq!(attributes_to_json_value(&span.attributes), expected_attrs);
    }

    #[rstest]
    #[tokio::test]
    /// 'should create a trace when streaming'
    async fn test_create_a_trace_when_streaming(registry: Registry, harness: TestHarness) {
        let test_flow = create_test_flow(&registry);
        let mut labels = HashMap::new();
        labels.insert("custom".to_string(), "label".to_string());
        let options = ActionRunOptions {
            telemetry_labels: Some(labels),
            ..Default::default()
        };

        let response = test_flow.stream("foo".to_string(), Some(options));
        let result = response.output.await.unwrap();
        assert_eq!(result, "bar foo");

        let spans = harness.get_spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        assert_eq!(span.name.as_ref(), "testFlow");

        let expected_attrs = json!({
            "genkit:input": "\"foo\"",
            "genkit:isRoot": true,
            "genkit:metadata:subtype": "flow",
            "genkit:metadata:context": "{}",
            "genkit:name": "testFlow",
            "genkit:output": "\"bar foo\"",
            "genkit:path": "/{testFlow,t:flow}",
            "genkit:state": "success",
            "genkit:type": "action",
            "custom": "label"
        });
        assert_eq!(attributes_to_json_value(&span.attributes), expected_attrs);
    }

    #[rstest]
    #[tokio::test]
    /// 'records traces of nested actions'
    async fn test_records_traces_of_nested_actions(registry: Registry, harness: TestHarness) {
        let test_action = define_action(
            &registry,
            ActionType::Tool,
            "testAction",
            |_: (), _: ActionFnArg<()>| async { Ok("bar".to_string()) },
        );
        let test_flow = define_flow(
            &registry,
            "testFlow",
            move |_: String, _: ActionFnArg<()>| {
                let test_action = test_action.clone();
                async move {
                    run("custom", move || async move {
                        let res = test_action.run((), None).await?.result;
                        Ok(format!("foo {}", res))
                    })
                    .await
                }
            },
        );

        let mut context = ActionContext::default();
        context
            .additional_context
            .insert("user".to_string(), json!("pavel"));
        let options = ActionRunOptions {
            context: Some(context),
            ..Default::default()
        };
        let result = test_flow
            .run("foo".to_string(), Some(options))
            .await
            .unwrap();
        assert_eq!(result.result, "foo bar");

        let spans = harness.get_spans();
        assert_eq!(spans.len(), 3);

        // Find spans by name to avoid depending on export order.
        let action_span = spans.iter().find(|s| s.name == "testAction").unwrap();
        let custom_span = spans.iter().find(|s| s.name == "custom").unwrap();
        let flow_span = spans.iter().find(|s| s.name == "testFlow").unwrap();

        // Assertions for testAction span
        let expected_action_attrs = json!({
            "genkit:input": "null",
            "genkit:metadata:subtype": "tool",
            "genkit:name": "testAction",
            "genkit:output": "\"bar\"",
            "genkit:path": "/{testFlow,t:flow}/{custom,t:flowStep}/{testAction,t:action,s:tool}",
            "genkit:state": "success",
            "genkit:type": "action"
        });
        assert_eq!(
            attributes_to_json_value(&action_span.attributes),
            expected_action_attrs
        );

        // Assertions for custom span
        let expected_custom_attrs = json!({
            "genkit:name": "custom",
            "genkit:output": "\"foo bar\"",
            "genkit:path": "/{testFlow,t:flow}/{custom,t:flowStep}",
            "genkit:state": "success",
            "genkit:type": "flowStep"
        });
        assert_eq!(
            attributes_to_json_value(&custom_span.attributes),
            expected_custom_attrs
        );

        // Assertions for testFlow span
        let expected_flow_attrs = json!({
            "genkit:input": "\"foo\"",
            "genkit:isRoot": true,
            "genkit:metadata:subtype": "flow",
            "genkit:metadata:context": "{\"user\":\"pavel\"}",
            "genkit:name": "testFlow",
            "genkit:output": "\"foo bar\"",
            "genkit:path": "/{testFlow,t:flow}",
            "genkit:state": "success",
            "genkit:type": "action"
        });
        assert_eq!(
            attributes_to_json_value(&flow_span.attributes),
            expected_flow_attrs
        );
    }
}
