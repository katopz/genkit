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
        #[derive(JsonSchema, Deserialize, Clone)]
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
