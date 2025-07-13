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
mod test {
    use genkit_core::{
        action::{
            define_action, ActionBuilder, ActionFnArg, ActionMiddleware, ActionMiddlewareNext,
            ActionRunOptions, StreamingResponse,
        },
        context::ActionContext,
        registry::{ActionType, Registry},
    };
    use rstest::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    #[fixture]
    fn registry() -> Arc<Mutex<Registry>> {
        Arc::new(Mutex::new(Registry::new()))
    }

    #[rstest]
    #[tokio::test]
    async fn test_apply_middleware() {
        let middleware1 = ActionMiddleware {
            f: Arc::new(
                |input: String, options, next: ActionMiddlewareNext<String, i32, ()>| {
                    Box::pin(async move {
                        let mut result = (next.f)(input + "middle1", options).await?;
                        result.result += 1;
                        Ok(result)
                    })
                },
            ),
            _phantom: std::marker::PhantomData,
        };

        let middleware2 = ActionMiddleware {
            f: Arc::new(
                |input: String, options, next: ActionMiddlewareNext<String, i32, ()>| {
                    Box::pin(async move {
                        let mut result = (next.f)(input + "middle2", options).await?;
                        result.result += 2;
                        Ok(result)
                    })
                },
            ),
            _phantom: std::marker::PhantomData,
        };

        let action = ActionBuilder::new(
            ActionType::Util,
            "foo",
            |input: String, _: ActionFnArg<()>| async move { Ok(input.len() as i32) },
        )
        .with_middleware(vec![middleware1, middleware2])
        .build();

        let result = action.run("foo".to_string(), None).await.unwrap();

        // "foomiddle2middle1".len() = 17. 17 + 1 + 2 = 20.
        // Middleware is applied in reverse order of definition to form a call stack.
        // The effective order of execution is middleware2, then middleware1.
        // input -> "foo"
        // middleware2 adds "middle2" -> "foomiddle2"
        // middleware1 adds "middle1" -> "foomiddle2middle1" (length 17)
        // action fn returns 17
        // middleware1 adds 1 -> 18
        // middleware2 adds 2 -> 20
        assert_eq!(result.result, 20);
    }

    #[rstest]
    #[tokio::test]
    async fn test_returns_telemetry_info(registry: Arc<Mutex<Registry>>) {
        let test_action = define_action(
            &registry.lock().unwrap(),
            ActionType::Util,
            "foo",
            |input: String, _: ActionFnArg<()>| async move { Ok(input.len() as i32) },
        );

        let action_result = test_action.run("foo".to_string(), None).await.unwrap();

        assert_eq!(action_result.result, 3);
        assert!(!action_result.telemetry.trace_id.is_empty());
        assert!(!action_result.telemetry.span_id.is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn test_run_action_with_options(registry: Arc<Mutex<Registry>>) {
        #[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
        struct MyContext {
            foo: String,
        }

        let passed_context = Arc::new(Mutex::new(None));
        let passed_context_clone = passed_context.clone();

        let test_action = define_action(
            &registry.lock().unwrap(),
            ActionType::Util,
            "foo",
            move |input: String, args: ActionFnArg<i32>| {
                let passed_context_clone_inner = passed_context.clone();
                async move {
                    if let Some(ctx) = args.context {
                        *passed_context_clone_inner.lock().unwrap() =
                            serde_json::from_value::<MyContext>(
                                ctx.get("my_context").unwrap().clone(),
                            )
                            .ok();
                    }
                    let _ = args.chunk_sender.send(1);
                    let _ = args.chunk_sender.send(2);
                    let _ = args.chunk_sender.send(3);
                    Ok(input.len() as i32)
                }
            },
        );

        let chunks = Arc::new(Mutex::new(Vec::new()));
        let chunks_clone = chunks.clone();

        let mut context_map = HashMap::new();
        context_map.insert("my_context".to_string(), json!({"foo": "bar"}));
        let context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let options = ActionRunOptions {
            on_chunk: Some(Arc::new(move |chunk| {
                if let Ok(c) = chunk {
                    chunks_clone.lock().unwrap().push(c);
                }
            })),
            context: Some(context),
            ..Default::default()
        };

        let result = test_action
            .run("1234".to_string(), Some(options))
            .await
            .unwrap();

        assert_eq!(result.result, 4);
        assert_eq!(
            *passed_context_clone.lock().unwrap(),
            Some(MyContext {
                foo: "bar".to_string()
            })
        );
        assert_eq!(*chunks.lock().unwrap(), vec![1, 2, 3]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_should_stream_the_response(registry: Arc<Mutex<Registry>>) {
        use futures::stream::TryStreamExt;
        #[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
        struct StreamCount {
            count: i32,
        }

        let test_action = define_action(
            &registry.lock().unwrap(),
            ActionType::Util,
            "hello",
            |input: String, args: ActionFnArg<StreamCount>| async move {
                let _ = args.chunk_sender.send(StreamCount { count: 1 });
                let _ = args.chunk_sender.send(StreamCount { count: 2 });
                let _ = args.chunk_sender.send(StreamCount { count: 3 });
                Ok(format!("hi {}", input))
            },
        );

        let streaming_response: StreamingResponse<String, StreamCount> =
            test_action.stream("Pavel".to_string(), None);

        let (chunks_result, output_result) = tokio::join!(
            streaming_response.stream.try_collect::<Vec<_>>(),
            streaming_response.output
        );

        let received_chunks = chunks_result.unwrap();
        let final_output = output_result.unwrap();

        assert_eq!(final_output, "hi Pavel".to_string());
        assert_eq!(
            received_chunks,
            vec![
                StreamCount { count: 1 },
                StreamCount { count: 2 },
                StreamCount { count: 3 }
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_should_inherit_context_from_parent(registry: Arc<Mutex<Registry>>) {
        #[derive(Serialize, Deserialize, Debug, PartialEq, Clone, JsonSchema)]
        struct Auth {
            email: String,
        }

        let child_action = define_action(
            &registry.lock().unwrap(),
            ActionType::Util,
            "child",
            |_: (), args: ActionFnArg<()>| async move {
                let email = args
                    .context
                    .and_then(|ctx| ctx.get("auth").cloned())
                    .and_then(|v| serde_json::from_value::<Auth>(v).ok())
                    .map(|auth| auth.email)
                    .unwrap_or_else(|| "unknown".to_string());
                Ok(format!("hi {}", email))
            },
        );

        let parent_action = define_action(
            &registry.lock().unwrap(),
            ActionType::Util,
            "parent",
            move |_: (), _: ActionFnArg<()>| {
                let child_action_clone = child_action.clone();
                async move {
                    // Because the parent's `run` will establish the context, the child's `run`
                    // will pick it up from the task-local storage.
                    let child_result = child_action_clone.run((), None).await.unwrap();
                    Ok(child_result.result)
                }
            },
        );

        let mut context_map = HashMap::new();
        context_map.insert("auth".to_string(), json!({"email": "a@b.c"}));
        let context = ActionContext {
            additional_context: context_map,
            ..Default::default()
        };

        let options = ActionRunOptions {
            context: Some(context),
            ..Default::default()
        };

        let result = parent_action.run((), Some(options)).await.unwrap();

        assert_eq!(result.result, "hi a@b.c");
    }

    #[rstest]
    #[tokio::test]
    async fn test_should_include_trace_info_in_context(registry: Arc<Mutex<Registry>>) {
        let test_action = define_action(
            &registry.lock().unwrap(),
            ActionType::Util,
            "foo",
            |_: (), args: ActionFnArg<()>| async move {
                Ok(format!(
                    "traceId={} spanId={}",
                    !args.trace.trace_id.is_empty(),
                    !args.trace.span_id.is_empty()
                ))
            },
        );

        let result = test_action.run((), None).await.unwrap();
        println!("{:?}", result.result);
        assert_eq!(result.result, "traceId=true spanId=true");
    }
}
