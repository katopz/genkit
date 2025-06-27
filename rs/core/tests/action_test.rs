// Integration tests for actions.
// In Rust, it's common to have unit tests within the module itself (e.g., in action.rs)
// and integration tests in the `tests/` directory. This file serves as an
// integration test for the action system.

use futures::StreamExt;
use genkit_core::action::{ActionBuilder, ActionFnArg};
use genkit_core::async_utils::channel;
use genkit_core::context::ActionContext;
use genkit_core::error;
use genkit_core::registry::{ActionType, Registry};
use genkit_core::tracing::TraceContext;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_util::sync::CancellationToken;

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
struct TestInput {
    name: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq)]
struct TestOutput {
    greeting: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Debug, PartialEq, Clone)]
struct TestStreamChunk {
    count: i32,
}

#[cfg(test)]
mod test {
    use crate::*;

    #[tokio::test]
    async fn test_action_execution_with_context() {
        let mut registry = Registry::new();

        // Define an action using the builder
        let test_action = ActionBuilder::new(
            ActionType::Util,
            "testUtil",
            |input: TestInput, args: ActionFnArg<TestStreamChunk>| async move {
                assert!(args.context.is_some());
                let context = args.context.unwrap();
                let auth_user = context
                    .auth
                    .unwrap()
                    .get("user")
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string();

                Ok(TestOutput {
                    greeting: format!("Hello, {} from {}!", input.name, auth_user),
                })
            },
        )
        .build(&mut registry);

        // Manually construct the arguments for invocation
        let (chunk_tx, _chunk_rx) = channel();
        let args = ActionFnArg {
            streaming_requested: false,
            chunk_sender: chunk_tx,
            context: Some(ActionContext {
                auth: Some(json!({"user": "test-user"})),
                ..Default::default()
            }),
            trace: TraceContext {
                trace_id: "trace-1".into(),
                span_id: "span-1".into(),
            },
            abort_signal: CancellationToken::new(),
        };

        let input = TestInput {
            name: "Genkit".into(),
        };

        // Directly call the action's underlying function
        let result = test_action.func.run(input, args).await.unwrap();

        assert_eq!(result.greeting, "Hello, Genkit from test-user!");
    }

    #[tokio::test]
    async fn test_action_streaming() {
        let mut registry = Registry::new();

        // Define a streaming action
        let streaming_action = ActionBuilder::new(
            ActionType::Flow,
            "streamingFlow",
            |input: TestInput, args: ActionFnArg<TestStreamChunk>| async move {
                for i in 1..=3 {
                    args.chunk_sender.send(TestStreamChunk { count: i });
                }
                // Close the sender to signal the end of the stream.
                // Note: The async_utils::Channel doesn't have an explicit close on the sender side.
                // The stream ends when the sender is dropped.
                Ok(TestOutput {
                    greeting: format!("Streamed for {}", input.name),
                })
            },
        )
        .build(&mut registry);

        let input = TestInput {
            name: "Streamer".into(),
        };
        let context = Some(ActionContext {
            auth: Some(json!({"user": "stream-user"})),
            ..Default::default()
        });

        // Use the .stream() helper method
        let mut response = streaming_action.stream(input, context);

        // Collect chunks from the stream
        let mut chunks = Vec::new();
        while let Some(chunk) = response.stream.next().await {
            chunks.push(chunk);
        }

        // Await the final output
        let output = response.output.await.unwrap();

        assert_eq!(
            chunks,
            vec![
                TestStreamChunk { count: 1 },
                TestStreamChunk { count: 2 },
                TestStreamChunk { count: 3 }
            ]
        );
        assert_eq!(output.greeting, "Streamed for Streamer");
    }

    #[tokio::test]
    async fn test_action_aborts_via_signal() {
        let mut registry = Registry::new();
        let long_running_action = ActionBuilder::new(
            ActionType::Util,
            "longRunning",
            |_: (), args: ActionFnArg<()>| async move {
                tokio::select! {
                    _ = args.abort_signal.cancelled() => {
                        // The operation was cancelled.
                        Err(genkit_core::error::Error::new_user_facing(
                            genkit_core::status::StatusCode::Cancelled,
                            "Operation was cancelled by the user.",
                            None
                        ))
                    },
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                        // This part should not be reached in this test.
                        Ok(())
                    }
                }
            },
        )
        .build(&mut registry);

        let cancel_token = CancellationToken::new();

        let (chunk_tx, _chunk_rx) = channel();
        let args = ActionFnArg {
            streaming_requested: false,
            chunk_sender: chunk_tx,
            context: None,
            trace: TraceContext {
                trace_id: "trace-cancel".into(),
                span_id: "span-cancel".into(),
            },
            abort_signal: cancel_token.clone(),
        };

        let handle = tokio::spawn(async move { long_running_action.func.run((), args).await });

        // Cancel the task after a short delay
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        cancel_token.cancel();

        let result = handle.await.unwrap();
        assert!(result.is_err());

        if let Err(error::Error::UserFacing(status)) = result {
            assert_eq!(status.code, genkit_core::status::StatusCode::Cancelled);
            assert_eq!(status.message, "Operation was cancelled by the user.");
        } else {
            panic!("Expected a UserFacing error with Cancelled status");
        }
    }
}
