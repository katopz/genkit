// Copyright 2025 Google LLC
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

use genkit_ai::{
    document::{Part, ToolRequest, ToolResponse},
    tool::{define_interrupt, define_tool, InterruptConfig, Resumable, ToolAction, ToolConfig},
};
use genkit_core::error::Result;
use genkit_core::{
    error::Error,
    registry::{ErasedAction, Registry},
};
use rstest::{fixture, rstest};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fmt::Debug, future::Future, pin::Pin, sync::Arc};

#[fixture]
fn registry() -> Registry {
    Registry::new()
}

/// Helper to retrieve a type-erased action from the registry and downcast it
/// to a concrete `ToolAction`. This is necessary because `define_tool` only
// registers the tool and doesn't return a handle to it.
async fn get_tool_action<I, O, S>(registry: &Registry, name: &str) -> Arc<ToolAction<I, O, S>>
where
    I: 'static,
    O: 'static,
    S: 'static,
{
    let erased_action = registry
        .lookup_action(&format!("/tool/{}", name))
        .await
        .unwrap();
    let any_action = erased_action.as_any();
    any_action
        .downcast_ref::<ToolAction<I, O, S>>()
        .map(|concrete| Arc::new(ToolAction(concrete.0.clone())))
        .unwrap()
}

#[derive(Default, JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
struct TestOutput {
    foo: String,
}

async fn assert_interrupt(result: Result<impl Serialize + Debug>, expected_metadata: &str) {
    assert!(result.is_err());
    if let Err(Error::Internal { message, .. }) = result {
        assert_eq!(message, format!("INTERRUPT::{}", expected_metadata));
    } else {
        panic!(
            "Expected an internal error for interrupt, but got {:?}",
            result
        );
    }
}

#[rstest]
#[tokio::test]
/// Corresponds to: it('should throw a simple interrupt with no metadata', ...)
async fn test_interrupt_with_no_metadata(registry: Registry) {
    define_interrupt::<(), ()>(
        &registry,
        InterruptConfig {
            name: "simple".to_string(),
            description: "simple interrupt".to_string(),
            ..Default::default()
        },
    );

    let tool = get_tool_action::<(), (), ()>(&registry, "simple").await;
    let result = tool.run((), None).await;
    assert_interrupt(result, "null").await;
}

#[rstest]
#[tokio::test]
/// Corresponds to: it('should throw a simple interrupt with fixed metadata', ...)
async fn test_interrupt_with_fixed_metadata(registry: Registry) {
    let factory = Arc::new(|_: ()| {
        Box::pin(async { Some(json!({ "foo": "bar" })) })
            as Pin<Box<dyn Future<Output = Option<Value>> + Send>>
    });
    define_interrupt::<(), ()>(
        &registry,
        InterruptConfig {
            name: "simple".to_string(),
            description: "simple interrupt".to_string(),
            request_metadata_factory: Some(factory),
            ..Default::default()
        },
    );
    let tool = get_tool_action::<(), (), ()>(&registry, "simple").await;
    let result = tool.run((), None).await;
    assert_interrupt(result, r#"{"foo":"bar"}"#).await;
}

#[rstest]
#[tokio::test]
/// Corresponds to: it('should throw a simple interrupt with function-returned metadata', ...)
async fn test_interrupt_with_function_metadata(registry: Registry) {
    let factory = Arc::new(|input: String| {
        Box::pin(async move { Some(json!({ "foo": input })) })
            as Pin<Box<dyn Future<Output = Option<Value>> + Send>>
    });

    define_interrupt::<String, ()>(
        &registry,
        InterruptConfig {
            name: "simple".to_string(),
            description: "simple interrupt".to_string(),
            input_schema: Some("".to_string()),
            request_metadata_factory: Some(factory),
            ..Default::default()
        },
    );
    let tool = get_tool_action::<String, (), ()>(&registry, "simple").await;
    let result = tool.run("bar".to_string(), None).await;
    assert_interrupt(result, r#"{"foo":"bar"}"#).await;
}

#[rstest]
#[tokio::test]
/// Corresponds to: it('should throw a simple interrupt with async function-returned metadata', ...)
async fn test_interrupt_with_async_function_metadata(registry: Registry) {
    // The implementation is identical to the sync version because the factory
    // already returns a Future.
    let factory = Arc::new(|input: String| {
        Box::pin(async move { Some(json!({ "foo": input })) })
            as Pin<Box<dyn Future<Output = Option<Value>> + Send>>
    });

    define_interrupt::<String, ()>(
        &registry,
        InterruptConfig {
            name: "simple".to_string(),
            description: "simple interrupt".to_string(),
            input_schema: Some("".to_string()),
            request_metadata_factory: Some(factory),
            ..Default::default()
        },
    );
    let tool = get_tool_action::<String, (), ()>(&registry, "simple").await;
    let result = tool.run("bar".to_string(), None).await;
    assert_interrupt(result, r#"{"foo":"bar"}"#).await;
}

#[rstest]
#[tokio::test]
/// Corresponds to: it('should register the reply schema / json schema as the output schema of the tool', ...)
async fn test_interrupt_registers_output_schema(registry: Registry) {
    define_interrupt::<(), TestOutput>(
        &registry,
        InterruptConfig {
            name: "simple".to_string(),
            description: "simple".to_string(),
            output_schema: Some(TestOutput {
                foo: "bar".to_string(),
            }),
            ..Default::default()
        },
    );

    let tool = get_tool_action::<(), TestOutput, ()>(&registry, "simple").await;
    let metadata = tool.metadata();

    let expected_schema = serde_json::to_value(schema_for!(TestOutput)).unwrap();
    assert_eq!(metadata.output_schema.as_ref().unwrap(), &expected_schema);
}

#[rstest]
#[tokio::test]
/// Corresponds to: .respond() -> 'constructs a ToolResponsePart'
async fn test_respond_constructs_tool_response_part(registry: Registry) {
    define_tool(
        &registry,
        ToolConfig::<(), String> {
            name: "test".to_string(),
            description: "test".to_string(),
            ..Default::default()
        },
        |_, _| async { Ok("".to_string()) },
    );
    let tool = get_tool_action::<(), String, ()>(&registry, "test").await;
    let request_part = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            input: Some(json!({})),
            ..Default::default()
        }),
        ..Default::default()
    };
    let response = tool
        .respond(&request_part, "output".to_string(), None)
        .unwrap();

    let expected = Part {
        tool_response: Some(ToolResponse {
            name: "test".to_string(),
            output: Some(json!("output")),
            ..Default::default()
        }),
        metadata: Some([("interruptResponse".to_string(), json!(true))].into()),
        ..Default::default()
    };
    assert_eq!(response, expected);
}

#[rstest]
#[tokio::test]
/// Corresponds to: .respond() -> 'includes metadata'
async fn test_respond_includes_metadata(registry: Registry) {
    define_tool(
        &registry,
        ToolConfig::<(), String> {
            name: "test".to_string(),
            description: "test".to_string(),
            ..Default::default()
        },
        |_, _| async { Ok("".to_string()) },
    );
    let tool = get_tool_action::<(), String, ()>(&registry, "test").await;
    let request_part = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            input: Some(json!({})),
            ..Default::default()
        }),
        ..Default::default()
    };
    let response = tool
        .respond(
            &request_part,
            "output".to_string(),
            Some(json!({ "extra": "data" })),
        )
        .unwrap();

    let expected = Part {
        tool_response: Some(ToolResponse {
            name: "test".to_string(),
            output: Some(json!("output")),
            ..Default::default()
        }),
        metadata: Some([("interruptResponse".to_string(), json!({ "extra": "data" }))].into()),
        ..Default::default()
    };
    assert_eq!(response, expected);
}

#[rstest]
#[tokio::test]
/// Corresponds to: .respond() -> 'validates schema'
async fn test_respond_validates_schema(registry: Registry) {
    #[derive(Default, JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
    struct MyNumber(i32);

    define_tool(
        &registry,
        ToolConfig::<(), MyNumber> {
            name: "test".to_string(),
            description: "test".to_string(),
            output_schema: Some(MyNumber(0)),
            ..Default::default()
        },
        |_, options| async move { Err((options.interrupt)(None)) },
    );

    let tool = get_tool_action::<(), MyNumber, ()>(&registry, "test").await;
    let request_part = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };

    let valid_response = tool.respond(&request_part, MyNumber(55), None).unwrap();
    let expected = Part {
        tool_response: Some(ToolResponse {
            name: "test".to_string(),
            output: Some(json!(MyNumber(55))),
            ..Default::default()
        }),
        metadata: Some([("interruptResponse".to_string(), json!(true))].into()),
        ..Default::default()
    };
    assert_eq!(valid_response, expected);
}

#[rstest]
#[tokio::test]
/// Corresponds to: .restart() -> 'constructs a ToolRequestPart'
async fn test_restart_constructs_tool_request_part(registry: Registry) {
    define_tool(
        &registry,
        ToolConfig::<(), ()> {
            name: "test".to_string(),
            description: "test".to_string(),
            ..Default::default()
        },
        |_, _| async { Ok(()) },
    );
    let tool = get_tool_action::<(), (), ()>(&registry, "test").await;

    let request_part = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            input: Some(json!({})),
            ..Default::default()
        }),
        ..Default::default()
    };
    let restarted_part = tool.restart(&request_part, None, None).unwrap();
    let expected = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            input: Some(json!({})),
            ..Default::default()
        }),
        metadata: Some([("resumed".to_string(), json!(true))].into()),
        ..Default::default()
    };

    assert_eq!(restarted_part, expected);
}

#[rstest]
#[tokio::test]
/// Corresponds to: .restart() -> 'includes metadata'
async fn test_restart_includes_metadata(registry: Registry) {
    define_tool(
        &registry,
        ToolConfig::<(), ()> {
            name: "test".to_string(),
            description: "test".to_string(),
            ..Default::default()
        },
        |_, _| async { Ok(()) },
    );
    let tool = get_tool_action::<(), (), ()>(&registry, "test").await;

    let request_part = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            input: Some(json!({})),
            ..Default::default()
        }),
        ..Default::default()
    };
    let restarted_part = tool
        .restart(&request_part, Some(json!({ "extra": "data" })), None)
        .unwrap();
    let expected = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            input: Some(json!({})),
            ..Default::default()
        }),
        metadata: Some([("resumed".to_string(), json!({ "extra": "data" }))].into()),
        ..Default::default()
    };
    assert_eq!(restarted_part, expected);
}

#[rstest]
#[tokio::test]
/// Corresponds to: .restart() -> 'validates schema'
async fn test_restart_validates_schema(registry: Registry) {
    #[derive(Default, Clone, JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
    struct ValidatedInput {
        #[schemars(length(min = 5))]
        text: String,
    }

    define_tool(
        &registry,
        ToolConfig::<ValidatedInput, ()> {
            name: "test".to_string(),
            description: "test".to_string(),
            input_schema: Some(ValidatedInput { text: "".into() }),
            ..Default::default()
        },
        |_, options| async move { Err((options.interrupt)(None)) },
    );
    let tool = get_tool_action::<ValidatedInput, (), ()>(&registry, "test").await;

    let request_part = Part {
        tool_request: Some(ToolRequest {
            name: "test".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Test with invalid input
    let result = tool.restart(
        &request_part,
        None,
        Some(ValidatedInput {
            text: "four".into(),
        }),
    );
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), Error::Validation(_)));

    // Test with valid input
    let valid_result = tool
        .restart(
            &request_part,
            None,
            Some(ValidatedInput {
                text: "long enough".into(),
            }),
        )
        .unwrap();

    let expected_request = ToolRequest {
        name: "test".to_string(),
        input: Some(json!({"text": "long enough"})),
        ..Default::default()
    };
    let expected = Part {
        tool_request: Some(expected_request),
        metadata: Some(
            [
                ("resumed".to_string(), json!(true)),
                ("replacedInput".to_string(), json!({})),
            ]
            .into(),
        ),
        ..Default::default()
    };
    assert_eq!(valid_result, expected);
}
