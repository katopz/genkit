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
    tool::{define_interrupt, define_tool, Resumable, ToolAction, ToolConfig},
};
use genkit_core::{
    error::Error,
    registry::{ErasedAction, Registry},
};
use rstest::{fixture, rstest};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

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

#[rstest]
#[tokio::test]
async fn test_define_interrupt_throws_interrupt(mut registry: Registry) {
    define_interrupt::<(), ()>(
        &mut registry,
        ToolConfig {
            name: "simple".to_string(),
            description: "simple interrupt".to_string(),
            ..Default::default()
        },
    );

    let tool = get_tool_action::<(), (), ()>(&registry, "simple").await;
    let result = tool.run((), None).await;

    assert!(result.is_err());
    if let Err(Error::Internal { message, .. }) = result {
        // The runner for `define_interrupt` calls the interrupt function with `None`,
        // which serializes to the string "null".
        assert_eq!(message, "INTERRUPT::null");
    } else {
        panic!(
            "Expected an internal error for interrupt, but got {:?}",
            result
        );
    }
}

#[rstest]
#[tokio::test]
async fn test_define_interrupt_registers_output_schema(mut registry: Registry) {
    define_interrupt::<(), TestOutput>(
        &mut registry,
        ToolConfig {
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
async fn test_respond_constructs_tool_response_part(mut registry: Registry) {
    define_tool(
        &mut registry,
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
async fn test_respond_includes_metadata(mut registry: Registry) {
    define_tool(
        &mut registry,
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
async fn test_respond_validates_schema(mut registry: Registry) {
    #[derive(Default, JsonSchema, Serialize, Deserialize, Debug, PartialEq)]
    struct MyNumber(i32);

    define_tool(
        &mut registry,
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

    // The Rust `respond` method is strongly typed, so we can't pass invalid data
    // in the same way the TypeScript test does with `as any`.
    // Instead, we confirm that valid data passes the schema check inside `respond`.
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
async fn test_restart_constructs_tool_request_part(mut registry: Registry) {
    define_tool(
        &mut registry,
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
    let restarted_part = tool.restart(&request_part, None).unwrap();
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
async fn test_restart_includes_metadata(mut registry: Registry) {
    define_tool(
        &mut registry,
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
        .restart(&request_part, Some(json!({ "extra": "data" })))
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

// Note: The `restart().validates schema` test from the TypeScript suite was not ported
// because the Rust implementation of `Resumable::restart` does not include logic
// for replacing the input (`replaceInput`), which is what that test validates.
