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

use genkit_ai::{
    generate::GenerateResponse,
    message::MessageData,
    model::{CandidateData, FinishReason, GenerateRequest, GenerateResponseData},
    Part, Role, ToolRequest,
};
use rstest::rstest;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// #toJSON()

#[test]
fn test_to_json_data_serialization() {
    let response_data = GenerateResponseData {
        candidates: vec![CandidateData {
            message: MessageData {
                role: Role::Model,
                content: vec![Part::text(r#"{"name": "Bob"}"#)],
                ..Default::default()
            },
            finish_reason: Some(FinishReason::Stop),
            ..Default::default()
        }],
        ..Default::default()
    };
    let response = GenerateResponse::<Value>::new(&response_data, None);
    let json_data = response.to_json_data().unwrap();

    let expected_data = GenerateResponseData {
        candidates: vec![CandidateData {
            index: 0,
            message: MessageData {
                role: Role::Model,
                content: vec![Part::text(r#"{"name": "Bob"}"#)],
                ..Default::default()
            },
            finish_reason: Some(FinishReason::Stop),
            finish_message: None,
        }],
        usage: None,
        custom: None,
        operation: None,
        aggregated: None,
    };

    // GenerateResponseData does not implement `PartialEq` because CandidateData does not.
    // We can serialize both to JSON and compare the values for a deep comparison.
    let json_data_value = serde_json::to_value(json_data).unwrap();
    let expected_data_value = serde_json::to_value(expected_data).unwrap();
    assert_eq!(json_data_value, expected_data_value);
}

#[derive(Deserialize, Debug, PartialEq, Default, Serialize, Clone)]
struct TestOutput {
    name: String,
    age: Option<u32>,
}

// #output()
#[rstest]
#[case(
        "return structured data from the data part",
        MessageData {
            role: Role::Model,
            content: vec![Part { data: Some(json!({"name": "Alice", "age": 30})), ..Default::default() }],
            ..Default::default()
        },
        TestOutput { name: "Alice".to_string(), age: Some(30) }
    )]
#[case(
        "parse JSON from text when the data part is absent",
        MessageData {
            role: Role::Model,
            content: vec![Part::text(r#"{"name": "Bob"}"#)],
            ..Default::default()
        },
        TestOutput { name: "Bob".to_string(), age: None }
    )]
fn test_output_parsing(
    #[case] desc: &str,
    #[case] message_data: MessageData,
    #[case] expected_output: TestOutput,
) {
    let response_data = GenerateResponseData {
        candidates: vec![CandidateData {
            message: message_data,
            ..Default::default()
        }],
        ..Default::default()
    };
    let response = GenerateResponse::<TestOutput>::new(&response_data, None);
    let output = response.output().unwrap();
    assert_eq!(output, expected_output, "Test failed: {}", desc);
}

#[test]
// #assertValid()
fn test_assert_valid_blocked() {
    let response_data = GenerateResponseData {
        candidates: vec![CandidateData {
            finish_reason: Some(FinishReason::Blocked),
            finish_message: Some("Content was blocked".to_string()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let response = GenerateResponse::<Value>::new(&response_data, None);
    let err = response.assert_valid().unwrap_err();
    assert!(err.to_string().contains("Generation blocked"));
}

#[test]
fn test_assert_valid_no_message() {
    let response_data = GenerateResponseData {
        candidates: vec![],
        ..Default::default()
    };
    let response = GenerateResponse::<Value>::new(&response_data, None);
    let err = response.assert_valid().unwrap_err();
    assert!(err.to_string().contains("Model did not generate a message"));
}

#[derive(JsonSchema, Deserialize, Debug, PartialEq)]
struct ValidSchema {
    name: String,
    age: u32,
}

#[test]
fn test_assert_valid_schema_failure() {
    let response_data = GenerateResponseData {
        candidates: vec![CandidateData {
            message: MessageData {
                role: Role::Model,
                content: vec![Part::text(r#"{"name": "John", "age": "30"}"#)],
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };
    let schema_value = serde_json::to_value(schemars::schema_for!(ValidSchema)).unwrap();
    let request = GenerateRequest {
        messages: vec![],
        output: Some(json!({ "schema": schema_value })),
        ..Default::default()
    };
    let response = GenerateResponse::<Value>::new(&response_data, Some(request));
    let err = response.assert_valid_schema().unwrap_err();
    assert!(err.to_string().contains("Schema validation failed"));
}

#[test]
fn test_assert_valid_schema_success() {
    let response_data = GenerateResponseData {
        candidates: vec![CandidateData {
            message: MessageData {
                role: Role::Model,
                content: vec![Part::text(r#"{"name": "John", "age": 30}"#)],
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };
    let schema_value = serde_json::to_value(schemars::schema_for!(ValidSchema)).unwrap();
    let request = GenerateRequest {
        messages: vec![],
        output: Some(json!({ "schema": schema_value })),
        ..Default::default()
    };
    let response = GenerateResponse::<Value>::new(&response_data, Some(request));
    assert!(response.assert_valid_schema().is_ok());
}

// #toolRequests()
#[rstest]
#[case("returns empty array if no tools requests found", vec![Part::text(r#"{"abc":"123"}"#)], vec![])]
#[case("returns tool call if present",
        vec![Part {
            tool_request: Some(ToolRequest {
                name: "foo".to_string(),
                ref_id: Some("abc".to_string()),
                input: Some(json!("banana")),
            }),
            ..Default::default()
        }],
        vec![ToolRequest {
            name: "foo".to_string(),
            ref_id: Some("abc".to_string()),
            input: Some(json!("banana")),
        }]
    )]
#[case("returns all tool calls",
        vec![
            Part {
                tool_request: Some(ToolRequest {
                    name: "foo".to_string(),
                    ref_id: Some("abc".to_string()),
                    input: Some(json!("banana")),
                }),
                ..Default::default()
            },
            Part {
                tool_request: Some(ToolRequest {
                    name: "bar".to_string(),
                    ref_id: Some("bcd".to_string()),
                    input: Some(json!("apple")),
                }),
                ..Default::default()
            }
        ],
        vec![
            ToolRequest {
                name: "foo".to_string(),
                ref_id: Some("abc".to_string()),
                input: Some(json!("banana")),
            },
            ToolRequest {
                name: "bar".to_string(),
                ref_id: Some("bcd".to_string()),
                input: Some(json!("apple")),
            }
        ]
    )]
fn test_tool_requests_extraction(
    #[case] desc: &str,
    #[case] content: Vec<Part>,
    #[case] expected_requests: Vec<ToolRequest>,
) {
    let response_data = GenerateResponseData {
        candidates: vec![CandidateData {
            message: MessageData {
                role: Role::Model,
                content,
                ..Default::default()
            },
            ..Default::default()
        }],
        ..Default::default()
    };
    let response = GenerateResponse::<Value>::new(&response_data, None);
    let tool_requests: Vec<ToolRequest> = response
        .tool_requests()
        .unwrap()
        .into_iter()
        .cloned()
        .collect();
    assert_eq!(tool_requests, expected_requests, "Test failed: {}", desc);
}
