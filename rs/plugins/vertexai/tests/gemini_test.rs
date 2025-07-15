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

//! # Vertex AI Gemini Models
//!
//! This module provides the implementation for the Gemini family of models
//! on Vertex AI.

use genkit::{model::GenerateRequest, MessageData, Part};
use genkit_vertexai::model::gemini::{VertexCandidate, VertexContent, VertexPart};
use genkit_vertexai::model::helpers::{to_genkit_response, to_vertex_request};
use rstest::rstest;
use serde_json::json;

#[cfg(test)]
/// toGeminiMessages
mod to_gemini_message_tests {

    use super::*;

    #[rstest]
    #[case(
        "should transform genkit message (text content) correctly",
        GenerateRequest {
            messages: vec![MessageData::user(vec![Part::text("Tell a joke about dogs.")])],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Tell a joke about dogs."}]
            }]
        })
    )]
    #[case(
        "should transform genkit message (tool request content) correctly",
        GenerateRequest {
            messages: vec![MessageData::tool(vec![Part::tool_request(
                "tellAFunnyJoke",
                Some(json!({"topic": "dogs"})),
                None
            )])],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "name": "tellAFunnyJoke",
                        "args": { "topic": "dogs" }
                    }
                }]
            }]
        })
    )]
    #[case(
        "should transform genkit message (tool response content) correctly",
        GenerateRequest {
            messages: vec![MessageData::tool(vec![
                Part::tool_response(
                    "tellAFunnyJoke",
                    Some(json!("Why did the dogs cross the road?")),
                    None,
                ),
                Part::tool_response(
                    "tellAnotherFunnyJoke",
                    Some(json!("To get to the other side.")),
                    None,
                )
            ])],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "tool",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "tellAFunnyJoke",
                            "response": {
                                "content": "Why did the dogs cross the road?",
                                "name": "tellAFunnyJoke"
                            }
                        }
                    },
                    {
                        "functionResponse": {
                            "name": "tellAnotherFunnyJoke",
                            "response": {
                                "content": "To get to the other side.",
                                "name": "tellAnotherFunnyJoke"
                            }
                        }
                    }
                ]
            }]
        })
    )]
    fn test_to_vertex_request_messages(
        #[case] description: &str,
        #[case] input_request: GenerateRequest,
        #[case] expected_json: serde_json::Value,
    ) {
        let vertex_req = to_vertex_request(&input_request).unwrap();
        let result_json = serde_json::to_value(vertex_req).unwrap();
        let expected = expected_json.get("contents").unwrap();
        let result = result_json.get("contents").unwrap();

        assert_eq!(result, expected, "Failed test: {}", description);
    }
}

#[cfg(test)]
/// toGeminiSystemInstruction
mod to_gemini_system_instruction_tests {

    use super::*;

    #[test]
    fn test_to_vertex_request_with_system_instruction() {
        let req = GenerateRequest {
            messages: vec![
                MessageData::system(vec![Part::text("You are a cat expert.")]),
                MessageData::user(vec![Part::text("Tell me about cats.")]),
            ],
            ..Default::default()
        };

        let vertex_req = to_vertex_request(&req).unwrap();

        let expected_system_instruction = json!({
            "role": "user",
            "parts": [{"text": "You are a cat expert."}]
        });
        assert_eq!(
            serde_json::to_value(&vertex_req.system_instruction).unwrap(),
            serde_json::to_value(Some(expected_system_instruction)).unwrap()
        );

        let expected_contents = json!([
            {
                "role": "user",
                "parts": [{"text": "Tell me about cats."}]
            }
        ]);
        assert_eq!(
            serde_json::to_value(&vertex_req.contents).unwrap(),
            expected_contents
        );
    }

    #[rstest]
    #[case(
        "should transform from system to user",
        vec![VertexCandidate {
            content: VertexContent {
                role: "model".to_string(),
                parts: vec![VertexPart { text: Some("A funny joke.".to_string()), ..Default::default() }],
            },
            finish_reason: Some("STOP".to_string()),
        }],
        json!([{
            "index": 0,
            "message": {
                "role": "model",
                "content": [{"text": "A funny joke."}]
            },
            "finishReason": "stop"
        }])
    )]
    fn test_to_genkit_response(
        #[case] description: &str,
        #[case] candidates: Vec<VertexCandidate>,
        #[case] expected_candidates: serde_json::Value,
    ) {
        use genkit_vertexai::model::gemini::VertexGeminiResponse;

        let vertex_response = VertexGeminiResponse {
            candidates,
            usage_metadata: None,
        };
        let genkit_response =
            to_genkit_response(&GenerateRequest::default(), vertex_response).unwrap();

        let mut result_json = serde_json::to_value(genkit_response.candidates).unwrap();

        if let Some(candidates_arr) = result_json.as_array_mut() {
            for candidate in candidates_arr {
                if let Some(c_obj) = candidate.as_object_mut() {
                    c_obj.remove("finishMessage");
                    if let Some(msg) = c_obj.get_mut("message") {
                        if let Some(msg_obj) = msg.as_object_mut() {
                            msg_obj.remove("metadata");
                            if let Some(content) = msg_obj.get_mut("content") {
                                if let Some(content_arr) = content.as_array_mut() {
                                    for item in content_arr {
                                        if let Some(item_obj) = item.as_object_mut() {
                                            if let Some(tr) = item_obj.get_mut("toolRequest") {
                                                if let Some(tr_obj) = tr.as_object_mut() {
                                                    tr_obj.remove("ref");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        assert_eq!(
            result_json, expected_candidates,
            "Failed test: {}",
            description
        );
    }
}
