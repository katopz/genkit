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

use genkit::{
    model::{FinishReason, GenerateRequest},
    Media, MessageData, Part, ToolResponse,
};
use genkit_vertexai::model::{
    gemini::VertexGeminiResponse,
    helpers::{to_genkit_response, to_vertex_request},
};
use rstest::rstest;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;

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
            messages: vec![MessageData::model(vec![Part::tool_request(
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
            ])],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "function",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "tellAFunnyJoke",
                            "response": {
                                "content": "Why did the dogs cross the road?",
                                "name": "tellAFunnyJoke"
                            }
                        }
                    }
                ]
            }]
        })
    )]
    #[case(
        "should transform genkit message (tool response content with ref) correctly",
        GenerateRequest {
            messages: vec![MessageData::tool(vec![
                Part {
                    tool_response: Some(ToolResponse {
                        name: "tellAFunnyJoke".to_string(),
                        output: Some(json!("Why did the dogs cross the road?")),
                        r#ref: Some("1".to_string()),
                    }),
                    ..Default::default()
                },
                Part {
                    tool_response: Some(ToolResponse {
                        name: "tellAnotherFunnyJoke".to_string(),
                        output: Some(json!("To get to the other side.")),
                        r#ref: Some("0".to_string()),
                    }),
                    ..Default::default()
                }
            ])],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "function",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "tellAnotherFunnyJoke",
                            "response": {
                                "content": "To get to the other side.",
                                "name": "tellAnotherFunnyJoke"
                            }
                        }
                    },
                    {
                        "functionResponse": {
                            "name": "tellAFunnyJoke",
                            "response": {
                                "content": "Why did the dogs cross the road?",
                                "name": "tellAFunnyJoke"
                            }
                        }
                    }
                ]
            }]
        })
    )]
    #[case(
        "should transform genkit message (inline base64 image content) correctly",
        GenerateRequest {
            messages: vec![MessageData::user(vec![
                Part::text("describe the following image:"),
                Part {
                    media: Some(Media {
                        content_type: Some("image/jpeg".to_string()),
                        url: "data:image/jpeg;base64,/9j/4QDe/9k=".to_string(),
                    }),
                    ..Default::default()
                },
            ])],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": "describe the following image:"},
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": "/9j/4QDe/9k="
                        }
                    }
                ]
            }]
        })
    )]
    #[case(
        "should re-populate thoughtSignature from reasoning metadata",
        GenerateRequest {
            messages: vec![
                MessageData::model(vec![
                    Part {
                        reasoning: Some("".to_string()),
                        metadata: Some(HashMap::from([(
                            "thoughtSignature".to_string(),
                            Value::String("abc123".to_string()),
                        )])),
                        ..Default::default()
                    }
                ])
            ],
            ..Default::default()
        },
        json!({
            "contents": [{
                "role": "model",
                "parts": [{"thought": true, "thoughtSignature": "abc123"}]
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

    #[rstest]
    #[case(
        "should transform from system to user",
        GenerateRequest {
            messages: vec![
                MessageData::system(vec![Part::text("You are an expert in all things cats.")]),
                MessageData::user(vec![Part::text("Tell me about cats.")])
            ],
            ..Default::default()
        },
        json!({
            "role": "user",
            "parts": [{"text": "You are an expert in all things cats."}]
        }),
        json!([
            {
                "role": "user",
                "parts": [{"text": "Tell me about cats."}]
            }
        ])
    )]
    #[case(
        "should transform from system to user with multiple parts",
        GenerateRequest {
            messages: vec![
                MessageData::system(vec![
                    Part::text("You are an expert in all things animals."),
                    Part::text("You love cats.")
                ]),
                MessageData::user(vec![Part::text("Tell me about dogs.")])
            ],
            ..Default::default()
        },
        json!({
            "role": "user",
            "parts": [
                {"text": "You are an expert in all things animals."},
                {"text": "You love cats."}
            ]
        }),
        json!([
            {
                "role": "user",
                "parts": [{"text": "Tell me about dogs."}]
            }
        ])
    )]
    fn test_system_instruction_transformation(
        #[case] description: &str,
        #[case] input_request: GenerateRequest,
        #[case] expected_system_instruction: serde_json::Value,
        #[case] expected_contents: serde_json::Value,
    ) {
        let vertex_req = to_vertex_request(&input_request).unwrap();

        // Assert that the system instruction is correctly extracted and transformed.
        let system_instruction_json = serde_json::to_value(vertex_req.system_instruction).unwrap();
        let expected_si_json = serde_json::to_value(Some(expected_system_instruction)).unwrap();
        assert_eq!(
            system_instruction_json, expected_si_json,
            "Failed test (system_instruction): {}",
            description
        );

        // Assert that the remaining messages (contents) are correct.
        let contents_json = serde_json::to_value(vertex_req.contents).unwrap();
        assert_eq!(
            contents_json, expected_contents,
            "Failed test (contents): {}",
            description
        );
    }
}

#[cfg(test)]
/// fromGeminiCandidate
mod from_gemini_candidate_tests {
    use super::*;
    use genkit::model::CandidateData;
    use genkit_vertexai::model::gemini::{
        SafetyRating, VertexCandidate, VertexContent, VertexPart,
    };
    use serde_json::{Number, Value};

    // A temporary struct to help with comparing the finishReason as a lowercase string,
    // which matches the format of the ts test case.
    #[derive(Serialize, Debug)]
    #[serde(rename_all = "camelCase")]
    struct ComparableCandidate {
        index: u32,
        message: MessageData,
        finish_reason: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_message: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        custom: Option<serde_json::Value>,
    }

    impl From<CandidateData> for ComparableCandidate {
        fn from(c: CandidateData) -> Self {
            ComparableCandidate {
                index: c.index,
                message: c.message,
                finish_reason: c
                    .finish_reason
                    .unwrap_or(FinishReason::Unknown)
                    .to_string()
                    .to_lowercase(),
                finish_message: c.finish_message,
                custom: c.custom,
            }
        }
    }

    /// Recursively traverses a serde_json::Value and rounds any floating-point numbers.
    fn round_floats_in_json(value: &mut Value, precision: u32) {
        match value {
            Value::Object(map) => {
                for (_, v) in map {
                    round_floats_in_json(v, precision);
                }
            }
            Value::Array(arr) => {
                for v in arr {
                    round_floats_in_json(v, precision);
                }
            }
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    let factor = 10.0_f64.powi(precision as i32);
                    let rounded = (f * factor).round() / factor;
                    if let Some(new_n) = Number::from_f64(rounded) {
                        *n = new_n;
                    }
                }
            }
            _ => {}
        }
    }

    #[rstest]
    #[case(
        "should transform gemini candidate to genkit candidate (text parts) correctly",
        VertexCandidate {
            content: VertexContent {
                role: "model".to_string(),
                parts: vec![
                    VertexPart {
                        text: Some("Why did the dog go to the bank?\n\nTo get his bones cashed!".to_string()),
                        ..Default::default()
                    }
                ],
            },
            finish_reason: Some("STOP".to_string()),
            safety_ratings: Some(vec![
                SafetyRating {
                    category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.12074952),
                    severity: Some("HARM_SEVERITY_NEGLIGIBLE".to_string()),
                    severity_score: Some(0.18388656),
                },
                SafetyRating {
                    category: "HARM_CATEGORY_DANGEROUS_CONTENT".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.37874627),
                    severity: Some("HARM_SEVERITY_LOW".to_string()),
                    severity_score: Some(0.37227696),
                },
                SafetyRating {
                    category: "HARM_CATEGORY_HARASSMENT".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.3983479),
                    severity: Some("HARM_SEVERITY_LOW".to_string()),
                    severity_score: Some(0.22270013),
                },
                SafetyRating {
                    category: "HARM_CATEGORY_SEXUALLY_EXPLICIT".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: None,
                    severity: None,
                    severity_score: None,
                },
            ]),
            citation_metadata: None,
        },
        json!({
            "index": 0,
            "message": {
                "role": "model",
                "content": [
                    {
                        "text": "Why did the dog go to the bank?\n\nTo get his bones cashed!"
                    }
                ]
            },
            "finishReason": "stop",
            "custom": {
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.12074952,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.18388656
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.37874627,
                        "severity": "HARM_SEVERITY_LOW",
                        "severityScore": 0.37227696
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.39834791, // Adjusted for f32 precision
                        "severity": "HARM_SEVERITY_LOW",
                        "severityScore": 0.22270013
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }
        })
    )]
    #[case(
        "should transform gemini candidate to genkit candidate (function call parts) correctly",
        VertexCandidate {
            content: VertexContent {
                role: "model".to_string(),
                parts: vec![
                    VertexPart {
                        function_call: Some(serde_json::from_value(json!({"name": "tellAFunnyJoke", "args": {"topic": "dog"}})).unwrap()),
                        ..Default::default()
                    }
                ],
            },
            finish_reason: Some("STOP".to_string()),
            safety_ratings: Some(vec![
                SafetyRating {
                    category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.11858909),
                    severity: Some("HARM_SEVERITY_NEGLIGIBLE".to_string()),
                    severity_score: Some(0.11456649),
                },
                SafetyRating {
                    category: "HARM_CATEGORY_DANGEROUS_CONTENT".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.13857833),
                    severity: Some("HARM_SEVERITY_NEGLIGIBLE".to_string()),
                    severity_score: Some(0.11417085),
                },
                SafetyRating {
                    category: "HARM_CATEGORY_HARASSMENT".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.28012377),
                    severity: Some("HARM_SEVERITY_NEGLIGIBLE".to_string()),
                    severity_score: Some(0.112405084),
                },
                SafetyRating {
                    category: "HARM_CATEGORY_SEXUALLY_EXPLICIT".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: None,
                    severity: None,
                    severity_score: None,
                },
            ]),
            citation_metadata: None,
        },
        json!({
            "index": 0,
            "message": {
                "role": "model",
                "content": [
                    {
                        "toolRequest": {
                            "name": "tellAFunnyJoke",
                            "input": {
                                "topic": "dog"
                            },
                            "ref": "0"
                        }
                    }
                ]
            },
            "finishReason": "stop",
            "custom": {
                "safetyRatings": [
                     {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.11858909,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.11456649
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.13857833,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.11417085
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.28012377,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.112405084
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }
        })
    )]
    #[case(
        "should transform gemini candidate to genkit candidate (thought parts) correctly",
        VertexCandidate {
            content: VertexContent {
                role: "model".to_string(),
                parts: vec![
                    VertexPart {
                        thought: Some(true),
                        thought_signature: Some("abc123".to_string()),
                        ..Default::default()
                    },
                    VertexPart {
                        thought: Some(true),
                        text: Some("thought with text".to_string()),
                        thought_signature: Some("def456".to_string()),
                        ..Default::default()
                    },
                ],
            },
            finish_reason: Some("STOP".to_string()),
            safety_ratings: Some(vec![
                SafetyRating {
                    category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
                    probability: "NEGLIGIBLE".to_string(),
                    probability_score: Some(0.11858909),
                    severity: Some("HARM_SEVERITY_NEGLIGIBLE".to_string()),
                    severity_score: Some(0.11456649),
                },
            ]),
            citation_metadata: None,
        },
        json!({
            "index": 0,
            "message": {
                "role": "model",
                "content": [
                    {
                        "reasoning": "",
                        "metadata": {
                            "thoughtSignature": "abc123"
                        }
                    },
                    {
                        "reasoning": "thought with text",
                        "metadata": {
                            "thoughtSignature": "def456"
                        }
                    }
                ]
            },
            "finishReason": "stop",
            "custom": {
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "probability": "NEGLIGIBLE",
                        "probabilityScore": 0.11858909,
                        "severity": "HARM_SEVERITY_NEGLIGIBLE",
                        "severityScore": 0.11456649
                    }
                ]
            }
        })
    )]
    fn test_from_gemini_candidate(
        #[case] description: &str,
        #[case] gemini_candidate: VertexCandidate,
        #[case] mut expected_output: serde_json::Value,
    ) {
        let response = to_genkit_response(
            &GenerateRequest::default(),
            VertexGeminiResponse {
                candidates: vec![gemini_candidate],
                usage_metadata: None,
            },
        )
        .unwrap();

        let genkit_candidate = response.candidates.into_iter().next().unwrap();
        let comparable_candidate: ComparableCandidate = genkit_candidate.into();

        let mut result_json = serde_json::to_value(comparable_candidate).unwrap();

        // Round floats to handle precision differences.
        round_floats_in_json(&mut result_json, 8);
        round_floats_in_json(&mut expected_output, 8);

        assert_eq!(result_json, expected_output, "Failed test: {}", description);
    }
}

#[cfg(test)]
/// cleanSchema
mod clean_schema_tests {
    use genkit_vertexai::model::helpers::clean_schema;

    use super::*;

    #[test]
    fn test_strips_nulls_from_type() {
        let schema = json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "subtitle": {
                    "type": ["string", "null"]
                }
            },
            "required": ["title"],
            "additionalProperties": true,
            "$schema": "http://json-schema.org/draft-07/schema#"
        });

        let cleaned = clean_schema(schema);

        let expected = json!({
            "type": "object",
            "properties": {
                "title": {
                    "type": "string"
                },
                "subtitle": {
                    "type": "string"
                }
            },
            "required": ["title"]
        });

        assert_eq!(cleaned, expected);
    }
}

#[cfg(test)]
/// toGeminiTool
mod to_gemini_tool_tests {
    use genkit::ToolDefinition;
    use genkit_vertexai::model::helpers::to_gemini_tool;
    use serde_json::json;

    #[test]
    fn test_converts_schema_correctly() {
        let tool_def = ToolDefinition {
            name: "foo".to_string(),
            description: "tool foo".to_string(),
            input_schema: Some(json!({
                "type": "object",
                "properties": {
                    "simpleString": {
                        "type": ["string", "null"],
                        "description": "a string"
                    },
                    "simpleNumber": {
                        "type": "number",
                        "description": "a number"
                    },
                    "simpleBoolean": {
                        "type": "boolean",
                        "description": "a boolean"
                    },
                    "simpleArray": {
                        "type": "array",
                        "description": "an array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "simpleEnum": {
                        "type": "string",
                        "description": "an enum",
                        "enum": ["choice_a", "choice_b"]
                    }
                },
                "required": ["simpleString", "simpleNumber"]
            })),
            ..Default::default()
        };

        let result = to_gemini_tool(&tool_def).unwrap();
        let result_json = serde_json::to_value(result).unwrap();

        let expected_json = json!({
            "name": "foo",
            "description": "tool foo",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "simpleString": {
                        "type": "STRING",
                        "description": "a string",
                        "nullable": true
                    },
                    "simpleNumber": {
                        "type": "NUMBER",
                        "description": "a number"
                    },
                    "simpleBoolean": {
                        "type": "BOOLEAN",
                        "description": "a boolean"
                    },
                    "simpleArray": {
                        "type": "ARRAY",
                        "description": "an array",
                        "items": {
                            "type": "STRING"
                        }
                    },
                    "simpleEnum": {
                        "type": "STRING",
                        "description": "an enum",
                        "enum": ["choice_a", "choice_b"]
                    }
                },
                "required": ["simpleString", "simpleNumber"]
            }
        });

        // The properties can be in a different order, so we need to compare them carefully.
        let result_props = result_json["parameters"]["properties"].as_object().unwrap();
        let expected_props = expected_json["parameters"]["properties"]
            .as_object()
            .unwrap();

        assert_eq!(result_props.len(), expected_props.len());
        for (key, val) in expected_props {
            assert_eq!(
                result_props.get(key).unwrap(),
                val,
                "Mismatch in property: {}",
                key
            );
        }

        // Check other fields
        assert_eq!(result_json["name"], expected_json["name"]);
        assert_eq!(result_json["description"], expected_json["description"]);
        assert_eq!(
            result_json["parameters"]["type"],
            expected_json["parameters"]["type"]
        );
        assert_eq!(
            result_json["parameters"]["required"],
            expected_json["parameters"]["required"]
        );
    }
}
