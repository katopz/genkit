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

//! # Model Middleware Tests

use genkit_ai::document::{Document, Part};
use genkit_ai::message::{MessageData, Role};
use genkit_ai::model::{
    augment_with_context, simulate_system_prompt, validate_support, GenerateRequest,
    GenerateResponseData, ModelInfoSupports, NextFn,
};
use serde_json::Value;
use std::pin::Pin;

// A mock `next` function for testing middleware. It returns a response
// containing the modified request, so we can inspect what the middleware did.
fn mock_next() -> NextFn {
    Box::new(|req| {
        let response_data = GenerateResponseData {
            candidates: vec![crate::model::CandidateData {
                index: 0,
                // We encode the modified request into the custom field to retrieve it easily.
                message: MessageData {
                    role: Role::Model,
                    content: vec![Part {
                        custom: Some(serde_json::to_value(req).unwrap()),
                        ..Default::default()
                    }],
                    metadata: None,
                },
                finish_reason: Some(crate::model::FinishReason::Stop),
                finish_message: None,
            }],
            ..Default::default()
        };
        Box::pin(async { Ok(response_data) })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use genkit_ai::tool::ToolDefinition;

    #[tokio::test]
    async fn test_validate_support() {
        let supports = ModelInfoSupports {
            media: Some(false),
            tools: Some(false),
            multiturn: Some(false),
            ..Default::default()
        };
        let middleware = validate_support("test-model".to_string(), supports);

        // Test media rejection
        let media_req = GenerateRequest {
            messages: vec![MessageData {
                role: Role::User,
                content: vec![Part {
                    media: Some(crate::document::Media {
                        content_type: None,
                        url: "http://example.com/img.png".to_string(),
                    }),
                    ..Default::default()
                }],
                metadata: None,
            }],
            ..Default::default()
        };
        assert!(middleware(media_req, mock_next()).await.is_err());

        // Test tool rejection
        let tool_req = GenerateRequest {
            messages: vec![MessageData::default()],
            tools: Some(vec![ToolDefinition {
                name: "test".into(),
                description: "test".into(),
                input_schema: None,
                output_schema: None,
            }]),
            ..Default::default()
        };
        assert!(middleware(tool_req, mock_next()).await.is_err());

        // Test multiturn rejection
        let multiturn_req = GenerateRequest {
            messages: vec![MessageData::default(), MessageData::default()],
            ..Default::default()
        };
        assert!(middleware(multiturn_req, mock_next()).await.is_err());

        // Test valid request
        let valid_req = GenerateRequest {
            messages: vec![MessageData::default()],
            ..Default::default()
        };
        assert!(middleware(valid_req, mock_next()).await.is_ok());
    }

    #[tokio::test]
    async fn test_simulate_system_prompt() {
        let middleware = simulate_system_prompt(None);
        let req = GenerateRequest {
            messages: vec![
                MessageData {
                    role: Role::System,
                    content: vec![Part {
                        text: Some("Be a pirate.".to_string()),
                        ..Default::default()
                    }],
                    metadata: None,
                },
                MessageData {
                    role: Role::User,
                    content: vec![Part {
                        text: Some("Hello".to_string()),
                        ..Default::default()
                    }],
                    metadata: None,
                },
            ],
            ..Default::default()
        };

        let result = middleware(req, mock_next()).await.unwrap();
        let final_request: GenerateRequest = serde_json::from_value(
            result.candidates[0].message.content[0]
                .custom
                .clone()
                .unwrap(),
        )
        .unwrap();

        assert_eq!(final_request.messages.len(), 3);
        assert_eq!(final_request.messages[0].role, Role::User);
        assert!(final_request.messages[0].text().contains("system: "));
        assert!(final_request.messages[0].text().contains("Be a pirate."));
        assert_eq!(final_request.messages[1].role, Role::Model);
        assert_eq!(final_request.messages[1].text(), "Understood.");
        assert_eq!(final_request.messages[2].role, Role::User);
        assert_eq!(final_request.messages[2].text(), "Hello");
    }

    #[tokio::test]
    async fn test_augment_with_context() {
        let middleware = augment_with_context(None);
        let req = GenerateRequest {
            messages: vec![MessageData {
                role: Role::User,
                content: vec![Part {
                    text: Some("What is Genkit?".to_string()),
                    ..Default::default()
                }],
                metadata: None,
            }],
            docs: Some(vec![Document::from_text(
                "Genkit is an AI framework.".to_string(),
                None,
            )]),
            ..Default::default()
        };

        let result = middleware(req, mock_next()).await.unwrap();
        let final_request: GenerateRequest = serde_json::from_value(
            result.candidates[0].message.content[0]
                .custom
                .clone()
                .unwrap(),
        )
        .unwrap();

        let last_message = final_request.messages.last().unwrap();
        assert_eq!(last_message.content.len(), 2); // Original text part + context part

        let context_part = &last_message.content[1];
        let context_text = context_part.text.as_ref().unwrap();

        assert!(context_text.contains("Use the following information"));
        assert!(context_text.contains("- [0]: Genkit is an AI framework."));
        assert_eq!(
            context_part
                .metadata
                .as_ref()
                .unwrap()
                .get("purpose")
                .unwrap(),
            &Value::String("context".to_string())
        );
    }
}
