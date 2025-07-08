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

//! # OpenAI Compatibility Layer for Model Garden
//!
//! This module provides a compatibility layer to allow Genkit to interact
//! with models in the Vertex AI Model Garden that expose an OpenAI-compatible API.

use genkit_ai::model::{
    define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponse, ModelAction,
    ModelRef,
};
use genkit_ai::{tool::ToolDefinition, MessageData, Part, Role};
use genkit_core::Registry;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Note: This implementation will require a dependency on an OpenAI client library,
// such as `async-openai`. The types below are placeholders inspired by that library's
// common structure. For a real implementation, these would be replaced by the
// actual types from the chosen crate.

// Placeholder structs - these would come from the `async-openai` crate.
pub mod openai_types {
    use super::*;
    use serde_json::Value;

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatCompletionRequestMessage {
        pub role: String,
        pub content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_calls: Option<Vec<ToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_call_id: Option<String>,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ToolCall {
        pub id: String,
        pub r#type: String,
        pub function: FunctionCall,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FunctionCall {
        pub name: String,
        pub arguments: String,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ChatCompletionTool {
        pub r#type: String,
        pub function: FunctionDefinition,
    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct FunctionDefinition {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub description: Option<String>,
        pub parameters: Value,
    }

    #[derive(Debug, Serialize)]
    pub struct CreateChatCompletionRequest {
        pub model: String,
        pub messages: Vec<ChatCompletionRequestMessage>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tools: Option<Vec<ChatCompletionTool>>,
        // Other params like temperature, top_p, etc. would go here
    }

    #[derive(Debug, Deserialize)]
    pub struct ChatCompletionChoice {
        pub index: u32,
        pub message: ChatCompletionResponseMessage,
        pub finish_reason: String,
    }

    #[derive(Debug, Deserialize)]
    #[allow(unused)]
    pub struct ChatCompletionResponseMessage {
        pub role: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_calls: Option<Vec<ToolCall>>,
    }

    #[derive(Debug, Deserialize)]
    #[allow(unused)]
    pub struct ChatCompletionResponse {
        pub id: String,
        pub object: String,
        pub created: u64,
        pub model: String,
        pub choices: Vec<ChatCompletionChoice>,
        // usage would go here
    }
}

use self::openai_types::*;

/// Configuration schema for OpenAI-compatible models.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct OpenAIConfig {
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    // Other OpenAI-specific fields can be added here.
}

fn to_openai_role(role: Role) -> String {
    match role {
        Role::Model => "assistant".to_string(),
        Role::User => "user".to_string(),
        Role::System => "system".to_string(),
        Role::Tool => "tool".to_string(),
    }
}

pub fn to_openai_tool(tool: &ToolDefinition) -> ChatCompletionTool {
    ChatCompletionTool {
        r#type: "function".to_string(),
        function: FunctionDefinition {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            parameters: tool
                .input_schema
                .clone()
                .unwrap_or(serde_json::json!({"type": "object", "properties": {}})),
        },
    }
}

pub fn to_openai_messages(
    messages: &[MessageData],
) -> Vec<openai_types::ChatCompletionRequestMessage> {
    // This is a simplified conversion. A real implementation would need to handle
    // multi-part messages, tool calls, and tool responses correctly.
    messages
        .iter()
        .map(|m| {
            let content = m
                .content
                .iter()
                .filter_map(|p| p.text.clone())
                .collect::<Vec<String>>()
                .join("\n");
            openai_types::ChatCompletionRequestMessage {
                role: to_openai_role(m.role.clone()),
                content,
                tool_calls: None,
                tool_call_id: None,
            }
        })
        .collect()
}

pub fn from_openai_choice(choice: openai_types::ChatCompletionChoice) -> CandidateData {
    let content = choice
        .message
        .content
        .map(|text| vec![Part::text(text)])
        .unwrap_or_default();

    let finish_reason = match choice.finish_reason.as_str() {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::Blocked,
        "tool_calls" => FinishReason::Stop,
        _ => FinishReason::Unknown,
    };

    CandidateData {
        index: choice.index,
        message: genkit_ai::MessageData {
            role: Role::Model,
            content,
            metadata: None,
        },
        finish_reason: Some(finish_reason),
        finish_message: None,
    }
}

/// Creates a `ModelAction` for an OpenAI-compatible model from the Model Garden.
#[allow(unused)]
pub fn openai_compatible_model(
    model_ref: ModelRef<serde_json::Value>,
    // The factory would produce a configured OpenAI client.
    // _client_factory: fn() -> T, where T is an OpenAI client.
) -> ModelAction {
    let model_name = model_ref.name.clone();

    let runner = move |req: GenerateRequest, _| {
        let _model_name = model_name.clone();
        async move {
            // 1. Create an OpenAI client using the factory.
            // let client = client_factory();

            // 2. Convert Genkit request to OpenAI request.
            let messages: Vec<ChatCompletionRequestMessage> = to_openai_messages(&req.messages);

            let tools = req
                .tools
                .as_ref()
                .map(|t| t.iter().map(to_openai_tool).collect());

            let _openai_request = CreateChatCompletionRequest {
                model: "placeholder-model-name".to_string(), // Would get from model_ref or config
                messages,
                tools,
            };

            // 3. Make the API call.
            // let response = client.chat().create(openai_request).await?;
            // This is a mock response.
            let mock_response_json = r#"{
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "\n\nHello there, how may I assist you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            }"#;
            let response: ChatCompletionResponse = serde_json::from_str(mock_response_json)
                .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))?;

            // 4. Convert OpenAI response to Genkit response.
            let candidates = response
                .choices
                .into_iter()
                .map(from_openai_choice)
                .collect();

            Ok(GenerateResponse {
                candidates,
                usage: None, // Would be mapped from response.usage
                ..Default::default()
            })
        }
    };

    let mut registry = Registry::default();
    define_model(
        &mut registry,
        genkit_ai::model::DefineModelOptions {
            name: model_ref.name.clone(),
            ..Default::default()
        },
        runner,
    )
}
