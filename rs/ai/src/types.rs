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

//! # AI-related Types
//!
//! This module defines common types used in generative AI interactions,
//! such as `LlmStats` and `Tool`. It is the Rust equivalent of `types.ts`.

use genkit_core::registry::ErasedAction;
use schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{self, Value};
use std::sync::Arc;

/// Statistics related to a Language Model (LLM) call.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LlmStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_token_count: Option<u32>,
}

/// A definition of a tool that can be provided to a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Tool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: Value,
}

/// Represents a single tool call requested by a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: Value,
}

/// Represents the response from a Language Model (LLM).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LlmResponse {
    pub completion: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub stats: LlmStats,
}

/// Converts a slice of `ErasedAction`s to the `Tool` wire format expected by models.
pub fn to_tool_wire_format(actions: &[Arc<dyn ErasedAction>]) -> Vec<Tool> {
    actions
        .iter()
        .map(|action| {
            let meta = action.metadata();

            let short_name = if let Some(idx) = meta.name.rfind('/') {
                &meta.name[idx + 1..]
            } else {
                &meta.name
            };

            let input_schema = meta.input_schema.clone().map_or(Value::Null, |s| {
                serde_json::to_value(s).unwrap_or(Value::Null)
            });
            let output_schema = meta.output_schema.clone().map_or(Value::Null, |s| {
                serde_json::to_value(s).unwrap_or(Value::Null)
            });

            Tool {
                name: short_name.to_string(),
                description: meta.description.clone(),
                schema: serde_json::json!({
                    "input": input_schema,
                    "output": output_schema
                }),
            }
        })
        .collect()
}
