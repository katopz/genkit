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

//! # Genkit Tool
//!
//! This module defines the structures and traits for creating and using tools
//! within the Genkit framework. Tools are functions that models can call to
// attain external information or perform actions. It is the Rust equivalent of
// `@genkit-ai/ai/tool`.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

/// Describes the schema of a tool that can be passed to a model.
///
/// The model uses this definition to understand what the tool does, what arguments
/// it expects, and what it returns. This corresponds to `ToolDefinition` in the
/// TypeScript SDK.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    /// The name of the tool. Must be a valid function name.
    pub name: String,
    /// A description of what the tool does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A JSON schema describing the input parameters of the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    /// A JSON schema describing the output of the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

/// A trait representing an executable tool.
///
/// This trait is implemented by structs that encapsulate the logic for a specific
/// tool. It provides both the tool's definition (for the model) and the means
/// to execute it. This is the conceptual equivalent of `ToolAction` in TypeScript.
pub trait Tool: Send + Sync {
    /// Returns the schema definition of the tool.
    fn definition(&self) -> &ToolDefinition;

    /// Executes the tool with the given input.
    ///
    /// The input and output are `serde_json::Value` to allow for generic
    /// tool execution. Implementations should handle validation and conversion
    /// to their specific input/output types.
    fn execute<'a>(
        &'a self,
        input: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;
}

/// A simple example of a tool implementation.
///
/// This is a placeholder to show how one might implement the `Tool` trait. A
/// `define_tool` macro or function would likely be provided to simplify this.
pub struct ExampleTool {
    definition: ToolDefinition,
}

impl ExampleTool {
    pub fn new() -> Self {
        Self {
            definition: ToolDefinition {
                name: "exampleTool".to_string(),
                description: Some("An example tool that echoes the input.".to_string()),
                // In a real scenario, you might use `serde_json::json!` to define a schema.
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "payload": { "type": "string" }
                    },
                    "required": ["payload"]
                })),
                output_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "echo": { "type": "string" }
                    }
                })),
            },
        }
    }
}

impl Default for ExampleTool {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for ExampleTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    fn execute<'a>(
        &'a self,
        input: &'a Value,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(async move {
            // In a real tool, you would deserialize `input` into a specific struct,
            // perform some logic, and then serialize the result.
            let payload = input
                .get("payload")
                .and_then(Value::as_str)
                .unwrap_or("default_payload");

            let output = serde_json::json!({
                "echo": format!("You sent: {}", payload)
            });
            Ok(output)
        })
    }
}
