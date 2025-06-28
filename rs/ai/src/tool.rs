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

//! # Model Tooling
//!
//! This module provides the structures and functions for defining and using
//! tools with generative models. It is the Rust equivalent of `tool.ts`.

use genkit_core::action::{Action, ActionBuilder};
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::ErasedAction;
use genkit_core::registry::{ActionType, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, Value};
use std::future::Future;
use std::sync::Arc;

/// A definition of a tool that can be provided to a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

/// The `Action` type for a tool.
pub type ToolAction<I = Value, O = Value, S = ()> = Action<I, O, S>;

/// A reference to a tool, which can be a name or a concrete action.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", untagged)]
pub enum ToolArgument {
    Name(String),
    #[serde(skip)]
    Action(Arc<ToolAction>),
}

impl Default for ToolArgument {
    fn default() -> Self {
        ToolArgument::Name(String::new())
    }
}

impl std::fmt::Debug for ToolArgument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => f.debug_tuple("Name").field(s).finish(),
            Self::Action(a) => f.debug_tuple("Action").field(&a.meta.name).finish(),
        }
    }
}

impl From<String> for ToolArgument {
    fn from(s: String) -> Self {
        ToolArgument::Name(s)
    }
}

impl<'a> From<&'a str> for ToolArgument {
    fn from(s: &'a str) -> Self {
        ToolArgument::Name(s.to_string())
    }
}

impl From<Arc<ToolAction>> for ToolArgument {
    fn from(action: Arc<ToolAction>) -> Self {
        ToolArgument::Action(action)
    }
}

/// Configuration for defining a tool.
pub struct ToolConfig<I, O> {
    pub name: String,
    pub description: String,
    pub input_schema: Option<I>,
    pub output_schema: Option<O>,
}

/// Defines a new tool and registers it as a Genkit action.
pub fn define_tool<I, O, F, Fut>(
    registry: &mut Registry,
    name: &str,
    description: &str,
    runner: F,
) -> Arc<ToolAction<I, O>>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ActionContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    let action = ActionBuilder::new(ActionType::Tool, name.to_string(), move |input, args| {
        runner(input, args.context.unwrap_or_default())
    })
    .with_description(description.to_string())
    .build(registry);
    Arc::new(action)
}

/// Resolves a slice of `ToolArgument`s into a `Vec` of `ToolAction`s.
pub async fn resolve_tools(
    registry: &Registry,
    tools: Option<&[ToolArgument]>,
) -> Result<Vec<Arc<ToolAction>>> {
    let Some(tools) = tools else {
        return Ok(Vec::new());
    };

    let mut resolved_tools = Vec::new();
    for tool_arg in tools {
        match tool_arg {
            ToolArgument::Name(name) => {
                let action = registry
                    .lookup_action(&format!("/tool/{}", name))
                    .await
                    .ok_or_else(|| Error::new_internal(format!("Tool '{}' not found", name)))?;
                let tool_action =
                    any_downcast::downcast_arc::<ToolAction>(action).map_err(|_| {
                        Error::new_internal(format!("Tool '{}' is not a valid ToolAction", name))
                    })?;
                resolved_tools.push(tool_action);
            }
            ToolArgument::Action(action) => {
                resolved_tools.push(action.clone());
            }
        }
    }
    Ok(resolved_tools)
}

// A stub for the assumed downcasting library, to make the code snippet valid.
// In a real scenario, this would be a dependency.
mod any_downcast {
    use super::ErasedAction;
    use std::sync::Arc;

    pub fn downcast_arc<T: 'static>(
        arc: Arc<dyn ErasedAction>,
    ) -> std::result::Result<Arc<T>, Arc<dyn ErasedAction>> {
        if arc.as_any().is::<T>() {
            unsafe {
                let raw = Arc::into_raw(arc);
                let ptr = raw as *const T;
                Ok(Arc::from_raw(ptr))
            }
        } else {
            Err(arc)
        }
    }
}

/// Converts a `ToolAction` to the `ToolDefinition` wire format.
pub fn to_tool_definition(tool: &ToolAction) -> Result<ToolDefinition> {
    let metadata = tool.metadata();
    let name = &metadata.name;
    let short_name = name.split('/').next_back().unwrap_or(name);

    let input_schema = metadata
        .input_schema
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| Error::new_internal(format!("Failed to serialize input schema: {}", e)))?;

    let output_schema = metadata
        .output_schema
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|e| Error::new_internal(format!("Failed to serialize output schema: {}", e)))?;

    Ok(ToolDefinition {
        name: short_name.to_string(),
        description: metadata.description.clone().unwrap_or_default(),
        input_schema,
        output_schema,
    })
}
