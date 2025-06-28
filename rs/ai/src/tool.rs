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

use crate::document::Part;
use async_trait::async_trait;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, Value};
use std::any::Any;
use std::future::Future;
use std::ops::Deref;
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

/// A wrapper for a tool `Action` that provides tool-specific functionality.
#[derive(Clone)]
pub struct ToolAction<I = Value, O = Value, S = ()>(pub Action<I, O, S>);

impl<I, O, S> Deref for ToolAction<I, O, S> {
    type Target = Action<I, O, S>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl<I, O, S> ErasedAction for ToolAction<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<Value> {
        let result = self.0.run_http_json(input, context).await?;
        serde_json::to_value(result)
            .map_err(|e| Error::new_internal(format!("Failed to serialize tool output: {}", e)))
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn metadata(&self) -> &genkit_core::action::ActionMetadata {
        self.0.metadata()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A reference to a tool, which can be a name or a concrete action.
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase", untagged)]
pub enum ToolArgument {
    Name(String),
    #[serde(skip)]
    Action(Arc<dyn ErasedAction>),
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
            Self::Action(a) => f.debug_tuple("Action").field(&a.name()).finish(),
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

impl From<Arc<dyn ErasedAction>> for ToolArgument {
    fn from(action: Arc<dyn ErasedAction>) -> Self {
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
) -> ToolAction<I, O, ()>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ActionContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    ToolAction(
        ActionBuilder::new(ActionType::Tool, name.to_string(), move |input, args| {
            runner(input, args.context.unwrap_or_default())
        })
        .with_description(description.to_string())
        .build(registry),
    )
}

/// Resolves a slice of `ToolArgument`s into a `Vec` of `ToolAction`s.
pub async fn resolve_tools(
    registry: &Registry,
    tools: Option<&[ToolArgument]>,
) -> Result<Vec<Arc<dyn ErasedAction>>> {
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
                resolved_tools.push(action);
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

/// Converts an `ErasedAction` to the `ToolDefinition` wire format.
pub fn to_tool_definition(tool: &dyn ErasedAction) -> Result<ToolDefinition> {
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

/// Checks if a `Part` contains a tool request.
pub fn is_tool_request(part: &Part) -> bool {
    part.tool_request.is_some()
}
