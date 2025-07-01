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

use crate::document::{Part, ToolRequest, ToolRequestPart, ToolResponsePart};
use async_trait::async_trait;
use genkit_core::action::{detached_action, Action, ActionBuilder, ActionFnArg};
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use genkit_core::schema::{parse_schema, ProvidedSchema};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, Value};
use std::any::Any;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use thiserror::Error;

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl From<ToolDefinition> for ToolRequest {
    fn from(def: ToolDefinition) -> Self {
        ToolRequest {
            name: def.name,
            ref_id: None,
            input: def.input_schema,
        }
    }
}

/// An error thrown to interrupt a tool's execution.
/// The framework will catch this and return a response with an interrupt reason.
#[derive(Debug, Error)]
#[error("Tool execution was interrupted.")]
pub struct ToolInterruptError {
    pub metadata: Option<Value>,
}

/// Options passed to the tool implementation function.
pub struct ToolFnOptions {
    /// A function that can be called during tool execution that will result in the tool
    /// getting interrupted. The returned `Error` should be propagated from the tool function.
    pub interrupt: Box<dyn Fn(Option<Value>) -> Error + Send + Sync>,
    pub context: ActionContext,
}

/// The signature for a tool's implementation logic.
pub type ToolFn<I, O> =
    dyn Fn(I, ToolFnOptions) -> Pin<Box<dyn Future<Output = Result<O>> + Send>> + Send + Sync;

pub trait Resumable<I, O> {
    fn respond(&self, interrupt: &ToolRequestPart, output_data: O) -> Result<ToolResponsePart>;
    fn restart(
        &self,
        interrupt: &ToolRequestPart,
        resumed_metadata: Option<Value>,
    ) -> Result<ToolRequestPart>;
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

impl<I, O, S> Resumable<I, O> for ToolAction<I, O, S>
where
    I: JsonSchema,
    O: Serialize + JsonSchema,
{
    fn respond(&self, interrupt: &ToolRequestPart, output_data: O) -> Result<ToolResponsePart> {
        let schema_def = self.0.meta.output_schema.clone().ok_or_else(|| {
            Error::new_internal("Tool has no output schema for response validation.")
        })?;
        let data_value = serde_json::to_value(output_data)
            .map_err(|e| Error::new_internal(format!("Failed to serialize tool output: {}", e)))?;
        parse_schema::<Value>(data_value.clone(), ProvidedSchema::FromType(schema_def))?;

        Ok(Part {
            tool_response: Some(crate::document::ToolResponse {
                name: interrupt.tool_request.as_ref().unwrap().name.clone(),
                ref_id: interrupt.tool_request.as_ref().unwrap().ref_id.clone(),
                output: Some(data_value),
            }),
            metadata: Some(
                [(
                    "interruptResponse".to_string(),
                    serde_json::Value::Bool(true),
                )]
                .iter()
                .cloned()
                .collect(),
            ),
            ..Default::default()
        })
    }

    fn restart(
        &self,
        interrupt: &ToolRequestPart,
        resumed_metadata: Option<Value>,
    ) -> Result<ToolRequestPart> {
        let mut metadata = interrupt.metadata.clone().unwrap_or_default();
        metadata.insert(
            "resumed".to_string(),
            resumed_metadata.unwrap_or(Value::Bool(true)),
        );
        Ok(Part {
            tool_request: interrupt.tool_request.clone(),
            metadata: Some(metadata),
            ..Default::default()
        })
    }
}

#[async_trait]
impl<I, O, S> ErasedAction for ToolAction<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
{
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<Value> {
        self.0.run_http_json(input, context).await
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<genkit_core::action::StreamingResponse<Value, Value>> {
        self.0.stream_http_json(input, context)
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
/// Configuration for defining a tool.
#[derive(Default)]
pub struct ToolConfig<I = (), O = ()> {
    pub name: String,
    pub description: String,
    pub input_schema: Option<I>,
    pub output_schema: Option<O>,
    pub metadata: Option<Value>,
}

/// Defines a new tool and registers it as a Genkit action.
pub fn define_tool<I, O, F, Fut>(
    registry: &mut Registry,
    config: ToolConfig<I, O>,
    runner: F,
) -> ToolAction<I, O, ()>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    let action = ActionBuilder::new(
        ActionType::Tool,
        config.name.clone(),
        move |input, args: ActionFnArg<()>| {
            let runner = Arc::new(runner.clone());
            let runner_clone = runner.clone();
            async move {
                let options = ToolFnOptions {
                    interrupt: Box::new(|metadata| {
                        Error::new_internal(format!(
                            "INTERRUPT::{}",
                            serde_json::to_string(&metadata.unwrap_or(Value::Null)).unwrap()
                        ))
                    }),
                    context: args.context.unwrap_or_default(),
                };
                runner_clone(input, options).await
            }
        },
    )
    .with_description(config.description)
    .build();

    let tool_action = ToolAction(action);
    registry
        .register_action(config.name, tool_action.clone())
        .unwrap();
    tool_action
}

/// Configuration for an interrupt.
pub type InterruptConfig<I, O> = ToolConfig<I, O>;

/// Defines a tool that interrupts the flow to wait for user input.
pub fn define_interrupt<I, O>(
    registry: &mut Registry,
    config: InterruptConfig<I, O>,
) -> ToolAction<I, O, ()>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    define_tool(registry, config, |_, ctx: ToolFnOptions| async move {
        Err((ctx.interrupt)(None))
    })
}

/// Defines a dynamic tool that is not registered with the framework globally.
pub fn dynamic_tool<I, O, F, Fut>(config: ToolConfig<I, O>, runner: F) -> ToolAction<I, O, ()>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    let runner_arc = Arc::new(runner);
    let action = detached_action(
        ActionType::Tool,
        config.name.clone(),
        move |input, args: ActionFnArg<()>| {
            let runner_clone = runner_arc.clone();
            let options = ToolFnOptions {
                interrupt: Box::new(|metadata| {
                    Error::new_internal(format!(
                        "INTERRUPT::{}",
                        serde_json::to_string(&metadata.unwrap_or(Value::Null)).unwrap()
                    ))
                }),
                context: args.context.unwrap_or_default(),
            };
            async move { runner_clone(input, options).await }
        },
    );

    ToolAction(action)
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

    let mut tool_metadata = metadata.metadata.clone();
    if name != short_name {
        tool_metadata.insert(
            "originalName".to_string(),
            serde_json::Value::String(name.to_string()),
        );
    }

    Ok(ToolDefinition {
        name: short_name.to_string(),
        description: metadata.description.clone().unwrap_or_default(),
        input_schema,
        output_schema,
        metadata: Some(serde_json::Value::Object(
            tool_metadata.into_iter().collect(),
        )),
    })
}

/// Checks if a `Part` contains a tool request.
pub fn is_tool_request(part: &Part) -> bool {
    part.tool_request.is_some()
}
