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

use crate::document::{Part, ToolRequestPart, ToolResponsePart};
use async_trait::async_trait;
use genkit_core::action::{detached_action, Action, ActionBuilder, ActionFnArg};
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use genkit_core::schema::{parse_schema, ProvidedSchema};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, json, Value};
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
    fn respond(
        &self,
        interrupt: &ToolRequestPart,
        output_data: O,
        response_metadata: Option<Value>,
    ) -> Result<ToolResponsePart>;
    fn restart(
        &self,
        interrupt: &ToolRequestPart,
        resumed_metadata: Option<Value>,
        replace_input: Option<I>,
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
    I: JsonSchema + Serialize,
    O: Serialize + JsonSchema,
{
    fn respond(
        &self,
        interrupt: &ToolRequestPart,
        output_data: O,
        response_metadata: Option<Value>,
    ) -> Result<ToolResponsePart> {
        let schema_def = self.0.meta.output_schema.clone().ok_or_else(|| {
            Error::new_internal("Tool has no output schema for response validation.")
        })?;
        let data_value = serde_json::to_value(output_data)
            .map_err(|e| Error::new_internal(format!("Failed to serialize tool output: {}", e)))?;
        parse_schema::<Value>(data_value.clone(), ProvidedSchema::FromType(schema_def))?;

        let interrupt_response_value = response_metadata.unwrap_or(Value::Bool(true));

        Ok(Part {
            tool_response: Some(crate::document::ToolResponse {
                name: interrupt.tool_request.as_ref().unwrap().name.clone(),
                ref_id: interrupt.tool_request.as_ref().unwrap().ref_id.clone(),
                output: Some(data_value),
            }),
            metadata: Some(
                [("interruptResponse".to_string(), interrupt_response_value)]
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
        replace_input: Option<I>,
    ) -> Result<ToolRequestPart> {
        let mut metadata = interrupt.metadata.clone().unwrap_or_default();
        metadata.insert(
            "resumed".to_string(),
            resumed_metadata.unwrap_or(Value::Bool(true)),
        );

        let mut tool_request = interrupt.tool_request.clone().unwrap();

        if let Some(new_input) = replace_input {
            let schema_def = self.0.meta.input_schema.clone().ok_or_else(|| {
                Error::new_internal("Tool has no input schema for restart validation.")
            })?;
            let input_value = serde_json::to_value(new_input).map_err(|e| {
                Error::new_internal(format!("Failed to serialize new input: {}", e))
            })?;

            parse_schema::<Value>(input_value.clone(), ProvidedSchema::FromType(schema_def))?;

            tool_request.input = Some(input_value);
            metadata.insert("replacedInput".to_string(), json!({}));
        }

        Ok(Part {
            tool_request: Some(tool_request),
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

impl<I, O, S> From<ToolAction<I, O, S>> for ToolArgument
where
    I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
{
    fn from(action: ToolAction<I, O, S>) -> Self {
        ToolArgument::Action(Arc::new(action))
    }
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
#[derive(Default, JsonSchema)]
pub struct ToolConfig<I = (), O = ()> {
    pub name: String,
    pub description: String,
    pub input_schema: Option<I>,
    pub output_schema: Option<O>,
    pub metadata: Option<Value>,
}

/// Defines a new tool and registers it as a Genkit action.
pub fn define_tool<I, O, F, Fut>(registry: &mut Registry, config: ToolConfig<I, O>, runner: F)
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
    println!(
        "define_tool: registering action named `{}` of type {:?}",
        config.name,
        std::any::Any::type_id(&tool_action)
    );
    registry.register_action(&config.name, tool_action).unwrap();
}

/// Configuration for an interrupt.
pub type InterruptConfig<I, O> = ToolConfig<I, O>;

/// Defines a tool that interrupts the flow to wait for user input.
pub fn define_interrupt<I, O>(registry: &mut Registry, config: InterruptConfig<I, O>)
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    define_tool(registry, config, |_, ctx: ToolFnOptions| async move {
        Err((ctx.interrupt)(None))
    });
}

/// Represents a tool definition that is not yet registered with the framework.
pub struct DynamicTool<I, O, F, Fut>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    config: ToolConfig<I, O>,
    runner: F,
}

impl<I, O, F, Fut> DynamicTool<I, O, F, Fut>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    /// Attaches the dynamic tool to a registry, making it an executable action.
    pub fn attach(self, registry: &mut Registry) -> ToolArgument {
        let runner_arc = Arc::new(self.runner);
        let action = detached_action(
            ActionType::Tool,
            self.config.name.clone(),
            move |input, args: ActionFnArg<()>| {
                let runner_clone = runner_arc.clone();
                let options = ToolFnOptions {
                    interrupt: Box::new(|metadata| {
                        Error::new_internal(format!(
                            "INTERRUPT::{}",
                            serde_json::to_string(&metadata.unwrap_or(serde_json::Value::Null))
                                .unwrap()
                        ))
                    }),
                    context: args.context.unwrap_or_default(),
                };
                async move { runner_clone(input, options).await }
            },
        );

        registry
            .register_action(&self.config.name, action.clone())
            .unwrap();

        ToolArgument::from(ToolAction(action))
    }
}

impl<I, O, F, Fut> DynamicTool<I, O, F, Fut>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    /// Attaches the dynamic tool to a registry, making it an executable action.
    pub fn to_tool_argument(self) -> ToolArgument {
        let runner_arc = Arc::new(self.runner);
        let action = detached_action(
            ActionType::Tool,
            self.config.name.clone(),
            move |input, args: ActionFnArg<()>| {
                let runner_clone = runner_arc.clone();
                let options = ToolFnOptions {
                    interrupt: Box::new(|metadata| {
                        Error::new_internal(format!(
                            "INTERRUPT::{}",
                            serde_json::to_string(&metadata.unwrap_or(serde_json::Value::Null))
                                .unwrap()
                        ))
                    }),
                    context: args.context.unwrap_or_default(),
                };
                async move { runner_clone(input, options).await }
            },
        );

        ToolArgument::from(ToolAction(action))
    }
}

/// Defines a dynamic tool that is not registered with the framework globally.
pub fn dynamic_tool<I, O, F, Fut>(config: ToolConfig<I, O>, runner: F) -> DynamicTool<I, O, F, Fut>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + Clone + 'static,
    Fut: Future<Output = Result<O>> + Send + 'static,
{
    DynamicTool { config, runner }
}

/// Defines a dynamic tool that will always interrupt when called.
///
/// This is useful for tools that require human-in-the-loop intervention
/// or whose implementation exists outside the current execution flow.
pub fn dynamic_tool_without_runner<I, O>(config: ToolConfig<I, O>) -> ToolArgument
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
{
    dynamic_tool(config, |_, options: ToolFnOptions| async move {
        Err((options.interrupt)(None))
    })
    .to_tool_argument()
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
