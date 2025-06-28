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

//! # Prompts
//!
//! This module provides the `define_prompt` function and related structures for
//! creating and executing type-safe, templated prompts. It is the Rust
// equivalent of `prompt.ts`.

use crate::generate::{
    generate, to_generate_request, GenerateOptions, GenerateResponse, GenerateStreamResponse,
};
use crate::message::{MessageData, Role};
use crate::model::Model;
use crate::tool::ToolAction;
use genkit_core::action::ActionBuilder;
use genkit_core::error::Result;
use genkit_core::registry::{ActionType, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::sync::Arc;
//
// Configuration
//

/// Configuration for a prompt action.
// We use generics here to eventually support typed inputs and outputs.
// For now, they will often default to `Value`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptConfig<I = Value, O = Value, C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    pub model: Model,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<I>,
    // In Rust, we'll represent templates as simple strings for now.
    // A full implementation would use a templating engine like Handlebars or Tera.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<super::generate::OutputOptions>,
    // Other fields from TS version like `messages`, `docs`, `tools`, etc. can be added here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _marker_o: Option<std::marker::PhantomData<O>>,
}

/// Options for generating a response from a prompt.
// This is essentially a subset of `GenerateOptions`.
#[derive(Debug, Clone, Default)]
pub struct PromptGenerateOptions {
    pub config: Option<Value>,
    // In a full implementation, this would include fields like `context`, `history`, etc.
}

//
// Executable Prompt
//

/// A prompt that can be executed as a function.
#[derive(Clone)]
pub struct ExecutablePrompt<I = Value, O = Value, C = Value> {
    config: Arc<PromptConfig<I, O, C>>,
    registry: Arc<Registry>,
}

impl<I, O, C> ExecutablePrompt<I, O, C>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    /// Generates a response by rendering the prompt and calling the model.
    pub async fn generate(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions>,
    ) -> Result<GenerateResponse<O>> {
        let render_opts = self.render(input, opts).await?;
        generate(&self.registry, render_opts).await
    }

    /// Generates a streaming response.
    pub fn stream(
        self,
        input: I,
        opts: Option<PromptGenerateOptions>,
    ) -> GenerateStreamResponse<O> {
        // This is a bit complex because `render` is async but `stream` is not.
        // We spawn a task to handle the async rendering and then stream management.
        let (tx, rx) = tokio::sync::mpsc::channel(32);

        let response_handle = tokio::spawn(async move {
            // 1. Render the prompt to get the final GenerateOptions
            let render_opts = self.render(input, opts).await?;

            // 2. Call generate_stream with the rendered options
            let mut stream_response = crate::generate::generate_stream(&self.registry, render_opts);

            // 3. Pipe chunks from the model's stream to our output channel
            use futures::stream::StreamExt;
            while let Some(chunk_result) = stream_response.stream.next().await {
                if tx.send(chunk_result).await.is_err() {
                    // The receiver was dropped, so we can stop forwarding chunks.
                    break;
                }
            }

            // 4. Await the final full response from the model and return it.
            // This is what will be returned when `.await` is called on the JoinHandle.
            stream_response.response.await.map_err(|join_err| {
                genkit_core::error::Error::new_internal(format!(
                    "Stream processing task failed: {}",
                    join_err
                ))
            })?
        });

        GenerateStreamResponse {
            stream: Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)),
            response: response_handle,
        }
    }

    /// Renders the prompt template into a `GenerateOptions` struct.
    pub async fn render(
        &self,
        _input: I,
        opts: Option<PromptGenerateOptions>,
    ) -> Result<GenerateOptions> {
        // This is a simplified rendering logic. A real implementation would
        // use a templating engine and handle variables from `_input`.
        let mut messages = Vec::new();
        if let Some(system_text) = &self.config.system {
            messages.push(MessageData {
                role: Role::System,
                content: vec![crate::document::Part {
                    text: Some(system_text.clone()),
                    ..Default::default()
                }],
                metadata: None,
            });
        }
        if let Some(prompt_text) = &self.config.prompt {
            messages.push(MessageData {
                role: Role::User,
                content: vec![crate::document::Part {
                    text: Some(prompt_text.clone()),
                    ..Default::default()
                }],
                metadata: None,
            });
        }

        Ok(GenerateOptions {
            model: Some(self.config.model.clone()),
            messages: Some(messages),
            config: opts.and_then(|o| o.config),
            output: self.config.output.clone(),
            ..Default::default()
        })
    }

    /// Returns the prompt usable as a tool.
    pub async fn as_tool(&self) -> Result<Arc<ToolAction<I, GenerateResponse<O>>>> {
        // This requires converting the prompt action into a tool-compatible action.
        let name = &self.config.name;

        // Try to downcast the Arc<dyn ErasedAction> to Arc<dyn Any + Send + Sync>
        // and then to the concrete Arc<ToolAction>.
        let action = self
            .registry
            .lookup_action(&format!("/prompt/{}", name))
            .await // Await here to get the Option
            .ok_or_else(|| {
                genkit_core::error::Error::new_internal(format!("Prompt '{}' not found", name))
            })?;

        let concrete_action = action
            .as_any()
            .downcast_ref::<ToolAction<I, GenerateResponse<O>>>()
            .ok_or_else(|| {
                genkit_core::error::Error::new_internal(format!(
                    "Mismatched type for prompt action '{}'. Expected ToolAction, got something else.",
                    name
                ))
            })?;
        Ok(Arc::new(concrete_action.clone()))
    }
}

//
// Define Prompt
//

/// Defines a prompt which can be used to generate content or render a request.
pub fn define_prompt<I, O, C>(
    registry: &mut Registry,
    config: PromptConfig<I, O, C>,
) -> ExecutablePrompt<I, O, C>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let config_arc = Arc::new(config);
    let registry_arc = Arc::new(registry.clone());

    // Define an action that *renders* the prompt to a GenerateRequest
    let prompt_action = {
        let config_clone = config_arc.clone();
        let registry_clone = registry_arc.clone();
        let name = config_clone.name.clone();

        ActionBuilder::new(
            ActionType::Prompt,
            name,
            move |input: I,
                  _args: genkit_core::action::ActionFnArg<
                crate::model::GenerateResponseChunkData,
            >| {
                let config = config_clone.clone();
                let registry = registry_clone.clone();
                async move {
                    let prompt = ExecutablePrompt { config, registry };
                    let gen_opts = prompt.render(input, None).await?;
                    to_generate_request(&prompt.registry, &gen_opts).await
                }
            },
        )
        .build(registry)
    };

    // The action is now in the registry. The ExecutablePrompt just needs the info to find it.
    let _ = prompt_action; // Ensure it's not dropped if we needed to store it somewhere.

    ExecutablePrompt {
        config: config_arc,
        registry: registry_arc,
    }
}

/// Helper to check if a value is an `ExecutablePrompt`.
/// In Rust, this is less necessary due to the static type system,
/// but can be useful for dynamic scenarios.
pub fn is_executable_prompt<I, O, C>(_: &ExecutablePrompt<I, O, C>) -> bool {
    true
}

/// A placeholder function for `prompt()`, which would look up a defined prompt.
pub async fn prompt<I, O, C>(registry: &Registry, name: &str) -> Result<ExecutablePrompt<I, O, C>>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    // A real implementation would look up the config from the registry,
    // which requires the registry to store more than just the action fn.
    // This is a placeholder demonstrating the lookup.
    let action = registry
        .lookup_action(&format!("/prompt/{}", name))
        .await
        .ok_or_else(|| {
            genkit_core::error::Error::new_internal(format!("Prompt '{}' not found", name))
        })?;

    // We can't easily reconstruct the ExecutablePrompt without storing the original config.
    // This highlights a limitation of the current simple registry design.
    let _ = action;
    unimplemented!("Looking up prompts by name is not fully supported yet without a way to retrieve the original config from the registry.")
}
