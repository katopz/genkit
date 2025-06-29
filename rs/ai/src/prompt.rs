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

use crate::document::Part;
use crate::generate::{
    generate, generate_stream, to_generate_request, GenerateOptions, GenerateResponse,
    GenerateStreamResponse, OutputOptions,
};
use crate::message::{MessageData, Role};
use crate::model::{GenerateRequest, Model};
use crate::tool::ToolArgument;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
// #[cfg(feature = "dotprompt-private")]
// use handlebars::HandlebarsHelper;
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// A type alias for an action that renders a prompt.
pub type PromptAction<I = Value> = Action<I, GenerateRequest, ()>;

/// Configuration for a prompt action.
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PromptConfig<I = Value, O = Value, C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Model>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<I>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<MessageData>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolArgument>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _marker: Option<std::marker::PhantomData<O>>,
}

/// Options for generating a response from a prompt.
#[derive(Clone, Default, Debug)]
pub struct PromptGenerateOptions<O = Value> {
    pub config: Option<Value>,
    pub messages: Option<Vec<MessageData>>,
    pub output: Option<OutputOptions>,
    pub _marker: std::marker::PhantomData<O>,
}

/// A prompt that can be executed as a function.
#[derive(Clone)]
pub struct ExecutablePrompt<I = Value, O = Value, C = Value> {
    config: Arc<PromptConfig<I, O, C>>,
    registry: Arc<Registry>,
}

impl<I, O, C> ExecutablePrompt<I, O, C>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: for<'de> Deserialize<'de>
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + Clone
        + Default
        + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    /// Generates a response by rendering the prompt and calling the model.
    pub async fn generate(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateResponse<O>> {
        let render_opts = self.render(input, opts).await?;
        generate::<O>(&self.registry, render_opts).await
    }

    /// Generates a streaming response.
    pub fn stream(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions<O>>,
    ) -> GenerateStreamResponse<O> {
        let registry_clone = self.registry.clone();
        let config_clone = self.config.clone();

        let response_handle = tokio::spawn(async move {
            let prompt = ExecutablePrompt {
                config: config_clone,
                registry: registry_clone,
            };
            prompt.render(input, opts).await
        });

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let final_response_handle = tokio::spawn(async move {
            let render_opts = response_handle.await.unwrap()?;
            let mut stream_resp = generate_stream::<O>(&Registry::new(), render_opts);
            while let Some(chunk) = futures_util::StreamExt::next(&mut stream_resp.stream).await {
                if tx.send(chunk).await.is_err() {
                    break;
                }
            }
            stream_resp.response.await.unwrap()
        });

        GenerateStreamResponse {
            stream: Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)),
            response: final_response_handle,
        }
    }

    /// Renders the prompt template into a `GenerateOptions` struct.
    pub async fn render(
        &self,
        _input: I, // In a real implementation, this would be used for templating.
        opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateOptions<O>> {
        let mut messages = Vec::new();
        if let Some(system_text) = &self.config.system {
            messages.push(MessageData {
                role: Role::System,
                content: vec![Part {
                    text: Some(system_text.clone()),
                    ..Default::default()
                }],
                metadata: None,
            });
        }
        if let Some(config_messages) = &self.config.messages {
            messages.extend(config_messages.clone());
        }
        if let Some(opts_messages) = opts.as_ref().and_then(|o| o.messages.as_ref()) {
            messages.extend(opts_messages.clone());
        }
        if let Some(prompt_text) = &self.config.prompt {
            messages.push(MessageData {
                role: Role::User,
                content: vec![Part {
                    text: Some(prompt_text.clone()),
                    ..Default::default()
                }],
                metadata: None,
            });
        }

        let gen_opts = GenerateOptions {
            model: self.config.model.clone(),
            messages: Some(messages),
            tools: self.config.tools.clone(),
            config: self
                .config
                .config
                .as_ref()
                .and_then(|c| serde_json::to_value(c).ok())
                .or_else(|| opts.and_then(|o| o.config)),
            output: self.config.output.clone(),
            ..Default::default()
        };
        Ok(gen_opts)
    }
}

/// Defines a prompt which can be used to generate content or render a request.
pub fn define_prompt<I, O, C>(
    registry: &mut Registry,
    config: PromptConfig<I, O, C>,
) -> ExecutablePrompt<I, O, C>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: for<'de> Deserialize<'de>
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + Clone
        + Default
        + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let config_arc = Arc::new(config);
    let registry_arc = Arc::new(registry.clone());

    let prompt_action = {
        let config_clone = config_arc.clone();
        let registry_clone = registry_arc.clone();
        let name = config_clone.name.clone();

        ActionBuilder::new(
            ActionType::Prompt,
            name,
            move |input: I, _ctx: genkit_core::action::ActionFnArg<()>| {
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

    let _ = prompt_action;

    ExecutablePrompt {
        config: config_arc,
        registry: registry_arc,
    }
}

/// Helper to check if a value is an `ExecutablePrompt`.
pub fn is_executable_prompt<I, O, C>(_: &ExecutablePrompt<I, O, C>) -> bool {
    true
}

// /// Defines a partial template for use in other Dotprompt templates.
// #[cfg(feature = "dotprompt-private")]
// pub fn define_partial(registry: &mut Registry, name: &str, source: &str) {
//     registry.dotprompt.define_partial(name, source);
// }

// /// Defines a custom Handlebars helper for use in prompt templates.
// #[cfg(feature = "dotprompt-private")]
// pub fn define_helper(
//     registry: &mut Registry,
//     name: &str,
//     helper: Box<dyn HandlebarsHelper + Send + Sync>,
// ) {
//     registry.dotprompt.define_helper(name, helper);
// }

// /// Loads all `.prompt` files from a given directory and registers them as prompts.
// ///
// /// This function recursively searches the specified directory.
// /// Files starting with `_` are treated as partials.
// #[cfg(feature = "dotprompt-private")]
// pub async fn load_prompt_folder(registry: &mut Registry, dir: &str, ns: &str) -> Result<()> {
//     let prompts_path = PathBuf::from(dir);
//     if prompts_path.exists() {
//         load_prompt_folder_recursively(registry, &prompts_path, ns, &PathBuf::new()).await?;
//     }
//     Ok(())
// }

// #[cfg(feature = "dotprompt-private")]
// async fn load_prompt_folder_recursively(
//     registry: &mut Registry,
//     base_path: &Path,
//     ns: &str,
//     sub_dir: &Path,
// ) -> Result<()> {
//     let current_path = base_path.join(sub_dir);
//     let mut entries = tokio::fs::read_dir(current_path).await?;

//     while let Some(entry) = entries.next_entry().await? {
//         let path = entry.path();
//         if path.is_dir() {
//             let relative_path = path.strip_prefix(base_path).unwrap().to_path_buf();
//             load_prompt_folder_recursively(registry, base_path, ns, &relative_path).await?;
//         } else if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
//             if file_name.ends_with(".prompt") {
//                 let source = tokio::fs::read_to_string(&path).await?;
//                 if file_name.starts_with('_') {
//                     if let Some(partial_name) = file_name
//                         .strip_prefix('_')
//                         .and_then(|s| s.strip_suffix(".prompt"))
//                     {
//                         define_partial(registry, partial_name, &source);
//                     }
//                 } else {
//                     let prefix = sub_dir.to_str().unwrap_or("").replace('\\', "/");
//                     load_prompt(
//                         registry,
//                         source,
//                         prefix,
//                         file_name.to_string(),
//                         ns.to_string(),
//                     )
//                     .await?;
//                 }
//             }
//         }
//     }
//     Ok(())
// }

// #[cfg(feature = "dotprompt-private")]
// async fn load_prompt(
//     registry: &mut Registry,
//     source: String,
//     prefix: String,
//     filename: String,
//     ns: String,
// ) -> Result<()> {
//     let mut name = filename.strip_suffix(".prompt").unwrap().to_string();
//     let mut variant: Option<String> = None;
//     if let Some(dot_index) = name.rfind('.') {
//         variant = Some(name[dot_index + 1..].to_string());
//         name = name[..dot_index].to_string();
//     }

//     let parsed_prompt = registry.dotprompt.parse(&source)?;
//     let mut prompt_metadata = registry.dotprompt.render_metadata(&parsed_prompt).await?;
//     if let Some(v) = variant.clone() {
//         prompt_metadata.variant = Some(v);
//     }

//     let full_name = format!(
//         "{}/{}{}",
//         ns,
//         if !prefix.is_empty() {
//             format!("{}/", prefix)
//         } else {
//             "".to_string()
//         },
//         name
//     );

//     let config = PromptConfig {
//         name: full_name,
//         variant,
//         model: prompt_metadata.model,
//         config: prompt_metadata.config,
//         description: prompt_metadata.description,
//         input_schema: prompt_metadata.input.and_then(|i| i.schema),
//         output: prompt_metadata.output.map(|o| OutputOptions {
//             format: o.format,
//             schema: o.schema,
//             ..Default::default()
//         }),
//         messages: Some(parsed_prompt.messages),
//         tools: prompt_metadata.tools,
//         ..Default::default()
//     };

//     define_prompt::<Value, Value, Value>(registry, config);
//     Ok(())
// }

/// A placeholder function for `prompt()`, which would look up a defined prompt.
pub async fn prompt<I, O, C>(registry: &Registry, name: &str) -> Result<ExecutablePrompt<I, O, C>>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: for<'de> Deserialize<'de>
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + Clone
        + Default
        + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let action = registry
        .lookup_action(&format!("/prompt/{}", name))
        .await
        .ok_or_else(|| Error::new_internal(format!("Prompt '{}' not found", name)))?;
    let _ = action;
    unimplemented!(
        "Looking up prompts by name is not fully supported yet without a way to retrieve the original config from the registry."
    )
}
