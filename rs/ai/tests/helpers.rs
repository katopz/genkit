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

//! # Common Test Helpers for AI components
//!
//! This module provides shared utilities for testing, including mock models
//! and Registry setup. This mirrors the structure of `js/ai/tests/helpers.ts`.

use async_trait::async_trait;
use genkit_ai::{
    self, define_tool,
    model::{
        define_model, CandidateData, DefineModelOptions, FinishReason, GenerateRequest,
        GenerateResponseChunkData, GenerateResponseData, ModelInfoSupports,
    },
    MessageData, Part, Role, ToolConfig,
};
use genkit_core::{error::Result, plugin::Plugin, registry::Registry};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

////////////////////////////////////////////////////////////////////////////
// Programmable Model                                                     //
////////////////////////////////////////////////////////////////////////////

/// A type alias for the swappable handler inside our ProgrammableModel.
pub type ProgrammableModelHandler = Arc<
    Box<
        dyn Fn(
                GenerateRequest,
                Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
            ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
            + Send
            + Sync,
    >,
>;

/// A handle that tests can use to modify a model's behavior at runtime.
#[derive(Clone)]
pub struct ProgrammableModel {
    pub last_request: Arc<Mutex<Option<GenerateRequest>>>,
    pub handler: Arc<Mutex<ProgrammableModelHandler>>,
}

/// The Genkit Plugin that provides the programmable model for testing.
#[derive(Clone)]
pub struct ProgrammableModelPlugin {
    state: Arc<Mutex<Option<ProgrammableModel>>>,
    options: DefineModelOptions,
}

impl ProgrammableModelPlugin {
    pub fn new(options: Option<DefineModelOptions>) -> Self {
        let mut opts = options.unwrap_or_default();
        if opts.name.is_empty() {
            opts.name = "programmableModel".to_string();
        }
        Self {
            state: Arc::new(Mutex::new(None)),
            options: opts,
        }
    }

    pub fn get_handle(&self) -> ProgrammableModel {
        self.state
            .lock()
            .unwrap()
            .clone()
            .expect("Programmable model was not initialized by the plugin")
    }
}

#[async_trait]
impl Plugin for ProgrammableModelPlugin {
    fn name(&self) -> &'static str {
        "programmableModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let initial_handler: ProgrammableModelHandler = Arc::new(Box::new(|_req, _cb| {
            Box::pin(async { Ok(Default::default()) })
        }));

        let model_state = ProgrammableModel {
            last_request: Arc::new(Mutex::new(None)),
            handler: Arc::new(Mutex::new(initial_handler)),
        };

        *self.state.lock().unwrap() = Some(model_state.clone());

        let mut define_opts = self.options.clone();
        let mut supports = define_opts.supports.unwrap_or_default();
        if supports.tools.is_none() {
            supports.tools = Some(true);
        }
        if supports.system_role.is_none() {
            supports.system_role = Some(true);
        }
        define_opts.supports = Some(supports);

        define_model(registry, define_opts, move |req, streaming_callback| {
            let model_state = model_state.clone();
            async move {
                *model_state.last_request.lock().unwrap() = Some(req.clone());
                let handler = model_state.handler.lock().unwrap().clone();
                handler(req, streaming_callback).await
            }
        });

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////
// Echo Model                                                             //
////////////////////////////////////////////////////////////////////////////

/// A plugin that defines a model behaving like the TypeScript `echoModel`.
/// It concatenates all message history with commas and includes the config.
pub struct EchoModelPlugin {
    pub last_request: Arc<Mutex<Option<GenerateRequest>>>,
}

#[async_trait]
impl Plugin for EchoModelPlugin {
    fn name(&self) -> &'static str {
        "echoModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let last_request_clone = self.last_request.clone();
        define_model(
            registry,
            DefineModelOptions {
                name: "echoModel".to_string(),
                label: Some("TS-Compatible Echo Model".to_string()),
                supports: Some(ModelInfoSupports {
                    system_role: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
            move |req: GenerateRequest, streaming_callback| {
                let last_request_clone = last_request_clone.clone();
                async move {
                    *last_request_clone.lock().unwrap() = Some(req.clone());

                    if let Some(cb) = streaming_callback {
                        for i in (1..=3).rev() {
                            cb(GenerateResponseChunkData {
                                index: 0,
                                content: vec![Part::text(i.to_string())],
                                ..Default::default()
                            });
                        }
                    }

                    // 1. Collect all message parts into a vector of strings.
                    let all_text_parts: Vec<String> = req
                        .messages
                        .iter()
                        .map(|m| {
                            let prefix = match m.role {
                                Role::User | Role::Model => "".to_string(),
                                _ => format!("{}: ", m.role.to_string().to_lowercase()),
                            };
                            // In JS, joining parts of a message defaults to a comma.
                            let content = m
                                .content
                                .iter()
                                .map(|p| p.text.as_deref().unwrap_or(""))
                                .collect::<Vec<_>>()
                                .join(",");
                            format!("{}{}", prefix, content)
                        })
                        .collect();

                    // 2. Join all message strings with a comma, exactly like the TS version.
                    let concatenated_text = all_text_parts.join(",");

                    let config_str = serde_json::to_string(&req.config.unwrap_or_default())
                        .unwrap_or_else(|_| "null".to_string());

                    // 3. Construct the final response string.
                    let response_text = format!("Echo: {}", concatenated_text);
                    let config_text = format!("; config: {}", config_str);

                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            message: MessageData {
                                role: Role::Model,
                                content: vec![Part::text(response_text), Part::text(config_text)],
                                ..Default::default()
                            },
                            finish_reason: Some(FinishReason::Stop),
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                }
            },
        );
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////
// Test Fixture Setup Functions                                           //
////////////////////////////////////////////////////////////////////////////

/// Test fixture that provides a Registry with a TS-compatible `echoModel`.
/// Returns the Registry and a handle to inspect the last request sent to the model.
#[allow(unused)]
pub async fn registry_with_echo_model() -> (Arc<Registry>, Arc<Mutex<Option<GenerateRequest>>>) {
    let last_request = Arc::new(Mutex::new(None));
    let echo_plugin = EchoModelPlugin {
        last_request: last_request.clone(),
    };

    let mut registry = Registry::new();
    genkit_ai::configure_ai(&mut registry);
    registry.set_default_model("echoModel".to_string());
    echo_plugin.initialize(&mut registry).await.unwrap();

    (Arc::new(registry), last_request)
}

/// Test fixture that provides a Registry with the `programmableModel`.
/// Returns the Registry and a handle to program the model's behavior.
#[allow(unused)]
pub async fn registry_with_programmable_model_options(
    options: Option<DefineModelOptions>,
) -> (Arc<Registry>, ProgrammableModel) {
    let pm_plugin = ProgrammableModelPlugin::new(options);

    let mut registry = Registry::new();
    genkit_ai::configure_ai(&mut registry);
    let model_name = pm_plugin.options.name.clone();
    registry.set_default_model(model_name);
    pm_plugin.initialize(&mut registry).await.unwrap();

    let handle = pm_plugin.get_handle();
    (Arc::new(registry), handle)
}

/// Test fixture that provides a Registry with the `programmableModel`.
/// Returns the Registry and a handle to program the model's behavior.
#[allow(unused)]
pub async fn registry_with_programmable_model() -> (Arc<Registry>, ProgrammableModel) {
    registry_with_programmable_model_options(None).await
}

/// Test fixture that provides a Registry with an `echoModel` and `toolA`.
#[allow(unused)]
pub async fn registry_with_echo_model_and_tool(
) -> (Arc<Registry>, Arc<Mutex<Option<GenerateRequest>>>) {
    let (registry_arc, last_request) = registry_with_echo_model().await;
    let mut registry = (*registry_arc).clone();

    define_tool(
        &mut registry,
        ToolConfig {
            name: "toolA".to_string(),
            description: "toolA descr".to_string(),
            ..Default::default()
        },
        |_: (), _: genkit_ai::tool::ToolFnOptions| async { Ok("a".to_string()) },
    );

    (Arc::new(registry), last_request)
}
