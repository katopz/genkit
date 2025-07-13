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

use genkit_ai::prompt::{define_prompt, PromptConfig, PromptGenerateOptions};
use genkit_ai::session::Session;
use genkit_core::error::Result;
use genkit_core::Context;
use serde_json::Value;
use std::sync::Arc;

// Import test helpers
#[path = "../helpers.rs"]
pub mod helpers;
#[derive(Clone)]
pub struct TestCase {
    pub name: String,
    pub config: PromptConfig<Value, Value, Value>,
    pub input: Value,
    pub options: Option<PromptGenerateOptions<Value>>,
    pub state: Option<Value>,
    pub context: Option<Context>,
    pub want_text: String,
    pub want_rendered: Value,
}

pub async fn test_runner(case: Box<TestCase>) -> Result<()> {
    let (registry, _last_request) = helpers::registry_with_echo_model().await;
    let p = define_prompt(&registry, case.config);

    let session = if let Some(state) = case.state.clone() {
        let session = Session::new(registry.clone(), None, None, Some(state)).await?;
        Some(Arc::new(session))
    } else {
        None
    };

    // Test generation
    let p_clone = p.clone();
    let input_clone = case.input.clone();
    let options_clone = case.options.clone();
    let response = if let Some(session_arc) = session.clone() {
        session_arc
            .run(move || async move { p_clone.generate(input_clone, options_clone).await })
            .await?
    } else {
        let context = case.context.clone().unwrap_or_default();
        let fut = async move { p_clone.generate(input_clone, options_clone).await };
        genkit_core::context::run_with_context(context, fut).await?
    };

    assert_eq!(
        response.text()?,
        case.want_text,
        "Failed generate: {}",
        case.name
    );

    // Test render
    let p_clone_render = p.clone();
    let input_clone_render = case.input.clone();
    let options_clone_render = case.options.clone();
    let rendered = if let Some(session_arc) = session {
        session_arc
            .run(move || async move {
                p_clone_render
                    .render(input_clone_render, options_clone_render)
                    .await
            })
            .await?
    } else {
        let context = case.context.unwrap_or_default();
        let fut = async move {
            p_clone_render
                .render(input_clone_render, options_clone_render)
                .await
        };
        genkit_core::context::run_with_context(context, fut).await?
    };

    let rendered_json = serde_json::to_value(&rendered)?;
    assert_eq!(
        rendered_json, case.want_rendered,
        "Failed render: {}",
        case.name
    );

    Ok(())
}
