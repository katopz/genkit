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
use serde_json::Value;
use std::sync::Arc;

// Import test helpers
#[path = "../helpers.rs"]
pub mod helpers;

pub struct TestCase {
    pub name: String,
    pub config: PromptConfig<Value, Value, Value>,
    pub input: Value,
    pub options: Option<PromptGenerateOptions<Value>>,
    pub state: Option<Value>,
    pub want_text: String,
    pub want_rendered: Value,
}

pub async fn test_runner(case: TestCase) -> Result<()> {
    let (registry, _last_request) = helpers::registry_with_echo_model().await;
    let mut mut_registry = (*registry).clone();

    let p = define_prompt(&mut mut_registry, case.config);

    let session = Session::new(registry.clone(), None, None, case.state.clone()).await?;
    let session_arc = Arc::new(session);

    // Test generation
    let p_gen = p.clone();
    let input_gen = case.input.clone();
    let options_gen = case.options.clone();
    let response = session_arc
        .clone()
        .run(async move { p_gen.generate(input_gen, options_gen).await })
        .await?;
    assert_eq!(
        response.text()?,
        case.want_text,
        "Failed generate: {}",
        case.name
    );

    // Test render
    let p_render = p.clone();
    let input_render = case.input.clone();
    let options_render = case.options.clone();
    let rendered = session_arc
        .run(async move { p_render.render(input_render, options_render).await })
        .await?;

    let rendered_json = serde_json::to_value(&rendered)?;
    assert_eq!(
        rendered_json, case.want_rendered,
        "Failed render: {}",
        case.name
    );

    Ok(())
}
