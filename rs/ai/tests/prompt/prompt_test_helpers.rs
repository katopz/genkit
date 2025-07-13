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

    // Test generation in a separate block to manage lifetimes
    {
        let p_clone = p.clone();
        let input = case.input.clone();
        let options = case.options.clone();
        let context = case.context.clone().unwrap_or_default();

        let response = if let Some(session_arc) = session.clone() {
            let fut = async move { p_clone.generate(input, options).await };
            session_arc.run(fut).await?
        } else {
            let fut = async move { p_clone.generate(input, options).await };
            genkit_core::context::run_with_context(context, fut).await?
        };

        assert_eq!(
            response.text()?,
            case.want_text,
            "Failed generate: {}",
            case.name
        );
    }

    // Test render in a separate block
    {
        let p_clone = p.clone();
        let input = case.input.clone();
        let options = case.options.clone();
        let context = case.context.unwrap_or_default();

        let rendered = if let Some(session_arc) = session {
            let fut = async move { p_clone.render(input, options).await };
            session_arc.run(fut).await?
        } else {
            let fut = async move { p_clone.render(input, options).await };
            genkit_core::context::run_with_context(context, fut).await?
        };

        let rendered_json = serde_json::to_value(&rendered)?;
        assert_eq!(
            rendered_json, case.want_rendered,
            "Failed render: {}",
            case.name
        );
    }

    Ok(())
}
