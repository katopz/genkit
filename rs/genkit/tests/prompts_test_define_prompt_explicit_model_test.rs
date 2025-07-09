//! Copyright 2024 Google LLC
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

mod helpers;

use genkit::{model::Model, prompt::PromptConfig, Genkit};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
struct TestInput {
    name: String,
}

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
}

#[rstest]
#[tokio::test]
/// 'calls dotprompt with explicit model'
async fn test_calls_prompt_with_explicit_model(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_explicit_model_test".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            prompt: Some("hi {{name}}".to_string()),
            ..Default::default()
        })
        .await;

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi Genkit; config: {}");
}
