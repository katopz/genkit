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

mod helpers;

use genkit::{prompt::PromptConfig, Genkit, Part, Role};
use genkit_ai::MessageData;
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
/// 'renders dotprompt messages'
async fn test_renders_prompt_messages(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit.define_prompt::<TestInput, Value, Value>(PromptConfig {
        name: "hi_render_test".to_string(),
        prompt: Some("hi {{name}}".to_string()),
        ..Default::default()
    });

    let response = hi_prompt
        .render(
            TestInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    let expected_messages = Some(vec![MessageData {
        role: Role::User,
        content: vec![Part::text("hi Genkit")],
        ..Default::default()
    }]);

    assert_eq!(response.messages, expected_messages);
    assert!(response.model.is_none());
}
