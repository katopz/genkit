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

use genkit::{
    config::Config,
    genkit,
    model::{
        define_model, generate, render_prompt, run_prompt, CandidateData, DefineModelOptions,
        GenerateRequest, GenerateResponseData, MessageData, ModelRef, OutputConfig, Part, Role,
    },
    prompt::{define_prompt, prompt, PromptData},
    registry::Registry,
    Genkit,
};
use genkit_core::error::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

// Test helpers module
mod helpers {
    use super::*;

    // Defines a simple model that echoes input.
    pub fn define_echo_model(registry: &mut Registry) {
        let model_action = define_model(
            DefineModelOptions {
                name: "echoModel".to_string(),
                ..Default::default()
            },
            move |req: GenerateRequest, _| {
                Box::pin(async move {
                    let mut text_to_echo = Vec::new();
                    for m in req.messages.iter() {
                        let content_text = m
                            .content
                            .iter()
                            .filter_map(|p| p.text.as_ref())
                            .map(|s| s.as_str())
                            .collect::<Vec<&str>>()
                            .join(",");
                        text_to_echo.push(content_text);
                    }

                    let config_str = serde_json::to_string(&req.config).unwrap_or("{}".to_string());

                    let response_message = MessageData {
                        role: Role::Model,
                        content: vec![
                            Part::text(format!("Echo: {}", text_to_echo.join(","))),
                            Part::text(format!("; config: {}", config_str)),
                        ],
                        ..Default::default()
                    };

                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            message: response_message,
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                })
            },
        );
        let action_name = model_action.name().to_string();
        registry.register_action(Arc::new(model_action)).unwrap();
    }

    pub fn define_static_response_model(registry: &mut Registry, response_text: &'static str) {
        let model_action = define_model(
            DefineModelOptions {
                name: "staticResponseModel".to_string(),
                ..Default::default()
            },
            move |_: GenerateRequest, _| {
                let response_message = MessageData {
                    role: Role::Model,
                    content: vec![Part::text(response_text)],
                    ..Default::default()
                };
                Box::pin(async {
                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            message: response_message,
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                })
            },
        );
        let action_name = model_action.name().to_string();
        registry.register_action(Arc::new(model_action)).unwrap();
    }
}

fn setup_default() -> Genkit {
    let mut ai = genkit(Default::default());
    ai.config_mut().model = Some("echoModel".to_string());
    helpers::define_echo_model(&mut ai.registry());
    ai
}

fn setup_for_files() -> Genkit {
    let mut config = Config::default();
    config.prompt_dir = Some("tests/prompts".to_string());
    let mut ai = genkit(config);
    helpers::define_echo_model(&mut ai.registry());
    helpers::define_static_response_model(&mut ai.registry(), "```json\n{\"bar\": \"baz\"}\n```");
    ai
}

#[derive(Deserialize, Debug, PartialEq)]
struct NameInput {
    name: String,
}

#[tokio::test]
async fn calls_dotprompt_with_default_model() -> Result<()> {
    let ai = setup_default();
    let hi_prompt = define_prompt(
        PromptData::new("hi").with_template("hi {{ name }}"),
        |input: NameInput| async move { Ok(json!({"name": input.name})) },
    )?;

    let response = run_prompt(
        &hi_prompt,
        &ai,
        NameInput {
            name: "Genkit".into(),
        },
    )
    .await?;
    assert_eq!(response.text()?, "Echo: hi Genkit; config: {}");
    Ok(())
}

#[tokio::test]
async fn calls_dotprompt_with_config() -> Result<()> {
    let ai = setup_default();
    let mut prompt_data = PromptData::new("hi").with_template("hi {{ name }}");
    let mut config = HashMap::new();
    config.insert("temperature".to_string(), json!(11));
    prompt_data.config = Some(config);

    let hi_prompt = define_prompt(prompt_data, |input: NameInput| async move {
        Ok(json!({"name": input.name}))
    })?;

    let response = run_prompt(
        &hi_prompt,
        &ai,
        NameInput {
            name: "Genkit".into(),
        },
    )
    .await?;
    assert_eq!(
        response.text()?,
        "Echo: hi Genkit; config: {\"temperature\":11}"
    );
    Ok(())
}

#[derive(Deserialize, Debug, PartialEq)]
struct FooOutput {
    bar: String,
}

#[tokio::test]
async fn infers_output_schema() -> Result<()> {
    let mut ai = genkit(Default::default());
    helpers::define_static_response_model(&mut ai.registry(), "```json\n{\"bar\": \"baz\"}\n```");

    let mut prompt_data = PromptData::new("hi").with_template("hi {{ name }}");
    prompt_data.model = Some("staticResponseModel".into());

    let mut output_config = OutputConfig::new_json();
    output_config.schema = Some(json!({
        "type": "object",
        "properties": { "bar": { "type": "string" } },
        "required": ["bar"]
    }));
    prompt_data.output = Some(output_config);

    let hi_prompt = define_prompt(prompt_data, |input: NameInput| async move {
        Ok(json!({"name": input.name}))
    })?;

    let response = run_prompt(
        &hi_prompt,
        &ai,
        NameInput {
            name: "Genkit".into(),
        },
    )
    .await?;
    let output: FooOutput = serde_json::from_value(response.output.unwrap())?;

    assert_eq!(output, FooOutput { bar: "baz".into() });
    Ok(())
}

#[tokio::test]
async fn renders_dotprompt_messages() -> Result<()> {
    let ai = setup_default();
    let hi_prompt = define_prompt(
        PromptData::new("hi").with_template("hi {{ name }}"),
        |input: NameInput| async move { Ok(json!({"name": input.name})) },
    )?;

    let response = render_prompt(
        &hi_prompt,
        &ai,
        NameInput {
            name: "Genkit".into(),
        },
    )
    .await?;

    let expected_messages = vec![MessageData {
        role: Role::User,
        content: vec![Part::text("hi Genkit")],
        ..Default::default()
    }];
    assert_eq!(response.messages, Some(expected_messages));
    assert!(response.config.is_none());
    Ok(())
}

// Tests for file-based prompts
#[tokio::test]
async fn loads_prompt_from_the_folder() -> Result<()> {
    let ai = setup_for_files();
    let test_prompt = prompt(&ai, "test")?;

    let response = generate(&ai, test_prompt.request().clone()).await?;

    assert_eq!(
        response.text()?,
        "Echo: Hello from the prompt file; config: {\"temperature\":11}"
    );

    let rendered = test_prompt.render_request(json!({}))?.unwrap();
    assert_eq!(
        rendered.messages.unwrap()[0].content[0].text.as_deref(),
        Some("Hello from the prompt file")
    );
    assert_eq!(
        rendered.config.unwrap().get("temperature").unwrap(),
        &json!(11)
    );

    Ok(())
}

#[tokio::test]
async fn loads_from_from_the_sub_folder() -> Result<()> {
    let ai = setup_for_files();
    let test_prompt = prompt(&ai, "sub/test")?;

    let response = generate(&ai, test_prompt.request().clone()).await?;

    assert_eq!(
        response.text()?,
        "Echo: Hello from the sub folder prompt file; config: {\"temperature\":12}"
    );
    Ok(())
}

#[tokio::test]
async fn loads_from_from_the_folder_with_all_the_options() -> Result<()> {
    let ai = setup_for_files();
    let test_prompt = prompt(&ai, "kitchensink")?;

    let request = test_prompt
        .render_request(json!({"subject": "banana", "history": ""}))?
        .unwrap();

    assert_eq!(
        request.model.unwrap(),
        ModelRef::from("googleai/gemini-5.0-ultimate-pro-plus")
    );
    assert_eq!(
        request.config.as_ref().unwrap().get("temperature"),
        Some(&json!(11))
    );
    assert_eq!(request.max_tool_turns, Some(77));
    assert_eq!(request.tools.as_ref().unwrap().len(), 2);
    assert_eq!(
        request.messages.as_ref().unwrap()[0],
        MessageData {
            role: Role::System,
            content: vec![Part::text(" Hello ")],
            ..Default::default()
        }
    );
    assert_eq!(
        request.messages.as_ref().unwrap()[1],
        MessageData {
            role: Role::Model,
            content: vec![Part::text(" from the prompt file banana")],
            ..Default::default()
        }
    );
    Ok(())
}

#[tokio::test]
async fn loads_a_variant_from_the_folder() -> Result<()> {
    let ai = setup_for_files();
    let test_prompt = prompt(&ai, "test.variant")?;

    let response = generate(&ai, test_prompt.request().clone()).await?;

    assert_eq!(
        response.text()?,
        "Echo: Hello from a variant of the hello prompt; config: {\"temperature\":13}"
    );
    Ok(())
}

#[tokio::test]
async fn loads_prompt_with_output_schema() -> Result<()> {
    let ai = setup_for_files();
    let test_prompt = prompt(&ai, "output")?;

    let request = test_prompt
        .render_request(json!({"name": "Genkit"}))?
        .unwrap();

    let response = generate(&ai, request).await?;
    let output: FooOutput = serde_json::from_value(response.output.unwrap())?;

    assert_eq!(output, FooOutput { bar: "baz".into() });
    Ok(())
}
