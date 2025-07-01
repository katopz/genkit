use async_trait::async_trait;
use genkit::{error::Result, plugin::Plugin, registry::Registry, Genkit, GenkitOptions};
use genkit_ai::{
    generate::{generate, GenerateOptions},
    model::{
        self, define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
    },
    prompt,
};
use serde_json::Value;
use std::sync::Arc;

/// A simple plugin to define our echo model for testing.
struct EchoModelPlugin;

#[async_trait]
impl Plugin for EchoModelPlugin {
    fn name(&self) -> &'static str {
        "echoModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        // Define a simple model that just echoes back the last message.
        define_model(
            registry,
            model::DefineModelOptions {
                name: "echoModel".to_string(),
                label: Some("Echo Model".to_string()),
                ..Default::default()
            },
            |req: GenerateRequest, _streaming_callback| async move {
                let last_message = req.messages.last().cloned().unwrap_or_default();
                let response = GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        message: last_message,
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    }],
                    ..Default::default()
                };
                Ok(response)
            },
        );
        Ok(())
    }
}

#[tokio::test]
async fn test_genkit_initialization_and_generate() -> Result<()> {
    // 1. Initialize Genkit with the EchoModelPlugin and a default model.
    let echo_plugin = Arc::new(EchoModelPlugin);
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![echo_plugin],
        default_model: Some("echoModel".to_string()),
    })
    .await?;

    // 2. Call `generate` with the "echoModel" specified explicitly (still works).
    let response = generate::<Value>(
        genkit.registry(),
        GenerateOptions {
            model: Some(model::Model::Name("echoModel".to_string())),
            prompt: Some(vec![genkit_ai::document::Part::text("hello explicit")]),
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(response.text()?, "hello explicit");

    // 3. Call `generate` *without* a model, relying on the default.
    let response_default = generate::<Value>(
        genkit.registry(),
        GenerateOptions {
            prompt: Some(vec![genkit_ai::document::Part::text("hello default")]),
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(response_default.text()?, "hello default");

    // 4. Define and use a prompt that relies on the default model.
    let my_prompt = prompt::define_prompt::<(), Value, ()>(
        &mut genkit.registry().clone(),
        prompt::PromptConfig {
            name: "myPrompt".to_string(),
            // No model specified here, should use default.
            model: None,
            prompt: Some("This is a test prompt".to_string()),
            ..Default::default()
        },
    );

    // Generate content using the executable prompt.
    let response_from_prompt = my_prompt.generate((), None).await?;
    assert_eq!(response_from_prompt.text()?, "This is a test prompt");

    Ok(())
}
