use async_trait::async_trait;
use genkit::{error::Result, plugin::Plugin, registry::Registry, Genkit, GenkitOptions};
use genkit_ai::{
    generate::{generate, GenerateOptions},
    model::{
        self, define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
    },
    prompt,
};
use rstest::*;
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

/// This fixture function acts as the "beforeEach" block.
/// It will be run for each test that uses the `genkit_instance` argument.
#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    let echo_plugin = Arc::new(EchoModelPlugin);
    Genkit::init(GenkitOptions {
        plugins: vec![echo_plugin],
        default_model: Some("echoModel".to_string()),
        context: None,
    })
    .await
    .unwrap()
}

#[rstest]
#[tokio::test]
async fn test_generate_with_explicit_model(#[future] genkit_instance: Arc<Genkit>) -> Result<()> {
    let genkit = genkit_instance.await;
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
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_generate_with_default_model(#[future] genkit_instance: Arc<Genkit>) -> Result<()> {
    let genkit = genkit_instance.await;
    let response = generate::<Value>(
        genkit.registry(),
        GenerateOptions {
            // No model specified, relying on the default from the fixture.
            prompt: Some(vec![genkit_ai::document::Part::text("hello default")]),
            ..Default::default()
        },
    )
    .await?;
    assert_eq!(response.text()?, "hello default");
    Ok(())
}

#[rstest]
#[tokio::test]
async fn test_prompt_with_default_model(#[future] genkit_instance: Arc<Genkit>) -> Result<()> {
    let genkit = genkit_instance.await;
    // We need a mutable reference to the registry for define_prompt.
    // Since the Arc gives us shared ownership, we can clone it.
    let mut registry = genkit.registry().clone();

    let my_prompt = prompt::define_prompt::<(), Value, ()>(
        &mut registry,
        prompt::PromptConfig {
            name: "myPrompt".to_string(),
            // No model specified here, should use default from the registry.
            model: None,
            prompt: Some("This is a test prompt".to_string()),
            ..Default::default()
        },
    );

    let response = my_prompt.generate((), None).await?;
    assert_eq!(response.text()?, "This is a test prompt");
    Ok(())
}
