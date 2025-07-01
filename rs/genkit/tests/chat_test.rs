#[path = "helpers.rs"]
mod helpers;

use genkit::{error::Result, Genkit};
use genkit_ai::{
    generate::{generate, GenerateOptions},
    model, prompt,
};
use rstest::*;
use serde_json::Value;
use std::sync::Arc;

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
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
