use async_trait::async_trait;
use genkit::{
    error::Result,
    model::{Candidate, FinishReason, Message, Part, Role},
    plugin::Plugin,
    registry::Registry,
    Genkit, GenkitOptions,
};
use genkit_ai::{
    define_model,
    model::{
        CandidateData, DefineModelOptions, GenerateRequest, GenerateResponseData, ModelInfoSupports,
    },
    GenerateResponseChunkData, MessageData,
};
use std::sync::Arc;

/// A simple plugin to define our echo model for testing.
pub struct EchoModelPlugin;

#[async_trait]
impl Plugin for EchoModelPlugin {
    fn name(&self) -> &'static str {
        "echoModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        // Define a simple model that just echoes back the last message.
        define_model(
            registry,
            DefineModelOptions {
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

#[allow(unused)]
pub async fn genkit_instance_with_echo_model() -> Arc<Genkit> {
    let echo_plugin = Arc::new(EchoModelPlugin);
    Genkit::init(GenkitOptions {
        plugins: vec![echo_plugin],
        default_model: Some("echoModel".to_string()),
        context: None,
    })
    .await
    .unwrap()
}

// Helper to define an echo model for testing.
#[allow(unused)]
pub fn define_echo_model(registry: &mut Registry, constrained_support: &str) {
    let supports = if constrained_support == "all" {
        Some(ModelInfoSupports {
            output: Some(vec!["banana".to_string()]),
            ..Default::default()
        })
    } else {
        Some(ModelInfoSupports {
            output: Some(vec![]),
            ..Default::default()
        })
    };
    let model_opts = DefineModelOptions {
        name: "echoModel".to_string(),
        label: Some("Echo Model".to_string()),
        supports,
        ..Default::default()
    };
    define_model(registry, model_opts, |req, streaming_callback| async move {
        let last_msg_text = req.messages.last().cloned().unwrap_or_default();

        // If a streaming callback is provided, we send down the countdown chunks.
        if let Some(cb) = streaming_callback {
            let chunks_data = vec![
                GenerateResponseChunkData {
                    index: 0,
                    content: vec![genkit::model::Part::text("3")],
                    ..Default::default()
                },
                GenerateResponseChunkData {
                    index: 0,
                    content: vec![genkit::model::Part::text("2")],
                    ..Default::default()
                },
                GenerateResponseChunkData {
                    index: 0,
                    content: vec![genkit::model::Part::text("1")],
                    ..Default::default()
                },
            ];

            for data in chunks_data {
                cb(data);
            }
        }

        // Both streaming and non-streaming calls return a final response.
        let text = format!(
            "Echo: {}",
            Message::<String>::new(last_msg_text, None).text()
        );
        Ok(GenerateResponseData {
            candidates: vec![Candidate {
                index: 0,
                finish_reason: Some(FinishReason::Stop),
                message: MessageData {
                    role: Role::Model,
                    content: vec![Part::text(text)],
                    metadata: None,
                },
                ..Default::default()
            }],
            ..Default::default()
        })
    });
}
