use async_trait::async_trait;
use genkit::{error::Result, registry::Registry};
use genkit::{
    model::{Candidate, FinishReason, Part, Role},
    plugin::Plugin,
    Genkit, GenkitOptions,
};
use genkit_ai::{
    define_model,
    model::{DefineModelOptions, GenerateRequest, GenerateResponseData, ModelInfoSupports},
    GenerateResponseChunkData,
};
use genkit_ai::{model::CandidateData, MessageData};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};

#[allow(unused)]
pub async fn run_async<F, O>(f: F) -> O
where
    F: FnOnce() -> O,
{
    sleep(Duration::from_millis(0)).await;
    f()
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
        let last_msg = req.messages.last().cloned().unwrap_or_default();

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

        // Filter out instructional parts before creating the echo text.
        let filtered_text = last_msg
            .content
            .iter()
            .filter(|p| {
                p.metadata.as_ref().map_or_else(
                    || true,
                    |meta| {
                        meta.get("purpose")
                            != Some(&serde_json::Value::String("output".to_string()))
                    },
                )
            })
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<&str>>()
            .join("");

        // Both streaming and non-streaming calls return a final response.
        let text = format!("Echo: {}", filtered_text);
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

// A type alias for the swappable handler inside our ProgrammableModel.
pub type ProgrammableModelHandler = Arc<
    Box<
        dyn Fn(
                GenerateRequest,
                // The handler receives an Option of a boxed, dynamic callback function.
                Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
            ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
            + Send
            + Sync,
    >,
>;

// In TS, the StreamingCallback is a simple (chunk) => void function.
// We're defining the type for the callback here.
#[allow(unused)]
pub type StreamingCallback = Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>;

// The handler for a programmable model. It's an async function.
#[allow(unused)]
pub type ResponseHandlerFn = Box<
    dyn Fn(
            GenerateRequest,
            Option<StreamingCallback>,
        ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send
        + Sync,
>;

/// A handle that tests can use to modify the model's behavior at runtime.
#[derive(Clone)]
pub struct ProgrammableModel {
    pub last_request: Arc<Mutex<Option<GenerateRequest>>>,
    pub handler: Arc<Mutex<ProgrammableModelHandler>>,
}

/// The Genkit Plugin that provides the programmable model for testing.
#[derive(Clone, Default)]
pub struct ProgrammableModelPlugin {
    state: Arc<Mutex<Option<ProgrammableModel>>>,
}

impl ProgrammableModelPlugin {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_handle(&self) -> ProgrammableModel {
        self.state
            .lock()
            .unwrap()
            .clone()
            .expect("Programmable model was not initialized by the plugin")
    }
}

#[async_trait]
impl Plugin for ProgrammableModelPlugin {
    fn name(&self) -> &'static str {
        "programmableModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let initial_handler: ProgrammableModelHandler = Arc::new(Box::new(|_req, _cb| {
            Box::pin(async { Ok(Default::default()) })
        }));

        let model_state = ProgrammableModel {
            last_request: Arc::new(Mutex::new(None)),
            handler: Arc::new(Mutex::new(initial_handler)),
        };

        *self.state.lock().unwrap() = Some(model_state.clone());

        define_model(
            registry,
            DefineModelOptions {
                name: "programmableModel".to_string(),
                supports: Some(ModelInfoSupports {
                    tools: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
            move |req, streaming_callback| {
                let model_state = model_state.clone();
                async move {
                    *model_state.last_request.lock().unwrap() = Some(req.clone());

                    let cb_wrapper: Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>> =
                        streaming_callback.map(|cb| Box::new(cb) as Box<_>);

                    let handler = model_state.handler.lock().unwrap().clone();
                    handler(req, cb_wrapper).await
                }
            },
        );

        Ok(())
    }
}

/// A simple plugin that provides an `echoModel` for testing.
pub struct EchoModelPlugin;

#[async_trait]
impl Plugin for EchoModelPlugin {
    fn name(&self) -> &'static str {
        "echoModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        define_model(
            registry,
            DefineModelOptions {
                name: "echoModel".to_string(),
                ..Default::default()
            },
            |req: GenerateRequest, _cb| async move {
                let config_str = if let Some(config) = req.config.as_ref().filter(|c| !c.is_null())
                {
                    serde_json::to_string(config)
                        .map_err(|e| genkit::error::Error::new_internal(e.to_string()))?
                } else {
                    "{}".to_string()
                };

                let all_text = req
                    .messages
                    .iter()
                    .flat_map(|m| &m.content)
                    .filter_map(|p| p.text.as_deref())
                    .collect::<Vec<_>>()
                    .join("");

                let text = format!("Echo: {}; config: {}", all_text, config_str);

                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        finish_reason: Some(FinishReason::Stop),
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text(text)],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
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

#[allow(unused)]
pub async fn genkit_with_programmable_model() -> (Arc<Genkit>, ProgrammableModel) {
    let pm_plugin = Arc::new(ProgrammableModelPlugin::new());
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![pm_plugin.clone() as Arc<dyn Plugin>],
        default_model: Some("programmableModel".to_string()),
        ..Default::default()
    })
    .await
    .unwrap();
    let handle = pm_plugin.get_handle();
    (genkit, handle)
}

// A plugin that defines a model behaving like the TypeScript `echoModel` for testing.
// This is defined locally in the test to allow for inspecting the last request.
struct TsEchoModelPlugin {
    last_request: Arc<Mutex<Option<GenerateRequest>>>,
}

#[async_trait]
impl Plugin for TsEchoModelPlugin {
    fn name(&self) -> &'static str {
        "tsEchoModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let last_request_clone = self.last_request.clone();
        define_model(
            registry,
            DefineModelOptions {
                name: "echoModel".to_string(),
                label: Some("TS-Compatible Echo Model".to_string()),
                supports: Some(ModelInfoSupports {
                    system_role: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
            move |req: GenerateRequest, streaming_callback| {
                let last_request_clone = last_request_clone.clone();
                async move {
                    // Store the last request for inspection.
                    let mut last_req = last_request_clone.lock().unwrap();
                    *last_req = Some(req.clone());

                    // Handle streaming by sending down a countdown.
                    if let Some(cb) = streaming_callback {
                        for i in (1..=3).rev() {
                            cb(GenerateResponseChunkData {
                                index: 0,
                                content: vec![genkit::model::Part::text(i.to_string())],
                                ..Default::default()
                            });
                        }
                    }

                    // Construct the response text by concatenating messages, similar to the TS version.
                    let concatenated_messages = req
                        .messages
                        .iter()
                        .map(|m| {
                            let prefix = match m.role {
                                Role::User | Role::Model => "".to_string(),
                                _ => format!("{}: ", m.role.to_string().to_lowercase()),
                            };
                            let content = m
                                .content
                                .iter()
                                .filter_map(|p| p.text.as_deref())
                                .collect::<Vec<_>>()
                                .join(",");
                            format!("{}{}", prefix, content)
                        })
                        .collect::<Vec<_>>()
                        .join(",");

                    let config_str = match &req.config {
                        Some(c) if !c.is_null() => {
                            serde_json::to_string(c).unwrap_or_else(|_| "{}".to_string())
                        }
                        _ => "{}".to_string(),
                    };

                    let response_text_part = Part::text(format!("Echo: {}", concatenated_messages));

                    let config_part = Part::text(format!("; config: {}", config_str));

                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            message: MessageData {
                                role: Role::Model,
                                content: vec![response_text_part, config_part],
                                ..Default::default()
                            },
                            finish_reason: Some(FinishReason::Stop),
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                }
            },
        );
        Ok(())
    }
}

#[allow(unused)]
pub async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    let last_request = Arc::new(Mutex::new(None));
    let echo_plugin = Arc::new(TsEchoModelPlugin {
        last_request: last_request.clone(),
    });

    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![echo_plugin as Arc<dyn Plugin>],
        default_model: Some("echoModel".to_string()),
        ..Default::default()
    })
    .await
    .unwrap();
    (genkit, last_request)
}
