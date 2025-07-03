use async_trait::async_trait;
use genkit::{error::Result, registry::Registry};
use genkit::{
    model::{Candidate, FinishReason, Message, Part, Role},
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

// In TS, the StreamingCallback is a simple (chunk) => void function.
// We're defining the type for the callback here.
pub type StreamingCallback = Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>;

// The handler for a programmable model. It's an async function.
pub type ResponseHandlerFn = Box<
    dyn Fn(
            GenerateRequest,
            Option<StreamingCallback>,
        ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send
        + Sync,
>;

#[derive(Clone)]
#[allow(unused)]
pub struct ProgrammableModel {
    pub last_request: Arc<Mutex<Option<GenerateRequest>>>,
    pub handler: Arc<Mutex<Arc<ResponseHandlerFn>>>,
}

impl Default for ProgrammableModel {
    fn default() -> Self {
        Self {
            last_request: Default::default(),
            handler: Arc::new(Mutex::new(Arc::new(Box::new(|_req, _cb| {
                Box::pin(async { Ok(Default::default()) })
            })))),
        }
    }
}

#[allow(unused)]
pub fn define_programmable_model(registry: &mut Registry) -> ProgrammableModel {
    let state = ProgrammableModel::default();

    let model_state = state.clone();
    let model_opts = DefineModelOptions {
        name: "programmableModel".to_string(),
        supports: Some(ModelInfoSupports {
            tools: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };

    define_model(registry, model_opts, move |req, streaming_callback| {
        let model_state = model_state.clone();
        async move {
            *model_state.last_request.lock().unwrap() = Some(req.clone());
            // This is tricky. The streaming_callback from define_model is an Option<impl Fn(...)>.
            // We can't easily store it or pass it to our Box<dyn Fn(...)>.
            // We need to wrap it.
            let cb_wrapper: Option<StreamingCallback> =
                streaming_callback.map(|cb| Box::new(cb) as StreamingCallback);

            let handler = model_state.handler.lock().unwrap().clone();
            handler(req, cb_wrapper).await
        }
    });

    state
}

/// A plugin for defining the programmable model for testing.
#[derive(Clone)]
pub struct ProgrammableModelPlugin {
    // This state allows the test to get a handle to the model's state
    // after it has been initialized through the plugin system.
    pub state: Arc<Mutex<Option<ProgrammableModel>>>,
}

impl Default for ProgrammableModelPlugin {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(None)),
        }
    }
}

#[allow(unused)]
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

#[async_trait]
impl Plugin for ProgrammableModelPlugin {
    fn name(&self) -> &'static str {
        "programmableModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let pm = define_programmable_model(registry);
        *self.state.lock().unwrap() = Some(pm);
        Ok(())
    }
}

#[allow(unused)]
pub async fn genkit_with_programmable_model() -> (Arc<Genkit>, ProgrammableModel) {
    let pm_plugin = Arc::new(ProgrammableModelPlugin::new());
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![pm_plugin.clone() as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();
    let handle = pm_plugin.get_handle();
    (genkit, handle)
}
