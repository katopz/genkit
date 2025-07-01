use genkit::{error::Result, registry::Registry};
use genkit_ai::{
    define_model,
    model::{DefineModelOptions, GenerateRequest, GenerateResponseData, ModelInfoSupports},
    GenerateResponseChunkData,
};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};

pub async fn run_async<F, O>(f: F) -> O
where
    F: FnOnce() -> O,
{
    sleep(Duration::from_millis(0)).await;
    f()
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
pub struct ProgrammableModel {
    pub last_request: Arc<Mutex<Option<GenerateRequest>>>,
    pub handler: Arc<Mutex<ResponseHandlerFn>>,
}

impl Default for ProgrammableModel {
    fn default() -> Self {
        Self {
            last_request: Default::default(),
            handler: Arc::new(Mutex::new(Box::new(|_req, _cb| {
                Box::pin(async { Ok(Default::default()) })
            }))),
        }
    }
}

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

            let fut = {
                let handler = model_state.handler.lock().unwrap();
                handler(req, cb_wrapper)
            };
            fut.await
        }
    });

    state
}
